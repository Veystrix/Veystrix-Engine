#include "io_pager_async.hpp"

#include <dispatch/dispatch.h>
#include <errno.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace veystrix {
namespace {

// The single C1 completion ready-queue: exactly one mutex and one condition
// variable for the whole process. No additional sync object is introduced.
struct AsyncState {
  std::atomic<int> next_ticket{1};        // atomic ticket allocator
  std::atomic<int> last_completed{0};     // "the atomic ticket" published by notify
  std::mutex mu;                          // C1 mutex #1 (only one)
  std::condition_variable cv;             // C1 condition variable #1 (only one)
  struct Completion {
    bool done = false;
    int errors = 0;
  };
  std::unordered_map<int, Completion> completions;
};

AsyncState& async_state() {
  static AsyncState s;
  return s;
}

// Per-batch state shared (by shared_ptr) with every dispatch block, so the
// caller's `reads` vector and callback never dangle when submit returns.
struct Batch {
  std::vector<PagerRead> reads;
  std::function<void(int ticket)> on_done;
  std::atomic<int> next{0};
  std::atomic<int> errors{0};
  int ticket = kInvalidTicket;
};

// C1 hard rule: user-interactive queue, width 8. The caller's width is only a
// request for fewer lanes; it can never raise the worker count past 8.
// VEYSTRIX_CPP_ASYNC_WIDTH raises the cap (measurement knob, max 64).
int async_width_cap() {
  static const int cap = [] {
    const char* e = std::getenv("VEYSTRIX_CPP_ASYNC_WIDTH");
    const int v = e ? std::atoi(e) : 8;
    return v >= 1 && v <= 64 ? v : 8;
  }();
  return cap;
}

}  // namespace

AsyncTicket submit_batched_reads_async(
    const std::vector<PagerRead>& reads, int width,
    std::function<void(int ticket)> on_done) {
  AsyncState& st = async_state();
  const int ticket = st.next_ticket.fetch_add(1, std::memory_order_relaxed);

  auto batch = std::make_shared<Batch>();
  batch->reads = reads;                 // copy: caller may destroy its vector
  batch->on_done = std::move(on_done);
  batch->ticket = ticket;

  if (width < 1) width = 1;
  if (width > async_width_cap()) width = async_width_cap();

  dispatch_group_t group = dispatch_group_create();
  dispatch_queue_t queue =
      dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

  for (int lane = 0; lane < width; ++lane) {
    dispatch_group_async(group, queue, ^{
      for (;;) {
        const int index = batch->next.fetch_add(1, std::memory_order_relaxed);
        if (index >= static_cast<int>(batch->reads.size())) break;
        const PagerRead& r = batch->reads[static_cast<std::size_t>(index)];
        char* dst = reinterpret_cast<char*>(static_cast<std::uintptr_t>(r.dst));
        std::int64_t done = 0;
        while (done < r.length) {
          ssize_t got = ::pread(r.fd, dst + done,
                                static_cast<std::size_t>(r.length - done),
                                static_cast<off_t>(r.offset + done));
          if (got > 0) {
            done += got;
          } else if (got < 0 && errno == EINTR) {
            continue;
          } else {
            batch->errors.fetch_add(1, std::memory_order_relaxed);
            break;
          }
        }
      }
    });
  }

  dispatch_group_notify(group, queue, ^{
    const int errors = batch->errors.load(std::memory_order_relaxed);
    // The notify block runs only after every lane block has finished, so all
    // destination bytes are written here. Fire the callback FIRST: a driver
    // that wakes from wait_ticket is then guaranteed the callback already
    // ran ("ticket fires once" is observable at completion time).
    if (batch->on_done) batch->on_done(ticket);
    // Publish completion under the one C1 mutex, then wake any driver waiting
    // on the one C1 CV. The mutex lock/unlock is the release edge; wait_ticket's
    // lock acquisition is the matching acquire edge, so bytes written by the
    // lane blocks are visible before the driver builds any graph.
    {
      std::lock_guard<std::mutex> lk(st.mu);
      AsyncState::Completion& c = st.completions[ticket];
      c.errors = errors;
      c.done = true;
    }
    st.last_completed.store(ticket, std::memory_order_release);  // atomic ticket
    st.cv.notify_all();
  });

  dispatch_release(group);
  return ticket;
}

bool wait_ticket(AsyncTicket ticket) {
  AsyncState& st = async_state();
  std::unique_lock<std::mutex> lk(st.mu);
  st.cv.wait(lk, [&st, ticket]() {
    auto it = st.completions.find(ticket);
    return it != st.completions.end() && it->second.done;
  });
  return st.completions[ticket].errors == 0;
}

bool wait_ticket_for(AsyncTicket ticket, long timeout_us, bool* ok) {
  AsyncState& st = async_state();
  std::unique_lock<std::mutex> lk(st.mu);
  const bool done = st.cv.wait_for(lk, std::chrono::microseconds(timeout_us), [&st, ticket]() {
    auto it = st.completions.find(ticket);
    return it != st.completions.end() && it->second.done;
  });
  if (done && ok) *ok = st.completions[ticket].errors == 0;
  return done;
}

void release_ticket(AsyncTicket ticket) {
  AsyncState& st = async_state();
  std::lock_guard<std::mutex> lk(st.mu);
  auto it = st.completions.find(ticket);
  if (it != st.completions.end() && it->second.done) st.completions.erase(it);
}

bool ticket_complete(AsyncTicket ticket) {
  AsyncState& st = async_state();
  std::lock_guard<std::mutex> lk(st.mu);
  auto it = st.completions.find(ticket);
  return it != st.completions.end() && it->second.done;
}

int ticket_errors(AsyncTicket ticket) {
  AsyncState& st = async_state();
  std::lock_guard<std::mutex> lk(st.mu);
  auto it = st.completions.find(ticket);
  return (it != st.completions.end() && it->second.done) ? it->second.errors : 0;
}

int reset_async_pager_state() {
  AsyncState& st = async_state();
  std::lock_guard<std::mutex> lk(st.mu);
  const int n = static_cast<int>(st.completions.size());
  st.completions.clear();
  st.next_ticket.store(1, std::memory_order_relaxed);
  st.last_completed.store(0, std::memory_order_release);
  return n;
}

}  // namespace veystrix
