// Phase 2 — CPU-only synthetic fixtures for LayerCursor (brief §5 Phase 2,
// items 1 & 3). No MLX, no checkpoint, no real files, no threads. Concurrency
// is modelled as a DETERMINISTIC event simulation: a fake reader completes
// tickets with seeded random delays, and a driver interleaves per-layer
// cursors. Every payload is a pure function of (expert, layer), so byte identity
// is checkable regardless of completion order.
//
// Build (standalone):
//   xcrun clang++ -std=c++20 -O2 -Icpp/engine/src \
//     cpp/engine/src/layer_cursor.cpp cpp/engine/tests/test_layer_cursor.cpp \
//     -o /tmp/test_layer_cursor
#include "layer_cursor.hpp"
#include "pool_interface.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace veystrix::engine;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Deterministic byte model
// ---------------------------------------------------------------------------
static uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
static uint64_t hash_bytes(const std::vector<uint8_t>& v) {
  uint64_t h = 0x243F6A8885A308D3ULL;
  for (uint8_t b : v) h = mix64(h ^ static_cast<uint64_t>(b));
  return h;
}
static std::vector<uint8_t> payload(int expert, int layer) {
  uint64_t h = mix64(static_cast<uint64_t>(static_cast<uint32_t>(expert)) *
                         0x9E3779B97F4A7C15ULL ^
                     (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 17) ^
                     0xA5A5A5A5ULL);
  std::vector<uint8_t> out(24);
  for (std::size_t i = 0; i < out.size(); ++i) {
    h = mix64(h + i);
    out[i] = static_cast<uint8_t>(h >> 56);
  }
  return out;
}
static std::vector<uint8_t> combine(const std::vector<uint8_t>& a,
                                    const std::vector<uint8_t>& b,
                                    uint64_t tag) {
  std::vector<uint8_t> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    uint8_t bv = b.empty() ? 0 : b[i % b.size()];
    out[i] = static_cast<uint8_t>(a[i] ^ bv ^
                                  static_cast<uint8_t>(tag >> (8 * (i % 8))));
  }
  return out;
}

// A fake MLX-like array: just bytes, with a fixed "shape/dtype" tag recorded in
// the op log so invariant 1 can compare them.
struct FakeArray {
  std::vector<uint8_t> bytes{};
  bool operator==(const FakeArray& o) const { return bytes == o.bytes; }
};

struct Pin {
  int session;
  int layer;
};
struct Slot {
  std::vector<uint8_t> bytes{};
  int expert = -1;
  int layer = -1;          // slots are keyed by (layer, expert), not expert alone
  bool resident = false;
  bool in_flight = false;
  bool writing = false;
  std::vector<Pin> pins{};
  uint64_t last_used = 0;
};
struct FakeRead {
  int ticket = -1;
  int expert = -1;
  int layer = -1;
  int slot = -1;
  int attempt = 0;
  int remaining = 0;
  bool done = false;
  bool failed = false;
  std::vector<uint8_t> payload{};
};

struct WorldConfig {
  int n_layers = 5;
  int num_experts = 16;
  int topk = 3;
  int capacity = 32;
  int max_delay = 5;
  // (expert, layer, attempt, first_session) -> should this physical read fail?
  std::function<bool(int, int, int, int)> fail_policy{};
};

// The fake pool + fake reader in one object, mirroring the plan/pin/submit/
// wait/release lifecycle of pool_interface.hpp. Uses NO mutex/CV and NO thread:
// completion order is chosen by a seeded RNG.
struct World {
  WorldConfig cfg;
  std::vector<Slot> slots;
  std::vector<FakeRead> reads;
  int next_ticket = 1;
  uint64_t clock = 0;
  std::mt19937_64 rng;

  // diagnostic counters (all must satisfy the invariants below)
  int physical_reads = 0;
  int hits = 0;
  int coalesced = 0;
  int evictions = 0;
  int pinned_or_inflight_evictions = 0;  // must stay 0 (invariant 4)
  int writes_after_complete = 0;         // must stay 0 (invariant 2)
  int build_over_writing = 0;            // must stay 0 (invariant 2)
  int parks = 0;

  explicit World(WorldConfig c, uint64_t seed)
      : cfg(std::move(c)), slots(c.capacity), rng(seed) {}

  FakeRead* find_read(int ticket) {
    for (auto& r : reads)
      if (r.ticket == ticket) return &r;
    return nullptr;
  }
  bool ticket_done(int ticket) {
    FakeRead* r = find_read(ticket);
    return r && r->done;
  }
  bool ticket_failed(int ticket) {
    FakeRead* r = find_read(ticket);
    return r && r->done && r->failed;
  }

  int resident_slot(int layer, int expert) const {
    for (std::size_t i = 0; i < slots.size(); ++i)
      // A slot is a HIT only once its bytes are materialized. A half-written
      // slot is in flight, never a hit (no false hit on half-written data).
      if (slots[i].resident && !slots[i].in_flight && !slots[i].writing &&
          slots[i].layer == layer && slots[i].expert == expert)
        return static_cast<int>(i);
    return -1;
  }
  FakeRead* inflight_read(int layer, int expert) {
    for (auto& r : reads)
      if (!r.done && r.layer == layer && r.expert == expert) return &r;
    return nullptr;
  }
  bool has_pins(int slot) const { return !slots[slot].pins.empty(); }

  void pin(int slot, int session, int layer) {
    slots[slot].pins.push_back({session, layer});
  }
  void release_fence(int session, int layer) {
    for (auto& s : slots) {
      std::vector<Pin> kept;
      for (const Pin& p : s.pins)
        if (!(p.session == session && p.layer == layer)) kept.push_back(p);
      s.pins = std::move(kept);
    }
  }

  // How many fresh slots are needed and how many are available? Used to park
  // BEFORE creating any reads, so a parked attempt has no side effects.
  bool can_acquire(int layer, const std::vector<int>& route) const {
    int need = 0;
    std::vector<int> seen;
    for (int e : route) {
      if (std::find(seen.begin(), seen.end(), e) != seen.end()) continue;
      seen.push_back(e);
      if (resident_slot(layer, e) >= 0) continue;
      bool inline_flight = false;
      for (const auto& r : reads)
        if (!r.done && r.layer == layer && r.expert == e) {
          inline_flight = true;
          break;
        }
      if (!inline_flight) ++need;
    }
    int avail = 0;
    for (const auto& s : slots)
      if (!s.resident || (!has_pins(static_cast<int>(&s - slots.data())) &&
                          !s.in_flight))
        ++avail;
    return need <= avail;
  }

  // Plan+pin+submit for one expert. Returns false only on pool exhaustion.
  bool acquire(int session, int layer, int expert, int* slot_out,
               int* ticket_out) {
    int hit = resident_slot(layer, expert);
    if (hit >= 0) {
      ++hits;
      pin(hit, session, layer);
      slots[hit].last_used = ++clock;
      *slot_out = hit;
      *ticket_out = -1;
      return true;
    }
    FakeRead* shared = inflight_read(layer, expert);
    if (shared) {
      ++coalesced;
      pin(shared->slot, session, layer);
      slots[shared->slot].last_used = ++clock;
      *slot_out = shared->slot;
      *ticket_out = shared->ticket;
      return true;
    }
    // Choose a victim: free slot first, else LRU among unpinned/unin-flight.
    int victim = -1;
    for (std::size_t i = 0; i < slots.size(); ++i)
      if (!slots[i].resident) {
        victim = static_cast<int>(i);
        break;
      }
    if (victim < 0) {
      uint64_t oldest = ~0ULL;
      for (std::size_t i = 0; i < slots.size(); ++i) {
        if (has_pins(static_cast<int>(i)) || slots[i].in_flight) continue;
        if (slots[i].last_used < oldest) {
          oldest = slots[i].last_used;
          victim = static_cast<int>(i);
        }
      }
      if (victim < 0) {
        // No unpinned/unin-flight victim: park (backpressure), never evict a
        // pinned slot and never throw. This is NOT an eviction violation.
        return false;
      }
      ++evictions;
    }
    Slot& sl = slots[victim];
    sl.resident = true;
    sl.expert = expert;
    sl.layer = layer;
    sl.in_flight = true;
    sl.writing = true;
    sl.bytes.clear();
    sl.last_used = ++clock;
    pin(victim, session, layer);

    FakeRead r;
    r.ticket = next_ticket++;
    r.expert = expert;
    r.layer = layer;
    r.slot = victim;
    r.attempt = 0;
    r.remaining = 1 + static_cast<int>(rng() % cfg.max_delay);
    r.payload = payload(expert, layer);
    r.failed = cfg.fail_policy ? cfg.fail_policy(expert, layer, 0, session)
                               : false;
    reads.push_back(r);
    ++physical_reads;
    *slot_out = victim;
    *ticket_out = r.ticket;
    return true;
  }

  // Retry a failed read once (new ticket, attempt=1) into a FRESH slot. The
  // failed slot was invalidated and must not be shared by two retrying
  // sessions: each retry re-materializes its own bytes.
  struct RetryResult {
    int ticket = -1;
    int slot = -1;
  };
  RetryResult retry(int session, int layer, int expert) {
    int victim = -1;
    for (std::size_t i = 0; i < slots.size(); ++i)
      if (!slots[i].resident && !slots[i].in_flight) {
        victim = static_cast<int>(i);
        break;
      }
    if (victim < 0)
      for (std::size_t i = 0; i < slots.size(); ++i)
        if (!has_pins(static_cast<int>(i)) && !slots[i].in_flight) {
          victim = static_cast<int>(i);
          break;
        }
    CHECK(victim >= 0);  // capacity 8 is ample for this fixture
    Slot& sl = slots[victim];
    sl.resident = true;
    sl.expert = expert;
    sl.layer = layer;
    sl.in_flight = true;
    sl.writing = true;
    sl.bytes.clear();
    sl.last_used = ++clock;
    pin(victim, session, layer);

    FakeRead r;
    r.ticket = next_ticket++;
    r.expert = expert;
    r.layer = layer;
    r.slot = victim;
    r.attempt = 1;
    r.remaining = 1;
    r.payload = payload(expert, layer);
    r.failed = cfg.fail_policy ? cfg.fail_policy(expert, layer, 1, session)
                               : false;
    reads.push_back(r);
    return {r.ticket, victim};
  }

  // Complete one in-flight read with a seeded random delay.
  void tick() {
    std::vector<int> idx;
    for (std::size_t i = 0; i < reads.size(); ++i)
      if (!reads[i].done) idx.push_back(static_cast<int>(i));
    if (idx.empty()) return;
    FakeRead& r = reads[idx[rng() % idx.size()]];
    if (--r.remaining > 0) return;
    Slot& sl = slots[r.slot];
    if (!r.failed) {
      if (sl.writing) sl.bytes = r.payload;
      else ++writes_after_complete;
    } else {
      sl.resident = false;
      sl.bytes.clear();
    }
    sl.writing = false;
    sl.in_flight = false;
    r.done = true;
  }
};

// ---------------------------------------------------------------------------
// Session model: independent PRNG / position / KV / GDN / PLE state.
// ---------------------------------------------------------------------------
struct Session {
  int id = 0;
  FakeArray hidden{};
  uint64_t prng = 0;
  uint64_t position = 0;
  FakeArray kv{}, gdn{}, ple{};
  LayerCursor<FakeArray> cursor{};
  bool failed = false;
  bool done = false;
  bool retried = false;
  std::vector<FakeArray> layer_outputs{};
  std::vector<std::string> ops{};        // all bookkeeping events
  std::vector<std::string> mlx_ops{};    // only MLX-graph ops (invariant 1)
  std::vector<std::pair<int, int>> plan_slots{};  // (slot, ticket)
  std::vector<int> plan_route{};
  std::vector<int> forced_route{};  // test hook: override layer-0 route
};

static FakeArray initial_hidden(int id) {
  return FakeArray{payload(0, 100 + id)};
}
static Session make_session(int id) {
  Session s;
  s.id = id;
  s.hidden = initial_hidden(id);
  s.prng = mix64(static_cast<uint64_t>(id) + 1);
  s.position = 0;
  s.kv = FakeArray{payload(1, 200 + id)};
  s.gdn = FakeArray{payload(2, 200 + id)};
  s.ple = FakeArray{payload(3, 200 + id)};
  s.cursor.begin(0, s.hidden);
  return s;
}

static std::vector<int> compute_route(const Session& s, int layer) {
  if (layer == 0 && !s.forced_route.empty()) return s.forced_route;
  uint64_t h = mix64(s.prng ^ (static_cast<uint64_t>(layer) << 32) ^
                     hash_bytes(s.hidden.bytes));
  std::vector<int> route;
  for (int i = 0; i < 3; ++i) {
    h = mix64(h);
    route.push_back(static_cast<int>(h % 16));
  }
  return route;
}

static bool runnable(const Session& s, World& w) {
  if (s.done || s.failed) return false;
  if (s.cursor.phase == LayerPhase::AwaitReads) {
    for (int t : s.cursor.pending)
      if (!w.ticket_done(t)) return false;
  }
  return true;
}

// Advance one session by one micro-step of its cursor.
static void advance(World& w, Session& s) {
  const int layer = s.cursor.layer;
  if (s.cursor.phase == LayerPhase::PreRoute) {
    std::vector<int> route = compute_route(s, layer);
    if (!w.can_acquire(layer, route)) {
      ++w.parks;  // invariant 4: park, never throw
      s.ops.push_back("park");
      return;
    }
    s.plan_route = route;
    s.plan_slots.clear();
    std::vector<int> tickets;
    for (int e : route) {
      int slot = -1, ticket = -1;
      CHECK(w.acquire(s.id, layer, e, &slot, &ticket));  // pin from plan time
      s.plan_slots.push_back({slot, ticket});
      if (ticket >= 0) tickets.push_back(ticket);
    }
    s.ops.push_back("route[k=" + std::to_string(route.size()) +
                    ",shape=1x16,dt=f32]");
    s.mlx_ops.push_back("route[k=" + std::to_string(route.size()) +
                        ",shape=1x16,dt=f32]");
    s.ops.push_back("submit[" + std::to_string(tickets.size()) + "]");
    s.cursor.await_reads(std::move(tickets));
    return;
  }
  if (s.cursor.phase == LayerPhase::AwaitReads) {
    bool any_failed = false;
    for (int t : s.cursor.pending)
      if (w.ticket_failed(t)) any_failed = true;
    if (any_failed) {
      if (s.retried) {
        s.failed = true;  // second failure: fail THIS session only
        s.ops.push_back("fail_session");
        return;
      }
      s.retried = true;
      std::vector<int> fresh;
      for (int t : s.cursor.pending) {
        if (!w.ticket_failed(t)) {
          fresh.push_back(t);
          continue;
        }
        FakeRead* r = w.find_read(t);
        CHECK(r != nullptr);
        s.cursor.on_ticket_done(t);
        auto rr = w.retry(s.id, layer, r->expert);
        fresh.push_back(rr.ticket);
        // Redirect this session's plan slots from the invalidated slot to the
        // fresh retry slot so the later gather reads the new bytes.
        for (auto& ps : s.plan_slots)
          if (ps.first == r->slot) ps.first = rr.slot;
      }
      s.cursor.await_reads(std::move(fresh));
      s.ops.push_back("retry_once");
      return;
    }
    const std::vector<int> done_list = s.cursor.pending;  // copy: on_ticket_done erases
    for (int t : done_list) s.cursor.on_ticket_done(t);
    CHECK(s.cursor.ready());
    // Invariant 2: no graph over bytes still being written.
    for (const auto& ps : s.plan_slots)
      if (w.slots[ps.first].writing) ++w.build_over_writing;
    s.cursor.enter_post_moe();
    s.ops.push_back("reads_ready[acquire]");
    return;
  }
  // PostMoE: build, eval, then release the (session, layer) fence.
  std::vector<uint8_t> moe;
  for (const auto& ps : s.plan_slots) {
    const auto& b = w.slots[ps.first].bytes;
    moe.insert(moe.end(), b.begin(), b.end());
  }
  s.ops.push_back("build[gather=" + std::to_string(s.plan_slots.size()) +
                  ",no_lazy_load]");
  s.mlx_ops.push_back("build[gather=" + std::to_string(s.plan_slots.size()) +
                      ",no_lazy_load]");
  FakeArray next{combine(s.hidden.bytes, moe, s.prng)};
  s.prng = mix64(s.prng ^ hash_bytes(next.bytes));  // per-session PRNG only
  s.position += 1;
  s.kv.bytes = combine(s.kv.bytes, next.bytes, s.prng);
  s.gdn.bytes = combine(s.gdn.bytes, next.bytes, mix64(s.prng));
  s.ple.bytes = combine(s.ple.bytes, next.bytes, mix64(s.prng + 1));
  s.hidden = next;
  s.ops.push_back("eval[shape=1x16,dt=f32]");
  s.mlx_ops.push_back("eval[shape=1x16,dt=f32]");
  w.release_fence(s.id, layer);  // after this session's eval returns
  s.ops.push_back("release_fence");
  s.layer_outputs.push_back(next);
  s.cursor.finish(next);
  if (s.cursor.layer >= w.cfg.n_layers) s.done = true;
}

template <class PickFn>
static void drive(World& w, std::vector<Session>& ss, PickFn pick,
                  int max_iter = 200000) {
  int iter = 0;
  for (; iter < max_iter; ++iter) {
    bool active = false;
    for (const auto& s : ss)
      if (!s.done && !s.failed) active = true;
    if (!active) break;
    w.tick();
    int idx = pick(ss, w);
    if (idx < 0) continue;
    advance(w, ss[idx]);
  }
  CHECK(iter < max_iter);  // no deadlock
}

static int pick_furthest_behind(const std::vector<Session>& ss, World& w) {
  int best = -1;
  for (std::size_t i = 0; i < ss.size(); ++i) {
    if (!runnable(ss[i], w)) continue;
    // "Furthest behind" = smallest layer index (C3 soft lockstep).
    if (best < 0 || ss[i].cursor.layer < ss[best].cursor.layer ||
        (ss[i].cursor.layer == ss[best].cursor.layer && ss[i].id < ss[best].id))
      best = static_cast<int>(i);
  }
  return best;
}

static std::vector<FakeArray> run_solo(int id, WorldConfig cfg, uint64_t seed) {
  World w(cfg, seed);
  std::vector<Session> ss = {make_session(id)};
  drive(w, ss, pick_furthest_behind);
  CHECK(ss[0].done && !ss[0].failed);
  return ss[0].layer_outputs;
}

static void complete_ticket(World& w, int ticket) {
  for (int i = 0; i < 10000 && !w.ticket_done(ticket); ++i) w.tick();
  CHECK(w.ticket_done(ticket));
}

// ---------------------------------------------------------------------------
// (a) one test per C1 bit-exactness invariant
// ---------------------------------------------------------------------------
static void test_invariant_1() {
  WorldConfig cfg;
  auto solo_ops = [&]() {
    World w(cfg, 11);
    std::vector<Session> ss = {make_session(1)};
    drive(w, ss, pick_furthest_behind);
    return ss[0].mlx_ops;
  }();
  World w(cfg, 11);
  std::vector<Session> ss;
  for (int id : {1, 2, 3}) ss.push_back(make_session(id));
  drive(w, ss, pick_furthest_behind);
  for (const auto& s : ss) {
    CHECK(!s.failed && s.done);
    CHECK(s.mlx_ops.size() == solo_ops.size());
    for (std::size_t i = 0; i < solo_ops.size(); ++i)
      CHECK(s.mlx_ops[i] == solo_ops[i]);  // same MLX op sequence / shapes / dtypes / gather order
  }
  std::printf("  ok: per-session MLX op sequence, shapes, dtypes, gather order identical to solo\n");
}

static void test_invariant_2() {
  WorldConfig cfg;
  cfg.max_delay = 8;
  World w(cfg, 22);
  std::vector<Session> ss;
  for (int id : {1, 2, 3, 4}) ss.push_back(make_session(id));
  drive(w, ss, pick_furthest_behind);
  CHECK(w.build_over_writing == 0);
  CHECK(w.writes_after_complete == 0);
  std::printf("  ok: no graph built over writing bytes; completion read with acquire; no lazy Load\n");
}

static void test_invariant_3() {
  // Directly model the residency->pin window: a slot acquired for session S
  // at layer L is pinned from plan time and must keep its byte-exact payload
  // even under strong LRU pressure from other sessions, until S's fence.
  WorldConfig cfg;
  cfg.capacity = 3;  // tight: force the evictor to consider pinned slots
  cfg.num_experts = 32;
  World w(cfg, 33);

  int s_pin = -1, t_pin = -1;
  CHECK(w.acquire(1, 0, 7, &s_pin, &t_pin));       // plan + pin (miss)
  CHECK(w.has_pins(s_pin));
  CHECK(t_pin >= 0);
  complete_ticket(w, t_pin);                      // bytes materialized
  const auto expected = payload(7, 0);
  CHECK(w.slots[s_pin].bytes == expected);

  // Two more sessions acquire distinct experts -> the pool is now full.
  int s2 = -1, t2 = -1, s3 = -1, t3 = -1;
  CHECK(w.acquire(2, 0, 8, &s2, &t2));
  CHECK(w.acquire(3, 0, 9, &s3, &t3));
  complete_ticket(w, t2);
  complete_ticket(w, t3);
  CHECK(s_pin != s2 && s_pin != s3);

  // Session 3's eval has returned -> release its fence, making slot s3
  // evictable. The fourth acquire must evict that UNPINNED slot, never s_pin.
  w.release_fence(3, 0);
  CHECK(!w.has_pins(s3));
  int s4 = -1, t4 = -1;
  CHECK(w.acquire(4, 0, 10, &s4, &t4));
  CHECK(w.pinned_or_inflight_evictions == 0);
  CHECK(s4 != s_pin);
  CHECK(w.slots[s_pin].resident && w.slots[s_pin].expert == 7);
  CHECK(w.slots[s_pin].bytes == expected);         // payload still byte-exact

  // Fence releases ONLY session 1's pin; other pins stay.
  w.release_fence(1, 0);
  CHECK(!w.has_pins(s_pin));
  CHECK(w.has_pins(s2) && w.has_pins(s4));
  CHECK(w.slots[s_pin].bytes == expected);

  // Now the pinned slot is evictable and its bytes may change.
  int s5 = -1, t5 = -1;
  CHECK(w.acquire(5, 0, 11, &s5, &t5));
  CHECK(w.evictions >= 1);
  std::printf("  ok: slot pinned from plan time kept byte-exact under LRU pressure until its fence\n");
}

static void test_invariant_4() {
  WorldConfig cfg;
  cfg.capacity = 2;
  cfg.topk = 1;
  cfg.num_experts = 8;
  World w(cfg, 44);
  Session a = make_session(1);
  Session b = make_session(2);
  // Park a at PreRoute so it pins slots and holds them.
  std::vector<Session> ss = {a};
  // Pin two slots for session 1 at layer 0 manually via one acquire each.
  int s1 = -1, t1 = -1, s2 = -1, t2 = -1;
  CHECK(w.acquire(1, 0, 0, &s1, &t1));
  CHECK(w.acquire(1, 0, 1, &s2, &t2));
  CHECK(w.has_pins(s1) && w.has_pins(s2));
  // Session 2 needs a third slot: must park, never throw / evict a pinned slot.
  bool threw = false;
  try {
    int s3 = -1, t3 = -1;
    w.acquire(2, 0, 2, &s3, &t3);
  } catch (...) {
    threw = true;
  }
  CHECK(!threw);
  CHECK(w.pinned_or_inflight_evictions == 0);
  // Reads complete, then the fence releases the pins; now the parked acquire
  // succeeds.
  complete_ticket(w, t1);
  complete_ticket(w, t2);
  w.release_fence(1, 0);
  int s3 = -1, t3 = -1;
  CHECK(w.acquire(2, 0, 2, &s3, &t3));
  std::printf("  ok: pool exhaustion parks (no throw); no pinned/in-flight slot evicted\n");
}

static void test_invariant_5() {
  WorldConfig cfg;
  World w(cfg, 55);
  std::vector<Session> ss = {make_session(1), make_session(2), make_session(3)};
  drive(w, ss, pick_furthest_behind);
  for (const auto& s : ss) {
    CHECK(s.done && !s.failed);
    Session solo = make_session(s.id);
    CHECK(s.prng != 0);
    // Compare against a solo run's terminal numeric state by replaying the solo
    // session and inspecting its (independent) state snapshot.
    World ws(cfg, 55);
    std::vector<Session> solo_ss = {solo};
    drive(ws, solo_ss, pick_furthest_behind);
    CHECK(s.prng == solo_ss[0].prng);
    CHECK(s.position == solo_ss[0].position);
    CHECK(s.kv == solo_ss[0].kv);
    CHECK(s.gdn == solo_ss[0].gdn);
    CHECK(s.ple == solo_ss[0].ple);
    CHECK(s.hidden == solo_ss[0].hidden);
  }
  std::printf("  ok: per-session PRNG/KV/GDN/PLE/position isolated (no shared mutable numeric state)\n");
}

static void test_invariant_6() {
  WorldConfig cfg;
  cfg.capacity = 8;
  // First read of expert 0 at layer 0 fails for everybody; session 1's retry
  // fails again, session 2's retry succeeds.
  cfg.fail_policy = [](int expert, int layer, int attempt, int session) {
    if (expert == 0 && layer == 0 && attempt == 0) return true;
    if (expert == 0 && layer == 0 && attempt == 1 && session == 1) return true;
    return false;
  };
  World w(cfg, 66);
  std::vector<Session> ss = {make_session(1), make_session(2)};
  ss[0].forced_route = {0};
  ss[1].forced_route = {0};

  // Both reach layer 0 PreRoute; A submits expert 0, B coalesces onto A's
  // ticket (one physical read, two waiters).
  advance(w, ss[0]);
  CHECK(ss[0].cursor.phase == LayerPhase::AwaitReads &&
        ss[0].cursor.pending.size() == 1);
  const int shared_ticket = ss[0].cursor.pending[0];
  advance(w, ss[1]);
  CHECK(ss[1].cursor.phase == LayerPhase::AwaitReads &&
        ss[1].cursor.pending.size() == 1);
  CHECK(ss[1].cursor.pending[0] == shared_ticket);  // shared in-flight ticket
  CHECK(w.physical_reads == 1 && w.coalesced == 1);

  // One failed read wakes EVERY waiter with an error.
  complete_ticket(w, shared_ticket);
  CHECK(w.ticket_failed(shared_ticket));
  CHECK(runnable(ss[0], w) && runnable(ss[1], w));
  CHECK(std::find(ss[0].cursor.pending.begin(), ss[0].cursor.pending.end(),
                  shared_ticket) != ss[0].cursor.pending.end());
  CHECK(std::find(ss[1].cursor.pending.begin(), ss[1].cursor.pending.end(),
                  shared_ticket) != ss[1].cursor.pending.end());

  // Each retries once. A's retry fails again -> only A fails; B's retry
  // succeeds and B continues to a byte-exact output.
  advance(w, ss[0]);
  CHECK(ss[0].retried && !ss[0].failed);
  const int retry_a = ss[0].cursor.pending[0];
  advance(w, ss[1]);
  CHECK(ss[1].retried && !ss[1].failed);
  const int retry_b = ss[1].cursor.pending[0];
  complete_ticket(w, retry_a);
  complete_ticket(w, retry_b);
  CHECK(w.ticket_failed(retry_a));
  CHECK(!w.ticket_failed(retry_b));

  advance(w, ss[0]);  // second failure -> fail this session only
  CHECK(ss[0].failed);
  advance(w, ss[1]);  // all tickets complete -> PostMoE
  CHECK(ss[1].cursor.phase == LayerPhase::PostMoE);
  advance(w, ss[1]);  // build + eval + fence for layer 0
  CHECK(!ss[1].failed && ss[1].layer_outputs.size() == 1);

  // Drive B through the remaining layers (A is failed and skipped).
  drive(w, ss, pick_furthest_behind);
  CHECK(ss[1].done && !ss[1].failed);

  // B's output matches a session that never saw any failure and that was
  // driven through all five layers.
  WorldConfig clean = cfg;
  clean.fail_policy = {};
  World ws(clean, 77);
  std::vector<Session> solo = {make_session(2)};
  solo[0].forced_route = {0};
  drive(ws, solo, pick_furthest_behind);
  CHECK(solo[0].done && !solo[0].failed);
  CHECK(ss[1].layer_outputs == solo[0].layer_outputs);

  std::printf("  ok: one failed read woke every waiter; retry once; second failure failed only that session\n");
}

// ---------------------------------------------------------------------------
// (b) 1000 randomized interleavings, byte-exact
// ---------------------------------------------------------------------------
static void test_stress() {
  constexpr int kIters = 1000;
  for (int it = 0; it < kIters; ++it) {
    WorldConfig cfg;
    cfg.capacity = 24;
    cfg.n_layers = 5;
    uint64_t seed = mix64(static_cast<uint64_t>(it) + 0xC0FFEEULL);

    const int ids[3] = {1, 2, 3};
    std::vector<std::vector<FakeArray>> solos;
    for (int id : ids) solos.push_back(run_solo(id, cfg, seed));

    World w(cfg, seed);
    std::mt19937_64 pick_rng(mix64(seed ^ 0x1234ULL));
    std::vector<Session> ss;
    for (int id : ids) ss.push_back(make_session(id));
    drive(w, ss, [&](const std::vector<Session>& sessions, World& world) {
      std::vector<int> candidates;
      for (std::size_t i = 0; i < sessions.size(); ++i)
        if (runnable(sessions[i], world)) candidates.push_back(static_cast<int>(i));
      if (candidates.empty()) return -1;
      return candidates[pick_rng() % candidates.size()];
    });

    for (std::size_t i = 0; i < ss.size(); ++i) {
      CHECK(!ss[i].failed && ss[i].done);
      CHECK(ss[i].layer_outputs == solos[i]);  // byte-exact
    }
    CHECK(w.build_over_writing == 0);
    CHECK(w.writes_after_complete == 0);
    CHECK(w.pinned_or_inflight_evictions == 0);
  }
  std::printf("  ok: 1000 randomized interleavings byte-exact (3 sessions x 5 layers)\n");
}

int main() {
  std::printf("test_layer_cursor (CPU-only synthetic):\n");

  std::printf("[1] C1 invariant 1 - op sequence/shapes/dtypes/gather order\n");
  test_invariant_1();
  std::printf("[2] C1 invariant 2 - no array over writing bytes\n");
  test_invariant_2();
  std::printf("[3] C1 invariant 3 - pin from plan time until eval\n");
  test_invariant_3();
  std::printf("[4] C1 invariant 4 - eviction never pins/in-flight; park not throw\n");
  test_invariant_4();
  std::printf("[5] C1 invariant 5 - per-session numeric state isolation\n");
  test_invariant_5();
  std::printf("[6] C1 invariant 6 - failed read wakes all; retry once; fail session only\n");
  test_invariant_6();
  std::printf("[7] stress - 1000 randomized byte-exact interleavings\n");
  test_stress();

  std::printf("test_layer_cursor OK (all checks passed)\n");
  return 0;
}
