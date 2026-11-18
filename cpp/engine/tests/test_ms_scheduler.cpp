// Multi-session Phase 1 — tiny CPU-only fixtures (synthetic, no checkpoint,
// no MLX, no benchmark, no new threads). Mirrors the TinySession style of
// test_cooperative_decode.cpp but each fixture owns its own SessionState/PRNG.
//
// Checks (contract §implementation):
//  (a) N=2..4 interleaved sessions produce bit-identical per-session token
//      streams to solo cooperative_run.
//  (b) a session that yields at layer L consumes no further steps until the
//      reads-ready callback reports completion.
//  (c) work-conserving: the driver advances runnable sessions while one waits.
//  (d) admission refuses when the page cache would fall below its floor.
//  (e) per-session PRNG independence: B's draws are unchanged by A's draws.
#include "admission.hpp"
#include "cooperative_decode.hpp"
#include "ms_scheduler.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace veystrix::engine;

static NumericsProfile frozen_profile() {
  NumericsProfile p;
  p.prefill_chunk = 256;
  p.kv_tier = "int8";
  p.kv_mode = "quantized";
  p.adaptive_top_k = 20;
  p.freeze();
  return p;
}

// One deterministic session: token[step] = draw(own PRNG) ^ mix(seed, step).
// Pure per-session state; nothing depends on other sessions or call order.
struct Fixture {
  SessionState session;
  int length;
  std::vector<uint64_t> tokens;
  bool yield_once = false;
  int yield_layer = -1;

  Fixture(int id, unsigned long long key, int len, bool yield = false, int layer = -1)
      : session(id, key, frozen_profile()), length(len),
        yield_once(yield), yield_layer(layer) {}
};

static bool fixture_step(void* p, int step, unsigned long long seed) {
  auto* f = static_cast<Fixture*>(p);
  if (f->yield_once && step == 0) {
    // Explicit yield at the configured layer: this step is consumed but emits
    // no token. The scheduler reads the request from this session only.
    f->session.request_yield(f->yield_layer);
    f->yield_once = false;
    return false;
  }
  uint64_t x = f->session.draw_u64() ^
               (seed + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(step + 1));
  f->tokens.push_back(x);
  f->session.advance_position();
  return static_cast<int>(f->tokens.size()) >= f->length;
}

static bool streams_equal(const std::vector<uint64_t>& a,
                          const std::vector<uint64_t>& b) {
  return a == b;
}

static bool check_a_bit_identity() {
  const int lengths[] = {6, 7, 8};
  const unsigned long long seeds[] = {11ULL, 22ULL, 33ULL, 44ULL};
  for (int N = 2; N <= 4; ++N) {
    const int LEN = lengths[N - 2];
    // Solo reference: run each session alone with cooperative_run.
    std::vector<std::vector<uint64_t>> solo(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
      Fixture f(i, seeds[i], LEN);
      CoopSession c{&f, fixture_step, seeds[i]};
      std::vector<CoopSession> one{c};
      cooperative_run(one);
      solo[static_cast<std::size_t>(i)] = f.tokens;
    }

    // Interleaved: one driver, no yields, all sessions scheduled together.
    MsScheduler sched;
    std::vector<std::unique_ptr<Fixture>> states;
    states.reserve(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
      states.push_back(std::make_unique<Fixture>(i, seeds[i], LEN));
      ScheduledSession ss;
      ss.id = i;
      ss.state = states.back().get();
      ss.session = &states.back()->session;
      ss.step_fn = fixture_step;
      ss.seed = seeds[i];
      sched.add(ss);
    }
    const int executed = sched.drive();
    if (executed != N * LEN) {
      std::fprintf(stderr, "FAIL(a): N=%d steps %d != %d\n", N, executed, N * LEN);
      return false;
    }
    for (int i = 0; i < N; ++i) {
      if (!streams_equal(solo[static_cast<std::size_t>(i)],
                         states[static_cast<std::size_t>(i)]->tokens)) {
        std::fprintf(stderr, "FAIL(a): N=%d session %d stream differs from solo\n", N, i);
        return false;
      }
      if (sched.steps_done(i) != LEN) {
        std::fprintf(stderr, "FAIL(a): N=%d session %d step count %d != %d\n",
                     N, i, sched.steps_done(i), LEN);
        return false;
      }
    }
  }
  std::printf("test_ms_scheduler OK: (a) bit-identity N=2,3,4\n");
  return true;
}

static bool check_b_yield() {
  MsScheduler sched;
  auto f = std::make_unique<Fixture>(0, 123ULL, 6, /*yield=*/true, /*layer=*/7);
  ScheduledSession ss;
  ss.id = 0;
  ss.state = f.get();
  ss.session = &f->session;
  ss.step_fn = fixture_step;
  ss.seed = 123ULL;
  sched.add(ss);

  // reads-ready never fires: only the initial yielding step is consumed.
  const int e1 = sched.drive(5, [](int, int) { return false; });
  if (e1 != 1 || sched.steps_done(0) != 1 || !f->tokens.empty() ||
      sched.phase(0) != SessionPhase::WaitingForExpertReads ||
      sched.wait_layer(0) != 7) {
    std::fprintf(stderr, "FAIL(b): yield not held (e=%d steps=%d layer=%d)\n",
                 e1, sched.steps_done(0), sched.wait_layer(0));
    return false;
  }
  // Draining with no runnable session must not spin.
  if (sched.drive(100, [](int, int) { return false; }) != 0) {
    std::fprintf(stderr, "FAIL(b): driver spun on a waiting-only session\n");
    return false;
  }
  // Report reads ready: exactly one further step runs.
  const int e2 = sched.drive(1, [](int, int) { return true; });
  if (e2 != 1 || sched.steps_done(0) != 2 || f->tokens.size() != 1 ||
      sched.phase(0) != SessionPhase::Ready) {
    std::fprintf(stderr, "FAIL(b): reads-ready did not resume session\n");
    return false;
  }
  std::printf("test_ms_scheduler OK: (b) yield at layer 7 held until reads-ready\n");
  return true;
}

static bool check_c_work_conserving() {
  MsScheduler sched;
  auto waiting = std::make_unique<Fixture>(0, 55ULL, 6, /*yield=*/true, /*layer=*/3);
  auto running = std::make_unique<Fixture>(1, 66ULL, 8);

  ScheduledSession a;
  a.id = 0; a.state = waiting.get(); a.session = &waiting->session;
  a.step_fn = fixture_step; a.seed = 55ULL;
  sched.add(a);
  ScheduledSession b;
  b.id = 1; b.state = running.get(); b.session = &running->session;
  b.step_fn = fixture_step; b.seed = 66ULL;
  sched.add(b);

  // Three rounds with the reads never ready: session 0 stays at its yield
  // step, session 1 advances every round.
  const int executed = sched.drive(3, [](int, int) { return false; });
  if (executed != 4 || sched.steps_done(0) != 1 || sched.steps_done(1) != 3 ||
      running->tokens.size() != 3 || !waiting->tokens.empty()) {
    std::fprintf(stderr,
                 "FAIL(c): not work-conserving (exec=%d waiting=%d running=%d)\n",
                 executed, sched.steps_done(0), sched.steps_done(1));
    return false;
  }
  std::printf("test_ms_scheduler OK: (c) work-conserving while one session waits\n");
  return true;
}

static bool check_d_admission() {
  const NumericsProfile p = frozen_profile();
  AdmissionArgs args;
  args.capacity_mb = 5000.0;
  args.model_resident_mb = 1000.0;
  args.expert_pool_mb = 500.0;
  args.pinned_pool_mb = 0.0;
  args.page_cache_floor_mb = 3200.0;
  args.session_state_mb = 250.0;
  args.min_available_mb = 0.0;  // isolate the page-cache-floor rule

  const AdmissionDecision first = evaluate_admission(args, 0, p);
  if (!first.admitted || first.resulting_sessions != 1 ||
      first.projected_page_cache_mb != 3250.0) {
    std::fprintf(stderr, "FAIL(d): first session should be admitted\n");
    return false;
  }
  const AdmissionDecision second = evaluate_admission(args, 1, p);
  if (second.admitted || second.resulting_sessions != 2 ||
      second.projected_page_cache_mb != 3000.0 ||
      second.reason != "page cache floor") {
    std::fprintf(stderr, "FAIL(d): N+1 must be refused below page-cache floor\n");
    return false;
  }
  // Standing 3 GiB rule ([cited]) is an independent refusal reason.
  args.page_cache_floor_mb = 200.0;
  args.min_available_mb = 3072.0;
  const AdmissionDecision lowavail = evaluate_admission(args, 1, p);
  if (lowavail.admitted || lowavail.reason.find("3 GiB") == std::string::npos) {
    std::fprintf(stderr, "FAIL(d): 3 GiB rule not enforced\n");
    return false;
  }
  // Unfrozen profile is refused, never mutated ([idea] contract §3 rule 3).
  const AdmissionDecision unfrozen = evaluate_admission(args, 0, NumericsProfile{});
  if (unfrozen.admitted || unfrozen.reason.find("not frozen") == std::string::npos) {
    std::fprintf(stderr, "FAIL(d): unfrozen profile not refused\n");
    return false;
  }
  std::printf("test_ms_scheduler OK: (d) admission refuses below page-cache floor\n");
  return true;
}

static bool check_e_prng_independence() {
  const unsigned long long kA = 111ULL, kB = 222ULL;
  const NumericsProfile p = frozen_profile();
  SessionState a(0, kA, p), b(1, kB, p);
  SessionState a_ref(0, kA, p), b_ref(1, kB, p);
  std::vector<uint64_t> a_draws, b_draws, a_ref_draws, b_ref_draws;
  for (int i = 0; i < 8; ++i) {
    a_draws.push_back(a.draw_u64());  // A draws interleaved with B
    b_draws.push_back(b.draw_u64());
  }
  for (int i = 0; i < 8; ++i) {
    a_ref_draws.push_back(a_ref.draw_u64());  // each alone
    b_ref_draws.push_back(b_ref.draw_u64());
  }
  if (!streams_equal(b_draws, b_ref_draws)) {
    std::fprintf(stderr, "FAIL(e): B's draws changed by A's draws\n");
    return false;
  }
  if (!streams_equal(a_draws, a_ref_draws)) {
    std::fprintf(stderr, "FAIL(e): A's draws changed by B's draws\n");
    return false;
  }
  if (a.prng_key() != kA || b.prng_key() != kB ||
      a.prng_state() != a_ref.prng_state() || b.prng_state() != b_ref.prng_state()) {
    std::fprintf(stderr, "FAIL(e): PRNG key/state was advanced across sessions\n");
    return false;
  }
  std::printf("test_ms_scheduler OK: (e) per-session PRNG independence\n");
  return true;
}

int main() {
  int passed = 0;
  if (check_a_bit_identity()) ++passed;
  if (check_b_yield()) ++passed;
  if (check_c_work_conserving()) ++passed;
  if (check_d_admission()) ++passed;
  if (check_e_prng_independence()) ++passed;
  if (passed != 5) {
    std::fprintf(stderr, "test_ms_scheduler FAILED (%d/5)\n", passed);
    return 1;
  }
  std::printf("test_ms_scheduler OK (5/5)\n");
  return 0;
}
