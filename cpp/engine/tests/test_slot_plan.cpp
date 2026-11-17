// Pure-CPU tests for the shared slot bookkeeping (no MLX, no I/O).
// Regression for Bug 1 (flattened-list miss count), a seeded 10k-step
// randomized invariant check, and fail-closed pool exhaustion.
#include "slot_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using veystrix::engine::SlotStep;
using veystrix::engine::plan_slots;

static int failures = 0;
#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);        \
      ++failures;                                                           \
    }                                                                       \
  } while (0)

// Apply the caller's miss assignment and assert the post-step invariant.
static void commit_and_check(const std::vector<int32_t>& route, SlotStep& step,
                             std::vector<int32_t>& slot_expert,
                             std::vector<int32_t>& free_slots, int num_slots,
                             const char* where) {
  for (size_t i = 0; i < step.miss_e.size(); ++i) {
    CHECK(step.miss_s[i] >= 0 && step.miss_s[i] < num_slots, "miss slot range");
    slot_expert[step.miss_s[i]] = step.miss_e[i];
    auto it = std::find(free_slots.begin(), free_slots.end(), step.miss_s[i]);
    if (it != free_slots.end()) free_slots.erase(it);
  }
  CHECK(step.slots.size() == route.size(), "slots/route size");
  for (size_t j = 0; j < route.size(); ++j) {
    int32_t s = step.slots[j];
    if (s < 0 || s >= num_slots) {
      std::fprintf(stderr, "FAIL: %s: route[%zu] slot %d out of range\n", where,
                   j, s);
      ++failures;
      continue;
    }
    if (slot_expert[s] != route[j]) {
      std::fprintf(stderr,
                   "FAIL: %s: slot %d holds %d, want %d\n", where, s,
                   slot_expert[s], route[j]);
      ++failures;
    }
  }
  // free_slots must be exactly the complement of occupied slots.
  size_t occupied = 0;
  for (int32_t e : slot_expert) occupied += (e >= 0);
  CHECK(free_slots.size() == static_cast<size_t>(num_slots) - occupied,
        "free_slots complement");
  std::vector<int32_t> sorted = free_slots;
  std::sort(sorted.begin(), sorted.end());
  CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
        "free_slots unique");
}

// Exact replica of the OLD miss counter (the buggy line the fix removes):
//   for (e : needed) if (std::find(pairs.begin(), pairs.end(), e)==end) ++misses;
// where pairs is the FLATTENED [expert, slot, ...] list. Kept in the test to
// demonstrate the regression guard is meaningful: the old code undercounts.
static size_t old_flat_miss_count(const std::vector<int32_t>& needed,
                                  const std::vector<int32_t>& pairs) {
  size_t misses = 0;
  for (int32_t e : needed)
    if (std::find(pairs.begin(), pairs.end(), e) == pairs.end()) ++misses;
  return misses;
}

static void test_bug1_regression() {
  // Pool FULL with experts 100..115 in slots 0..15. Routed experts 3 and 7
  // are NOT resident. The old code did:
  //   for (e : needed) if (std::find(pairs.begin(), pairs.end(), e)==end) ++misses;
  // where pairs = [100,0,101,1,...,115,15] (flattened expert,slot). Because
  // slot numbers 3 and 7 appear in pairs, std::find "finds" experts 3 and 7,
  // so the miss count was 1 (only 200) instead of 3 -> one slot too few freed
  // -> plan_remap returned -1 -> gather_qmm read out of bounds + heap corruption.
  const int num_slots = 16;
  std::vector<int32_t> slot_expert(num_slots), free_slots;
  std::vector<int> slot_ref(num_slots, 0);
  std::vector<uint64_t> slot_last(num_slots, 0);
  for (int s = 0; s < num_slots; ++s) slot_expert[s] = 100 + s;  // full pool

  std::vector<int32_t> route = {3, 7, 200};
  // Demonstrate the old code fails this very scenario: the flattened pairs list
  // contains slot numbers 3 and 7, so std::find wrongly treats experts 3 and 7
  // as resident and undercounts misses as 1 instead of 3.
  std::vector<int32_t> pairs;
  for (int s = 0; s < num_slots; ++s) {
    pairs.push_back(slot_expert[s]);
    pairs.push_back(s);
  }
  size_t old_misses = old_flat_miss_count(route, pairs);
  CHECK(old_misses == 1, "old flat-list counter undercounts (regression proof)");
  std::printf("  old miss counter would report %zu misses (correct: 3)\n",
              old_misses);

  SlotStep step = plan_slots(route, slot_expert, slot_ref, slot_last,
                             free_slots, num_slots);

  CHECK(step.miss_e.size() == 3, "bug1: three misses");
  std::vector<int32_t> want = {3, 7, 200};
  CHECK(step.miss_e == want, "bug1: miss_e == {3,7,200}");
  CHECK(step.miss_s.size() == 3, "bug1: three miss slots");
  CHECK(step.evicted.size() == 3, "bug1: three evictions");
  commit_and_check(route, step, slot_expert, free_slots, num_slots,
                   "bug1_regression");
  std::printf("test_slot_plan: bug1 regression OK\n");
}

static void test_randomized() {
  const int E = 512, K = 10, num_slots = 16;
  std::mt19937 rng(0xC0FFEEu);
  std::uniform_int_distribution<int> pick(0, E - 1);
  std::vector<int32_t> slot_expert(num_slots, -1), free_slots(num_slots);
  std::vector<int> slot_ref(num_slots, 0);
  std::vector<uint64_t> slot_last(num_slots, 0);
  for (int s = 0; s < num_slots; ++s) free_slots[s] = s;

  uint64_t tick = 0;
  for (int it = 0; it < 10000; ++it) {
    std::vector<int32_t> route(K);
    for (int j = 0; j < K; ++j) route[j] = pick(rng);
    std::shuffle(route.begin(), route.end(), rng);

    SlotStep step = plan_slots(route, slot_expert, slot_ref, slot_last,
                               free_slots, num_slots);
    commit_and_check(route, step, slot_expert, free_slots, num_slots,
                     "randomized");
    for (int32_t s : step.slots) slot_last[s] = ++tick;
    if (failures) break;
  }
  std::printf("test_slot_plan: randomized 10000 steps OK\n");
}

static void test_exhaustion() {
  const int num_slots = 4;
  std::vector<int32_t> slot_expert(num_slots, -1), free_slots(num_slots);
  std::vector<int> slot_ref(num_slots, 0);
  std::vector<uint64_t> slot_last(num_slots, 0);
  for (int s = 0; s < num_slots; ++s) free_slots[s] = s;
  // 5 distinct experts cannot fit in 4 slots even with full eviction.
  std::vector<int32_t> route = {0, 1, 2, 3, 4};
  bool threw = false;
  try {
    plan_slots(route, slot_expert, slot_ref, slot_last, free_slots, num_slots);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw, "exhaustion: K > num_slots throws");
  CHECK(free_slots.size() == static_cast<size_t>(num_slots),
        "exhaustion: no partial mutation of free_slots");
  std::printf("test_slot_plan: exhaustion OK\n");
}

int main() {
  test_bug1_regression();
  test_randomized();
  test_exhaustion();
  if (failures) {
    std::fprintf(stderr, "test_slot_plan FAILED (%d)\n", failures);
    return 1;
  }
  std::printf("test_slot_plan OK (all checks passed)\n");
  return 0;
}
