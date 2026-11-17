// Bit-exact, hardware-free test of the portable pager's remap logic.
#include "io_pager.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace veystrix;

int main() {
  // Route with a resident expert, two new experts (one repeated), in route order.
  const std::vector<int32_t> route = {7, 2, 9, 2, 7};
  const std::vector<int32_t> resident = {7, 0};            // expert 7 is in slot 0
  const std::vector<int32_t> free_slots = {3, 4, 5};
  const int64_t header = 8, row_bytes = 64, stride = 128;
  std::vector<uint8_t> buf(16 * stride, 0);

  PagerPlan p = plan_remap(route, resident, free_slots, 42, header, row_bytes,
                           buf.data(), stride);

  // Slots: 7->0 (resident), 2->3 (new), 9->4 (new), 2->3 (dup), 7->0 (resident)
  const std::vector<int32_t> want_slots = {0, 3, 4, 3, 0};
  assert(p.slots == want_slots);
  // Two reads only: expert 2 (slot 3) and expert 9 (slot 4); deduplicated.
  assert(p.reads.size() == 2);
  assert(p.reads[0].offset == header + 2 * row_bytes);
  assert(p.reads[0].dst == reinterpret_cast<uint64_t>(buf.data() + 3 * stride));
  assert(p.reads[1].offset == header + 9 * row_bytes);
  assert(p.reads[1].dst == reinterpret_cast<uint64_t>(buf.data() + 4 * stride));

  // Determinism: same inputs -> identical plan.
  PagerPlan q = plan_remap(route, resident, free_slots, 42, header, row_bytes,
                           buf.data(), stride);
  assert(q.slots == p.slots);
  assert(q.reads.size() == p.reads.size());
  for (size_t i = 0; i < p.reads.size(); ++i) {
    assert(q.reads[i].offset == p.reads[i].offset);
    assert(q.reads[i].dst == p.reads[i].dst);
    assert(q.reads[i].pad == 0);
    assert(q.reads[i].length == p.reads[i].length);
  }

  // Fail closed: not enough free slots -> -1, never a partial wrong mapping.
  PagerPlan r = plan_remap({1, 2, 3}, {}, {10}, 42, header, row_bytes,
                           buf.data(), stride);
  assert(r.slots[0] == 10);
  assert(r.slots[1] == -1);
  assert(r.slots[2] == -1);
  assert(r.reads.size() == 1);

  std::cout << "test_io_pager OK\n";
  return 0;
}
