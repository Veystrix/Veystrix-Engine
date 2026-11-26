// Bit-exact end-to-end test of the portable pager:
// 1. Write a fake expert shards file with known rows.
// 2. plan_remap routes (with dupes + a resident expert).
// 3. run_batched_reads fills the slots via native batched pread.
// 4. Wrap slots as an mx::array zero-copy and confirm bytes match the source.
#include "io_pager.hpp"

#include <mlx/mlx.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix;

static const int64_t HEADER = 8;
static const int64_t ROW = 64;       // 64 bytes = 16 uint32
static const int32_t NEXP = 12;

int main() {
  // Fake expert data: row e is a repeat of (e*16 + j), j=0..15.
  std::vector<uint32_t> expert_data(NEXP * 16);
  for (int e = 0; e < NEXP; ++e)
    for (int j = 0; j < 16; ++j) expert_data[e * 16 + j] = e * 1000 + j;

  std::string path = "/tmp/veystrix_pager_test.bin";
  {
    std::ofstream f(path, std::ios::binary);
    std::vector<uint8_t> header(HEADER, 0);
    f.write(reinterpret_cast<const char*>(header.data()), HEADER);
    f.write(reinterpret_cast<const char*>(expert_data.data()),
            expert_data.size() * sizeof(uint32_t));
  }
  int fd = open(path.c_str(), O_RDONLY);
  assert(fd >= 0);

  // Route: expert 5 (new), 1 (new), 5 (dup), 9 (new). Expert 3 is "resident" in slot 0.
  const std::vector<int32_t> route = {5, 1, 5, 9};
  const std::vector<int32_t> resident = {3, 0};
  const std::vector<int32_t> free_slots = {2, 4, 6};
  const int64_t stride = ROW;
  std::vector<uint8_t> slots(8 * stride, 0xCD);

  PagerPlan plan = plan_remap(route, resident, free_slots, fd, HEADER, ROW,
                              slots.data(), stride);
  // 5->2, 1->4, 5->2 (dup), 9->6. Three unique reads.
  assert(plan.slots == std::vector<int32_t>({2, 4, 2, 6}));
  assert(plan.reads.size() == 3);
  int failed = run_batched_reads(plan.reads, 4);
  assert(failed == 0);

  // Zero-copy MX handoff: wrap slot 2 (expert 5) as [16] uint32.
  mx::array view(slots.data() + 2 * stride, {16}, mx::uint32, [](void*) {});
  mx::eval(view);
  const uint32_t* got = view.data<uint32_t>();
  for (int j = 0; j < 16; ++j) {
    assert(got[j] == expert_data[5 * 16 + j]);
  }
  // Slot 4 (expert 1) and slot 6 (expert 9) also exact.
  mx::array v4(slots.data() + 4 * stride, {16}, mx::uint32, [](void*) {});
  mx::array v6(slots.data() + 6 * stride, {16}, mx::uint32, [](void*) {});
  mx::eval(v4, v6);
  for (int j = 0; j < 16; ++j) {
    assert(v4.data<uint32_t>()[j] == expert_data[1 * 16 + j]);
    assert(v6.data<uint32_t>()[j] == expert_data[9 * 16 + j]);
  }

  close(fd);
  std::remove(path.c_str());
  std::printf("test_pager_read OK\n");
  return 0;
}
