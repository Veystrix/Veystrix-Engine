// D3 gate: nemo_snapshot round-trip is bit-exact and fails closed on
// token/fingerprint mismatch or corruption. No checkpoint or GPU model needed.
#include "nemo_snapshot.hpp"
#include "prefix_cache.hpp"

#include <mlx/mlx.h>

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static int failures = 0;
static void check(bool ok, const char* what) {
  if (!ok) { std::cerr << "FAIL: " << what << "\n"; ++failures; }
}
static void expect_throw(const std::string& dir, const std::string& fp,
                         const std::vector<int32_t>& toks, const char* what) {
  try {
    nemotron_runtime::SessionCacheState s =
        nemo_snapshot_restore(dir, fp, toks);
    (void)s;
    std::cerr << "FAIL: " << what << " (did not fail closed)\n";
    ++failures;
  } catch (const std::exception&) { /* expected */ }
}

int main(int argc, char** argv) {
  const std::string dir = (argc > 1 ? std::string(argv[1]) : "results/cpp/nemo-snapshot-test") + "/snap";
  const std::string fp = "nemotron-test-fingerprint";
  const std::vector<int32_t> toks{3, 5, 7};

  // Tiny geometry: 2 Mamba layers, 1 attention layer, K=4, conv_dim=8,
  // H=2, Dh=4, Ds=8. int8 path exercises the packed block layout.
  nemotron_runtime::SessionCacheState state;
  mx::array c = mx::ones(mx::Shape{1, 3, 8}, mx::bfloat16);
  mx::array s = mx::ones(mx::Shape{1, 2, 4, 8}, mx::float32);
  state.mamba_caches.push_back(MambaCacheState(c, s));
  state.mamba_caches.push_back(MambaCacheState(c, s));
  state.attn_int8.push_back(NemoInt8Cache{});
  auto& ac = state.attn_int8[0];
  ac.offset = 5;
  ac.block_tokens = 4096;
  ac.blocks.emplace_back(
      mx::ones(mx::Shape{1, 1, 2, 2}, mx::uint32),
      mx::ones(mx::Shape{1, 1, 2, 2}, mx::bfloat16),
      mx::zeros(mx::Shape{1, 1, 2, 2}, mx::bfloat16),
      mx::ones(mx::Shape{1, 1, 2, 2}, mx::uint32),
      mx::ones(mx::Shape{1, 1, 2, 2}, mx::bfloat16),
      mx::zeros(mx::Shape{1, 1, 2, 2}, mx::bfloat16), 5);

  NemoSnapshot snap = nemo_snapshot_capture(state, toks, fp, /*int8=*/true);
  nemo_snapshot_save(snap, dir);
  nemotron_runtime::SessionCacheState restored = nemo_snapshot_restore(dir, fp, toks);
  check(restored.mamba_caches.size() == 2, "mamba layer count");
  check(restored.attn_int8.size() == 1, "attention layer count");
  check(restored.attn_int8[0].offset == 5, "attention offset");
  check(restored.attn_int8[0].blocks.size() == 1, "int8 block count");
  check(restored.attn_int8[0].blocks[0].offset == 5, "int8 block offset");
  for (const auto& mc : restored.mamba_caches) {
    TensorBlob got = to_blob(mc.ssm_state);
    check(got.bytes == to_blob(s).bytes, "mamba ssm bit-exact");
    TensorBlob gc = to_blob(mc.conv_state);
    check(gc.bytes == to_blob(c).bytes, "mamba conv bit-exact");
  }
  expect_throw(dir, "wrong-fingerprint", toks, "wrong fingerprint");
  expect_throw(dir, fp, std::vector<int32_t>{3, 5}, "wrong tokens");
  expect_throw(dir, fp, std::vector<int32_t>{3, 5, 7, 9}, "non-prefix tokens");
  if (failures == 0)
    std::cout << "nemo_snapshot: round-trip bit-exact + fail-closed gates passed\n";
  return failures == 0 ? 0 : 1;
}
