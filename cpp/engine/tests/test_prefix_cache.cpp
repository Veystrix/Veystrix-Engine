// CPU-only unit test for the bit-exact prefix cache (no model, no SSD benchmark).
// Verifies capture -> save -> load -> restore is byte-identical, copy-on-write,
// and fails closed on token/fingerprint mismatch.
#include "prefix_cache.hpp"

#include <mlx/mlx.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static mx::array randf(std::initializer_list<int> shape, unsigned long long& seed, float scale) {
  size_t n = 1; for (int d : shape) n *= size_t(d);
  std::vector<float> v(n);
  for (auto& x : v) { seed = seed*6364136223846793005ULL+1442695040888963407ULL; x = float(int(seed>>33))/1000000000.f*scale; }
  return mx::array(v.data(), mx::Shape(shape.begin(), shape.end()), mx::float32);
}

static bool arrays_equal(const mx::array& a, const mx::array& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  mx::array af = mx::astype(mx::reshape(a, {-1}), mx::uint32);
  mx::array bf = mx::astype(mx::reshape(b, {-1}), mx::uint32);
  mx::eval(af, bf);
  const uint32_t* pa = af.data<uint32_t>(); const uint32_t* pb = bf.data<uint32_t>();
  for (size_t i = 0; i < af.size(); ++i) if (pa[i] != pb[i]) return false;
  return true;
}

int main() {
  const std::string fp = "unit-fingerprint-v1";
  const std::vector<int32_t> tokens{11, 22, 33, 44, 55};
  unsigned long long seed = 123ULL;

  // Build two tiny layers: one linear (gdn+qsa), one full-attention (gdn+full+qsa).
  std::vector<DecoderFullLayerCache> caches;
  {
    auto g = GDNCacheState(randf({1,3,8}, seed, 1.f), randf({1,4,6,6}, seed, 1.f));
    QSACache q; q.index_keys = randf({1,5,4}, seed, 1.f); q.index_position_ids = randf({1,5}, seed, 10.f); q.offset = 5;
    caches.emplace_back(DecoderLayerCache(std::move(g), std::move(q)), FullAttentionCache{});
  }
  {
    auto g = GDNCacheState(randf({1,3,8}, seed, 1.f), randf({1,4,6,6}, seed, 1.f));
    QSACache q; q.index_keys = randf({1,5,4}, seed, 1.f); q.index_position_ids = randf({1,5}, seed, 10.f); q.offset = 5;
    FullAttentionCache f;
    f.keys = randf({1,2,5,4}, seed, 1.f);
    // Non-contiguous transposed values (as in the real model's KV cache):
    // regression test that capture materializes strided views contiguously.
    mx::array vbase = randf({1,2,4,5}, seed, 1.f);
    f.values = mx::transpose(vbase, {0,1,3,2});  // [1,2,5,4], strided
    f.offset = 5;
    caches.emplace_back(DecoderLayerCache(std::move(g), std::move(q)), std::move(f));
  }

  PrefixSnapshot snap = prefix_capture(caches, tokens, fp);
  if (snap.layers.size() != 2) { std::fprintf(stderr, "FAIL: layer count\n"); return 1; }
  const std::string k1 = snap.chain_key;
  const std::string k2 = prefix_chain_key(fp, tokens);
  if (k1 != k2) { std::fprintf(stderr, "FAIL: chain key\n"); return 1; }

  std::vector<DecoderFullLayerCache> restored = prefix_restore(snap, 2);
  for (size_t l = 0; l < 2; ++l) {
    if (!arrays_equal(caches[l].base.gdn.conv_state, restored[l].base.gdn.conv_state) ||
        !arrays_equal(caches[l].base.gdn.recurrent_state, restored[l].base.gdn.recurrent_state)) {
      std::fprintf(stderr, "FAIL: gdn layer %zu\n", l); return 1;
    }
    if (!arrays_equal(*caches[l].base.qsa.index_keys, *restored[l].base.qsa.index_keys) ||
        !arrays_equal(*caches[l].base.qsa.index_position_ids, *restored[l].base.qsa.index_position_ids)) {
      std::fprintf(stderr, "FAIL: qsa layer %zu\n", l); return 1;
    }
    if (caches[l].full.keys.has_value() != restored[l].full.keys.has_value()) { std::fprintf(stderr,"FAIL: full presence\n"); return 1; }
    if (caches[l].full.keys.has_value() &&
        (!arrays_equal(*caches[l].full.keys, *restored[l].full.keys) ||
         !arrays_equal(*caches[l].full.values, *restored[l].full.values))) {
      std::fprintf(stderr, "FAIL: full layer %zu\n", l); return 1;
    }
  }

  // Save / load round trip.
  const std::string dir = "/tmp/veystrix_prefix_cache_test";
  prefix_save(snap, dir);
  PrefixSnapshot loaded = prefix_load(dir, fp, tokens);
  if (loaded.chain_key != k1) { std::fprintf(stderr, "FAIL: loaded chain key\n"); return 1; }
  std::vector<DecoderFullLayerCache> r2 = prefix_restore(loaded, 2);
  for (size_t l = 0; l < 2; ++l)
    if (!arrays_equal(caches[l].base.gdn.recurrent_state, r2[l].base.gdn.recurrent_state)) {
      std::fprintf(stderr, "FAIL: save/load layer %zu\n", l); return 1;
    }

  // Copy-on-write: mutating a restored array must not change a fresh restore.
  {
    mx::array fresh = prefix_restore(snap, 2)[0].base.gdn.recurrent_state;
    mx::array mutated = mx::astype(fresh, mx::float32) + mx::array(1.0f, mx::float32);
    mx::eval(mutated);
    std::vector<DecoderFullLayerCache> again = prefix_restore(snap, 2);
    if (!arrays_equal(caches[0].base.gdn.recurrent_state, again[0].base.gdn.recurrent_state)) {
      std::fprintf(stderr, "FAIL: copy-on-write\n"); return 1;
    }
  }

  // Fail closed on wrong token / fingerprint.
  bool threw = false;
  try { prefix_load(dir, fp, std::vector<int32_t>{11,22,33,44,99}); } catch (...) { threw = true; }
  if (!threw) { std::fprintf(stderr, "FAIL: token mismatch not rejected\n"); return 1; }
  threw = false;
  try { prefix_load(dir, "wrong-fingerprint", tokens); } catch (...) { threw = true; }
  if (!threw) { std::fprintf(stderr, "FAIL: fingerprint mismatch not rejected\n"); return 1; }

  std::printf("test_prefix_cache OK (capture/save/load/restore bit-exact, COW, fail-closed)\n");
  return 0;
}
