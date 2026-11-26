// probe_nemo_moe_batch -- CPU/MLX-only probe (no checkpoint, no guard).
//
// QUESTION (design doc Stage 0b for Nemotron): is moe_block_relu2 batch-invariant?
//   A = moe_block_relu2(x[N,H], route[N,K], scores[N,K], up, down)   -- one call
//   B = concat([ moe_block_relu2(x[i:i+1], route[i:i+1], scores[i:i+1], ...)
//                for i in 0..N ])                                    -- N calls at M=1
// If A == B bit-for-bit, the batched MoE round preserves the solo reference.
//
// Build (uses the engine library built by build_nemotron_multisession.sh):
//   MLX=.../mlx
//   xcrun clang++ -std=c++20 -O2 -I"$MLX/include" -Icpp/engine/src \
//     cpp/engine/tools/probe_nemo_moe_batch.cpp /tmp/nemo_ms_obj/libvey.a \
//     -L"$MLX/lib" -lmlx -Wl,-rpath,"$MLX/lib" -o /tmp/probe_nemo_moe_batch
#include <mlx/mlx.h>
#include "engine.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

static uint64_t st = 0x9e3779b97f4a7c15ULL;
static uint32_t nu32() { st = st*6364136223846793005ULL + 1442695040888963403ULL; return (uint32_t)(st>>32); }

static uint32_t f32bits(float f){ uint32_t u; __builtin_memcpy(&u,&f,4); return u; }

int main() {
  const int N = 4, K = 6, H = 256, E = 128, G = 64, BITS = 4;
  // Synthetic Q4 weights: packed uint32 [E, out, in/8], scales/biases bf16 [E,out,in/G].
  auto wshape = mx::Shape{E, H, H/8};
  auto sshape = mx::Shape{E, H, H/G};
  std::vector<uint32_t> wd((size_t)E*H*(H/8));
  for (auto& v : wd) v = nu32();
  std::vector<mx::bfloat16_t> sd((size_t)E*H*(H/G)), bd((size_t)E*H*(H/G));
  for (auto& v : sd) v = mx::bfloat16_t((float)((int)(nu32()%2001)-1000)/1000.0f);
  for (auto& v : bd) v = mx::bfloat16_t((float)((int)(nu32()%2001)-1000)/1000.0f);
  Q4Projection up(mx::array(wd.data(), wshape, mx::uint32),
                  mx::array(sd.data(), sshape, mx::bfloat16),
                  mx::array(bd.data(), sshape, mx::bfloat16));
  Q4Projection down(mx::array(wd.data(), wshape, mx::uint32),
                    mx::array(sd.data(), sshape, mx::bfloat16),
                    mx::array(bd.data(), sshape, mx::bfloat16));

  std::vector<float> xd((size_t)N*H);
  for (auto& v : xd) v = (float)((int)(nu32()%2001)-1000)/1000.0f;
  mx::array x(xd.data(), mx::Shape{N,H}, mx::float32);
  std::vector<int32_t> rd((size_t)N*K);
  std::vector<float> scd((size_t)N*K);
  for (int i=0;i<N*K;++i){ rd[i]=nu32()%E; scd[i]=(float)(nu32()%100)/100.0f; }
  mx::array route(rd.data(), mx::Shape{N,K}, mx::int32);
  mx::array scores(scd.data(), mx::Shape{N,K}, mx::float32);

  // A: one batched call.
  mx::array A = moe_block_relu2(x, route, scores, up, down);
  mx::eval(A);

  // B: N separate M=1 calls, concatenated.
  std::vector<mx::array> parts;
  for (int i=0;i<N;++i){
    mx::array xi = mx::reshape(mx::slice(x, mx::Shape{i,0}, mx::Shape{i+1,H}), mx::Shape{1,H});
    mx::array ri = mx::reshape(mx::slice(route, mx::Shape{i,0}, mx::Shape{i+1,K}), mx::Shape{1,K});
    mx::array si = mx::reshape(mx::slice(scores, mx::Shape{i,0}, mx::Shape{i+1,K}), mx::Shape{1,K});
    parts.push_back(moe_block_relu2(xi, ri, si, up, down));
  }
  mx::array B = mx::concatenate(parts, 0);
  mx::eval(B);

  mx::array Af = mx::reshape(mx::astype(A, mx::float32), mx::Shape{-1});
  mx::array Bf = mx::reshape(mx::astype(B, mx::float32), mx::Shape{-1});
  mx::eval(Af, Bf);
  const float* pa = Af.data<float>(); const float* pb = Bf.data<float>();
  size_t total = Af.size(), diff = 0; uint32_t maxulp=0; float maxabs=0;
  for (size_t i=0;i<total;++i){
    uint32_t ba=f32bits(pa[i]), bb=f32bits(pb[i]);
    if (ba!=bb){ ++diff; uint32_t oa=(ba&0x80000000u)?~ba:(ba|0x80000000u), ob=(bb&0x80000000u)?~bb:(bb|0x80000000u);
      uint32_t u=oa>=ob?oa-ob:ob-oa; if(u>maxulp)maxulp=u; float d=pa[i]-pb[i]; if(d<0)d=-d; if(d>maxabs)maxabs=d; }
  }
  std::printf("moe_block_relu2 batch-invariance: total=%zu differing=%zu (%.4f%%) max_ulp=%u max_abs=%g\n",
              total, diff, total?100.0*diff/total:0.0, maxulp, maxabs);
  std::printf("VERDICT: %s\n", diff==0?"BATCH-INVARIANT (bit-exact)":"DIVERGENT");
  return diff==0?0:1;
}
