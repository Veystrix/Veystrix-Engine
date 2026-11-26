#include "engine.hpp"
#include "gdn_kernel.h"

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <stdexcept>

namespace veystrix::engine {

DeltaStepOutput gated_delta_step_kernel(const mx::array& q, const mx::array& k,
                                        const mx::array& v, const mx::array& g,
                                        const mx::array& beta, const mx::array& state) {
  // Exact decode kernel: mirrors gated_delta_kernel(q,k,v,g,beta,state) with
  // scalar gate (g.ndim==3 -> [B,T,Hv]) and no mask, using the SAME source
  // string the Python engine compiles (verified byte-identical).
  // q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv]; state: [B,Hv,Dv,Dk].
  int B = q.shape(0), T = q.shape(1);
  int Hk = q.shape(2), Dk = q.shape(3);
  int Hv = v.shape(2), Dv = v.shape(3);

  try {
    auto kernel = mx::fast::metal_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        kGdnDeltaKernelSource);

    mx::array Tarr(static_cast<int32_t>(T));  // scalar int input
    std::vector<mx::array> inputs{q, k, v, g, beta, state, Tarr};
    std::vector<mx::Shape> out_shapes{mx::Shape{B, T, Hv, Dv}, mx::Shape(state.shape())};
    std::vector<mx::Dtype> out_dtypes{q.dtype(), state.dtype()};
    auto grid = std::make_tuple(32, Dv, B * Hv);
    auto threadgroup = std::make_tuple(32, 4, 1);
    std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmpl = {
        {"InT", q.dtype()}, {"StT", state.dtype()},
        {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv},
    };
    std::vector<mx::array> out = kernel(inputs, out_shapes, out_dtypes,
                                        grid, threadgroup, tmpl,
                                        std::nullopt, false, mx::StreamOrDevice{});
    return {out[0], out[1]};
  } catch (const std::exception&) {
    // No Metal — fall back to the bit-similar ops reference.
    return gated_delta_step(q, k, v, g, beta, state);
  }
}

}  // namespace veystrix::engine
