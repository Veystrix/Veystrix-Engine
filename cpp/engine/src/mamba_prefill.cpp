#include "mamba_prefill.hpp"

#include "engine.hpp"  // swiglu, linear, hc_silu

#include <mlx/mlx.h>
#include <mlx/fast.h>

#include <cmath>
#include <vector>

namespace mx = mlx::core;
namespace veystrix::engine {

namespace {

// mlx_lm.models.ssm.compute_dt (same compiled graph as mamba_kernel.cpp).
mx::array prefill_compute_dt(const mx::array& dt_raw, const mx::array& dt_bias,
                             float tmin, float tmax) {
  static auto fn = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array dt = mx::astype(in[0], mx::float32);
        dt = mx::logaddexp(dt + in[1], mx::array(0.0f));
        dt = mx::clip(dt, in[2], in[3]);
        return {dt};
      },
      /*shapeless=*/true);
  return fn({dt_raw, dt_bias, mx::array(tmin), mx::array(tmax)})[0];
}

// ssm.py:segsum(x) with mask=None.
//   x: [B,H,L] -> [B,H,L,L] (strict lower triangular cumsum along axis -2)
mx::array segsum(const mx::array& x) {
  const int L = static_cast<int>(x.shape(-1));
  mx::array rep = mx::repeat(mx::expand_dims(x, -1), L, /*axis=*/-1);
  mx::array tri = mx::tril(rep, -1);
  return mx::cumsum(tri, -2);
}

// ssm.py:ssm_attn (the seq_len>1 branch of ssm_update). Faithful op-for-op port.
struct SSMOut {
  mx::array y;      // [B,S,H,Dh]
  mx::array state;  // [B,H,Dh,Ds]
};

SSMOut ssm_attn(const mx::array& x, const mx::array& A_log,
                const mx::array& B, const mx::array& C, const mx::array& D,
                const mx::array& dt_raw, const mx::array& dt_bias,
                const mx::array& state_in, float tmin, float tmax,
                int step) {
  const int b = static_cast<int>(x.shape(0));
  const int l = static_cast<int>(x.shape(1));
  const int h = static_cast<int>(x.shape(2));
  const int dh = static_cast<int>(x.shape(3));
  const int g = static_cast<int>(B.shape(2));
  const int ds = static_cast<int>(B.shape(3));
  const int repeats = h / g;

  mx::array dt = prefill_compute_dt(dt_raw, dt_bias, tmin, tmax);  // [b,l,h]
  mx::array A = mx::negative(mx::exp(A_log));                     // [h] f32
  A = mx::astype(A, dt.dtype());
  mx::array A_reshaped = mx::reshape(A, mx::Shape{1, 1, h});
  mx::array dtA = dt * A_reshaped;                                // [b,l,h]
  mx::array dtx = mx::reshape(dt, mx::Shape{b, l, h, 1}) * x;     // [b,l,h,dh]

  mx::array state = state_in;
  std::vector<mx::array> ys;

  for (int i = 0; i < l; i += step) {
    const int s = std::min(step, l - i);
    auto sl = [&](const mx::array& v) {
      return mx::slice(v, {0, i, 0, 0}, {b, i + s, h, dh}, {1, 1, 1, 1});
    };
    mx::array dtx_c = sl(dtx);
    mx::array dtA_c =
        mx::slice(dtA, {0, i, 0}, {b, i + s, h}, {1, 1, 1});       // [b,s,h]
    mx::array B_c =
        mx::slice(B, {0, i, 0, 0}, {b, i + s, g, ds}, {1, 1, 1, 1});
    mx::array C_c =
        mx::slice(C, {0, i, 0, 0}, {b, i + s, g, ds}, {1, 1, 1, 1});

    // B = transpose(B, (0,2,3,1)) -> [b,g,ds,s]
      mx::array Bt = mx::transpose(B_c, {0, 2, 3, 1});
    // CB = swapaxes(C,1,2) @ B  -> [b,g,s,ds] @ [b,g,ds,s] -> [b,g,s,s]
    mx::array CB = mx::matmul(mx::swapaxes(C_c, 1, 2), Bt);
    CB = mx::repeat(CB, repeats, /*axis=*/1);                      // [b,h,s,s]

    // decay = exp(segsum(dtA.swapaxes(1,2)))  -> [b,h,s,s]
    mx::array decay = mx::exp(segsum(mx::swapaxes(dtA_c, 1, 2)));
      mx::array attn = mx::tril(CB * decay, 0);                      // [b,h,s,s]

    // y = attn @ dtx.swapaxes(1,2) -> [b,h,s,dh]; y.swapaxes(1,2) -> [b,s,h,dh]
      mx::array y = mx::matmul(attn, mx::swapaxes(dtx_c, 1, 2));
    y = mx::swapaxes(y, 1, 2);

    // decay = decay[:,:,-1:,:].transpose(0,3,1,2) -> [b,s,h,1]
    decay = mx::slice(decay, {0, 0, s - 1, 0}, {b, h, s, s}, {1, 1, 1, 1});
    decay = mx::reshape(decay, mx::Shape{b, h, 1, s});
    decay = mx::transpose(decay, {0, 3, 1, 2});                    // [b,s,h,1]

    // B = repeat(B,h//g,axis=1).swapaxes(2,3) -> [b,h,s,ds]
    mx::array Bh = mx::repeat(Bt, repeats, /*axis=*/1);
    Bh = mx::swapaxes(Bh, 2, 3);                                   // [b,h,s,ds]

    // dtxdecay = (dtx*decay).swapaxes(1,2).swapaxes(2,3) -> [b,h,dh,s]
    mx::array dtxdecay = dtx_c * decay;
    dtxdecay = mx::swapaxes(mx::swapaxes(dtxdecay, 1, 2), 2, 3);
      mx::array next_state = mx::matmul(dtxdecay, Bh);               // [b,h,dh,ds]

      if (state.size() != 0) {
      mx::array cumsum = mx::cumsum(dtA_c, /*axis=*/-2);            // [b,s,h]
      mx::array exp_cum = mx::exp(cumsum);
      mx::array last =
          mx::slice(exp_cum, {0, s - 1, 0}, {b, s, h}, {1, 1, 1});
      last = mx::reshape(last, mx::Shape{b, h, 1, 1});  // Python [b,h,1,1]
      next_state = next_state + last * state;

      // y_prev = (state.reshape(b,1,g,repeats,dh,ds) @
      //            C.reshape(b,s,g,1,ds,1)).squeeze(-1).flatten(2,3)
      mx::array st5 = mx::reshape(state, mx::Shape{b, 1, g, repeats, dh, ds});
      mx::array C6 = mx::reshape(C_c, mx::Shape{b, s, g, 1, ds, 1});
      mx::array yprev = mx::matmul(st5, C6);
      yprev = mx::reshape(mx::squeeze(yprev, -1), mx::Shape{b, s, h, dh});
      y = y + mx::reshape(exp_cum, mx::Shape{b, s, h, 1}) * yprev;
    }
    ys.push_back(mx::astype(y, x.dtype()));
    state = next_state;
  }

  mx::array out = ys.size() == 1 ? ys[0] : mx::concatenate(ys, 1);
  mx::array Ds = mx::reshape(D, mx::Shape{1, 1, h, 1});
  out = out + x * Ds;
  return {out, state};
}

// Depthwise conv1d over the padded sequence, mirroring nn.Conv1d(groups=C):
// out[b,t,c] = sum_k xpad[b,t+k,c] * weight[c,k,0] + bias[c].
// Uses the same f32 accumulation as mamba_conv.cpp: cast to f32, sum over K,
// cast back, then add bias in the activation dtype.
mx::array full_depthwise_conv(const mx::array& xpad,
                              const mx::array& raw_weight,
                              const mx::array& bias) {
  const int B = static_cast<int>(xpad.shape(0));
  const int T = static_cast<int>(xpad.shape(1));
  const int C = static_cast<int>(xpad.shape(2));
  const int K = static_cast<int>(raw_weight.shape(1));
  // MLX conv1d is the EXACT kernel Python's nn.Conv1d calls (groups=C).
  mx::array conv = mx::conv1d(xpad, raw_weight, /*stride=*/1,
                              /*padding=*/0, /*dilation=*/1,
                              /*groups=*/C);
  if (bias.size() != 0) conv = conv + bias;
  return hc_silu(conv);
}

}  // namespace

MambaLayerOutput mamba_layer_prefill(const mx::array& x,
                                     const MambaWeights& w,
                                     const MambaCacheState& cache,
                                     const MambaConfig& cfg) {
  const int B = static_cast<int>(x.shape(0));
  const int S = static_cast<int>(x.shape(1));
  const int I = cfg.intermediate_size;
  const int CD = cfg.conv_dim;
  const int H = cfg.num_heads;
  const int G = cfg.n_groups;
  const int Ds = cfg.ssm_state_size;
  const int Dh = cfg.head_dim;
  const int K = cfg.conv_kernel;

  mx::array projected = linear(x, w.in_proj);
  std::vector<mx::array> parts =
      mx::split(projected, mx::Shape{I, I + CD}, /*axis=*/-1);
  mx::array gate = parts[0], conv_input = parts[1], dt = parts[2];

  // _conv: pad with the cached conv state (K-1 tokens), full conv, silu,
  // and advance the cache to the last K-1 padded tokens.
  mx::array padded = mx::concatenate({cache.conv_state, conv_input}, /*axis=*/1);
  mx::array new_conv = mx::slice(padded, {0, padded.shape(1) - (K - 1), 0},
                                 {B, padded.shape(1), CD}, {1, 1, 1});
  mx::array conv_out = full_depthwise_conv(padded, w.conv1d_weight,
                                           w.conv1d_bias);  // [B,S,CD]

  std::vector<mx::array> ssm_parts =
      mx::split(conv_out, mx::Shape{I, I + G * Ds}, /*axis=*/-1);
  mx::array hs = ssm_parts[0], Bv = ssm_parts[1], Cv = ssm_parts[2];

  // mixer._ssm reshapes to [B,S,H,Dh] / [B,S,G,Ds] before ssm_update.
  hs = mx::reshape(hs, mx::Shape{B, S, H, Dh});
  Bv = mx::reshape(Bv, mx::Shape{B, S, G, Ds});
  Cv = mx::reshape(Cv, mx::Shape{B, S, G, Ds});
  SSMOut ssm = ssm_attn(hs, w.A_log, Bv, Cv, w.D, dt, w.dt_bias,
                        cache.ssm_state, cfg.time_step_min,
                        cfg.time_step_max, /*step=*/256);

  // MambaRMSNormGated: swiglu(gate, y) -> unflatten group_size -> rms_norm ->
  // weight * flatten.
  mx::array y_flat = mx::reshape(ssm.y, mx::Shape{B, S, I});  // mixer._ssm reshape
  mx::array gated = swiglu(gate, y_flat);
  const int group_size = I / G;
  mx::array unf = mx::reshape(gated, mx::Shape{B, S, G, group_size});
  mx::array normed = mx::fast::rms_norm(unf, /*weight=*/std::nullopt, cfg.eps);
  normed = mx::reshape(normed, mx::Shape{B, S, I});
  normed = mx::reshape(w.norm_weight, mx::Shape{1, 1, I}) * normed;

  mx::array out = linear(normed, w.out_proj);
  return {out, new_conv, ssm.state};
}

}  // namespace veystrix::engine
