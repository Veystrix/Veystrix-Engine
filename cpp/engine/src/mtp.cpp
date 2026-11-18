#include "mtp.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace veystrix::engine {
namespace {

// Qwen4ExpRMSNorm with group_size=hidden_size (the hyper-connection norm).
// Python: reshape [...,HC,H]; per-group RMS over last axis; gain = 1 + weight.
mx::array hc_norm_one_centered(const mx::array& x, const mx::array& raw_weight,
                               int hc_count, int hidden_size, float eps) {
  mx::Shape groups(x.shape().begin(), x.shape().end() - 1);
  groups.push_back(hc_count);
  groups.push_back(hidden_size);
  mx::array w2d = mx::reshape(raw_weight, mx::Shape{hc_count, hidden_size});
  // qsa_rms_norm does: f32, rsqrt(mean(square)+eps), *(1+weight), cast back,
  // all on the grouped [...,HC,H] view. Reshape back to the flat [...,HC*H].
  return mx::reshape(qsa_rms_norm(mx::reshape(x, groups), w2d, eps), x.shape());
}

struct GatedResult {
  mx::array mixed_input;
  std::optional<mx::array> injection_weights;
};

// Qwen4ExpGatedResidual.__call__ (one-centered hc_norm, silu low-rank mix).
// `inject_w` absent => use_combine=False (the MTP hyper-connection mixer).
GatedResult gated_residual_mtp(const mx::array& hyper_input,
                               const mx::array& raw_norm_weight,
                               const mx::array& down_w, const mx::array& up_w,
                               const std::optional<mx::array>& inject_w,
                               int hc_count, int hidden_size, float eps) {
  mx::array normed = hc_norm_one_centered(hyper_input, raw_norm_weight,
                                          hc_count, hidden_size, eps);
  int hc_hidden = hc_count * hidden_size;
  mx::Shape groups(hyper_input.shape().begin(), hyper_input.shape().end() - 1);
  groups.push_back(hc_count);
  groups.push_back(hidden_size);

  // mix = sigmoid(up(silu(down(normed) / hc_count)))
  mx::array hc_arr = mx::array(hc_count);  // int32 scalar: keeps dtype in Python
  mx::array low = mx::divide(linear(normed, Linear(down_w)), hc_arr);
  mx::array mix = mx::sigmoid(linear(hc_silu(low), Linear(up_w)));
  mx::array mixed_input = mx::mean(
      mx::multiply(mx::reshape(mix, groups), mx::reshape(normed, groups)), -2);

  if (inject_w) {
    // injection = 2 * sigmoid(block_inject(normed) / hc_count)
    mx::array two = mx::array(2);  // int32 scalar
    mx::array inj = mx::multiply(
        two, mx::sigmoid(mx::divide(linear(normed, Linear(*inject_w)), hc_arr)));
    return GatedResult{mixed_input, inj};
  }
  return GatedResult{mixed_input, std::nullopt};
}

// Fold an injection (branch output broadcast over HC streams) back into the
// residual: hyper_input + (branch[...,None,:] * inj[...,None]).reshape(shape).
mx::array inject_residual(const mx::array& hyper_input, const mx::array& branch,
                          const mx::array& inj) {
  mx::array b = mx::expand_dims(branch, -2);  // [..., 1, H]
  mx::array i = mx::expand_dims(inj, -1);     // [..., HC, 1]
  return hyper_input + mx::reshape(mx::multiply(b, i), hyper_input.shape());
}

// One dense Qwen4ExpAttention step with a fresh cache (single token). The QSA
// indexer (1 token, no complete blocks) returns None, so this is a plain dense
// softmax over the single key; rotary is evaluated at `cfg.position`.
mx::array dense_attention(const MTPConfig& cfg, const MTPWeights& w,
                          const mx::array& x) {
  const int B = static_cast<int>(x.shape(0));
  const int S = static_cast<int>(x.shape(1));
  const int nH = cfg.n_heads;
  const int nKV = cfg.n_kv_heads;
  const int D = cfg.head_dim;

  mx::array qproj = linear(x, Linear(w.q_proj));             // [B,S,nH*2D]
  mx::array kproj = linear(x, Linear(w.k_proj));             // [B,S,nKV*D]
  mx::array vproj = linear(x, Linear(w.v_proj));             // [B,S,nKV*D]

  mx::array q4 = mx::reshape(qproj, mx::Shape{B, S, nH, 2 * D});
  std::vector<mx::array> halves = mx::split(q4, 2, -1);
  mx::array q = halves[0];        // [B, S, nH, D]
  mx::array gate = halves[1];     // [B, S, nH, D]
  gate = mx::reshape(gate, mx::Shape{B, S, nH * D});

  // Qwen4ExpAttention.q_norm/k_norm are one-centered Qwen4ExpRMSNorm.
  q = mx::transpose(qsa_rms_norm(q, w.q_norm, cfg.eps), {0, 2, 1, 3});
  mx::array k = mx::transpose(
      qsa_rms_norm(mx::reshape(kproj, mx::Shape{B, S, nKV, D}), w.k_norm,
                   cfg.eps),
      {0, 2, 1, 3});
  mx::array v = mx::transpose(mx::reshape(vproj, mx::Shape{B, S, nKV, D}),
                              {0, 2, 1, 3});

  // MRoPE position ids: 3 axes, all equal to cfg.position for text positions.
  std::vector<int32_t> pos(static_cast<size_t>(3) * B * S,
                           static_cast<int32_t>(cfg.position));
  mx::array position_ids(pos.begin(), mx::Shape{3, B, S}, mx::int32);
  QSARotaryParams rp;
  rp.dim = static_cast<int>(D * cfg.partial_rotary_factor);
  rp.base = cfg.rope_base;
  rp.mrope_section = cfg.mrope_section;
  rp.max_position_embeddings = cfg.max_position_embeddings;
  q = qsa_apply_rope(q, position_ids, rp);
  k = qsa_apply_rope(k, position_ids, rp);

  // No mask (Python mask is None for a fresh cache with S=1). SDPA broadcasts
  // nKV -> nH (GQA); with a single key the softmax is 1, so out == v.
  const float scale = static_cast<float>(std::pow(static_cast<double>(D), -0.5));
  mx::array out = mx::fast::scaled_dot_product_attention(
      q, k, v, scale, /*mask_mode=*/"", /*mask_arr=*/std::nullopt,
      /*sinks=*/std::nullopt);
  out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}),
                    mx::Shape{B, S, nH * D});
  // o_proj(out * sigmoid(gate)), matching Qwen3_5Attention.__call__.
  return linear(mx::multiply(out, mx::sigmoid(gate)), Linear(w.o_proj));
}

// Dense Qwen3_5MoeSparseMoeBlock: router + SwitchGLU experts + shared expert.
mx::array dense_moe(const MTPConfig& cfg, const MTPWeights& w,
                    const mx::array& x) {
  // layer.mlp gate: softmax(precise) + argpartition + renorm (same as router()).
  RouterOutput r = router(linear(x, Linear(w.gate)), cfg.top_k, 0.0f);

  // SwitchGLU (no sort: top_k < 64): x[...,None,None,H], gather_mm with the
  // per-expert weights stored [E, out, in] and swapped to [E, in, out].
  mx::array x4 = mx::expand_dims(x, std::vector<int>{-2, -3});
  mx::array up = mx::gather_mm(x4, mx::swapaxes(w.sw_up, -1, -2),
                               std::nullopt, r.indices, false);
  mx::array gm = mx::gather_mm(x4, mx::swapaxes(w.sw_gate, -1, -2),
                               std::nullopt, r.indices, false);
  mx::array act = swiglu(gm, up);  // up * silu(gate)
  mx::array down = mx::gather_mm(act, mx::swapaxes(w.sw_down, -1, -2),
                                 std::nullopt, r.indices, false);
  mx::array y = mx::squeeze(down, -2);                  // [B,S,topk,H]
  y = mx::sum(mx::multiply(y, mx::expand_dims(r.scores, -1)), -2);  // [B,S,H]

  // shared_y = sigmoid(shared_expert_gate(x)) * shared_expert(x)
  mx::array sg = mx::sigmoid(linear(x, Linear(w.shared_gate_logits)));
  mx::array sh_gate = linear(x, Linear(w.shared_gate));
  mx::array sh_up = linear(x, Linear(w.shared_up));
  mx::array sh = linear(swiglu(sh_gate, sh_up), Linear(w.shared_down));
  return y + mx::multiply(sg, sh);
}

}  // namespace

mx::array mtp_fuse_inputs(const MTPConfig& cfg, const MTPWeights& w,
                          const mx::array& token_embeddings,
                          const mx::array& hidden_states) {
  int hc_hidden = cfg.hc_count * cfg.hidden_size;
  if (hidden_states.shape(hidden_states.shape().size() - 1) != hc_hidden) {
    throw std::invalid_argument(
        "mtp_fuse_inputs: hidden last dim != hc_count * hidden_size");
  }
  // fc_embedding(rmsnorm(embed)) -> [B,S,H]
  mx::array proj_emb = linear(
      qsa_rms_norm(token_embeddings, w.pre_fc_norm_embedding, cfg.eps),
      Linear(w.fc_embedding));
  // rmsnorm(hidden) -> [B,S,HC*H] -> reshape [B,S,HC,H] -> fc_hidden (broadcast)
  mx::array streams = mx::reshape(
      qsa_rms_norm(hidden_states, w.pre_fc_norm_hidden, cfg.eps),
      [&]() {
        mx::Shape s(hidden_states.shape().begin(), hidden_states.shape().end());
        s[s.size() - 1] = cfg.hc_count;
        s.push_back(cfg.hidden_size);
        return s;
      }());
  mx::array proj_hidden = linear(streams, Linear(w.fc_hidden));
  mx::array fused = mx::add(mx::expand_dims(proj_emb, -2), proj_hidden);
  mx::Shape out_shape(hidden_states.shape().begin(), hidden_states.shape().end());
  out_shape[out_shape.size() - 1] = hc_hidden;
  return mx::reshape(fused, out_shape);
}

mx::array mtp_draft_forward(const MTPConfig& cfg, const MTPWeights& w,
                            const mx::array& hidden_states,
                            const mx::array& next_token_ids) {
  // nn.Embedding lookup: gather embed_tokens.weight rows at next_token_ids.
  // w.emb is [V,H]; take(..., axis=-2) with ids [B,S] yields [B,S,H].
  mx::array token_emb =
      mx::take(w.emb, mx::astype(next_token_ids, mx::int32), -2);

  mx::array fused = mtp_fuse_inputs(cfg, w, token_emb, hidden_states);

  // ---- layer: attention hyper-connection + QSA attention -------------------
  GatedResult ag = gated_residual_mtp(fused, w.attn_hc_norm, w.attn_down,
                                      w.attn_up, w.attn_inject, cfg.hc_count,
                                      cfg.hidden_size, cfg.eps);
  mx::array attn_branch = dense_attention(cfg, w, ag.mixed_input);
  mx::array h1 = inject_residual(fused, attn_branch, *ag.injection_weights);

  // ---- layer: MLP hyper-connection + dense MoE -----------------------------
  GatedResult mg = gated_residual_mtp(h1, w.mlp_hc_norm, w.mlp_down, w.mlp_up,
                                      w.mlp_inject, cfg.hc_count,
                                      cfg.hidden_size, cfg.eps);
  mx::array moe_branch = dense_moe(cfg, w, mg.mixed_input);
  mx::array h2 = inject_residual(h1, moe_branch, *mg.injection_weights);

  // ---- hyper-connection mixer (use_combine=False) + tied LM head -----------
  GatedResult mix = gated_residual_mtp(h2, w.mix_hc_norm, w.mix_down, w.mix_up,
                                       std::nullopt, cfg.hc_count,
                                       cfg.hidden_size, cfg.eps);
  return linear(mix.mixed_input, Linear(w.emb));  // embed_tokens.as_linear
}

MTPVerifyResult mtp_verify(const std::vector<mx::array>& rows,
                           const std::vector<int32_t>& drafts,
                           const SamplingParams& params, uint64_t seed,
                           const std::vector<int32_t>& seen) {
  mx::random::seed(seed);
  std::vector<int32_t> history(seen.begin(), seen.end());

  // Draw one token from a (possibly overridden) masked row and remap through
  // the top-k survivor ids, exactly like verify_point_mass's categorical.
  auto draw_from = [](const Distribution& dist,
                      const mx::array& masked) -> int32_t {
    mx::array drawn = mx::random::categorical(masked, -1);
    mx::array token = dist.has_ids ? mx::take(dist.ids, drawn, -1) : drawn;
    mx::eval(token);
    return static_cast<int32_t>(
        mx::astype(mx::reshape(token, {-1}), mx::int32).item<int32_t>());
  };

  for (size_t index = 0; index < drafts.size(); ++index) {
    int32_t draft = drafts[index];
    Distribution dist = sampling_distribution(rows[index], params, history);

    // probabilities = softmax(masked); hit = (kept == draft) | arange == draft
    mx::array probabilities = mx::softmax(dist.masked, -1);
    mx::array draft_arr(static_cast<int32_t>(draft));
    mx::array hit = dist.has_ids
                        ? mx::equal(dist.ids, draft_arr)
                        : mx::equal(mx::arange(dist.masked.shape(0)), draft_arr);
    mx::array chance = mx::sum(mx::where(hit, probabilities, mx::array(0.0f)));
    mx::array draw = mx::random::uniform({});  // scalar, matches Python no-arg
    mx::eval(chance, draw);                    // Python: mx.eval(chance, draw)
    float c = chance.item<float>();
    float d = draw.item<float>();

    // A draft holding (almost) all the mass stands even if the draw is high.
    if (d < c || c > 0.999999f) {
      if (params.presence_penalty != 0.0f) history.push_back(draft);
      continue;
    }
    // Rejected: redraw from the target with the draft token removed.
    mx::array residual = mx::where(
        hit, mx::array(-std::numeric_limits<float>::infinity()), dist.masked);
    return MTPVerifyResult{static_cast<int>(index), draw_from(dist, residual)};
  }

  // All drafts accepted: draw the next token from the final target row.
  Distribution final_dist =
      sampling_distribution(rows[drafts.size()], params, history);
  return MTPVerifyResult{static_cast<int>(drafts.size()),
                         draw_from(final_dist, final_dist.masked)};
}

}  // namespace veystrix::engine
