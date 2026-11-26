#include "generic/inkling_dspark.hpp"

#include <mlx/compile.h>
#include <mlx/fast.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace veystrix::engine::generic {

namespace {

constexpr int64_t kDSparkKvStep = 256;

// Resolve a DSpark tensor prefix across possible checkpoint layouts ("", "mtp.", "model.").
std::string detect_dspark_prefix(const Loader& L) {
  for (const char* pfx : {"", "mtp.", "model."}) {
    if (L.has(std::string(pfx) + "fuse_weight") && L.has(std::string(pfx) + "mask_embed"))
      return pfx;
  }
  return "";
}

int dspark_quant_bits() {
  const char* e = std::getenv("VEYSTRIX_INKLING_DSPARK_BITS");
  if (!e || !*e) return 4;
  const int b = std::atoi(e);
  return (b == 4 || b == 8 || b == 16) ? b : 4;
}

float dspark_timing_ms(const char* name, float fallback) {
  const char* e = std::getenv(name);
  if (!e || !*e) return fallback;
  const float v = static_cast<float>(std::atof(e));
  return v > 0.0f ? v : fallback;
}

QLinear load_dspark_linear(const Loader& L, const std::string& path, int bits) {
  if (L.has(path + ".scales")) return L.linear(path);
  QLinear q;
  mx::array w = L.tensor(path + ".weight");
  if ((bits == 4 || bits == 8) && w.ndim() == 2 && (w.shape(1) % 64 == 0)) {
    auto qs = mx::quantize(w, /*group_size=*/64, /*bits=*/bits, "affine");
    q.w = qs[0];
    q.scales = qs[1];
    q.biases = qs[2];
    q.q = QuantSpec{true, 64, bits, "affine"};
    mx::eval(std::vector<mx::array>{*q.w, *q.scales, *q.biases});
  } else {
    q.w = w;
    mx::eval(*q.w);
  }
  if (L.has(path + ".bias")) {
    q.bias = L.tensor(path + ".bias");
    mx::eval(*q.bias);
  }
  return q;
}

QEmbedding load_dspark_embedding(const Loader& L, const std::string& path) {
  if (L.has(path + ".scales")) return L.embedding(path);
  QEmbedding e;
  e.w = L.tensor(path + ".weight");
  return e;
}

mx::array apply_rope_default(const mx::array& x, int head_dim, float theta, int64_t offset) {
  return mx::fast::rope(x, head_dim, /*traditional=*/false, theta, /*scale=*/1.0f,
                        static_cast<int>(offset));
}

QLinear concat_qlinears(const std::vector<QLinear>& parts) {
  QLinear out;
  out.q = parts[0].q;
  auto cat0 = [&](auto getter) -> std::optional<mx::array> {
    std::vector<mx::array> arrs;
    for (const auto& p : parts) {
      const auto& opt = getter(p);
      if (!opt) return std::nullopt;
      arrs.push_back(*opt);
    }
    return mx::contiguous(mx::concatenate(arrs, 0));
  };
  out.w = cat0([](const QLinear& p) -> const auto& { return p.w; });
  out.scales = cat0([](const QLinear& p) -> const auto& { return p.scales; });
  out.biases = cat0([](const QLinear& p) -> const auto& { return p.biases; });
  out.bias = cat0([](const QLinear& p) -> const auto& { return p.bias; });
  std::vector<mx::array> ev;
  if (out.w) ev.push_back(*out.w);
  if (out.scales) ev.push_back(*out.scales);
  if (out.biases) ev.push_back(*out.biases);
  if (out.bias) ev.push_back(*out.bias);
  if (!ev.empty()) mx::eval(ev);
  return out;
}

mx::array swiglu_mlp_fused(const mx::array& x, const QLinear& gate_up, const QLinear& down, int inter) {
  mx::array gu = gate_up(x);
  const int B = gu.shape(0), S = gu.shape(1);
  mx::array g = mx::slice(gu, {0, 0, 0}, {B, S, inter});
  mx::array u = mx::slice(gu, {0, 0, inter}, {B, S, 2 * inter});
  static auto silu_mul_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array g0 = mx::astype(in[0], mx::float32);
        mx::array act = mx::multiply(mx::multiply(g0, mx::sigmoid(g0)), mx::astype(in[1], mx::float32));
        return {mx::astype(act, in[0].dtype())};
      },
      true);
  return down(silu_mul_c({g, u})[0]);
}

}  // namespace

std::unique_ptr<Inkling2DSpark> Inkling2DSpark::try_load(const Loader& L, const Json& recipe) {
  const Json* mtp = recipe.find("mtp");
  if (!mtp || !mtp->is_object()) return nullptr;
  if (!mtp->get_bool("present", false)) return nullptr;
  if (mtp->get_str("draft_type", "") != "dspark") return nullptr;

  const std::string pfx = detect_dspark_prefix(L);
  if (!L.has(pfx + "fuse_weight") || !L.has(pfx + "mask_embed")) return nullptr;

  auto d = std::unique_ptr<Inkling2DSpark>(new Inkling2DSpark());
  DSparkConfig& c = d->cfg_;
  c.hidden_size = static_cast<int>(mtp->get_i64("hidden_size", recipe.at("hidden_size").i64("hidden_size")));
  c.num_hidden_layers = static_cast<int>(mtp->get_i64("num_hidden_layers", 3));
  c.num_attention_heads = static_cast<int>(mtp->get_i64("num_attention_heads", 24));
  c.num_key_value_heads = static_cast<int>(mtp->get_i64("num_key_value_heads", 8));
  c.head_dim = static_cast<int>(mtp->get_i64("head_dim", 128));
  c.intermediate_size = static_cast<int>(mtp->get_i64("intermediate_size", 9216));
  c.rms_norm_eps = static_cast<float>(mtp->get_num("rms_norm_eps", 1e-6));
  c.rope_theta = static_cast<float>(mtp->get_num("rope_theta", 1000000.0));
  c.sliding_window_size = static_cast<int>(mtp->get_i64("sliding_window_size", 1024));
  c.block_size = static_cast<int>(mtp->get_i64("block_size", 8));
  c.markov_rank = static_cast<int>(mtp->get_i64("markov_rank", 256));
  if (const Json* locs = mtp->find("local_layer_ids"); locs && locs->is_array()) {
    for (const Json& v : locs->arr("local_layer_ids")) c.local_layer_ids.insert(v.i32("local_layer_id"));
  }
  if (const Json* tgs = mtp->find("target_layer_ids"); tgs && tgs->is_array()) {
    for (const Json& v : tgs->arr("target_layer_ids")) c.target_layer_ids.push_back(v.i32("target_layer_id"));
  }
  if (c.target_layer_ids.size() < 2) return nullptr;
  d->scale_ = static_cast<float>(std::pow(static_cast<double>(c.head_dim), -0.5));

  const int qbits = dspark_quant_bits();
  int64_t retained_bytes = 0;

  mx::array fw = mx::astype(L.tensor(pfx + "fuse_weight"), mx::float32);
  d->fuse_alpha_ = mx::contiguous(mx::softmax(fw, -1, /*precise=*/true));
  d->mask_embed_ = mx::contiguous(mx::reshape(L.tensor(pfx + "mask_embed"), {1, 1, c.hidden_size}));
  d->final_norm_ = L.tensor(pfx + "norm.weight");
  retained_bytes += d->fuse_alpha_.nbytes() + d->mask_embed_.nbytes() + d->final_norm_.nbytes();

  std::vector<mx::array> to_eval{d->fuse_alpha_, d->mask_embed_, d->final_norm_};

  d->layers_.reserve(static_cast<size_t>(c.num_hidden_layers));
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    const std::string lp = pfx + "layers." + std::to_string(i) + ".";
    Layer ly;
    ly.is_local = c.local_layer_ids.count(i) > 0;
    ly.input_ln = L.tensor(lp + "input_layernorm.weight");
    ly.post_ln = L.tensor(lp + "post_attention_layernorm.weight");
    ly.dflare_norm = L.tensor(lp + "dflare_norm.weight");
    QLinear qp = load_dspark_linear(L, lp + "self_attn.q_proj", qbits);
    QLinear kp = load_dspark_linear(L, lp + "self_attn.k_proj", qbits);
    QLinear vp = load_dspark_linear(L, lp + "self_attn.v_proj", qbits);
    ly.qkv_proj = concat_qlinears({qp, kp, vp});
    ly.k_proj_ctx = load_dspark_linear(L, lp + "self_attn.k_proj_ctx", qbits);
    ly.v_proj_ctx = load_dspark_linear(L, lp + "self_attn.v_proj_ctx", qbits);
    ly.o_proj = load_dspark_linear(L, lp + "self_attn.o_proj", qbits);
    ly.q_norm = L.tensor(lp + "self_attn.q_norm.weight");
    ly.k_norm = L.tensor(lp + "self_attn.k_norm.weight");
    ly.k_norm_ctx = L.tensor(lp + "self_attn.k_norm_ctx.weight");
    QLinear gp = load_dspark_linear(L, lp + "mlp.gate_proj", qbits);
    QLinear up = load_dspark_linear(L, lp + "mlp.up_proj", qbits);
    ly.gate_up_proj = concat_qlinears({gp, up});
    ly.down_proj = load_dspark_linear(L, lp + "mlp.down_proj", qbits);
    retained_bytes += ly.qkv_proj.nbytes() +
                      ly.k_proj_ctx.nbytes() + ly.v_proj_ctx.nbytes() + ly.o_proj.nbytes() +
                      ly.gate_up_proj.nbytes() + ly.down_proj.nbytes() +
                      ly.input_ln.nbytes() + ly.post_ln.nbytes() + ly.dflare_norm.nbytes() +
                      ly.q_norm.nbytes() + ly.k_norm.nbytes() + ly.k_norm_ctx.nbytes();
    to_eval.push_back(ly.input_ln);
    to_eval.push_back(ly.post_ln);
    to_eval.push_back(ly.dflare_norm);
    to_eval.push_back(ly.q_norm);
    to_eval.push_back(ly.k_norm);
    to_eval.push_back(ly.k_norm_ctx);
    d->layers_.push_back(std::move(ly));
    mx::clear_cache();
  }

  // Recurrent Markov head: markov_w1, rnn_proj, hidden_out, dflare_norm.
  const std::string mp = pfx + "markov_head.";
  d->markov_.markov_w1 = load_dspark_embedding(L, mp + "markov_w1");
  d->markov_.dflare_norm = L.tensor(mp + "dflare_norm.weight");
  mx::array rnn_w = L.tensor(mp + "rnn_proj.weight");  // [3r, 2r + 2H]
  const int r = c.markov_rank;
  const int r3 = 3 * r;
  const int r2 = 2 * r;
  d->markov_.rnn_w_sp = mx::contiguous(mx::transpose(mx::slice(rnn_w, {0, 0}, {r3, r2})));
  d->markov_.rnn_w_hf = mx::contiguous(mx::transpose(mx::slice(rnn_w, {0, r2}, {r3, rnn_w.shape(1)})));
  d->markov_.rnn_bias = mx::contiguous(mx::astype(L.tensor(mp + "rnn_proj.bias"), mx::float32));
  d->markov_.hidden_out_w = mx::contiguous(mx::transpose(L.tensor(mp + "hidden_out.weight")));
  d->markov_.hidden_out_b = mx::contiguous(L.tensor(mp + "hidden_out.bias"));
  if (d->markov_.markov_w1.w) retained_bytes += d->markov_.markov_w1.w->nbytes();
  retained_bytes += d->markov_.dflare_norm.nbytes() + d->markov_.rnn_w_sp.nbytes() +
                    d->markov_.rnn_w_hf.nbytes() + d->markov_.rnn_bias.nbytes() +
                    d->markov_.hidden_out_w.nbytes() + d->markov_.hidden_out_b.nbytes();

  to_eval.push_back(d->markov_.dflare_norm);
  to_eval.push_back(d->markov_.rnn_w_sp);
  to_eval.push_back(d->markov_.rnn_w_hf);
  to_eval.push_back(d->markov_.rnn_bias);
  to_eval.push_back(d->markov_.hidden_out_w);
  to_eval.push_back(d->markov_.hidden_out_b);
  mx::eval(to_eval);
  mx::clear_cache();

  d->weight_bytes_ = retained_bytes;
  return d;
}

void Inkling2DSpark::attach_lm_head(const std::function<mx::array(const mx::array&)>& lm_head_fn,
                                    int unpadded_vocab) {
  // lm_head_fn([r, H]) -> [r, V]; transpose -> [V_trim, r], quantize to 8-bit affine (group 64).
  mx::array fw_rv = lm_head_fn(markov_.hidden_out_w);  // [r, V]
  const int V_trim = (unpadded_vocab > 0 && unpadded_vocab < fw_rv.shape(-1)) ? unpadded_vocab : fw_rv.shape(-1);
  mx::array fw_vr = mx::contiguous(mx::transpose(mx::slice(fw_rv, {0, 0}, {cfg_.markov_rank, V_trim})));  // [V_trim, r]
  auto q = mx::quantize(fw_vr, /*group_size=*/64, /*bits=*/8, "affine");
  mx::array fb = lm_head_fn(mx::reshape(markov_.hidden_out_b, {1, cfg_.hidden_size}));  // [1, V]
  fb = mx::contiguous(mx::slice(fb, {0, 0}, {1, V_trim}));
  markov_.fold_w = q[0];
  markov_.fold_s = q[1];
  markov_.fold_b = q[2];
  markov_.fold_bias = fb;
  markov_.folded = true;
  mx::eval(std::vector<mx::array>{*markov_.fold_w, *markov_.fold_s, *markov_.fold_b, *markov_.fold_bias});
  mx::clear_cache();
  weight_bytes_ += markov_.fold_w->nbytes() + markov_.fold_s->nbytes() + markov_.fold_b->nbytes() +
                   markov_.fold_bias->nbytes();
}

void Inkling2DSpark::reset(DSparkState& st) const {
  st.layers.assign(static_cast<size_t>(cfg_.num_hidden_layers), DSparkLayerKV{});
  st.ctx_len = 0;
  st.last_features.reset();
}

void Inkling2DSpark::truncate(DSparkState& st, int64_t new_n) const {
  if (new_n >= st.ctx_len) return;
  st.ctx_len = std::max<int64_t>(0, new_n);
  for (size_t i = 0; i < st.layers.size(); ++i) {
    auto& lkv = st.layers[i];
    const bool is_local = (i < layers_.size()) && layers_[i].is_local;
    if (lkv.k && lkv.len > 0) {
      const int64_t cap_limit = is_local ? std::min<int64_t>(cfg_.sliding_window_size, new_n) : new_n;
      lkv.len = std::clamp<int64_t>(cap_limit, 0, lkv.len);
    }
  }
}

NamedArrays Inkling2DSpark::save(const DSparkState& st, const std::string& pfx) const {
  NamedArrays out;
  out.emplace_back(pfx + ".ctx_len", mx::array(static_cast<int64_t>(st.ctx_len), mx::int64));
  for (size_t i = 0; i < st.layers.size(); ++i) {
    const auto& lkv = st.layers[i];
    if (lkv.k && lkv.len > 0 && lkv.v) {
      const std::string lp = pfx + ".l" + std::to_string(i);
      out.emplace_back(lp + ".k", *lkv.k);
      out.emplace_back(lp + ".v", *lkv.v);
      out.emplace_back(lp + ".len", mx::array(static_cast<int64_t>(lkv.len), mx::int64));
    }
  }
  if (st.last_features) {
    out.emplace_back(pfx + ".last_features", *st.last_features);
  }
  return out;
}

void Inkling2DSpark::load(DSparkState& st, const NamedArrays& a, const std::string& pfx) const {
  reset(st);
  for (const auto& item : a) {
    if (item.first == pfx + ".ctx_len") {
      mx::eval(item.second);
      st.ctx_len = item.second.data<int64_t>()[0];
    } else if (item.first == pfx + ".last_features") {
      st.last_features = item.second;
    } else {
      for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        const std::string lp = pfx + ".l" + std::to_string(i);
        if (item.first == lp + ".k") st.layers[static_cast<size_t>(i)].k = item.second;
        else if (item.first == lp + ".v") st.layers[static_cast<size_t>(i)].v = item.second;
        else if (item.first == lp + ".len") {
          mx::eval(item.second);
          st.layers[static_cast<size_t>(i)].len = item.second.data<int64_t>()[0];
          st.layers[static_cast<size_t>(i)].cap = st.layers[static_cast<size_t>(i)].len;
        }
      }
    }
  }
}

mx::array Inkling2DSpark::mix_features(const mx::array& features_sth, int row,
                                       const mx::array& norm_w) const {
  // features_sth: [S, T, H] BF16; fuse_alpha_[row]: [T] F32 -> [1, S, H] BF16.
  const int T = static_cast<int>(cfg_.target_layer_ids.size());
  mx::array alpha = mx::reshape(mx::slice(fuse_alpha_, {row, 0}, {row + 1, T}), {1, T, 1});
  mx::array mixed = mx::sum(mx::multiply(mx::astype(features_sth, mx::float32), alpha), 1);  // [S, H] F32
  mx::array normed = mx::fast::rms_norm(mx::astype(mixed, norm_w.dtype()), norm_w, cfg_.rms_norm_eps);
  return mx::expand_dims(normed, 0);  // [1, S, H]
}

void Inkling2DSpark::append_layer_kv(DSparkLayerKV& kv, const mx::array& k, const mx::array& v,
                                     bool is_local) const {
  // k, v: [1, Hkv, S, D] BF16.
  const int64_t S = k.shape(2);
  if (S <= 0) return;
  // Bound both local (1024) and global (4096) drafter KV caches so DSpark stays <=24 MiB even at 1M context.
  const int64_t sw = is_local ? static_cast<int64_t>(cfg_.sliding_window_size) : int64_t{4096};
  if (!kv.k || kv.len == 0) {
    if (S > sw) {
      kv.k = mx::contiguous(mx::slice(k, {0, 0, static_cast<int>(S - sw), 0},
                                      {1, cfg_.num_key_value_heads, static_cast<int>(S), cfg_.head_dim}));
      kv.v = mx::contiguous(mx::slice(v, {0, 0, static_cast<int>(S - sw), 0},
                                      {1, cfg_.num_key_value_heads, static_cast<int>(S), cfg_.head_dim}));
      kv.len = sw;
      kv.cap = sw;
    } else {
      kv.k = mx::contiguous(k);
      kv.v = mx::contiguous(v);
      kv.len = S;
      kv.cap = S;
    }
    return;
  }
  mx::array kc = mx::concatenate({mx::slice(*kv.k, {0, 0, 0, 0},
                                            {1, cfg_.num_key_value_heads, static_cast<int>(kv.len), cfg_.head_dim}),
                                  k},
                                 2);
  mx::array vc = mx::concatenate({mx::slice(*kv.v, {0, 0, 0, 0},
                                            {1, cfg_.num_key_value_heads, static_cast<int>(kv.len), cfg_.head_dim}),
                                  v},
                                 2);
  const int64_t total = kv.len + S;
  if (total > sw) {
    kc = mx::contiguous(mx::slice(kc, {0, 0, static_cast<int>(total - sw), 0},
                                  {1, cfg_.num_key_value_heads, static_cast<int>(total), cfg_.head_dim}));
    vc = mx::contiguous(mx::slice(vc, {0, 0, static_cast<int>(total - sw), 0},
                                  {1, cfg_.num_key_value_heads, static_cast<int>(total), cfg_.head_dim}));
    kv.len = sw;
    kv.cap = sw;
  } else {
    kv.len = total;
    kv.cap = total;
  }
  kv.k = kc;
  kv.v = vc;
}

void Inkling2DSpark::commit_context(DSparkState& st, const mx::array& target_features_sth) const {
  const int S = target_features_sth.shape(0);
  if (S <= 0) return;
  if (static_cast<int>(st.layers.size()) != cfg_.num_hidden_layers) reset(st);
  std::vector<mx::array> outs;
  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    const Layer& ly = layers_[static_cast<size_t>(i)];
    mx::array ctx = mix_features(target_features_sth, i, ly.dflare_norm);  // [1, S, H]
    mx::array k = ly.k_proj_ctx(ctx);                                      // [1, S, Hkv*D]
    mx::array v = ly.v_proj_ctx(ctx);
    k = mx::reshape(k, {1, S, cfg_.num_key_value_heads, cfg_.head_dim});
    k = mx::fast::rms_norm(k, ly.k_norm_ctx, cfg_.rms_norm_eps);
    k = mx::transpose(k, {0, 2, 1, 3});  // [1, Hkv, S, D]
    k = apply_rope_default(k, cfg_.head_dim, cfg_.rope_theta, st.ctx_len);
    v = mx::transpose(mx::reshape(v, {1, S, cfg_.num_key_value_heads, cfg_.head_dim}), {0, 2, 1, 3});
    append_layer_kv(st.layers[static_cast<size_t>(i)], k, v, ly.is_local);
    outs.push_back(*st.layers[static_cast<size_t>(i)].k);
    outs.push_back(*st.layers[static_cast<size_t>(i)].v);
  }
  st.ctx_len += S;
  const int T = target_features_sth.shape(1);
  st.last_features = mx::contiguous(mx::slice(target_features_sth, {S - 1, 0, 0}, {S, T, cfg_.hidden_size}));
  outs.push_back(*st.last_features);
  if (S >= 32) mx::eval(outs);
  else mx::async_eval(outs);
}

std::vector<int32_t> Inkling2DSpark::draft_block(
    const DSparkState& st, int32_t anchor_token, int num_draft,
    const std::function<mx::array(const mx::array&)>& embed_normed_fn,
    const std::function<mx::array(const mx::array&)>& lm_head_fn,
    float logit_softcap, int unpadded_vocab, float min_cum_prob) const {
  const int K = std::clamp(num_draft, 0, max_draft_tokens());
  if (K <= 0 || !st.last_features) return {};
  const int B = K + 1;  // 1 anchor + K mask tokens

  // 1. Anchor embedding (normalized by target's embed_norm) + K mask_embed rows.
  mx::array anchor_id(&anchor_token, mx::Shape{1, 1}, mx::int32);
  mx::array anchor_emb = embed_normed_fn(anchor_id);  // [1, 1, H]
  mx::array masks = mx::broadcast_to(mask_embed_, {1, K, cfg_.hidden_size});
  mx::array x = mx::concatenate({anchor_emb, masks}, 1);  // [1, B, H]

  // 2. Run 3 dense draft layers with bidirectional block attention over [context K/V, block K/V].
  const int dq = cfg_.num_attention_heads * cfg_.head_dim;
  const int dk = cfg_.num_key_value_heads * cfg_.head_dim;
  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    const Layer& ly = layers_[static_cast<size_t>(i)];
    mx::array h = mx::fast::rms_norm(x, ly.input_ln, cfg_.rms_norm_eps);
    mx::array qkv = ly.qkv_proj(h);
    mx::array q = mx::reshape(mx::slice(qkv, {0, 0, 0}, {1, B, dq}), {1, B, cfg_.num_attention_heads, cfg_.head_dim});
    mx::array k_blk = mx::reshape(mx::slice(qkv, {0, 0, dq}, {1, B, dq + dk}), {1, B, cfg_.num_key_value_heads, cfg_.head_dim});
    mx::array v_blk = mx::reshape(mx::slice(qkv, {0, 0, dq + dk}, {1, B, dq + 2 * dk}), {1, B, cfg_.num_key_value_heads, cfg_.head_dim});
    q = mx::transpose(mx::fast::rms_norm(q, ly.q_norm, cfg_.rms_norm_eps), {0, 2, 1, 3});
    k_blk = mx::transpose(mx::fast::rms_norm(k_blk, ly.k_norm, cfg_.rms_norm_eps), {0, 2, 1, 3});
    v_blk = mx::transpose(v_blk, {0, 2, 1, 3});
    q = apply_rope_default(q, cfg_.head_dim, cfg_.rope_theta, st.ctx_len);
    k_blk = apply_rope_default(k_blk, cfg_.head_dim, cfg_.rope_theta, st.ctx_len);

    mx::array k_all = k_blk;
    mx::array v_all = v_blk;
    const DSparkLayerKV& lkv = st.layers[static_cast<size_t>(i)];
    if (lkv.k && lkv.len > 0) {
      mx::array k_ctx = mx::slice(*lkv.k, {0, 0, 0, 0},
                                  {1, cfg_.num_key_value_heads, static_cast<int>(lkv.len), cfg_.head_dim});
      mx::array v_ctx = mx::slice(*lkv.v, {0, 0, 0, 0},
                                  {1, cfg_.num_key_value_heads, static_cast<int>(lkv.len), cfg_.head_dim});
      k_all = mx::concatenate({k_ctx, k_blk}, 2);
      v_all = mx::concatenate({v_ctx, v_blk}, 2);
    }
    // Bidirectional block attention: no causal mask.
    mx::array attn_out = mx::fast::scaled_dot_product_attention(q, k_all, v_all, scale_, "");
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {1, B, dq});
    x = mx::add(x, ly.o_proj(attn_out));
    mx::array h_mlp = mx::fast::rms_norm(x, ly.post_ln, cfg_.rms_norm_eps);
    x = mx::add(x, swiglu_mlp_fused(h_mlp, ly.gate_up_proj, ly.down_proj, cfg_.intermediate_size));
  }

  mx::array h_final = mx::fast::rms_norm(x, final_norm_, cfg_.rms_norm_eps);  // [1, B, H]
  mx::array h_pred = mx::reshape(mx::slice(h_final, {0, 1, 0}, {1, B, cfg_.hidden_size}),
                                 {K, cfg_.hidden_size});  // [K, H]

  // 3. Recurrent Markov head: precompute [h_pred, f_rnn] contribution for all K steps at once.
  mx::array f_rnn = mx::reshape(
      mix_features(*st.last_features, cfg_.num_hidden_layers, markov_.dflare_norm),
      {1, cfg_.hidden_size});                                                        // [1, H]
  mx::array f_rnn_k = mx::broadcast_to(f_rnn, {K, cfg_.hidden_size});                // [K, H]
  mx::array hf = mx::concatenate({h_pred, f_rnn_k}, -1);                             // [K, 2H]
  mx::array ctx_rnn = mx::add(mx::astype(mx::matmul(hf, markov_.rnn_w_hf), mx::float32),
                              markov_.rnn_bias);                                     // [K, 3r]

  static auto markov_gate_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        // in: [g_raw, c_raw, o_raw, state (F32), ref_dtype]
        mx::array g = mx::sigmoid(in[0]);
        mx::array c = mx::tanh(in[1]);
        mx::array o = mx::tanh(in[2]);
        mx::array new_st = mx::add(mx::multiply(g, in[3]),
                                   mx::multiply(mx::subtract(mx::array(1.0f, mx::float32), g), c));
        return {new_st, mx::astype(o, in[4].dtype())};
      },
      true);

  // Precompute base logits for all K positions in ONE batched lm_head call (reads lm_head weights once).
  std::optional<mx::array> base_logits_kv;
  if (markov_.folded) {
    mx::array bl = lm_head_fn(h_pred);  // [K, V]
    const int V_trim = markov_.fold_bias->shape(-1);
    if (V_trim < bl.shape(-1)) bl = mx::slice(bl, {0, 0}, {K, V_trim});
    base_logits_kv = mx::add(bl, *markov_.fold_bias);  // fold_bias added once to all K rows
  }

  static auto top1_prob_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        // in: [lg, cap] -> [argmax_u32, max_prob_f32]
        mx::array lf = mx::astype(in[0], mx::float32);
        mx::array cap = in[1];
        lf = mx::where(mx::greater(cap, mx::array(0.0f, mx::float32)),
                       mx::multiply(cap, mx::tanh(mx::divide(lf, mx::maximum(cap, mx::array(1.0f, mx::float32))))),
                       lf);
        mx::array id = mx::argmax(lf, -1);
        mx::array mxv = mx::max(lf, -1);
        mx::array lse = mx::logsumexp(lf, -1);
        mx::array p = mx::exp(mx::subtract(mxv, lse));
        return {id, p};
      },
      true);

  const int r = cfg_.markov_rank;
  mx::array state = mx::zeros({1, r}, mx::float32);
  int32_t prev_tok = anchor_token;
  float cum_p = 1.0f;
  float exp_tok = 1.0f;
  // Measured streamed 16 GB latency model: T(0) = 72 ms (S=1), T(k) = 11 + 72 + 45*k ms.
  // Pick the draft length k in 0..K that maximizes expected tokens/ms over S=1.
  const float kS1Ms = dspark_timing_ms("VEYSTRIX_DSPARK_S1_MS", 72.0f);
  const float kDraftMs = dspark_timing_ms("VEYSTRIX_DSPARK_DRAFT_MS", 11.0f);
  const float kPerDraftMs = dspark_timing_ms("VEYSTRIX_DSPARK_PER_DRAFT_MS", 45.0f);
  float best_tps = 1.0f / kS1Ms;
  int best_k = 0;
  std::vector<int32_t> drafted;
  drafted.reserve(static_cast<size_t>(K));

  for (int step = 0; step < K; ++step) {
    mx::array prev_id(&prev_tok, mx::Shape{1, 1}, mx::int32);
    mx::array prev_emb = mx::reshape(markov_.markov_w1(prev_id), {1, r});  // [1, r] BF16
    mx::array sp = mx::concatenate({mx::astype(state, prev_emb.dtype()), prev_emb}, -1);  // [1, 2r]
    mx::array ctx_s = mx::slice(ctx_rnn, {step, 0}, {step + 1, 3 * r});
    mx::array raw = mx::add(ctx_s, mx::astype(mx::matmul(sp, markov_.rnn_w_sp), mx::float32));
    mx::array g_raw = mx::slice(raw, {0, 0}, {1, r});
    mx::array c_raw = mx::slice(raw, {0, r}, {1, 2 * r});
    mx::array o_raw = mx::slice(raw, {0, 2 * r}, {1, 3 * r});
    auto gl = markov_gate_c({g_raw, c_raw, o_raw, state, prev_emb});
    state = gl[0];
    mx::array latent = gl[1];  // [1, r] BF16
    mx::array lg = mx::array(0.0f);
    if (markov_.folded) {
      mx::array bias = mx::quantized_matmul(latent, *markov_.fold_w, *markov_.fold_s, *markov_.fold_b,
                                            /*transpose=*/true, /*group_size=*/64, /*bits=*/8, "affine");
      mx::array base_s = mx::slice(*base_logits_kv, {step, 0}, {step + 1, base_logits_kv->shape(-1)});
      lg = mx::add(base_s, bias);
    } else {
      mx::array h_corr = mx::add(mx::matmul(latent, markov_.hidden_out_w), markov_.hidden_out_b);
      mx::array h_step = mx::add(mx::slice(h_pred, {step, 0}, {step + 1, cfg_.hidden_size}), h_corr);
      lg = lm_head_fn(mx::expand_dims(h_step, 0));  // [1, 1, V]
      lg = mx::reshape(lg, {1, lg.shape(-1)});
      if (unpadded_vocab > 0 && unpadded_vocab < lg.shape(-1)) {
        lg = mx::slice(lg, {0, 0}, {1, unpadded_vocab});
      }
    }
    if (min_cum_prob > 0.0f) {
      auto tp = top1_prob_c({lg, mx::array(logit_softcap, mx::float32)});
      mx::eval(tp);
      const float p = tp[1].data<float>()[0];
      if (p < 0.15f || (cum_p * p < min_cum_prob && !drafted.empty())) {
        break;
      }
      cum_p *= p;
      prev_tok = static_cast<int32_t>(tp[0].data<uint32_t>()[0]);
      drafted.push_back(prev_tok);
      if (cum_p < min_cum_prob) break;
    } else {
      // Note: softcap(x) = cap * tanh(x / cap) is strictly monotonic, so argmax is invariant!
      mx::array next_id = mx::argmax(lg, -1);
      mx::eval(next_id);
      prev_tok = static_cast<int32_t>(next_id.data<uint32_t>()[0]);
      drafted.push_back(prev_tok);
    }
  }
  return drafted;
}

}  // namespace veystrix::engine::generic
