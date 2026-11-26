// Qwen4 (qwen4_exp: Qwen3.8 Flash-Next / Qwen 4 preview) blocks for the generic runtime.
//
// Each block is a thin adapter over the bit-exact primitives the custom Qwen runtime
// validated against the production Flash-Next Python stack (mlx_vlm qwen4_exp + the
// models/flashnext patches: one-centered RMSNorm gains, threshold-1.0 routing):
//
//   linear.gdn_qwen4  Qwen4ExpGatedDeltaNet       -> gdn_prefill (S > 1) / gdn_layer_forward (S == 1)
//   attn.qsa_gated    Qwen4ExpAttention (dense)    -> full_attention_dense op sequence, KV in KVStore
//   pre.ple_ngram     Qwen4ExpPLELayer             -> ple_forward (n-gram rows pread from the shards)
//
// The hyper-connection residual (Qwen4ExpGatedResidual) lives in generic_model.cpp
// (recipe "residual": {"type": "hyper"}), because it changes the layer wiring, not a block.
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

#include "decoder_full.hpp"
#include "engine.hpp"
#include "gdn_layer.hpp"
#include "generic/generic_model.hpp"
#include "ple.hpp"
#include "prefill.hpp"
#include "rope_fused_apply.hpp"

namespace veystrix::engine::generic {

engine::Linear to_engine_linear(const QLinear& q) {
  if (q.q.quantized) {
    if (q.q.mode != "affine") throw std::runtime_error("qwen4 blocks: only affine quantization is ported");
    if (!q.biases) throw std::runtime_error("qwen4 blocks: affine quantized linear without biases");
    return engine::Linear::quantized(*q.w, *q.scales, *q.biases, q.q.group_size, q.q.bits, q.bias);
  }
  engine::Linear L(*q.w);
  L.bias = q.bias;
  return L;
}

namespace {

std::string tensors_path(const BlockCtx& c) {
  return Loader::expand(c.params.at("tensors").str("tensors"), c.layer);
}

// =============================================================================
// linear.gdn_qwen4
// =============================================================================
class Qwen4GdnMixer final : public Mixer {
 public:
  explicit Qwen4GdnMixer(const BlockCtx& c)
      : w_(build(c)) {
    const Json& p = c.params;
    cfg_.num_k_heads = p.at("num_k_heads").i32("num_k_heads");
    cfg_.num_v_heads = p.at("num_v_heads").i32("num_v_heads");
    cfg_.head_k_dim = p.at("head_k_dim").i32("head_k_dim");
    cfg_.head_v_dim = p.at("head_v_dim").i32("head_v_dim");
    cfg_.conv_kernel = static_cast<int>(p.get_i64("conv_kernel", 4));
    cfg_.eps = static_cast<float>(p.get_num("eps", 1e-6));
    cfg_.key_dim = cfg_.num_k_heads * cfg_.head_k_dim;
    cfg_.value_dim = cfg_.num_v_heads * cfg_.head_v_dim;
    cfg_.hidden_size = static_cast<int>(c.loader.recipe().at("hidden_size").i64("hidden_size"));
    const std::string act = p.get_str("output_gate", "sigmoid");
    if (act == "sigmoid") cfg_.output_activation = GDNActivation::Sigmoid;
    else if (act == "silu" || act == "swish") cfg_.output_activation = GDNActivation::Silu;
    else throw std::runtime_error("linear.gdn_qwen4: output_gate '" + act + "' is not ported");
    conv_dim_ = 2 * cfg_.key_dim + cfg_.value_dim;
  }
  std::string type() const override { return "linear.gdn_qwen4"; }

  void reset(LayerState& st) const override {
    st.arrays.clear();
    st.kv.reset();
    st.tokens = 0;
  }

  mx::array forward(const mx::array& x, LayerState& st) const override {
    if (st.arrays.size() != 2) {
      // Python: conv_state zeros in the activation dtype, recurrent state float32.
      st.arrays = {mx::zeros({1, cfg_.conv_kernel - 1, conv_dim_}, x.dtype()),
                   mx::zeros({1, cfg_.num_v_heads, cfg_.head_v_dim, cfg_.head_k_dim}, mx::float32)};
    }
    GDNCacheState cache(st.arrays[0], st.arrays[1]);
    GDNLayerOutput out = x.shape(1) == 1 ? gdn_layer_forward(x, w_, cache, cfg_)
                                         : gdn_prefill(x, w_, cache, cfg_, nullptr);
    st.arrays = {out.conv_state, out.recurrent_state};
    st.tokens += x.shape(1);
    return out.output;
  }

  NamedArrays save(const LayerState& st, const std::string& pfx) const override {
    if (st.arrays.size() != 2) return {};
    return {{pfx + ".conv", mx::contiguous(st.arrays[0])}, {pfx + ".rec", mx::contiguous(st.arrays[1])}};
  }
  void load(LayerState& st, const std::string& pfx, const NamedArrays& a, int64_t tokens) const override {
    st.arrays.clear();
    std::optional<mx::array> conv, rec;
    for (const auto& kv : a) {
      if (kv.first == pfx + ".conv") conv = kv.second;
      if (kv.first == pfx + ".rec") rec = kv.second;
    }
    if (conv && rec) st.arrays = {*conv, *rec};
    st.tokens = tokens;
  }
  int64_t weight_bytes() const override { return bytes_; }

 private:
  GDNWeights build(const BlockCtx& c) {
    const std::string p = tensors_path(c);
    QLinear qkv = c.loader.linear(p + ".in_proj_qkv"), z = c.loader.linear(p + ".in_proj_z"),
            b = c.loader.linear(p + ".in_proj_b"), a = c.loader.linear(p + ".in_proj_a"),
            out = c.loader.linear(p + ".out_proj");
    mx::array conv = c.loader.tensor(p + ".conv1d.weight"), alog = c.loader.tensor(p + ".A_log"),
              dtb = c.loader.tensor(p + ".dt_bias"), nw = c.loader.tensor(p + ".norm.weight");
    mx::eval(std::vector<mx::array>{conv, alog, dtb, nw});
    bytes_ = qkv.nbytes() + z.nbytes() + b.nbytes() + a.nbytes() + out.nbytes() + conv.nbytes() + alog.nbytes() +
             dtb.nbytes() + nw.nbytes();
    return GDNWeights(to_engine_linear(qkv), to_engine_linear(z), to_engine_linear(b), to_engine_linear(a), conv,
                      alog, dtb, nw, to_engine_linear(out));
  }
  int64_t bytes_ = 0;
  GDNWeights w_;
  GDNConfig cfg_{};
  int conv_dim_ = 0;
};

// =============================================================================
// attn.qsa_gated — Qwen4ExpAttention with its QSA indexer (mlx_vlm qwen4_exp language.py
// Qwen4ExpQSAIndexer). While the session holds at most block_topk complete key blocks
// (indexer_budget / compress_ratio, i.e. about 2K tokens) the indexer returns no mask and the
// attention is the plain causal one. Beyond it every query row attends only to its top-k
// scored blocks plus its own incomplete tail block: the model's own sparse attention.
// =============================================================================
class Qwen4AttnMixer final : public Mixer {
 public:
  explicit Qwen4AttnMixer(const BlockCtx& c) : kv_(*c.kv) {
    const Json& p = c.params;
    const std::string t = tensors_path(c);
    heads_ = p.at("num_heads").i32("num_heads");
    kv_heads_ = p.at("num_kv_heads").i32("num_kv_heads");
    head_dim_ = p.at("head_dim").i32("head_dim");
    eps_ = static_cast<float>(p.get_num("eps", 1e-6));
    one_centered_ = p.get_str("gain", "one") == "one";
    const Json& r = p.at("rope");
    if (r.get_str("type", "") != "mrope_interleaved")
      throw std::runtime_error("attn.qsa_gated: rope type '" + r.get_str("type", "") + "' is not ported");
    rope_dim_ = static_cast<int>(r.at("dims").i64("rope.dims"));
    rope_base_ = static_cast<float>(r.at("base").num("rope.base"));
    for (const auto& v : r.at("mrope_section").arr("mrope_section")) section_.push_back(v.i32("mrope_section"));
    q_ = c.loader.linear(t + ".q_proj");
    k_ = c.loader.linear(t + ".k_proj");
    v_ = c.loader.linear(t + ".v_proj");
    o_ = c.loader.linear(t + ".o_proj");
    qn_ = c.loader.tensor(t + ".q_norm.weight");
    kn_ = c.loader.tensor(t + ".k_norm.weight");
    mx::eval(std::vector<mx::array>{qn_, kn_});
    Lq_ = to_engine_linear(q_);
    Lk_ = to_engine_linear(k_);
    Lv_ = to_engine_linear(v_);
    Lo_ = to_engine_linear(o_);
    scale_ = static_cast<float>(std::pow(static_cast<double>(head_dim_), -0.5));
    if (const Json* ix = p.find("indexer"); ix && ix->is_object() && c.loader.has(t + ".indexer.q_layernorm.weight")) {
      ix_heads_ = ix->at("n_heads").i32("indexer.n_heads");
      ix_kv_heads_ = ix->at("kv_heads").i32("indexer.kv_heads");
      ix_dim_ = ix->at("head_dim").i32("indexer.head_dim");
      ix_ratio_ = ix->at("compress_ratio").i32("indexer.compress_ratio");
      const int budget = ix->at("budget").i32("indexer.budget");
      if (ix_kv_heads_ != 1) throw std::runtime_error("attn.qsa_gated: indexer kv_heads != 1 is not ported");
      if (ix_ratio_ <= 0 || budget % ix_ratio_) throw std::runtime_error("attn.qsa_gated: bad indexer budget/ratio");
      ix_topk_ = budget / ix_ratio_;
      ixp_ = c.loader.linear(t + ".indexer.index_qk_proj");
      Lix_ = to_engine_linear(ixp_);
      iqn_ = c.loader.tensor(t + ".indexer.q_layernorm.weight");
      ikn_ = c.loader.tensor(t + ".indexer.k_layernorm.weight");
      mx::eval(std::vector<mx::array>{iqn_, ikn_});
      indexer_ = true;
    }
  }
  std::string type() const override { return "attn.qsa_gated"; }

  void reset(LayerState& st) const override {
    st.arrays.clear();
    st.kv.emplace(kv_);
    st.tokens = 0;
  }

  // Indexer state (st.arrays): [0] pooled keys of the complete blocks, already mean-pooled, k_layernorm'd and
  // rotated (exactly what Qwen4ExpQSAIndexer recomputes from QSAKVCache.index_keys on every call; each block's
  // ops are independent of the others), buffer [1, capacity, D]; [1] the raw keys of the incomplete last block
  // [1, past % R, D]. 64 B/token per layer instead of 256, and decode no longer re-pools the whole history.
  static int64_t rows_of(const mx::array& a) { return a.shape(1); }
  mx::array pooled_update(LayerState& st, const mx::array& raw, int past) const {
    const int R = ix_ratio_, D = ix_dim_;
    const int prev_blocks = past / R, pend = past % R;
    if (st.arrays.size() < 2) {
      st.arrays.assign(1, mx::zeros({1, 4096, D}, raw.dtype()));
      st.arrays.push_back(mx::zeros({1, 0, D}, raw.dtype()));
    }
    mx::array all = pend > 0 ? mx::concatenate({st.arrays[1], raw}, 1) : raw;
    const int n = pend + static_cast<int>(raw.shape(1));
    const int nb = n / R, blocks = prev_blocks + nb;
    if (nb > 0) {
      mx::array pooled = mx::reshape(mx::slice(all, {0, 0, 0}, {1, nb * R, D}), {1, nb, R, D});
      pooled = mx::astype(mx::mean(mx::astype(pooled, mx::float32), 2), raw.dtype());
      pooled = mx::expand_dims(qnorm(pooled, ikn_), 1);                       // [1, 1, nb, D]
      mx::array bpos = mx::multiply(mx::astype(mx::arange(prev_blocks, blocks), mx::int32), mx::array(R, mx::int32));
      pooled = engine::rope_apply_fused(pooled, mx::reshape(bpos, {1, nb}), rope_dim_, rope_base_, section_, true);
      pooled = mx::reshape(pooled, {1, nb, D});
      if (rows_of(st.arrays[0]) < blocks) {
        const int cap = ((blocks + 4095) / 4096) * 4096;
        mx::array buf = mx::zeros({1, cap, D}, raw.dtype());
        if (prev_blocks > 0)
          buf = mx::slice_update(buf, mx::slice(st.arrays[0], {0, 0, 0}, {1, prev_blocks, D}), {0, 0, 0},
                                 {1, prev_blocks, D});
        st.arrays[0] = buf;
      }
      st.arrays[0] = mx::slice_update(st.arrays[0], pooled, {0, prev_blocks, 0}, {1, blocks, D});
    }
    st.arrays[1] = mx::slice(all, {0, nb * R, 0}, {1, n, D});
    return mx::slice(st.arrays[0], {0, 0, 0}, {1, blocks, D});
  }

  // Qwen4ExpQSAIndexer.__call__ (op order mirrored); nullopt when it returns None.
  std::optional<mx::array> qsa_mask(const mx::array& x, LayerState& st, int past) const {
    const int B = x.shape(0), S = x.shape(1), Hi = ix_heads_, D = ix_dim_, R = ix_ratio_, K = ix_topk_;
    mx::array qk = mx::reshape(engine::linear(x, Lix_), {B, S, Hi + 1, D});
    mx::array query = mx::slice(qk, {0, 0, 0, 0}, {B, S, Hi, D});
    mx::array raw = mx::squeeze(mx::slice(qk, {0, 0, Hi, 0}, {B, S, Hi + 1, D}), 2);
    query = mx::transpose(qnorm(query, iqn_), {0, 2, 1, 3});
    const mx::array pooled_rows = pooled_update(st, raw, past);  // update_indexer runs before the None return
    const int key_len = past + S;
    const int blocks = key_len / R;
    if (blocks <= K) return std::nullopt;
    std::vector<int32_t> pos(static_cast<size_t>(S));
    for (int t = 0; t < S; ++t) pos[static_cast<size_t>(t)] = past + t;
    query = engine::rope_apply_fused(query, mx::array(pos.data(), mx::Shape{1, S}, mx::int32), rope_dim_, rope_base_,
                                     section_, true);
    mx::array pooled = mx::expand_dims(pooled_rows, 1);                // [B, 1, blocks, D]
    mx::array pooledT = mx::transpose(pooled, {0, 1, 3, 2});
    const mx::array zero(0.0f, mx::float32);
    const mx::array root(static_cast<float>(std::sqrt(static_cast<double>(D))), mx::float32);
    const mx::array ratio(R, mx::int32);
    const mx::array neg_inf(-std::numeric_limits<float>::infinity(), mx::float32);
    // Rows are independent; above the score tile they go in blocks to bound memory.
    const int qb = attention_query_block(S, blocks, Hi);
    std::vector<mx::array> outs;
    for (const auto& [a, b] : query_spans(S, qb)) {
      const int n = b - a;
      mx::array qa = n == S ? query : mx::slice(query, {0, 0, a, 0}, {B, Hi, b, D});
      mx::array scores = mx::matmul(qa, pooledT);                                     // [B, Hi, n, blocks]
      scores = mx::sum(mx::maximum(mx::astype(scores, mx::float32), zero), 1);          // [B, n, blocks]
      scores = mx::divide(scores, root);
      mx::array query_ends = mx::astype(mx::arange(past + a + 1, past + b + 1), mx::int32);   // [n]
      mx::array counts = mx::floor_divide(query_ends, ratio);
      mx::array valid = mx::less(mx::reshape(mx::astype(mx::arange(blocks), mx::int32), {1, 1, blocks}),
                                 mx::reshape(counts, {1, n, 1}));
      scores = mx::where(valid, scores, neg_inf);
      mx::array sel = mx::argpartition(scores, -K, -1);
      sel = mx::slice(sel, {0, 0, blocks - K}, {B, n, blocks});                        // selected blocks [B, n, K]
      // any(token_block == selected) as a block mask: scatter the K selected blocks, then each block covers
      // R consecutive tokens (tokens past the last complete block belong to no selectable block).
      mx::array bm = mx::put_along_axis(mx::zeros({B, n, blocks}, mx::bool_), sel, mx::array(true), -1);
      mx::array tok = mx::reshape(mx::repeat(mx::expand_dims(bm, -1), R, -1), {B, n, blocks * R});
      if (blocks * R < key_len)
        tok = mx::concatenate({tok, mx::zeros({B, n, key_len - blocks * R}, mx::bool_)}, -1);
      mx::array ti = mx::reshape(mx::astype(mx::arange(key_len), mx::int32), {1, 1, key_len});
      mx::array qe = mx::reshape(query_ends, {1, n, 1});
      mx::array ts = mx::reshape(mx::multiply(counts, ratio), {1, n, 1});
      mx::array tail = mx::logical_and(mx::greater_equal(ti, ts), mx::less(ti, qe));
      mx::array causal = mx::less(ti, qe);
      mx::array sparse = mx::reshape(mx::greater(counts, mx::array(K, mx::int32)), {1, n, 1});
      mx::array m = mx::where(sparse, mx::logical_or(tok, tail), causal);              // [B, n, key_len]
      if (n < S) mx::eval(m);
      outs.push_back(m);
    }
    mx::array m = outs.size() == 1 ? outs[0] : mx::concatenate(outs, 1);
    return mx::expand_dims(m, 1);                                                      // [B, 1, S, key_len]
  }

  // Qwen4ExpRMSNorm (patch_rmsnorm._rms_norm): float32 normalise, gain w (one-centered) or 1 + w.
  mx::array qnorm(const mx::array& a, const mx::array& gain) const {
    const mx::Dtype d = a.dtype();
    mx::array y = mx::astype(a, mx::float32);
    y = y * mx::rsqrt(mx::mean(mx::square(y), -1, true) + eps_);
    mx::array g = mx::astype(gain, mx::float32);
    if (!one_centered_) g = 1.0f + g;
    y = y * g;
    return mx::astype(y, d);
  }

  mx::array forward(const mx::array& x, LayerState& st) const override {
    if (!st.kv) st.kv.emplace(kv_);
    const int B = x.shape(0), S = x.shape(1);
    const int Hq = heads_, Hkv = kv_heads_, D = head_dim_;
    mx::array qproj = engine::linear(x, Lq_);
    mx::array kproj = engine::linear(x, Lk_);
    mx::array vproj = engine::linear(x, Lv_);
    mx::array q4 = mx::reshape(qproj, mx::Shape{B, S, Hq, 2 * D});
    std::vector<mx::array> halves = mx::split(q4, 2, -1);
    mx::array q = halves[0];
    mx::array gate = mx::reshape(halves[1], mx::Shape{B, S, Hq * D});
    q = mx::transpose(qnorm(q, qn_), {0, 2, 1, 3});
    mx::array k = mx::transpose(qnorm(mx::reshape(kproj, mx::Shape{B, S, Hkv, D}), kn_), {0, 2, 1, 3});
    mx::array v = mx::transpose(mx::reshape(vproj, mx::Shape{B, S, Hkv, D}), {0, 2, 1, 3});
    const int offset = static_cast<int>(st.kv->offset());
    const std::optional<mx::array> sparse = indexer_ ? qsa_mask(x, st, offset) : std::nullopt;
    std::vector<int32_t> pos(static_cast<size_t>(S));
    for (int t = 0; t < S; ++t) pos[static_cast<size_t>(t)] = offset + t;
    mx::array position_ids(pos.data(), mx::Shape{1, S}, mx::int32);
    q = engine::rope_apply_fused(q, position_ids, rope_dim_, rope_base_, section_, true);
    k = engine::rope_apply_fused(k, position_ids, rope_dim_, rope_base_, section_, true);
    mx::array out = st.kv->update_and_attend(q, k, v, scale_, /*causal_single=*/true, sparse);
    st.tokens += S;
    out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), mx::Shape{B, S, Hq * D});
    return engine::linear(out * mx::sigmoid(gate), Lo_);
  }

  NamedArrays save(const LayerState& st, const std::string& pfx) const override {
    if (!st.kv) return {};
    NamedArrays out = st.kv->save(pfx + ".kv");
    if (indexer_ && st.arrays.size() >= 2 && st.tokens > 0) {
      const int blocks = static_cast<int>(st.tokens / ix_ratio_);
      if (blocks > 0)
        out.emplace_back(pfx + ".index_pooled", mx::contiguous(mx::slice(st.arrays[0], {0, 0, 0}, {1, blocks, ix_dim_})));
      if (rows_of(st.arrays[1]) > 0) out.emplace_back(pfx + ".index_pending", mx::contiguous(st.arrays[1]));
    }
    return out;
  }
  void load(LayerState& st, const std::string& pfx, const NamedArrays& a, int64_t tokens) const override {
    st.arrays.clear();
    st.kv.emplace(kv_);
    st.kv->load(pfx + ".kv", a, tokens);
    st.tokens = tokens;
    if (indexer_ && tokens > 0) {
      std::optional<mx::array> pooled, pending;
      for (const auto& [name, arr] : a) {
        if (name == pfx + ".index_pooled") pooled = arr;
        if (name == pfx + ".index_pending") pending = arr;
      }
      const int blocks = static_cast<int>(tokens / ix_ratio_), pend = static_cast<int>(tokens % ix_ratio_);
      if ((blocks > 0 && !pooled) || (pend > 0 && !pending))
        throw std::runtime_error("attn.qsa_gated: snapshot has no indexer state (made before the QSA port)");
      const mx::Dtype dt = pooled ? pooled->dtype() : pending->dtype();
      mx::array buf = mx::zeros({1, std::max(4096, ((blocks + 4095) / 4096) * 4096), ix_dim_}, dt);
      if (blocks > 0) buf = mx::slice_update(buf, *pooled, {0, 0, 0}, {1, blocks, ix_dim_});
      st.arrays.assign(1, buf);
      st.arrays.push_back(pend > 0 ? *pending : mx::zeros({1, 0, ix_dim_}, dt));
    }
  }
  // K + V per token; the indexer adds one pooled D-vector per compress_ratio tokens (rounded up).
  int64_t kv_values_per_token() const override {
    return 2LL * kv_heads_ * head_dim_ + (indexer_ ? (ix_dim_ + ix_ratio_ - 1) / ix_ratio_ : 0);
  }
  int64_t weight_bytes() const override {
    return q_.nbytes() + k_.nbytes() + v_.nbytes() + o_.nbytes() + qn_.nbytes() + kn_.nbytes()
           + (indexer_ ? ixp_.nbytes() + iqn_.nbytes() + ikn_.nbytes() : 0);
  }

 private:
  KVPolicy kv_;
  int heads_ = 0, kv_heads_ = 0, head_dim_ = 0, rope_dim_ = 0;
  float eps_ = 1e-6f, scale_ = 1.0f, rope_base_ = 10000.0f;
  bool one_centered_ = true;
  std::vector<int> section_;
  QLinear q_, k_, v_, o_;
  engine::Linear Lq_, Lk_, Lv_, Lo_;
  mx::array qn_ = mx::array(0.0f), kn_ = mx::array(0.0f);
  // QSA indexer (absent in a recipe without "indexer": then the attention stays dense)
  bool indexer_ = false;
  int ix_heads_ = 0, ix_kv_heads_ = 1, ix_dim_ = 0, ix_ratio_ = 1, ix_topk_ = 0;
  QLinear ixp_;
  engine::Linear Lix_;
  mx::array iqn_ = mx::array(0.0f), ikn_ = mx::array(0.0f);
};

// =============================================================================
// pre.ple_ngram — Qwen4ExpPLELayer (added to the residual streams before the layer)
// =============================================================================
PLEElemType ple_elem(const std::string& st) {
  if (st == "U32") return PLEElemType::U32;
  if (st == "F32") return PLEElemType::F32;
  if (st == "F16") return PLEElemType::F16;
  if (st == "BF16") return PLEElemType::BF16;
  throw std::runtime_error("pre.ple_ngram: unsupported shard dtype " + st);
}

class PleNgram final : public PreBlock {
 public:
  explicit PleNgram(const BlockCtx& c) : w_(build(c)) {}
  std::string type() const override { return "pre.ple_ngram"; }

  mx::array forward(const mx::array& h, const std::vector<int32_t>& ids, LayerState& st) const override {
    std::vector<int64_t> v(ids.begin(), ids.end());
    mx::array tok(v.data(), mx::Shape{1, static_cast<int>(v.size())}, mx::int64);
    PLECache cache;
    if (st.pre.size() == 2) {
      cache.context = st.pre[0];
      cache.conv_state = st.pre[1];
    }
    mx::array out = ple_forward(h, tok, w_, cache, nullptr);
    if (cache.context && cache.conv_state) st.pre = {*cache.context, *cache.conv_state};
    return out;
  }
  int64_t weight_bytes() const override { return bytes_; }

 private:
  PLEWeights build(const BlockCtx& c) {
    const Json& p = c.params;
    const std::string t = tensors_path(c);
    const Json& recipe = c.loader.recipe();
    const int hidden = static_cast<int>(recipe.at("hidden_size").i64("hidden_size"));
    const int hc = static_cast<int>(recipe.at("residual").at("hc_count").i64("hc_count"));
    const int ns = static_cast<int>(p.get_i64("ngram_size", 3));
    const int ck = static_cast<int>(p.get_i64("conv_kernel", 4));
    const int parts = static_cast<int>(p.at("split_parts").i64("split_parts"));
    QLinear kp = c.loader.linear(t + ".key_proj"), vp = c.loader.linear(t + ".value_proj");
    std::vector<mx::array> dense = {c.loader.tensor(t + ".norm_key.weight"), c.loader.tensor(t + ".norm_query.weight"),
                                    c.loader.tensor(t + ".norm_conv.weight"), c.loader.tensor(t + ".conv1d.weight"),
                                    c.loader.tensor(t + ".ple_embedding.layer_multipliers"),
                                    c.loader.tensor(t + ".ple_embedding.ngram_heads_offsets"),
                                    c.loader.tensor(t + ".ple_embedding.ngram_heads_vocab_sizes")};
    mx::eval(dense);
    bytes_ = kp.nbytes() + vp.nbytes();
    for (const auto& a : dense) bytes_ += a.nbytes();
    // Embedding shards stay on disk: only the rows a step needs are pread.
    const std::string sp = t + ".ple_embedding.ngram_embedding.shard_";
    const QuantSpec q = c.loader.quant_for(sp + "0");
    if (!q.quantized) throw std::runtime_error("pre.ple_ngram: n-gram embedding shards must be quantized");
    std::vector<PLEStreamedShard> streamed;
    std::vector<int64_t> offsets{0};
    auto ref = [&](const std::string& name) {
      const TensorInfo& ti = c.loader.index().at(name);
      if (ti.shape.size() != 2) throw std::runtime_error("pre.ple_ngram: shard tensor is not 2-D: " + name);
      PLERowTensorRef r;
      r.fd = ti.fd;
      r.data_offset = ti.offset;
      r.row_bytes = ti.nbytes / ti.shape[0];
      r.elements_per_row = ti.shape[1];
      r.elem = ple_elem(ti.dtype);
      return r;
    };
    for (int s = 0; s < parts; ++s) {
      const std::string base = sp + std::to_string(s);
      PLEStreamedShard d;
      d.weight = ref(base + ".weight");
      d.scales = ref(base + ".scales");
      d.biases = ref(base + ".biases");
      streamed.push_back(d);
      offsets.push_back(offsets.back() + c.loader.index().at(base + ".weight").shape[0]);
    }
    PLEWeights w(to_engine_linear(kp), to_engine_linear(vp), dense[0], dense[1], dense[2], dense[3], dense[4],
                 dense[5], dense[6], {}, std::move(offsets), q.group_size, q.bits, ns,
                 static_cast<int>(p.get_i64("heads_per_ngram", 8)), hc, hidden,
                 static_cast<int>(p.get_i64("embed_dim", hidden)),
                 static_cast<int>(p.at("eos_token_id").i64("eos_token_id")),
                 static_cast<float>(p.get_num("eps", 1e-6)), ck, ns, (ck - 1) * ns, q.mode);
    w.norm_one_centered = p.get_str("gain", "one") == "one";
    w.set_streamed_shards(std::move(streamed));
    return w;
  }
  int64_t bytes_ = 0;
  PLEWeights w_;
};

}  // namespace

void register_qwen4_blocks(std::map<std::string, MixerFactory>& mixers, std::map<std::string, PreFactory>& pre) {
  mixers["linear.gdn_qwen4"] = [](const BlockCtx& c) { return std::unique_ptr<Mixer>(new Qwen4GdnMixer(c)); };
  mixers["attn.qsa_gated"] = [](const BlockCtx& c) { return std::unique_ptr<Mixer>(new Qwen4AttnMixer(c)); };
  pre["pre.ple_ngram"] = [](const BlockCtx& c) { return std::unique_ptr<PreBlock>(new PleNgram(c)); };
}

}  // namespace veystrix::engine::generic
