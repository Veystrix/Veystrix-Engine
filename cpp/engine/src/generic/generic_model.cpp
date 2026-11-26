#include "generic/generic_model.hpp"
#include "generic/inkling_dspark.hpp"

#include <mlx/compile.h>
#include <mlx/fast.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "cpp_profile.hpp"
#include "engine.hpp"
#include "memory_budget.hpp"
#include "row_fetch.hpp"

namespace veystrix::engine::generic {

namespace {

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("generic: cannot read " + path);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

// FNV-1a 64 of the canonical recipe text (identity for snapshots/stats).
std::string fnv_hex(const std::string& s) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

}  // namespace

GenericModel::~GenericModel() {
  rowfetch::evict_cached_files();
  mx::clear_cache();
}

GenericModel::GenericModel(const std::string& model_dir, const std::string& recipe_src,
                           const ModelOptions& opt)
    : opt_(opt) {
  // Release any cached MLX buffers and mmap file pages in RAM before sizing and loading.
  membudget::clear_mlx_cache();
  rowfetch::evict_cached_files();
  const std::string text = (!recipe_src.empty() && recipe_src[0] == '{') ? recipe_src : read_text(recipe_src);
  recipe_ = Json::parse(text);
  const std::string fmt = recipe_.get_str("format", "");
  // /1: norm residual, blocks, layers. /2 adds optional "residual" (hyper-connections), "layers.pre"
  // and "speed". /3 adds "loop" (visit-decoupled looped core) and "block_attn_res" residuals.
  if (fmt != "veystrix-recipe/1" && fmt != "veystrix-recipe/2" && fmt != "veystrix-recipe/3")
    throw std::runtime_error("generic: unsupported recipe format '" + fmt +
                             "' (this build reads veystrix-recipe/1, /2, and /3)");
  recipe_digest_ = fnv_hex(recipe_.dump());
  model_type_ = recipe_.get_str("model_type", "unknown");
  hidden_ = static_cast<int>(recipe_.at("hidden_size").i64("hidden_size"));
  vocab_ = static_cast<int>(recipe_.at("vocab_size").i64("vocab_size"));
  unpadded_vocab_ = static_cast<int>(recipe_.get_i64("unpadded_vocab_size", vocab_));
  final_logit_softcap_ = static_cast<float>(recipe_.get_num("final_logit_softcapping", 0.0));
  eps_ = static_cast<float>(recipe_.get_num("rms_norm_eps", 1e-6));
  const bool shift = recipe_.get_bool("norm_shift", false);

  idx_ = std::make_unique<StIndex>(model_dir);
  Loader L(*idx_, recipe_);

  embed_ = L.embedding(recipe_.at("embed").str("embed"));
  if (const Json* en = recipe_.find("embed_norm"); en && en->is_string())
    embed_norm_ = L.norm_weight(en->str("embed_norm"), shift);
  const Json* residual = recipe_.find("residual");
  const std::string res_type = (residual && residual->is_object()) ? residual->get_str("type", "") : "";
  hyper_ = res_type == "hyper";
  block_attn_res_ = res_type == "block_attn_res";
  if (!res_type.empty() && !hyper_ && !block_attn_res_)
    throw std::runtime_error("generic: residual type '" + res_type + "' is not ported");
  auto load_hc = [&](const std::string& path, bool inject) {
    HyperRes r;
    r.norm = L.tensor(path + ".hc_norm.weight");
    r.down = L.linear(path + ".input_mix_weight_down");
    r.up = L.linear(path + ".input_mix_weight_up");
    if (inject) r.inject = L.linear(path + ".block_inject_weight");
    return r;
  };
  if (hyper_) {
    hc_ = static_cast<int>(residual->at("hc_count").i64("residual.hc_count"));
    hc_one_centered_ = residual->get_str("gain", "one") == "one";
    final_hc_ = load_hc(residual->at("final").str("residual.final"), false);
  } else {
    final_norm_ = L.norm_weight(recipe_.at("final_norm").str("final_norm"), shift);
    if (block_attn_res_) {
      res_block_size_ = static_cast<int>(residual->get_i64("block_size", 6));
      res_norm_values_ = residual->get_bool("normalize_values", true);
      if (const Json* fq = residual->find("final_query"); fq && fq->is_string())
        final_res_query_ = L.tensor(fq->str("residual.final_query"));
    }
  }
  tied_ = recipe_.get_bool("tie_word_embeddings", false);
  if (!tied_) head_ = L.linear(recipe_.at("head").str("head"));

  const Json& blocks = recipe_.at("blocks");
  const Json& layers = recipe_.at("layers");
  const int count = static_cast<int>(layers.at("count").i64("layers.count"));
  const auto& pattern = layers.at("pattern").arr("layers.pattern");
  if (pattern.empty()) throw std::runtime_error("generic: empty layer pattern");
  const Json* overrides = layers.find("overrides");
  const Json* pre = layers.find("pre");

  if (const Json* lp = recipe_.find("loop"); lp && lp->is_object()) {
    looped_ = true;
    loop_core_start_ = static_cast<int>(lp->get_i64("core_start", 6));
    loop_core_end_ = static_cast<int>(lp->get_i64("core_end", 18));
    loop_passes_ = static_cast<int>(lp->get_i64("passes", 3));
    loop_gate_rank_ = static_cast<int>(lp->get_i64("gate_rank", 64));
    loop_share_kv_ = lp->get_bool("share_kv", false);
    for (int l = 0; l < loop_core_start_; ++l) { visit_layer_.push_back(l); visit_pass_.push_back(0); }
    for (int p = 0; p < loop_passes_; ++p)
      for (int l = loop_core_start_; l < loop_core_end_; ++l) { visit_layer_.push_back(l); visit_pass_.push_back(p); }
    for (int l = loop_core_end_; l < count; ++l) { visit_layer_.push_back(l); visit_pass_.push_back(0); }
  } else {
    for (int l = 0; l < count; ++l) { visit_layer_.push_back(l); visit_pass_.push_back(0); }
  }
  const membudget::InklingResidencyPlan res_plan =
      looped_ ? membudget::plan_inkling_residency(membudget::physical_ram_bytes(),
                                                  membudget::available_ram_bytes(),
                                                  membudget::swap_used_bytes(),
                                                  membudget::memory_pressure_level())
              : membudget::InklingResidencyPlan{};
  const int resident_core = [&] {
    const char* e = std::getenv("VEYSTRIX_INKLING_RESIDENT_CORE");
    if (e && *e) return std::max(0, std::atoi(e));
    return res_plan.core;
  }();
  const int resident_extra = [&] {
    const char* e = std::getenv("VEYSTRIX_INKLING_RESIDENT_EXTRA");
    if (e && *e) return std::max(0, std::atoi(e));
    return res_plan.extra;
  }();
  auto bake_sconv = [](const mx::array& w_ck) -> mx::array {
    const int C = w_ck.shape(0), W = w_ck.shape(1), hist = W - 1;
    mx::array wf = mx::astype(w_ck, mx::float32);
    mx::array w_head = mx::slice(wf, {0, 0}, {C, hist});
    mx::array w_last = mx::add(mx::slice(wf, {0, hist}, {C, W}), mx::array(1.0f, mx::float32));
    const float inv_norm = 1.0f / std::sqrt(static_cast<float>(W + 1));
    mx::array kern = mx::multiply(mx::concatenate({w_head, w_last}, 1), mx::array(inv_norm, mx::float32));
    return mx::contiguous(mx::reshape(kern, {C, W, 1}));
  };
  const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(hidden_));

  const bool explicit_res_core = (std::getenv("VEYSTRIX_INKLING_RESIDENT_CORE") != nullptr &&
                                  *std::getenv("VEYSTRIX_INKLING_RESIDENT_CORE") != '\0');
  const size_t phys_ram = membudget::physical_ram_bytes();
  const bool tight_16g_load = phys_ram < (size_t{20} << 30);
  const bool live_load_guard = looped_ && !opt_.resident_experts && !explicit_res_core;
  const size_t avail_floor = std::clamp(phys_ram / 10, (size_t{22} << 30) / 10, size_t{4} << 30);
  const size_t load_swap_start = live_load_guard ? membudget::swap_used_bytes() : 0;
  bool load_pressure_stop = false;

  for (int l = 0; l < count; ++l) {
    const Json* entry = &pattern[static_cast<size_t>(l) % pattern.size()];
    if (overrides && overrides->is_object())
      if (const Json* o = overrides->find(std::to_string(l))) entry = o;
    const std::string mixer_key = entry->at("mixer").str("mixer");
    const std::string mlp_key = entry->at("mlp").str("mlp");
    const Json& mdef = blocks.at(mixer_key);
    const Json& pdef = blocks.at(mlp_key);
    bool is_core_res = (looped_ && resident_core > 0 && l >= loop_core_start_ &&
                        l < std::min(loop_core_end_, loop_core_start_ + resident_core));
    bool is_prelude_res = (looped_ && resident_extra > 0 && l < std::min(loop_core_start_, resident_extra));
    bool is_epilogue_res = (looped_ && resident_extra > loop_core_start_ &&
                            l >= loop_core_end_ &&
                            l < std::min(count, loop_core_end_ + (resident_extra - loop_core_start_)));
    if ((is_core_res || is_prelude_res || is_epilogue_res) && live_load_guard) {
      if (!load_pressure_stop) {
        const int cur_p = membudget::memory_pressure_level();
        const size_t cur_avail = membudget::available_ram_bytes();
        const size_t cur_swap = membudget::swap_used_bytes();
        // Stop pinning more resident layers across any Mac tier if macOS hits level-4
        // red pressure, available RAM drops below the tier floor, or swap grows > 512 MiB.
        if (cur_p >= 4 || cur_avail < avail_floor ||
            cur_swap > load_swap_start + (size_t{512} << 20)) {
          load_pressure_stop = true;
          rowfetch::evict_cached_files();
          mx::clear_cache();
        }
      }
      if (load_pressure_stop) {
        is_core_res = false;
        is_prelude_res = false;
        is_epilogue_res = false;
      }
    }
    const bool layer_resident = opt_.resident_experts || is_core_res || is_prelude_res || is_epilogue_res;
    // Per-layer KV override (recipe cache.kv.layers["<l>"] = mode).
    KVPolicy kv = opt_.kv;
    BlockCtx mc{L, mdef, l, layer_resident, &kv};
    const std::string mtype = mdef.at("type").str("block type");
    auto mit = mixer_registry().find(mtype);
    if (mit == mixer_registry().end())
      throw std::runtime_error("generic: no block registered for mixer type '" + mtype + "'");
    mixers_.push_back(mit->second(mc));
    BlockCtx pc{L, pdef, l, layer_resident, &kv};
    const std::string ptype = pdef.at("type").str("block type");
    auto pit = mlp_registry().find(ptype);
    if (pit == mlp_registry().end())
      throw std::runtime_error("generic: no block registered for mlp type '" + ptype + "'");
    mlps_.push_back(pit->second(pc));
    if (hyper_) {
      attn_hc_.push_back(load_hc(Loader::expand(residual->at("mixer").str("residual.mixer"), l), true));
      mlp_hc_.push_back(load_hc(Loader::expand(residual->at("mlp").str("residual.mlp"), l), true));
    } else if (block_attn_res_) {
      const std::string lp = Loader::expand(mdef.at("tensors").str("block tensors"), l);
      if (L.has(lp + "attn_norm.weight")) in_norm_.push_back(L.norm_weight(lp + "attn_norm", shift));
      else in_norm_.push_back(mx::ones({hidden_}, mx::bfloat16));
      post_norm_.push_back(L.norm_weight(lp + "mlp_norm", shift));
      attn_res_q_.push_back(L.has(lp + "attn_input.query") ? std::optional<mx::array>(L.tensor(lp + "attn_input.query")) : std::nullopt);
      mlp_res_q_.push_back(L.has(lp + "mlp_input.query") ? std::optional<mx::array>(L.tensor(lp + "mlp_input.query")) : std::nullopt);
      attn_sconv_.push_back(L.has(lp + "attn_conv") ? std::optional<mx::array>(bake_sconv(L.tensor(lp + "attn_conv"))) : std::nullopt);
      mlp_sconv_.push_back(L.has(lp + "mlp_conv") ? std::optional<mx::array>(bake_sconv(L.tensor(lp + "mlp_conv"))) : std::nullopt);
      q_in_norm_.push_back(L.has(lp + "q_input_norm.weight") ? std::optional<mx::array>(L.norm_weight(lp + "q_input_norm", shift)) : std::nullopt);
      k_in_norm_.push_back(L.has(lp + "k_input_norm.weight") ? std::optional<mx::array>(L.norm_weight(lp + "k_input_norm", shift)) : std::nullopt);
      v_in_norm_.push_back(L.has(lp + "v_input_norm.weight") ? std::optional<mx::array>(L.norm_weight(lp + "v_input_norm", shift)) : std::nullopt);
      if (L.has(lp + "loop_down")) {
        LoopGate lg;
        lg.msg_query = L.tensor(lp + "loop_message.query");
        lg.down = L.tensor(lp + "loop_down");
        lg.gain_forget = L.tensor(lp + "loop_gain_forget");
        for (int t = 0; t < 4; ++t) {
          mx::array mq = mx::astype(mx::slice(lg.msg_query, {t, 0}, {t + 1, hidden_}), mx::float32);
          lg.mq_s.push_back(mx::contiguous(mx::multiply(mx::reshape(mq, {1, 1, 1, hidden_}), mx::array(inv_sqrt_h, mx::float32))));
          lg.w0.push_back(mx::contiguous(mx::astype(mx::reshape(mx::slice(lg.down, {t, 0, 0, 0}, {t + 1, 1, hidden_, loop_gate_rank_}), {hidden_, loop_gate_rank_}), mx::bfloat16)));
          lg.w1.push_back(mx::contiguous(mx::astype(mx::reshape(mx::slice(lg.down, {t, 1, 0, 0}, {t + 1, 2, hidden_, loop_gate_rank_}), {hidden_, loop_gate_rank_}), mx::bfloat16)));
          lg.wgf.push_back(mx::contiguous(mx::astype(mx::reshape(mx::slice(lg.gain_forget, {t, 0, 0}, {t + 1, loop_gate_rank_, 2 * hidden_}), {loop_gate_rank_, 2 * hidden_}), mx::bfloat16)));
        }
        mx::array mq3 = mx::astype(mx::slice(lg.msg_query, {0, 0}, {3, hidden_}), mx::float32);
        lg.mq_qkv_t = mx::contiguous(mx::multiply(mx::transpose(mq3), mx::array(inv_sqrt_h, mx::float32)));
        lg.w0_qkv = mx::contiguous(mx::concatenate({lg.w0[0], lg.w0[1], lg.w0[2]}, 1));
        lg.w1_qkv = mx::contiguous(mx::stack({lg.w1[0], lg.w1[1], lg.w1[2]}, 0));
        lg.wgf_qkv = mx::contiguous(mx::stack({lg.wgf[0], lg.wgf[1], lg.wgf[2]}, 0));
        loop_gates_.push_back(std::move(lg));
      } else {
        loop_gates_.push_back(std::nullopt);
      }
    } else {
      const Json& norms = recipe_.at("layer_norms");
      in_norm_.push_back(L.norm_weight(Loader::expand(norms.at("input").str("input norm"), l), shift));
      post_norm_.push_back(L.norm_weight(Loader::expand(norms.at("post").str("post norm"), l), shift));
    }
    pre_.emplace_back();
    if (pre && pre->is_object())
      if (const Json* key = pre->find(std::to_string(l))) {
        const Json& def = blocks.at(key->str("layers.pre"));
        const std::string ptype2 = def.at("type").str("block type");
        auto it = pre_registry().find(ptype2);
        if (it == pre_registry().end())
          throw std::runtime_error("generic: no block registered for pre-layer type '" + ptype2 + "'");
        BlockCtx bc{L, def, l, opt_.resident_experts, &kv};
        pre_.back() = it->second(bc);
      }
    if (layer_resident) {
      mx::clear_cache();
      if (tight_16g_load && is_core_res) {
        // Brief pacing yield so macOS VM reclamation settles gradually between
        // 550 MiB resident layer loads instead of triggering an instant swap cliff.
        if (membudget::available_ram_bytes() < (size_t{5} << 30) ||
            membudget::memory_pressure_level() >= 2) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
      }
    }
  }
  for (size_t l = 0; l + 1 < mlps_.size(); ++l) mlps_[l]->link_next(mlps_[l + 1].get());
  // Materialise every resident tensor now (one pread pass), so the first
  // request does not pay for it and the bytes are accounted.
  std::vector<mx::array> all;
  for (auto& a : in_norm_) all.push_back(a);
  for (auto& a : post_norm_) all.push_back(a);
  if (!hyper_) all.push_back(final_norm_);
  if (embed_norm_) all.push_back(*embed_norm_);
  if (final_res_query_) all.push_back(*final_res_query_);
  for (auto& opt : attn_res_q_) if (opt) all.push_back(*opt);
  for (auto& opt : mlp_res_q_) if (opt) all.push_back(*opt);
  for (auto& opt : attn_sconv_) if (opt) all.push_back(*opt);
  for (auto& opt : mlp_sconv_) if (opt) all.push_back(*opt);
  for (auto& opt : q_in_norm_) if (opt) all.push_back(*opt);
  for (auto& opt : k_in_norm_) if (opt) all.push_back(*opt);
  for (auto& opt : v_in_norm_) if (opt) all.push_back(*opt);
  for (auto& lg : loop_gates_) if (lg) {
    for (auto& a : lg->mq_s) all.push_back(a);
    for (auto& a : lg->w0) all.push_back(a);
    for (auto& a : lg->w1) all.push_back(a);
    for (auto& a : lg->wgf) all.push_back(a);
    all.push_back(lg->mq_qkv_t);
    all.push_back(lg->w0_qkv);
    all.push_back(lg->w1_qkv);
    all.push_back(lg->wgf_qkv);
  }
  auto hc_arrays = [&](const HyperRes& r) {
    all.push_back(r.norm);
    for (const QLinear* q : {&r.down, &r.up, r.inject ? &*r.inject : nullptr})
      if (q)
        for (const auto* a : {&q->w, &q->scales, &q->biases, &q->bias})
          if (*a) all.push_back(**a);
  };
  for (const auto& r : attn_hc_) hc_arrays(r);
  for (const auto& r : mlp_hc_) hc_arrays(r);
  if (final_hc_) hc_arrays(*final_hc_);
  mx::eval(all);
  mx::clear_cache();
  if (opt_.enable_dspark && block_attn_res_) {
    dspark_ = Inkling2DSpark::try_load(L, recipe_);
    if (dspark_) {
      dspark_->attach_lm_head([this](const mx::array& x) { return lm_head_raw(x); }, unpadded_vocab_);
      mx::clear_cache();
    }
  }
  if (streamed_bytes() == 0) {
    // Full-resident model: drop one-shot load fds and any cached shard pages in RAM.
    idx_->close_load_fds();
  }
  rowfetch::evict_cached_files();
  mx::clear_cache();
  resident_bytes_ = idx_->loaded_bytes();
}

mx::array GenericModel::embed_normalized(const mx::array& ids) const {
  mx::array emb = embed_(ids);
  if (embed_norm_) emb = mx::fast::rms_norm(emb, *embed_norm_, eps_);
  return emb;
}

mx::array GenericModel::lm_head_raw(const mx::array& x) const {
  return tied_ ? embed_.as_linear(x) : (*head_)(x);
}

int64_t GenericModel::kv_values_per_token() const {
  int64_t t = 0;
  for (size_t v = 0; v < visit_layer_.size(); ++v) {
    const auto& m = mixers_[static_cast<size_t>(visit_layer_[v])];
    if (m->sliding_window() == 0) t += m->kv_values_per_token();
  }
  return t;
}

int64_t GenericModel::streamed_bytes() const {
  int64_t t = 0;
  for (const auto& m : mlps_) t += m->streamed_bytes();
  return t;
}

std::string GenericModel::describe_blocks() const {
  std::ostringstream o;
  std::map<std::string, int> counts;
  for (const auto& m : mixers_) counts[m->type()]++;
  for (const auto& m : mlps_) counts[m->type()]++;
  bool first = true;
  for (const auto& kv : counts) {
    if (!first) o << ",";
    first = false;
    o << "\"" << kv.first << "\":" << kv.second;
  }
  return o.str();
}

void GenericModel::reset(SessionState& st) const {
  st.assign(visit_layer_.size(), LayerState{});
  for (size_t v = 0; v < visit_layer_.size(); ++v)
    mixers_[static_cast<size_t>(visit_layer_[v])]->reset(st[v]);
}

// VEYSTRIX_GENERIC_PREFILL_ADVISE=1: F_RDADVISE read-ahead of the whole next layer in prefill
// (off by default: on 16 GB it read ~17 GB from the SSD per prompt and made a 70-token prompt 5 s slower).
bool prefill_advise() {
  static const bool on = [] {
    const char* e = std::getenv("VEYSTRIX_GENERIC_PREFILL_ADVISE");
    return e && std::strcmp(e, "1") == 0;
  }();
  return on;
}

std::vector<mx::array> GenericModel::forward(const std::vector<std::vector<int32_t>>& tokens,
                                             const std::vector<SessionState*>& states,
                                             ForwardStats* stats, bool prefetch_next_layer) {
  const size_t n = tokens.size();
  if (n == 0 || states.size() != n) throw std::runtime_error("generic: forward lane mismatch");
  auto& pf = cppprof::profiler();
  std::vector<mx::array> h;
  for (size_t i = 0; i < n; ++i) {
    if (tokens[i].empty()) throw std::runtime_error("generic: empty lane");
    if (states[i]->size() != visit_layer_.size()) throw std::runtime_error("generic: state not reset");
    mx::array ids(tokens[i].data(), mx::Shape{1, static_cast<int>(tokens[i].size())}, mx::int32);
    mx::array emb = embed_(ids);
    if (embed_norm_) emb = mx::fast::rms_norm(emb, *embed_norm_, eps_);
    h.push_back(hyper_ ? mx::tile(emb, std::vector<int>{1, 1, hc_}) : emb);
  }
  if (block_attn_res_) {
    return forward_inkling(h, states, stats, prefetch_next_layer, /*all_logits=*/false,
                           /*capture_features=*/false, /*save_verify_history=*/false, nullptr);
  }
  MlpStats ms;

  std::vector<mx::array> inj;  // hyper: per-lane injection weights of the current branch
  for (size_t l = 0; l < mixers_.size(); ++l) {
    std::vector<mx::array> xs;
    inj.clear();
    for (size_t i = 0; i < n && hyper_; ++i) {
      // Qwen4ExpDecoderLayer: (PLE) -> attn hyper-connection -> branch -> inject -> mlp hyper-connection.
      LayerState& st = (*states[i])[l];
      if (pre_[l]) h[i] = h[i] + pre_[l]->forward(h[i], tokens[i], st);
      auto [mixed, w] = hyper_mix(h[i], attn_hc_[l]);
      mx::array r = mixers_[l]->forward(mixed, st);
      h[i] = h[i] + mx::reshape(mx::expand_dims(r, -2) * mx::expand_dims(w, -1), h[i].shape());
      auto [mixed2, w2] = hyper_mix(h[i], mlp_hc_[l]);
      xs.push_back(mixed2);
      inj.push_back(w2);
    }
    for (size_t i = 0; i < n && !hyper_; ++i) {
      // DecoderLayer: h = x + mixer(input_layernorm(x))
      mx::array r = mixers_[l]->forward(mx::fast::rms_norm(h[i], in_norm_[l], eps_), (*states[i])[l]);
      h[i] = mx::add(h[i], r);
      xs.push_back(mx::fast::rms_norm(h[i], post_norm_[l], eps_));
    }
    const int lane_tokens = n == 1 ? h[0].shape(1) : 0;
    const bool piped = prefetch_next_layer && l + 1 < mlps_.size() && lane_tokens > 0 &&
                       mlps_[l + 1]->prefill_prefetch_wanted(lane_tokens);
    if (prefetch_next_layer && l + 1 < mlps_.size() && !piped && prefill_advise())
      mlps_[l + 1]->prefetch_hint();
    const double sync_before = ms.route_sync_ms;
    std::vector<mx::array> ys = mlps_[l]->forward(xs, ms);
    pf.add_route_sync(static_cast<int>(l), ms.route_sync_ms - sync_before);
    // Background read of the next layer's experts while this layer and the next mixer compute.
    if (piped) mlps_[l + 1]->prefill_prefetch(lane_tokens);
    for (size_t i = 0; i < n; ++i)
      h[i] = hyper_ ? h[i] + mx::reshape(mx::expand_dims(ys[i], -2) * mx::expand_dims(inj[i], -1), h[i].shape())
                    : mx::add(h[i], ys[i]);
  }
  std::vector<mx::array> logits;
  for (size_t i = 0; i < n; ++i) {
    const int S = h[i].shape(1);
    mx::array x = mx::array(0.0f);
    if (hyper_) {
      // Final mixer over every position, then the last one (the custom runtime's order).
      mx::array mixed = hyper_final(h[i]);
      x = mx::slice(mixed, {0, S - 1, 0}, {1, S, hidden_});
    } else {
      mx::array last = mx::slice(h[i], {0, S - 1, 0}, {1, S, hidden_});
      x = mx::fast::rms_norm(last, final_norm_, eps_);
    }
    mx::array lg = tied_ ? embed_.as_linear(x) : (*head_)(x);
    logits.push_back(mx::reshape(lg, {1, vocab_}));
  }
  if (stats) {
    stats->expert_bytes += ms.expert_bytes;
    stats->route_sync_ms += ms.route_sync_ms;
    stats->read_ms += ms.read_ms;
  }
  return logits;
}

VerifyResult GenericModel::forward_with_features(const std::vector<int32_t>& tokens, SessionState& state,
                                                 bool all_logits, bool save_verify_history,
                                                 ForwardStats* stats, bool prefetch_next_layer) {
  if (tokens.empty()) throw std::runtime_error("generic: empty lane");
  if (state.size() != visit_layer_.size()) throw std::runtime_error("generic: state not reset");
  mx::array ids(tokens.data(), mx::Shape{1, static_cast<int>(tokens.size())}, mx::int32);
  mx::array emb = embed_normalized(ids);
  std::vector<mx::array> h{emb};
  std::vector<SessionState*> st_ptrs{&state};
  VerifyResult out;
  const bool cap = dspark_ != nullptr;
  auto lgs = forward_inkling(h, st_ptrs, stats, prefetch_next_layer, all_logits, cap,
                             save_verify_history, cap ? &out.features : nullptr);
  out.logits = lgs[0];
  return out;
}

bool GenericModel::trim_verify_state(SessionState& st, int64_t old_n, int keep, int total_verify) const {
  for (size_t v = 0; v < st.size(); ++v) {
    LayerState& ls = st[v];
    ls.save_conv_history = false;
    if (ls.kv) {
      if (!ls.kv->truncate(old_n + keep)) return false;
      ls.kv->compact_if_needed();
    }
    ls.tokens = old_n + keep;
    auto trim_vec = [&](std::vector<mx::array>& vec) {
      for (auto& a : vec) {
        if (a.ndim() == 3 && a.shape(1) > total_verify) {
          const int hist = a.shape(1) - total_verify;
          a = mx::contiguous(mx::slice(a, {0, keep, 0}, {a.shape(0), keep + hist, a.shape(2)}));
        }
      }
    };
    trim_vec(ls.arrays);
    trim_vec(ls.pre);
  }
  return true;
}

bool GenericModel::truncate(SessionState& st, int64_t target_tokens) const {
  if (target_tokens <= 0) {
    reset(st);
    return true;
  }
  for (size_t v = 0; v < st.size(); ++v) {
    LayerState& ls = st[v];
    ls.save_conv_history = false;
    if (ls.kv) {
      if (!ls.kv->truncate(target_tokens)) return false;
      ls.kv->compact_if_needed();
    }
    ls.tokens = std::min<int64_t>(ls.tokens, target_tokens);
  }
  return true;
}

VerifyResult GenericModel::prefill_stage_major(const std::vector<int32_t>& tokens, SessionState& state,
                                               ForwardStats* stats,
                                               const std::function<bool()>& cancel_check) {
  if (tokens.empty()) throw std::runtime_error("generic: empty lane");
  if (state.size() != visit_layer_.size()) throw std::runtime_error("generic: state not reset");
  if (cancel_check && cancel_check()) return VerifyResult{};

  mx::array ids(tokens.data(), mx::Shape{1, static_cast<int>(tokens.size())}, mx::int32);
  mx::array emb = embed_normalized(ids);
  const int S = static_cast<int>(tokens.size());

  auto rms_norm_f32 = [&](const mx::array& a) -> mx::array {
    static auto rn_c = mx::compile(
        [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
          mx::array f = mx::astype(in[0], mx::float32);
          mx::array n = mx::multiply(f, mx::rsqrt(mx::mean(mx::square(f), -1, /*keepdims=*/true) + in[1]));
          return {mx::astype(n, mx::bfloat16)};
        },
        true);
    return rn_c({a, mx::array(eps_, mx::float32)})[0];
  };

  auto residual_sconv = [](const mx::array& x, const mx::array& kern_cw1,
                           std::optional<mx::array>& st_slot) {
    const int B = x.shape(0), L = x.shape(1), C = x.shape(2);
    const int W = kern_cw1.shape(1);
    const int hist = W - 1;
    mx::array st = st_slot ? *st_slot : mx::zeros({B, hist, C}, x.dtype());
    mx::array inp = mx::concatenate({st, x}, 1);
    st_slot = mx::contiguous(mx::slice(inp, {0, L, 0}, {B, L + hist, C}));
    if (W == 4) {
      static auto tap4_c = mx::compile(
          [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
            mx::array y = mx::multiply(mx::astype(in[0], mx::float32), in[4]);
            y = mx::add(y, mx::multiply(mx::astype(in[1], mx::float32), in[5]));
            y = mx::add(y, mx::multiply(mx::astype(in[2], mx::float32), in[6]));
            y = mx::add(y, mx::multiply(mx::astype(in[3], mx::float32), in[7]));
            return {mx::astype(y, in[0].dtype())};
          },
          true);
      mx::array kwc = mx::expand_dims(mx::transpose(mx::squeeze(kern_cw1, -1)), 0);
      return tap4_c({mx::slice(inp, {0, 0, 0}, {B, L, C}), mx::slice(inp, {0, 1, 0}, {B, L + 1, C}),
                     mx::slice(inp, {0, 2, 0}, {B, L + 2, C}), mx::slice(inp, {0, 3, 0}, {B, L + 3, C}),
                     mx::slice(kwc, {0, 0, 0}, {1, 1, C}), mx::slice(kwc, {0, 1, 0}, {1, 2, C}),
                     mx::slice(kwc, {0, 2, 0}, {1, 3, C}), mx::slice(kwc, {0, 3, 0}, {1, 4, C})})[0];
    }
    mx::array y = mx::conv1d(mx::astype(inp, mx::float32), kern_cw1, 1, 0, 1, C);
    return mx::astype(y, x.dtype());
  };

  auto bank_last = [](const mx::array& bk) -> mx::array {
    const int N = bk.shape(0);
    return mx::squeeze(mx::slice(bk, {N - 1, 0, 0, 0}, {N, bk.shape(1), bk.shape(2), bk.shape(3)}), 0);
  };

  mx::array banks_norm = mx::expand_dims(rms_norm_f32(emb), 0);
  std::optional<mx::array> partial;
  std::optional<mx::array> prelude_banks;
  std::optional<mx::array> prev_sn, prev_vals;

  std::vector<mx::array> captured;
  const std::vector<int>* tids = dspark_ ? &dspark_->target_layer_ids() : nullptr;
  if (tids) captured.reserve(tids->size());
  size_t tid_cursor = 0;

  const bool tight_16g_pf = membudget::physical_ram_bytes() < (size_t{20} << 30);
  MlpStats ms;
  for (size_t v = 0; v < visit_layer_.size(); ++v) {
    if (cancel_check && cancel_check()) return VerifyResult{};
    const int l = visit_layer_[v];
    const int pass = visit_pass_[v];
    const bool is_core = (looped_ && l >= loop_core_start_ && l < loop_core_end_);
    const bool allow_temp_res = !tight_16g_pf ||
        (membudget::memory_pressure_level() < 2 &&
         membudget::available_ram_bytes() >= (size_t{28} << 30) / 10);
    const bool need_temp_res = !is_core && !mlps_[l]->is_resident() && allow_temp_res;

    if (need_temp_res) {
      mlps_[l]->load_resident();
    }

    if (looped_ && l == loop_core_start_ && pass > 0) {
      const int base = prelude_banks->shape(0);
      const int cur_n = banks_norm.shape(0);
      if (cur_n > base) {
        mx::array sn = mx::slice(banks_norm, {base, 0, 0, 0},
                                 {cur_n, banks_norm.shape(1), banks_norm.shape(2), banks_norm.shape(3)});
        prev_sn = sn;
        prev_vals = sn;
      } else {
        prev_sn.reset();
        prev_vals.reset();
      }
      banks_norm = *prelude_banks;
      partial.reset();
    }

    if (tids && tid_cursor < tids->size() && (*tids)[tid_cursor] == static_cast<int>(v)) {
      mx::array feat = partial ? *partial : mx::astype(bank_last(banks_norm), emb.dtype());
      captured.push_back(mx::reshape(feat, {S, hidden_}));
      ++tid_cursor;
    }

    state[v].save_conv_history = false;
    mx::array r = mx::array(0.0f);
    const bool has_msg = pass > 0 && loop_gates_[l] && prev_sn.has_value();
    const bool three_q = attn_res_q_[l] && attn_res_q_[l]->ndim() == 2 && attn_res_q_[l]->shape(0) == 3;
    if (three_q) {
      auto qkv = attn_res_mix_multi(banks_norm, partial, *attn_res_q_[l]);
      mx::array xq = qkv[0], xk = qkv[1], xv = qkv[2];
      if (has_msg) {
        xq = apply_loop_msg(xq, *prev_sn, *prev_vals, *loop_gates_[l], 0);
        xk = apply_loop_msg(xk, *prev_sn, *prev_vals, *loop_gates_[l], 1);
        xv = apply_loop_msg(xv, *prev_sn, *prev_vals, *loop_gates_[l], 2);
      }
      xq = mx::fast::rms_norm(xq, q_in_norm_[l] ? *q_in_norm_[l] : in_norm_[l], eps_);
      xk = mx::fast::rms_norm(xk, k_in_norm_[l] ? *k_in_norm_[l] : in_norm_[l], eps_);
      xv = mx::fast::rms_norm(xv, v_in_norm_[l] ? *v_in_norm_[l] : in_norm_[l], eps_);
      r = mixers_[l]->forward_qkv(xq, xk, xv, state[v]);
    } else {
      mx::array mix = attn_res_q_[l]
                          ? attn_res_mix(banks_norm, partial, *attn_res_q_[l])
                          : (partial ? *partial : mx::astype(bank_last(banks_norm), emb.dtype()));
      if (has_msg) {
        auto qkv_m = apply_loop_msg_qkv_shared(mix, *prev_sn, *loop_gates_[l]);
        mx::array xq = qkv_m[0], xk = qkv_m[1], xv = qkv_m[2];
        xq = mx::fast::rms_norm(xq, q_in_norm_[l] ? *q_in_norm_[l] : in_norm_[l], eps_);
        xk = mx::fast::rms_norm(xk, k_in_norm_[l] ? *k_in_norm_[l] : in_norm_[l], eps_);
        xv = mx::fast::rms_norm(xv, v_in_norm_[l] ? *v_in_norm_[l] : in_norm_[l], eps_);
        r = mixers_[l]->forward_qkv(xq, xk, xv, state[v]);
      } else if (q_in_norm_[l] && k_in_norm_[l] && v_in_norm_[l]) {
        mx::array xq = mx::fast::rms_norm(mix, *q_in_norm_[l], eps_);
        mx::array xk = mx::fast::rms_norm(mix, *k_in_norm_[l], eps_);
        mx::array xv = mx::fast::rms_norm(mix, *v_in_norm_[l], eps_);
        r = mixers_[l]->forward_qkv(xq, xk, xv, state[v]);
      } else {
        mx::array xn = mx::fast::rms_norm(mix, in_norm_[l], eps_);
        r = mixers_[l]->forward(xn, state[v]);
      }
    }

    if (attn_sconv_[l]) {
      auto& pre_st = state[v].pre;
      std::optional<mx::array> st_a = pre_st.size() >= 1 ? std::optional<mx::array>(pre_st[0]) : std::nullopt;
      r = residual_sconv(r, *attn_sconv_[l], st_a);
      if (pre_st.size() < 2) pre_st.resize(2, mx::zeros_like(*st_a));
      pre_st[0] = *st_a;
    }
    partial = partial ? mx::add(*partial, r) : r;
    mx::array mlp_in = mlp_res_q_[l] ? attn_res_mix(banks_norm, partial, *mlp_res_q_[l]) : *partial;
    if (has_msg)
      mlp_in = apply_loop_msg(mlp_in, *prev_sn, *prev_vals, *loop_gates_[l], 3);
    mx::array x_mlp = mx::fast::rms_norm(mlp_in, post_norm_[l], eps_);

    // On higher-RAM Macs (>= 20 GiB), overlap streamed SSD reads with Metal GPU compute:
    // start layer 18's read 4 layers early (layer 14 pass 2) and overlap next streamed layer.
    const bool allow_pf_overlap = !tight_16g_pf;
    if (allow_pf_overlap && looped_ && pass == loop_passes_ - 1 && l == loop_core_end_ - 4 &&
        loop_core_end_ < static_cast<int>(mlps_.size()) &&
        mlps_[loop_core_end_]->prefill_prefetch_wanted(S)) {
      mlps_[loop_core_end_]->prefill_prefetch(S);
    }
    const int next_l = (v + 1 < visit_layer_.size()) ? visit_layer_[v + 1] : -1;
    if (allow_pf_overlap && !need_temp_res && next_l >= 0 &&
        mlps_[next_l]->prefill_prefetch_wanted(S)) {
      mlps_[next_l]->prefill_prefetch(S);
    }

    mx::array mo = mx::array(0.0f);
    if (S <= 2048) {
      mo = mlps_[l]->forward({x_mlp}, ms)[0];
    } else {
      std::vector<mx::array> mo_parts;
      const int chunk_sz = 2048;
      for (int off = 0; off < S; off += chunk_sz) {
        const int take = std::min(chunk_sz, S - off);
        mx::array x_slice = mx::slice(x_mlp, {0, off, 0}, {1, off + take, hidden_});
        mo_parts.push_back(mlps_[l]->forward({x_slice}, ms)[0]);
        mx::eval(mo_parts.back());
      }
      mo = mx::concatenate(mo_parts, 1);
    }
    if (mlp_sconv_[l]) {
      auto& pre_st = state[v].pre;
      std::optional<mx::array> st_m = pre_st.size() >= 2 ? std::optional<mx::array>(pre_st[1]) : std::nullopt;
      mo = residual_sconv(mo, *mlp_sconv_[l], st_m);
      if (pre_st.size() < 2) pre_st.resize(2, mx::zeros_like(*st_m));
      pre_st[1] = *st_m;
    }
    partial = mx::add(*partial, mo);

    const bool at_block_end = (res_block_size_ > 0 && ((l + 1) % res_block_size_ == 0));
    if (at_block_end) {
      banks_norm = mx::concatenate({banks_norm, mx::expand_dims(rms_norm_f32(*partial), 0)}, 0);
      partial.reset();
    }

    // Materialize state and release resident buffers for streamed layers
    std::vector<mx::array> ev{partial ? *partial : banks_norm};
    for (auto& a : state[v].arrays) ev.push_back(a);
    for (auto& a : state[v].pre) ev.push_back(a);
    if (state[v].kv) state[v].kv->append_state_arrays(ev);
    if (!captured.empty()) ev.push_back(captured.back());
    mx::eval(ev);

    if (need_temp_res) {
      mlps_[l]->release_resident();
    }
    mx::clear_cache();

    if (looped_ && l == loop_core_start_ - 1 && pass == 0) {
      prelude_banks = banks_norm;
    }
  }

  if (stats) {
    stats->expert_bytes += ms.expert_bytes;
    stats->read_ms += ms.read_ms;
  }

  // Compute final logits for the last token position
  const bool need_all_final = (tids && tid_cursor < tids->size());
  mx::array x = mx::array(0.0f);
  if (need_all_final) {
    mx::array last_all = final_res_query_
                             ? attn_res_mix(banks_norm, partial, *final_res_query_)
                             : (partial ? *partial : mx::astype(bank_last(banks_norm), emb.dtype()));
    mx::array x_all = mx::fast::rms_norm(last_all, final_norm_, eps_);
    if (tids && tid_cursor < tids->size() && (*tids)[tid_cursor] == static_cast<int>(visit_layer_.size())) {
      captured.push_back(mx::reshape(x_all, {S, hidden_}));
      ++tid_cursor;
    }
    x = mx::slice(x_all, {0, S - 1, 0}, {1, S, hidden_});
  } else {
    mx::array last_bank = mx::slice(banks_norm, {0, 0, S - 1, 0}, {banks_norm.shape(0), 1, S, hidden_});
    std::optional<mx::array> last_partial =
        partial ? std::optional<mx::array>(mx::slice(*partial, {0, S - 1, 0}, {1, S, hidden_}))
                : std::nullopt;
    mx::array last = final_res_query_
                         ? attn_res_mix(last_bank, last_partial, *final_res_query_)
                         : (last_partial ? *last_partial : mx::astype(bank_last(last_bank), emb.dtype()));
    x = mx::fast::rms_norm(last, final_norm_, eps_);
  }

  mx::array lg = lm_head_raw(x);
  if (final_logit_softcap_ > 0.0f) {
    mx::array cap = mx::array(final_logit_softcap_, mx::float32);
    lg = mx::astype(mx::multiply(cap, mx::tanh(mx::divide(mx::astype(lg, mx::float32), cap))), lg.dtype());
  }
  lg = mx::reshape(lg, {1, vocab_});
  if (unpadded_vocab_ > 0 && unpadded_vocab_ < vocab_) {
    mx::array valid = mx::less(mx::arange(vocab_, mx::int32), mx::array(unpadded_vocab_, mx::int32));
    lg = mx::where(valid, lg, mx::array(-1e9f, lg.dtype()));
  }
  mx::eval(lg);

  VerifyResult out;
  out.logits = lg;
  if (tids && captured.size() == tids->size()) {
    out.features = mx::stack(captured, 1);
    mx::eval(*out.features);
  }
  return out;
}

std::vector<mx::array> GenericModel::forward_inkling(const std::vector<mx::array>& h,
                                                     const std::vector<SessionState*>& states,
                                                     ForwardStats* stats, bool prefetch_next_layer,
                                                     bool all_logits, bool capture_features,
                                                     bool save_verify_history,
                                                     std::optional<mx::array>* out_features) {
  const size_t n = h.size();
  auto& pf = cppprof::profiler();
  MlpStats ms;
  static const bool phase_timing = [] {
    const char* e = std::getenv("VEYSTRIX_INKLING_TIMING");
    return e && std::strcmp(e, "1") == 0;
  }();
  double t_res_ms = 0, t_attn_ms = 0, t_moe_ms = 0, t_head_ms = 0;
  auto now_clk = [] { return std::chrono::steady_clock::now(); };
  auto ms_diff = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

  auto residual_sconv = [save_verify_history](const mx::array& x, const mx::array& kern_cw1,
                                              std::optional<mx::array>& st_slot) {
    const int B = x.shape(0), L = x.shape(1), C = x.shape(2);
    const int W = kern_cw1.shape(1);
    const int hist = W - 1;
    mx::array st = st_slot ? *st_slot : mx::zeros({B, hist, C}, x.dtype());
    mx::array inp = mx::concatenate({st, x}, 1);
    st_slot = save_verify_history ? mx::contiguous(inp)
                                  : mx::contiguous(mx::slice(inp, {0, L, 0}, {B, L + hist, C}));
    if (L == 1) {
      static auto dot_c = mx::compile(
          [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
            mx::array kwc = mx::expand_dims(mx::transpose(mx::squeeze(in[1], -1)), 0);
            mx::array y = mx::sum(mx::multiply(mx::astype(in[0], mx::float32), kwc), 1, /*keepdims=*/true);
            return {mx::astype(y, in[0].dtype())};
          },
          true);
      return dot_c({inp, kern_cw1})[0];
    }
    if (W == 4) {
      static auto tap4_c = mx::compile(
          [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
            mx::array y = mx::multiply(mx::astype(in[0], mx::float32), in[4]);
            y = mx::add(y, mx::multiply(mx::astype(in[1], mx::float32), in[5]));
            y = mx::add(y, mx::multiply(mx::astype(in[2], mx::float32), in[6]));
            y = mx::add(y, mx::multiply(mx::astype(in[3], mx::float32), in[7]));
            return {mx::astype(y, in[0].dtype())};
          },
          true);
      mx::array kwc = mx::expand_dims(mx::transpose(mx::squeeze(kern_cw1, -1)), 0);
      return tap4_c({mx::slice(inp, {0, 0, 0}, {B, L, C}), mx::slice(inp, {0, 1, 0}, {B, L + 1, C}),
                     mx::slice(inp, {0, 2, 0}, {B, L + 2, C}), mx::slice(inp, {0, 3, 0}, {B, L + 3, C}),
                     mx::slice(kwc, {0, 0, 0}, {1, 1, C}), mx::slice(kwc, {0, 1, 0}, {1, 2, C}),
                     mx::slice(kwc, {0, 2, 0}, {1, 3, C}), mx::slice(kwc, {0, 3, 0}, {1, 4, C})})[0];
    }
    mx::array y = mx::conv1d(mx::astype(inp, mx::float32), kern_cw1, 1, 0, 1, C);
    return mx::astype(y, x.dtype());
  };

  auto rms_norm_f32 = [&](const mx::array& a) -> mx::array {
    static auto rn_c = mx::compile(
        [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
          mx::array f = mx::astype(in[0], mx::float32);
          mx::array n = mx::multiply(f, mx::rsqrt(mx::mean(mx::square(f), -1, /*keepdims=*/true) + in[1]));
          return {mx::astype(n, mx::bfloat16)};
        },
        true);
    return rn_c({a, mx::array(eps_, mx::float32)})[0];
  };

  std::vector<mx::array> captured;
  const std::vector<int>* tids = (capture_features && dspark_) ? &dspark_->target_layer_ids() : nullptr;
  if (tids) captured.reserve(tids->size());
  size_t tid_cursor = 0;

  auto bank_last = [](const mx::array& bk) -> mx::array {
    const int N = bk.shape(0);
    return mx::squeeze(mx::slice(bk, {N - 1, 0, 0, 0}, {N, bk.shape(1), bk.shape(2), bk.shape(3)}), 0);
  };
  std::vector<mx::array> banks_norm;
  banks_norm.reserve(n);
  std::vector<std::optional<mx::array>> partials(n);
  for (size_t i = 0; i < n; ++i) banks_norm.push_back(mx::expand_dims(rms_norm_f32(h[i]), 0));
  std::vector<std::optional<mx::array>> prelude_banks(n);
  std::vector<std::optional<mx::array>> prev_sn(n), prev_vals(n);

  if (prefetch_next_layer && n == 1 && h[0].shape(1) >= 32 && !visit_layer_.empty()) {
    const int l0 = visit_layer_[0];
    if (mlps_[l0]->prefill_prefetch_wanted(h[0].shape(1))) mlps_[l0]->prefill_prefetch(h[0].shape(1));
  }
  for (size_t v = 0; v < visit_layer_.size(); ++v) {
    const int l = visit_layer_[v];
    const int pass = visit_pass_[v];
    if (looped_ && l == loop_core_start_ && pass > 0) {
      for (size_t i = 0; i < n; ++i) {
        const int base = prelude_banks[i]->shape(0);
        const int cur_n = banks_norm[i].shape(0);
        if (cur_n > base) {
          mx::array sn = mx::slice(banks_norm[i], {base, 0, 0, 0},
                                   {cur_n, banks_norm[i].shape(1), banks_norm[i].shape(2), banks_norm[i].shape(3)});
          prev_sn[i] = sn;
          prev_vals[i] = sn;
        } else {
          prev_sn[i].reset();
          prev_vals[i].reset();
        }
        banks_norm[i] = *prelude_banks[i];
        partials[i].reset();
      }
    }
    if (tids && tid_cursor < tids->size() && (*tids)[tid_cursor] == static_cast<int>(v)) {
      mx::array feat = partials[0] ? *partials[0] : mx::astype(bank_last(banks_norm[0]), h[0].dtype());
      captured.push_back(mx::reshape(feat, {h[0].shape(1), hidden_}));
      ++tid_cursor;
    }
    std::vector<mx::array> xs;
    for (size_t i = 0; i < n; ++i) {
      auto t0 = now_clk();
      (*states[i])[v].save_conv_history = save_verify_history;
      mx::array r = mx::array(0.0f);
      const bool has_msg = pass > 0 && loop_gates_[l] && prev_sn[i].has_value();
      const bool three_q = attn_res_q_[l] && attn_res_q_[l]->ndim() == 2 && attn_res_q_[l]->shape(0) == 3;
      if (three_q) {
        auto qkv = attn_res_mix_multi(banks_norm[i], partials[i], *attn_res_q_[l]);
        mx::array xq = qkv[0], xk = qkv[1], xv = qkv[2];
        if (has_msg) {
          xq = apply_loop_msg(xq, *prev_sn[i], *prev_vals[i], *loop_gates_[l], 0);
          xk = apply_loop_msg(xk, *prev_sn[i], *prev_vals[i], *loop_gates_[l], 1);
          xv = apply_loop_msg(xv, *prev_sn[i], *prev_vals[i], *loop_gates_[l], 2);
        }
        xq = mx::fast::rms_norm(xq, q_in_norm_[l] ? *q_in_norm_[l] : in_norm_[l], eps_);
        xk = mx::fast::rms_norm(xk, k_in_norm_[l] ? *k_in_norm_[l] : in_norm_[l], eps_);
        xv = mx::fast::rms_norm(xv, v_in_norm_[l] ? *v_in_norm_[l] : in_norm_[l], eps_);
        if (phase_timing) { mx::eval({xq, xk, xv}); auto t1 = now_clk(); t_res_ms += ms_diff(t0, t1); t0 = t1; }
        r = mixers_[l]->forward_qkv(xq, xk, xv, (*states[i])[v]);
      } else {
        mx::array mix = attn_res_q_[l]
                            ? attn_res_mix(banks_norm[i], partials[i], *attn_res_q_[l])
                            : (partials[i] ? *partials[i] : mx::astype(bank_last(banks_norm[i]), h[i].dtype()));
        if (has_msg) {
          auto qkv_m = apply_loop_msg_qkv_shared(mix, *prev_sn[i], *loop_gates_[l]);
          mx::array xq = qkv_m[0], xk = qkv_m[1], xv = qkv_m[2];
          xq = mx::fast::rms_norm(xq, q_in_norm_[l] ? *q_in_norm_[l] : in_norm_[l], eps_);
          xk = mx::fast::rms_norm(xk, k_in_norm_[l] ? *k_in_norm_[l] : in_norm_[l], eps_);
          xv = mx::fast::rms_norm(xv, v_in_norm_[l] ? *v_in_norm_[l] : in_norm_[l], eps_);
          if (phase_timing) { mx::eval({xq, xk, xv}); auto t1 = now_clk(); t_res_ms += ms_diff(t0, t1); t0 = t1; }
          r = mixers_[l]->forward_qkv(xq, xk, xv, (*states[i])[v]);
        } else if (q_in_norm_[l] && k_in_norm_[l] && v_in_norm_[l]) {
          mx::array xq = mx::fast::rms_norm(mix, *q_in_norm_[l], eps_);
          mx::array xk = mx::fast::rms_norm(mix, *k_in_norm_[l], eps_);
          mx::array xv = mx::fast::rms_norm(mix, *v_in_norm_[l], eps_);
          if (phase_timing) { mx::eval({xq, xk, xv}); auto t1 = now_clk(); t_res_ms += ms_diff(t0, t1); t0 = t1; }
          r = mixers_[l]->forward_qkv(xq, xk, xv, (*states[i])[v]);
        } else {
          mx::array xn = mx::fast::rms_norm(mix, in_norm_[l], eps_);
          if (phase_timing) { mx::eval(xn); auto t1 = now_clk(); t_res_ms += ms_diff(t0, t1); t0 = t1; }
          r = mixers_[l]->forward(xn, (*states[i])[v]);
        }
      }
      (*states[i])[v].save_conv_history = false;
      if (phase_timing) { mx::eval(r); auto t1 = now_clk(); t_attn_ms += ms_diff(t0, t1); t0 = t1; }
      if (attn_sconv_[l]) {
        auto& pre_st = (*states[i])[v].pre;
        std::optional<mx::array> st_a = pre_st.size() >= 1 ? std::optional<mx::array>(pre_st[0]) : std::nullopt;
        r = residual_sconv(r, *attn_sconv_[l], st_a);
        if (pre_st.size() < 2) pre_st.resize(2, mx::zeros_like(*st_a));
        pre_st[0] = *st_a;
      }
      partials[i] = partials[i] ? mx::add(*partials[i], r) : r;
      mx::array mlp_in = mlp_res_q_[l] ? attn_res_mix(banks_norm[i], partials[i], *mlp_res_q_[l]) : *partials[i];
      if (has_msg)
        mlp_in = apply_loop_msg(mlp_in, *prev_sn[i], *prev_vals[i], *loop_gates_[l], 3);
      mx::array x_mlp = mx::fast::rms_norm(mlp_in, post_norm_[l], eps_);
      if (phase_timing) { mx::eval(x_mlp); t_res_ms += ms_diff(t0, now_clk()); }
      xs.push_back(x_mlp);
    }
    const int lane_tokens = n == 1 ? h[0].shape(1) : 0;
    const int next_l = (v + 1 < visit_layer_.size()) ? visit_layer_[v + 1] : -1;
    const bool piped = prefetch_next_layer && next_l >= 0 && lane_tokens > 0 &&
                       mlps_[next_l]->prefill_prefetch_wanted(lane_tokens);
    bool pre_started_piped = false;
    // Looped-core advantage: at the end of Loop Pass 1 (layer 17, pass 1), warm ALL 6 streamed
    // epilogue layers (18..23) in the OS page cache in parallel with Loop Pass 2's 12 GPU layers.
    if (looped_ && n == 1 && lane_tokens >= 1 && lane_tokens <= 8 &&
        l == loop_core_end_ - 1 && pass == loop_passes_ - 2 && loop_core_end_ < static_cast<int>(mlps_.size())) {
      std::vector<const Mlp*> epilogue_targets;
      for (int el = loop_core_end_; el < static_cast<int>(mlps_.size()); ++el) {
        if (!mlps_[el]->is_resident()) epilogue_targets.push_back(mlps_[el].get());
      }
      if (!epilogue_targets.empty()) mlps_[l]->prefetch_epilogue_from_loop(xs[0], epilogue_targets);
    }
    // During prefill, start layer 18's 547 MiB SSD read 4 layers early (layer 14, pass 2) so it
    // finishes 100% hidden behind layers 14..17's Metal GPU compute.
    if (prefetch_next_layer && looped_ && n == 1 && lane_tokens >= 32 &&
        pass == loop_passes_ - 1 && l == loop_core_end_ - 4 && loop_core_end_ < static_cast<int>(mlps_.size())) {
      if (mlps_[loop_core_end_]->prefill_prefetch_wanted(lane_tokens)) {
        mlps_[loop_core_end_]->prefill_prefetch(lane_tokens);
      }
    }
    const bool at_res_to_stream_boundary =
        (looped_ && n == 1 && next_l >= 0 && mlps_[l]->is_resident() && !mlps_[next_l]->is_resident());
    if (at_res_to_stream_boundary) {
      if (lane_tokens >= 1 && lane_tokens <= 8) {
        mlps_[l]->spec_prefetch_next(xs[0]);
      } else if (piped) {
        mlps_[next_l]->prefill_prefetch(lane_tokens);
        pre_started_piped = true;
      }
    }
    const double sync_before = ms.route_sync_ms;
    auto tm0 = now_clk();
    std::vector<mx::array> ys = mlps_[l]->forward(xs, ms);
    if (phase_timing) { mx::eval(ys); t_moe_ms += ms_diff(tm0, now_clk()); }
    pf.add_route_sync(l, ms.route_sync_ms - sync_before);
    if (piped && !pre_started_piped) mlps_[next_l]->prefill_prefetch(lane_tokens);
    const bool flush_streamed_prefill = (lane_tokens >= 32 && !mlps_[l]->is_resident());
    const bool flush_prefill_layer = (lane_tokens >= 256);
    for (size_t i = 0; i < n; ++i) {
      auto tr0 = now_clk();
      mx::array mo = ys[i];
      if (mlp_sconv_[l]) {
        auto& pre_st = (*states[i])[v].pre;
        std::optional<mx::array> st_m = pre_st.size() >= 2 ? std::optional<mx::array>(pre_st[1]) : std::nullopt;
        mo = residual_sconv(mo, *mlp_sconv_[l], st_m);
        if (pre_st.size() < 2) pre_st.resize(2, mx::zeros_like(*st_m));
        pre_st[1] = *st_m;
      }
      partials[i] = mx::add(*partials[i], mo);
      const bool at_block_end = (res_block_size_ > 0 && ((l + 1) % res_block_size_ == 0));
      if (at_block_end) {
        banks_norm[i] = mx::concatenate({banks_norm[i], mx::expand_dims(rms_norm_f32(*partials[i]), 0)}, 0);
        partials[i].reset();
      }
      if (flush_streamed_prefill || flush_prefill_layer) {
        std::vector<mx::array> ev{partials[i] ? *partials[i] : banks_norm[i]};
        LayerState& ls = (*states[i])[v];
        for (auto& a : ls.arrays) ev.push_back(a);
        for (auto& a : ls.pre) ev.push_back(a);
        if (ls.kv) ls.kv->append_state_arrays(ev);
        if (i == 0 && !captured.empty()) ev.push_back(captured.back());
        mx::eval(ev);
        if (flush_streamed_prefill || at_block_end) mx::clear_cache();
      } else if (at_block_end) {
        if (lane_tokens >= 32) {
          mx::eval(banks_norm[i]);
          mx::clear_cache();
        } else {
          mx::async_eval(banks_norm[i]);
        }
      } else if (phase_timing) {
        mx::eval(partials[i] ? *partials[i] : banks_norm[i]);
        t_res_ms += ms_diff(tr0, now_clk());
      }
    }
    if (looped_ && l == loop_core_start_ - 1 && pass == 0) {
      for (size_t i = 0; i < n; ++i) prelude_banks[i] = banks_norm[i];
    }
  }

  std::vector<mx::array> logits;
  for (size_t i = 0; i < n; ++i) {
    const int S = h[i].shape(1);
    mx::array x = mx::array(0.0f);
    const bool need_all_final = all_logits || (i == 0 && tids && tid_cursor < tids->size());
    if (need_all_final) {
      mx::array last_all = final_res_query_
                               ? attn_res_mix(banks_norm[i], partials[i], *final_res_query_)
                               : (partials[i] ? *partials[i] : mx::astype(bank_last(banks_norm[i]), h[i].dtype()));
      mx::array x_all = mx::fast::rms_norm(last_all, final_norm_, eps_);
      if (i == 0 && tids && tid_cursor < tids->size() && (*tids)[tid_cursor] == static_cast<int>(visit_layer_.size())) {
        captured.push_back(mx::reshape(x_all, {S, hidden_}));
        ++tid_cursor;
      }
      x = all_logits ? x_all : mx::slice(x_all, {0, S - 1, 0}, {1, S, hidden_});
    } else {
      mx::array last_bank = mx::slice(banks_norm[i], {0, 0, S - 1, 0}, {banks_norm[i].shape(0), 1, S, hidden_});
      std::optional<mx::array> last_partial =
          partials[i] ? std::optional<mx::array>(mx::slice(*partials[i], {0, S - 1, 0}, {1, S, hidden_}))
                      : std::nullopt;
      mx::array last = final_res_query_
                           ? attn_res_mix(last_bank, last_partial, *final_res_query_)
                           : (last_partial ? *last_partial : mx::astype(bank_last(last_bank), h[i].dtype()));
      x = mx::fast::rms_norm(last, final_norm_, eps_);
    }
    mx::array lg = lm_head_raw(x);
    if (final_logit_softcap_ > 0.0f) {
      mx::array cap = mx::array(final_logit_softcap_, mx::float32);
      lg = mx::astype(mx::multiply(cap, mx::tanh(mx::divide(mx::astype(lg, mx::float32), cap))), lg.dtype());
    }
    lg = mx::reshape(lg, {all_logits ? S : 1, vocab_});
    if (unpadded_vocab_ > 0 && unpadded_vocab_ < vocab_) {
      mx::array valid = mx::less(mx::arange(vocab_, mx::int32), mx::array(unpadded_vocab_, mx::int32));
      lg = mx::where(valid, lg, mx::array(-1e9f, lg.dtype()));
    }
    if (phase_timing) {
      auto th0 = now_clk();
      mx::eval(lg);
      t_head_ms += ms_diff(th0, now_clk());
    }
    logits.push_back(lg);
  }
  if (out_features && tids && captured.size() == tids->size()) {
    *out_features = mx::stack(captured, 1);  // [S, T, H]
  }
  if (phase_timing) {
    const int tok_n = n == 1 ? h[0].shape(1) : 0;
    static int dec_cnt = 0;
    if (tok_n >= 32 || (++dec_cnt % 16 == 0)) {
      std::fprintf(stderr, "[INKLING_PHASE tok=%d] res_loop=%.1fms attn=%.1fms moe=%.1fms head=%.1fms (total=%.1fms)\n",
                   tok_n, t_res_ms, t_attn_ms, t_moe_ms, t_head_ms,
                   t_res_ms + t_attn_ms + t_moe_ms + t_head_ms);
    }
  }
  if (stats) {
    stats->expert_bytes += ms.expert_bytes;
    stats->route_sync_ms += ms.route_sync_ms;
    stats->read_ms += ms.read_ms;
  }
  return logits;
}

NamedArrays GenericModel::save(const SessionState& st) const {
  NamedArrays out;
  for (size_t v = 0; v < visit_layer_.size(); ++v) {
    const size_t l = static_cast<size_t>(visit_layer_[v]);
    NamedArrays a = mixers_[l]->save(st[v], "layer" + std::to_string(v));
    out.insert(out.end(), a.begin(), a.end());
    for (size_t k = 0; k < st[v].pre.size(); ++k)
      out.emplace_back("layer" + std::to_string(v) + ".pre" + std::to_string(k), mx::contiguous(st[v].pre[k]));
  }
  return out;
}

void GenericModel::load(SessionState& st, const NamedArrays& a, int64_t tokens) const {
  st.assign(visit_layer_.size(), LayerState{});
  for (size_t v = 0; v < visit_layer_.size(); ++v) {
    const size_t l = static_cast<size_t>(visit_layer_[v]);
    mixers_[l]->load(st[v], "layer" + std::to_string(v), a, tokens);
    for (size_t k = 0;; ++k) {
      const std::string key = "layer" + std::to_string(v) + ".pre" + std::to_string(k);
      auto it = std::find_if(a.begin(), a.end(), [&](const auto& kv) { return kv.first == key; });
      if (it == a.end()) break;
      st[v].pre.push_back(it->second);
    }
  }
}

mx::array GenericModel::attn_res_mix(const mx::array& bank_stacked,
                                     const std::optional<mx::array>& partial,
                                     const mx::array& query) const {
  // bank_stacked: [N, B, S, H] BF16 (already RMS-normalized in FP32). 2x lower bandwidth than FP32.
  const int N = bank_stacked.shape(0);
  if (!partial && N == 1) return mx::squeeze(bank_stacked, 0);
  const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(hidden_));
  mx::array q_col = mx::astype(mx::multiply(mx::reshape(mx::astype(query, mx::float32), {hidden_, 1}),
                                            mx::array(inv_sqrt_h, mx::float32)),
                               bank_stacked.dtype());                  // [H, 1] BF16
  mx::array s_bank = mx::matmul(bank_stacked, q_col);                  // [N, B, S, 1] BF16
  if (!partial) {
    mx::array w = mx::astype(mx::softmax(mx::astype(s_bank, mx::float32), 0, /*precise=*/true), bank_stacked.dtype());
    return mx::sum(mx::multiply(w, bank_stacked), 0);
  }
  static auto rn_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array f = mx::astype(in[0], mx::float32);
        mx::array n = mx::multiply(f, mx::rsqrt(mx::mean(mx::square(f), -1, /*keepdims=*/true) + in[1]));
        return {mx::astype(n, mx::bfloat16)};
      },
      true);
  mx::array p_norm = rn_c({*partial, mx::array(eps_, mx::float32)})[0];  // [B, S, H] BF16
  mx::array s_part = mx::expand_dims(mx::matmul(p_norm, q_col), 0);      // [1, B, S, 1] BF16
  mx::array w_all = mx::astype(mx::softmax(mx::astype(mx::concatenate({s_bank, s_part}, 0), mx::float32), 0, /*precise=*/true),
                               bank_stacked.dtype());
  const int B = bank_stacked.shape(1), S = bank_stacked.shape(2);
  mx::array w_bank = mx::slice(w_all, {0, 0, 0, 0}, {N, B, S, 1});
  mx::array w_part = mx::squeeze(mx::slice(w_all, {N, 0, 0, 0}, {N + 1, B, S, 1}), 0);
  return mx::add(mx::sum(mx::multiply(w_bank, bank_stacked), 0), mx::multiply(w_part, p_norm));
}

std::vector<mx::array> GenericModel::attn_res_mix_multi(const mx::array& bank_stacked,
                                                        const std::optional<mx::array>& partial,
                                                        const mx::array& queries_qh) const {
  const int N = bank_stacked.shape(0);
  const int Q = queries_qh.shape(0);
  if (!partial && N == 1) {
    mx::array one = mx::squeeze(bank_stacked, 0);
    return std::vector<mx::array>(static_cast<size_t>(Q), one);
  }
  const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(hidden_));
  mx::array qt = mx::astype(mx::multiply(mx::transpose(mx::astype(queries_qh, mx::float32)),
                                         mx::array(inv_sqrt_h, mx::float32)),
                            bank_stacked.dtype());                     // [H, Q] BF16
  mx::array s_bank = mx::matmul(bank_stacked, qt);                     // [N, B, S, Q] BF16
  std::optional<mx::array> p_norm;
  mx::array weights = mx::array(0.0f);
  if (partial) {
    static auto rn_c = mx::compile(
        [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
          mx::array f = mx::astype(in[0], mx::float32);
          mx::array n = mx::multiply(f, mx::rsqrt(mx::mean(mx::square(f), -1, /*keepdims=*/true) + in[1]));
          return {mx::astype(n, mx::bfloat16)};
        },
        true);
    p_norm = rn_c({*partial, mx::array(eps_, mx::float32)})[0];
    mx::array s_part = mx::expand_dims(mx::matmul(*p_norm, qt), 0);    // [1, B, S, Q] BF16
    weights = mx::astype(mx::softmax(mx::astype(mx::concatenate({s_bank, s_part}, 0), mx::float32), 0, /*precise=*/true),
                         bank_stacked.dtype());
  } else {
    weights = mx::astype(mx::softmax(mx::astype(s_bank, mx::float32), 0, /*precise=*/true), bank_stacked.dtype());
  }
  const int B = bank_stacked.shape(1), S = bank_stacked.shape(2);
  std::vector<mx::array> outs;
  outs.reserve(static_cast<size_t>(Q));
  for (int q = 0; q < Q; ++q) {
    mx::array wb = mx::slice(weights, {0, 0, 0, q}, {N, B, S, q + 1});
    mx::array acc = mx::sum(mx::multiply(wb, bank_stacked), 0);
    if (p_norm) {
      mx::array wp = mx::squeeze(mx::slice(weights, {N, 0, 0, q}, {N + 1, B, S, q + 1}), 0);
      acc = mx::add(acc, mx::multiply(wp, *p_norm));
    }
    outs.push_back(acc);
  }
  return outs;
}

mx::array GenericModel::apply_loop_msg(const mx::array& x, const mx::array& prev_sn, const mx::array& /*prev_vals*/,
                                       const LoopGate& g, int target_idx,
                                       const std::optional<mx::array>& x_rms_cached) const {
  // Compiled message read (prev_sn == prev_vals when normalize_values=True, single read!).
  static auto m_rms_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array mf = mx::astype(in[0], mx::float32);
        mx::array mr = mx::multiply(mf, mx::rsqrt(mx::mean(mx::square(mf), -1, /*keepdims=*/true) + in[1]));
        return {mx::astype(mr, mx::bfloat16)};
      },
      true);
  static auto x_rms_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array xf = mx::astype(in[0], mx::float32);
        mx::array xr = mx::multiply(xf, mx::rsqrt(mx::mean(mx::square(xf), -1, /*keepdims=*/true) + in[1]));
        return {mx::astype(xr, mx::bfloat16)};
      },
      true);
  static auto gate_blend_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        // in: [x_bf16, msg_bf16, gain_pre_bf16, forget_pre_bf16]
        mx::array xf = mx::astype(in[0], mx::float32);
        mx::array gain = mx::tanh(mx::astype(in[2], mx::float32));
        mx::array forget = mx::tanh(mx::astype(in[3], mx::float32));
        mx::array out = mx::add(mx::multiply(mx::subtract(mx::array(1.0f, mx::float32), forget), xf),
                                mx::multiply(gain, mx::astype(in[1], mx::float32)));
        return {mx::astype(out, in[0].dtype())};
      },
      true);
  mx::array mq_col = mx::reshape(mx::astype(g.mq_s[target_idx], prev_sn.dtype()), {hidden_, 1});
  mx::array scores = mx::matmul(prev_sn, mq_col);  // [M, B, S, 1] BF16
  mx::array w = mx::astype(mx::softmax(mx::astype(scores, mx::float32), 0, /*precise=*/true), prev_sn.dtype());
  mx::array msg = mx::sum(mx::multiply(w, prev_sn), 0);  // [B, S, H] BF16
  mx::array m_rms = m_rms_c({msg, mx::array(eps_, mx::float32)})[0];
  mx::array x_rms_bf = x_rms_cached ? *x_rms_cached : x_rms_c({x, mx::array(eps_, mx::float32)})[0];
  mx::array h = mx::add(mx::matmul(x_rms_bf, g.w0[target_idx]), mx::matmul(m_rms, g.w1[target_idx]));
  mx::array gf_pre = mx::matmul(h, g.wgf[target_idx]);
  mx::array gain_pre = mx::slice(gf_pre, {0, 0, 0}, {gf_pre.shape(0), gf_pre.shape(1), hidden_});
  mx::array forget_pre = mx::slice(gf_pre, {0, 0, hidden_}, {gf_pre.shape(0), gf_pre.shape(1), 2 * hidden_});
  return gate_blend_c({x, msg, gain_pre, forget_pre})[0];
}

std::vector<mx::array> GenericModel::apply_loop_msg_qkv_shared(const mx::array& mix, const mx::array& prev_sn,
                                                               const LoopGate& g) const {
  // Batched Q/K/V message read: 1 matmul over prev_sn instead of 3, and 1 x_rms @ w0_qkv instead of 3.
  static auto x_rms_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array xf = mx::astype(in[0], mx::float32);
        mx::array xr = mx::multiply(xf, mx::rsqrt(mx::mean(mx::square(xf), -1, /*keepdims=*/true) + in[1]));
        return {xf, mx::astype(xr, mx::bfloat16)};
      },
      true);
  static auto m_rms_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array mr = mx::multiply(in[0], mx::rsqrt(mx::mean(mx::square(in[0]), -1, /*keepdims=*/true) + in[1]));
        return {mx::astype(mr, mx::bfloat16)};
      },
      true);
  static auto gate_blend3_c = mx::compile(
      [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        mx::array gain = mx::tanh(mx::astype(in[2], mx::float32));
        mx::array forget = mx::tanh(mx::astype(in[3], mx::float32));
        mx::array out = mx::add(mx::multiply(mx::subtract(mx::array(1.0f, mx::float32), forget), in[0]),
                                mx::multiply(gain, mx::astype(in[1], mx::float32)));
        return {mx::astype(out, in[4].dtype())};
      },
      true);
  const int B = mix.shape(0), S = mix.shape(1), BS = B * S, R = loop_gate_rank_;
  mx::array scores3 = mx::matmul(prev_sn, mx::astype(g.mq_qkv_t, prev_sn.dtype()));  // [M, B, S, 3] BF16
  mx::array weights3 = mx::astype(mx::softmax(mx::astype(scores3, mx::float32), 0, /*precise=*/true), prev_sn.dtype());
  mx::array msgs3 = mx::matmul(mx::transpose(weights3, {1, 2, 3, 0}), mx::transpose(prev_sn, {1, 2, 0, 3}));
  mx::array m_rms3 = m_rms_c({mx::astype(msgs3, mx::float32), mx::array(eps_, mx::float32)})[0];  // [B, S, 3, H] BF16
  auto xr = x_rms_c({mix, mx::array(eps_, mx::float32)});
  mx::array hx3 = mx::matmul(xr[1], g.w0_qkv);                    // [B, S, 3*R]
  mx::array m_3bsh = mx::reshape(mx::transpose(m_rms3, {2, 0, 1, 3}), {3, BS, hidden_});
  mx::array hx_3bsr = mx::transpose(mx::reshape(hx3, {BS, 3, R}), {1, 0, 2});
  mx::array h_3bsr = mx::add(hx_3bsr, mx::matmul(m_3bsh, g.w1_qkv));       // [3, BS, R]
  mx::array gf_3 = mx::matmul(h_3bsr, g.wgf_qkv);                          // [3, BS, 2*H]
  mx::array gain3 = mx::slice(gf_3, {0, 0, 0}, {3, BS, hidden_});
  mx::array forget3 = mx::slice(gf_3, {0, 0, hidden_}, {3, BS, 2 * hidden_});
  mx::array msgs_3bsh = mx::reshape(mx::transpose(msgs3, {2, 0, 1, 3}), {3, BS, hidden_});
  mx::array xf_1bsh = mx::reshape(xr[0], {1, BS, hidden_});
  mx::array out3 = gate_blend3_c({xf_1bsh, msgs_3bsh, gain3, forget3, mix})[0];  // [3, BS, H]
  return {mx::reshape(mx::slice(out3, {0, 0, 0}, {1, BS, hidden_}), {B, S, hidden_}),
          mx::reshape(mx::slice(out3, {1, 0, 0}, {2, BS, hidden_}), {B, S, hidden_}),
          mx::reshape(mx::slice(out3, {2, 0, 0}, {3, BS, hidden_}), {B, S, hidden_})};
}

// Qwen4ExpGatedResidual.__call__ (use_combine=True): grouped RMSNorm over the hc streams
// (patch_rmsnorm gain convention), low-rank sigmoid mix -> one input, 2*sigmoid injection
// weights. Same op sequence as engine::gated_residual (custom Qwen runtime).
std::pair<mx::array, mx::array> GenericModel::hyper_mix(const mx::array& h, const HyperRes& r) const {
  mx::Shape gs(h.shape().begin(), h.shape().end() - 1);
  gs.push_back(hc_);
  gs.push_back(hidden_);
  mx::array yg = mx::astype(mx::reshape(h, gs), mx::float32);
  yg = yg * mx::rsqrt(mx::mean(mx::square(yg), -1, /*keepdims=*/true) + eps_);
  mx::array gain = mx::astype(mx::reshape(r.norm, mx::Shape{hc_, hidden_}), mx::float32);
  if (!hc_one_centered_) gain = 1.0f + gain;
  yg = yg * gain;
  mx::array normed = mx::astype(mx::reshape(yg, h.shape()), h.dtype());
  // x / self.hc_count with a Python int keeps bf16: divide by an int32 array, not a float.
  mx::array hc_arr = mx::array(hc_);
  mx::array low = r.down(normed) / hc_arr;
  mx::array mix = mx::sigmoid(r.up(engine::hc_silu(low)));
  mx::array mixed = mx::mean(mx::reshape(mix, gs) * mx::reshape(normed, gs), -2);
  mx::array two = mx::array(2);
  mx::array w = two * mx::sigmoid((*r.inject)(normed) / hc_arr);
  return {mixed, w};
}

mx::array GenericModel::hyper_final(const mx::array& h) const {
  const HyperRes& r = *final_hc_;
  mx::Shape gs(h.shape().begin(), h.shape().end() - 1);
  gs.push_back(hc_);
  gs.push_back(hidden_);
  mx::array yg = mx::astype(mx::reshape(h, gs), mx::float32);
  yg = yg * mx::rsqrt(mx::mean(mx::square(yg), -1, true) + eps_);
  mx::array gain = mx::astype(mx::reshape(r.norm, mx::Shape{hc_, hidden_}), mx::float32);
  if (!hc_one_centered_) gain = 1.0f + gain;
  yg = yg * gain;
  mx::array normed = mx::astype(mx::reshape(yg, h.shape()), h.dtype());
  mx::array hc_arr = mx::array(hc_);
  mx::array low = r.down(normed) / hc_arr;
  mx::array mix = mx::sigmoid(r.up(engine::hc_silu(low)));
  return mx::mean(mx::reshape(mix, gs) * mx::reshape(normed, gs), -2);
}

}  // namespace veystrix::engine::generic
