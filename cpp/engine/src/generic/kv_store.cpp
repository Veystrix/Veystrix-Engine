#include "generic/kv_store.hpp"

#include <mlx/compile.h>
#include <mlx/fast.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace veystrix::engine::generic {

namespace {

constexpr int kStep = 256;  // capacity growth, same as mlx_lm KVCache.step

bool valid_bits(int b) { return b == 16 || b == 8 || b == 4 || b == 3 || b == 2; }

// The mlx_lm weak-scalar convention: a Python float multiplied into a bf16
// array is first converted to float32 then to the array dtype.
mx::array weak(double v, mx::Dtype dt) { return mx::array(static_cast<float>(v), dt); }

mx::array rows(const mx::array& a, int64_t from, int64_t to) {
  mx::Shape start(a.ndim(), 0);
  mx::Shape stop(a.shape());
  start[2] = static_cast<int>(from);
  stop[2] = static_cast<int>(to);
  return mx::slice(a, start, stop);
}

std::vector<mx::array> encode(const mx::array& x, int bits, int gs) {
  if (bits == 16) return {x};
  return mx::quantize(x, gs, bits, "affine");
}

// Zero buffers shaped like `enc` but with `cap` rows on axis 2.
std::vector<mx::array> zeros_like_rows(const std::vector<mx::array>& enc, int64_t cap) {
  std::vector<mx::array> out;
  for (const auto& e : enc) {
    mx::Shape s(e.shape());
    s[2] = static_cast<int>(cap);
    out.push_back(mx::zeros(s, e.dtype()));
  }
  return out;
}

}  // namespace

KVPolicy KVPolicy::from_mode(const std::string& mode, int64_t window, int prefix,
                             int hist_k_bits, int hist_v_bits) {
  KVPolicy p;
  p.name = mode.empty() ? "bf16" : mode;
  const std::string& m = p.name;
  if (m == "bf16") {
  } else if (m == "int8") {
    p.recent_tokens = 0; p.hist_k_bits = 8; p.hist_v_bits = 8;
  } else if (m == "k8v4") {
    p.recent_tokens = 0; p.hist_k_bits = 8; p.hist_v_bits = 4;
  } else if (m == "int4") {
    p.recent_tokens = 0; p.hist_k_bits = 4; p.hist_v_bits = 4;
  } else if (m == "k4v2") {
    p.recent_tokens = 0; p.hist_k_bits = 4; p.hist_v_bits = 2;
  } else if (m == "k3v2") {
    p.recent_tokens = 0; p.hist_k_bits = 3; p.hist_v_bits = 2;
  } else if (m == "int2") {
    p.recent_tokens = 0; p.hist_k_bits = 2; p.hist_v_bits = 2;
  } else if (m == "tiered") {  // bf16 recent window + INT8 history
    p.recent_tokens = 8192; p.recent_bits = 16; p.hist_k_bits = 8; p.hist_v_bits = 8;
  } else if (m == "tiered4") {  // flashnext shape: INT8 recent + INT4 history
    p.recent_tokens = 8192; p.recent_bits = 8; p.hist_k_bits = 4; p.hist_v_bits = 4;
  } else if (m == "tiered4+" || m == "tiered4p") {  // + protected INT8 prefix
    p.name = "tiered4+";
    p.prefix_tokens = 1024; p.prefix_bits = 8;
    p.recent_tokens = 8192; p.recent_bits = 8; p.hist_k_bits = 4; p.hist_v_bits = 4;
  } else if (m == "tiered-k8v4") {
    p.prefix_tokens = 1024; p.prefix_bits = 8;
    p.recent_tokens = 8192; p.recent_bits = 8; p.hist_k_bits = 8; p.hist_v_bits = 4;
  } else if (m == "sparse-paged") {
    p.prefix_tokens = 1024; p.prefix_bits = 16;
    p.recent_tokens = 8192; p.recent_bits = 16; p.hist_k_bits = 4; p.hist_v_bits = 4;
    p.sparse_paged = true;
  } else if (m == "sparse-paged-k4v2") {
    p.prefix_tokens = 1024; p.prefix_bits = 16;
    p.recent_tokens = 8192; p.recent_bits = 16; p.hist_k_bits = 4; p.hist_v_bits = 2;
    p.sparse_paged = true;
  } else if (m == "sparse-paged-k2v2") {
    p.prefix_tokens = 1024; p.prefix_bits = 16;
    p.recent_tokens = 8192; p.recent_bits = 16; p.hist_k_bits = 2; p.hist_v_bits = 2;
    p.sparse_paged = true;
  } else {
    throw std::runtime_error("kv: unknown mode '" + m +
                             "' (bf16|int8|k8v4|int4|k4v2|k3v2|int2|tiered|tiered4|tiered4+|tiered-k8v4|sparse-paged|sparse-paged-k4v2|sparse-paged-k2v2)");
  }
  if (window >= -1) p.recent_tokens = window;
  if (prefix >= 0) p.prefix_tokens = prefix;
  if (hist_k_bits > 0) p.hist_k_bits = hist_k_bits;
  if (hist_v_bits > 0) p.hist_v_bits = hist_v_bits;
  if (!valid_bits(p.prefix_bits) || !valid_bits(p.recent_bits) || !valid_bits(p.hist_k_bits) ||
      !valid_bits(p.hist_v_bits))
    throw std::runtime_error("kv: bits must be 16, 8, 4, 3 or 2");
  if (p.prefix_tokens % p.page != 0)
    throw std::runtime_error("kv: protected prefix must be a multiple of the page (256)");
  if (p.prefix_tokens > 0 && p.prefix_bits == 16 && p.recent_tokens < 0)
    p.prefix_tokens = 0;  // everything bf16 anyway
  return p;
}

std::string KVPolicy::describe() const {
  std::ostringstream o;
  o << name << ":p" << prefix_tokens << "b" << prefix_bits << ":w" << recent_tokens << "b"
    << recent_bits << ":h" << hist_k_bits << "/" << hist_v_bits << ":g" << group_size;
  if (sliding_window > 0) o << ":sw" << sliding_window;
  if (sparse_paged) o << ":sp" << sparse_top_pages;
  return o.str();
}

static double bpv(int bits, int gs) {
  if (bits == 16) return 2.0;
  return bits / 8.0 + 4.0 / gs;  // bf16 scale + bias per group
}

double KVPolicy::bytes_per_value(int64_t tokens) const {
  if (tokens <= 0) return 2.0;
  const double t = static_cast<double>(tokens);
  const double pre = std::min<double>(t, prefix_tokens);
  double rest = t - pre;
  if (recent_tokens < 0) return (pre * bpv(prefix_bits, group_size) + rest * bpv(recent_bits, group_size)) / t;
  const double rec = std::min<double>(rest, recent_tokens == 0 ? 0 : recent_tokens + page);
  const double mid = rest;  // middle copies are written for every non-prefix token
  // K and V each use one value per element; report the per-value mean.
  const double hist = 0.5 * (bpv(hist_k_bits, group_size) + bpv(hist_v_bits, group_size));
  return (pre * bpv(prefix_bits, group_size) + rec * bpv(recent_bits, group_size) + mid * hist) / t;
}

void KVStore::append_seg(KVSeg& seg, const mx::array& k, const mx::array& v, int64_t first_pos) {
  const int gs = policy_.group_size;
  const int64_t S = k.shape(2);
  std::vector<mx::array> ek = encode(k, seg.kbits, gs);
  std::vector<mx::array> ev = encode(v, seg.vbits, gs);
  if (seg.cap == 0 || seg.len == 0) seg.start = first_pos;
  if (first_pos != seg.start + seg.len)
    throw std::runtime_error("kv: non-contiguous segment append");
  const int64_t prev = seg.len;
  if (seg.k.empty() || prev + S > seg.cap) {
    const int64_t n_steps = (kStep + S - 1) / kStep;
    const int64_t grow = n_steps * kStep;
    std::vector<mx::array> nk = zeros_like_rows(ek, grow);
    std::vector<mx::array> nv = zeros_like_rows(ev, grow);
    if (!seg.k.empty()) {
      for (size_t i = 0; i < seg.k.size(); ++i) {
        mx::array a = (prev % kStep != 0) ? rows(seg.k[i], 0, prev) : seg.k[i];
        seg.k[i] = mx::concatenate({a, nk[i]}, 2);
      }
      for (size_t i = 0; i < seg.v.size(); ++i) {
        mx::array a = (prev % kStep != 0) ? rows(seg.v[i], 0, prev) : seg.v[i];
        seg.v[i] = mx::concatenate({a, nv[i]}, 2);
      }
      seg.cap = (prev % kStep != 0 ? prev : seg.cap) + grow;
    } else {
      seg.k = std::move(nk);
      seg.v = std::move(nv);
      seg.cap = grow;
    }
  }
  auto put = [&](std::vector<mx::array>& buf, const std::vector<mx::array>& enc) {
    for (size_t i = 0; i < buf.size(); ++i) {
      mx::Shape start(buf[i].ndim(), 0);
      mx::Shape stop(buf[i].shape());
      start[2] = static_cast<int>(prev);
      stop[2] = static_cast<int>(prev + S);
      buf[i] = mx::slice_update(buf[i], enc[i], start, stop);
    }
  };
  put(seg.k, ek);
  put(seg.v, ev);
  seg.len = prev + S;
}

int64_t KVStore::history_end() const {
  const int64_t P = policy_.prefix_tokens;
  const int64_t W = policy_.recent_tokens;
  if (W < 0) return std::min<int64_t>(n_, P);
  if (W == 0) return n_;
  const int64_t page = policy_.page;
  int64_t h = n_ > W ? ((n_ - W) / page) * page : 0;
  return std::max<int64_t>(std::min<int64_t>(n_, P), h);
}

void KVStore::compact_recent(int64_t keep_from) {
  // Drop recent rows before `keep_from` once enough are dead (bounded copy).
  KVSeg& r = recent_;
  if (r.k.empty()) return;
  const int64_t dead = keep_from - r.start;
  if (dead < 4 * kStep) return;
  // In sparse-paged mode with BF16 recent window, quantize aged-out pages to INT4 middle_
  // lazily here (once per 1024 aged tokens, directly from exact BF16 rows) instead of on
  // every single decode step.
  const bool defer_middle = policy_.sparse_paged && policy_.sliding_window == 0 && r.kbits == 16 && r.vbits == 16;
  if (defer_middle && keep_from > r.start) {
    const int64_t mid_written = middle_.cap > 0 ? (middle_.start + middle_.len) : policy_.prefix_tokens;
    if (keep_from > mid_written) {
      const int64_t src_lo = std::max<int64_t>(0, mid_written - r.start);
      const int64_t src_hi = keep_from - r.start;
      if (src_hi > src_lo) {
        mx::array k_age = rows(r.k[0], src_lo, src_hi);
        mx::array v_age = rows(r.v[0], src_lo, src_hi);
        append_seg(middle_, k_age, v_age, r.start + src_lo);
      }
    }
  }
  const int64_t live = r.len - dead;
  const int64_t cap = ((live + kStep - 1) / kStep + 1) * kStep;
  auto move = [&](std::vector<mx::array>& buf) {
    for (auto& a : buf) {
      mx::array keep = mx::contiguous(rows(a, dead, r.len));
      mx::Shape s(a.shape());
      s[2] = static_cast<int>(cap - live);
      a = mx::concatenate({keep, mx::zeros(s, a.dtype())}, 2);
    }
  };
  move(r.k);
  move(r.v);
  r.start = keep_from;
  r.len = live;
  r.cap = cap;
}

bool KVStore::use_dense_recent() const {
  return policy_.sparse_paged && policy_.sliding_window == 0 && policy_.recent_tokens > 0 &&
         policy_.prefix_tokens > 0 && policy_.prefix_bits == 16 && policy_.recent_bits == 16;
}

void KVStore::split_dense_prefix() {
  if (!dense_recent_) return;
  const int64_t P = policy_.prefix_tokens;
  const int64_t W = policy_.recent_tokens;
  if (n_ <= P + W) return;
  if (recent_.len >= P) {
    append_seg(prefix_, rows(recent_.k[0], 0, P), rows(recent_.v[0], 0, P), 0);
  }
  const int64_t keep_from = std::max<int64_t>(P, ((n_ - W) / policy_.page) * policy_.page);
  compact_recent(keep_from);
  dense_recent_ = false;
}

void KVStore::compact_if_needed() {
  if (dense_recent_) {
    if (n_ <= policy_.prefix_tokens + policy_.recent_tokens) return;
    split_dense_prefix();
  }
  const int64_t SW = policy_.sliding_window;
  const int64_t P = SW > 0 ? 0 : policy_.prefix_tokens;
  const int64_t W = SW > 0 ? SW : policy_.recent_tokens;
  const bool use_middle = SW == 0 && W >= 0;
  const bool use_recent = SW > 0 || W != 0;
  if (!use_recent || !(use_middle || SW > 0)) return;
  const int64_t H_compact = SW > 0 ? std::max<int64_t>(0, n_ - SW) : history_end();
  compact_recent(std::max(H_compact, P));
}

int attention_query_block(int L, int kL, int Hq) {
  // VEYSTRIX_KV_SCORE_TILE overrides the tile (tests force small blocks; port_check reads the same variable).
  static const int64_t tile = [] {
    const char* e = std::getenv("VEYSTRIX_KV_SCORE_TILE");
    const long long v = e ? std::atoll(e) : 0;
    return v > 0 ? static_cast<int64_t>(v) : kScoreTile;
  }();
  const int64_t per_row = static_cast<int64_t>(Hq) * std::max(kL, 1);
  int64_t qb = tile / per_row;
  if (qb >= L) return L;
  qb = std::max<int64_t>(16, (qb / 16) * 16);
  return static_cast<int>(std::min<int64_t>(qb, L));
}

std::vector<std::pair<int, int>> query_spans(int L, int qb) {
  std::vector<std::pair<int, int>> out;
  qb = std::max(qb, 1);
  for (int a = 0; a < L;) {
    int b = std::min(L, a + qb);
    if (L - b > 0 && L - b < 16) b = L;
    out.emplace_back(a, b);
    a = b;
  }
  return out;
}

mx::array KVStore::update_and_attend(const mx::array& queries, const mx::array& keys,
                                     const mx::array& values, float scale, bool causal_single,
                                     const std::optional<mx::array>& mask_arr) {
  if (!init_) {
    prefix_.kbits = prefix_.vbits = policy_.prefix_bits;
    middle_.kbits = policy_.hist_k_bits;
    middle_.vbits = policy_.hist_v_bits;
    recent_.kbits = recent_.vbits = policy_.recent_bits;
    dense_recent_ = use_dense_recent();
    init_ = true;
  }
  const int64_t S = keys.shape(2);
  const int64_t n0 = n_;
  const int64_t SW = policy_.sliding_window;
  const int64_t P = SW > 0 ? 0 : policy_.prefix_tokens;
  const int64_t W = SW > 0 ? SW : policy_.recent_tokens;
  const bool use_middle = SW == 0 && W >= 0;
  const bool use_recent = SW > 0 || W != 0;

  // ---- write: every token once into each format it will be read in ----
  const int64_t P_store = dense_recent_ ? 0 : P;
  const int64_t pre_end = std::clamp<int64_t>(P_store - n0, 0, S);  // rows [0, pre_end) -> prefix
  if (pre_end > 0)
    append_seg(prefix_, rows(keys, 0, pre_end), rows(values, 0, pre_end), n0);
  if (pre_end < S) {
    mx::array k = pre_end > 0 ? rows(keys, pre_end, S) : keys;
    mx::array v = pre_end > 0 ? rows(values, pre_end, S) : values;
    if (use_middle) append_seg(middle_, k, v, n0 + pre_end);
    if (use_recent) append_seg(recent_, k, v, n0 + pre_end);
  }
  n_ = n0 + S;
  if (dense_recent_) split_dense_prefix();

  // ---- regions for the read ----
  const int64_t H = SW > 0 ? std::max<int64_t>(0, n_ - SW) : history_end();
  const int64_t pre_n = std::min<int64_t>(n_, P);
  struct Part { const KVSeg* seg; int64_t from, to; };  // buffer rows
  std::vector<Part> parts;
  if (dense_recent_) {
    parts.push_back({&recent_, 0, n_});
  } else {
    if (pre_n > 0) parts.push_back({&prefix_, 0, pre_n});
    if (use_middle && H > P) parts.push_back({&middle_, 0, H - P});
    if (use_recent && n_ > std::max(H, P)) {
      const int64_t from = std::max(H, P);
      parts.push_back({&recent_, from - recent_.start, n_ - recent_.start});
    }
  }
  if (parts.empty()) throw std::runtime_error("kv: empty attention");

  const int gs = policy_.group_size;
  const int L = queries.shape(2);
  const std::string mask = (L > 1 || causal_single) ? "causal" : "";

  auto view = [&](const std::vector<mx::array>& buf, int64_t a, int64_t b) {
    std::vector<mx::array> out;
    for (const auto& x : buf) out.push_back(rows(x, a, b));
    return out;
  };

  mx::array result = [&]() -> mx::array {
    if (parts.size() == 1 && parts[0].seg->kbits == 16 && parts[0].seg->vbits == 16) {
      // mlx_lm KVCache.update_and_fetch + mx.fast.scaled_dot_product_attention.
      const auto k = view(parts[0].seg->k, parts[0].from, parts[0].to);
      const auto v = view(parts[0].seg->v, parts[0].from, parts[0].to);
      if (!mask_arr) return mx::fast::scaled_dot_product_attention(queries, k[0], v[0], scale, mask);
      // Array mask (QSA). One call over all rows, as mlx_vlm makes it, unless the unfused score tile would
      // exceed kScoreTile; then query rows go in blocks (rows are independent) to bound memory.
      const int kLm = static_cast<int>(k[0].shape(2));
      const int qbm = attention_query_block(L, kLm, queries.shape(1));
      if (qbm >= L) return mx::fast::scaled_dot_product_attention(queries, k[0], v[0], scale, "", *mask_arr);
      std::vector<mx::array> outs;
      for (const auto& [a, b] : query_spans(L, qbm)) {
        mx::array qa = mx::slice(queries, {0, 0, a, 0}, {queries.shape(0), queries.shape(1), b, queries.shape(3)});
        mx::array ma = mx::slice(*mask_arr, {0, 0, a, 0}, {mask_arr->shape(0), mask_arr->shape(1), b, kLm});
        if (!outs.empty()) mx::eval(outs.back());
        outs.push_back(mx::fast::scaled_dot_product_attention(qa, k[0], v[0], scale, "", ma));
      }
      return mx::concatenate(outs, 2);
    }
    // Quantized / multi-region path: mirrors base.quantized_scaled_dot_product_attention.
    const int B = queries.shape(0), Hq = queries.shape(1), D = queries.shape(3);
    const int Hkv = parts[0].seg->k[0].shape(1);
    const int rep = Hq / Hkv;
    mx::array q = mx::multiply(queries, weak(scale, queries.dtype()));
    if (rep > 1) q = mx::reshape(q, {B, Hkv, rep, L, D});
    std::vector<int> lens;
    std::vector<std::vector<mx::array>> kparts, vparts;
    for (const Part& p : parts) {
      auto k = view(p.seg->k, p.from, p.to);
      auto v = view(p.seg->v, p.from, p.to);
      if (rep > 1) {
        for (auto& a : k) a = mx::expand_dims(a, -3);
        for (auto& a : v) a = mx::expand_dims(a, -3);
      }
      lens.push_back(static_cast<int>(p.to - p.from));
      kparts.push_back(std::move(k));
      vparts.push_back(std::move(v));
    }
    // Scores are materialised here (no fused kernel over packed regions): Hq x L x kL values per layer. A
    // 2048-token prefill chunk at 128K context would be 16 GiB, so query rows go in blocks whose score tile
    // stays <= kScoreTile elements. Each row's math is unchanged (rows are independent); port_check's
    // RefTieredKV uses the same blocks.
    int kL = 0;
    for (int n : lens) kL += n;
    const int qb = attention_query_block(L, kL, Hq);
    auto attend_rows = [&](int a, int b) -> mx::array {
      mx::array qa = q;
      if (b - a != L) {
        mx::Shape s0(q.ndim(), 0), s1(q.shape());
        s0[q.ndim() - 2] = a; s1[q.ndim() - 2] = b;
        qa = mx::slice(q, s0, s1);
      }
      std::vector<mx::array> blk;
      for (size_t i = 0; i < parts.size(); ++i) {
        const auto& k = kparts[i];
        if (parts[i].seg->kbits == 16)
          blk.push_back(mx::matmul(qa, mx::swapaxes(k[0], -1, -2)));
        else
          blk.push_back(mx::quantized_matmul(qa, k[0], k[1], k[2], true, gs, parts[i].seg->kbits, "affine"));
      }
      mx::array sc = blk.size() == 1 ? blk[0] : mx::concatenate(blk, -1);
      if (mask_arr) {
        // [B, 1, L, kL] -> rows [a, b); with GQA repeat the scores are [B, Hkv, rep, L, kL].
        mx::array m = mx::slice(*mask_arr, {0, 0, a, 0}, {mask_arr->shape(0), 1, b, kL});
        if (rep > 1) m = mx::expand_dims(m, 2);
        sc = mx::where(m, sc, mx::array(mx::finfo(sc.dtype()).min, sc.dtype()));
      } else if (L > 1) {
        mx::array qi = mx::arange(kL - L + a, kL - L + b);
        mx::array ki = mx::arange(kL);
        mx::array m = mx::greater_equal(mx::expand_dims(qi, 1), mx::expand_dims(ki, 0));
        sc = mx::where(m, sc, mx::array(mx::finfo(sc.dtype()).min, sc.dtype()));
      }
      sc = mx::softmax(sc, -1, /*precise=*/true);
      std::optional<mx::array> out;
      int at = 0;
      for (size_t i = 0; i < parts.size(); ++i) {
        mx::array pi = parts.size() == 1 ? sc : mx::slice(sc, [&] {
          mx::Shape s(sc.ndim(), 0); s.back() = at; return s; }(), [&] {
          mx::Shape s(sc.shape()); s.back() = at + lens[i]; return s; }());
        at += lens[i];
        const auto& v = vparts[i];
        mx::array o = parts[i].seg->vbits == 16
                          ? mx::matmul(pi, v[0])
                          : mx::quantized_matmul(pi, v[0], v[1], v[2], false, gs, parts[i].seg->vbits,
                                                 "affine");
        out = out ? mx::add(*out, o) : o;
      }
      return *out;
    };
    const auto spans = query_spans(L, qb);
    mx::array o = attend_rows(spans[0].first, spans[0].second);
    if (spans.size() > 1) {
      std::vector<mx::array> outs{o};
      for (size_t i = 1; i < spans.size(); ++i) {
        mx::eval(outs.back());  // bound the live score tiles to one block
        outs.push_back(attend_rows(spans[i].first, spans[i].second));
      }
      o = mx::concatenate(outs, -2);
    }
    if (rep > 1) o = mx::reshape(o, {B, Hq, L, D});
    return o;
  }();

  if (use_recent && (use_middle || SW > 0)) compact_recent(std::max(H, P));
  return result;
}

void KVStore::update_page_index(const mx::array& k, int64_t first_pos) {
  if (!policy_.sparse_paged || policy_.sliding_window > 0) return;
  const int64_t pg = policy_.page;
  const int64_t S = k.shape(2);
  if (S <= 0 || pg <= 0) return;
  // Only index COMPLETED pages (p < (first_pos + S) / pg). Cold sparse selection only ever
  // reads completed pages (< H / pg, at least W=8192 tokens old), so skipping partial tail
  // pages eliminates per-token min/max graph chaining and truncate() recomputation.
  const int64_t new_n = first_pos + S;
  const int64_t complete_pages = new_n / pg;
  const int64_t P = policy_.prefix_tokens;
  for (int64_t p = static_cast<int64_t>(page_k_min_.size()); p < complete_pages; ++p) {
    const int64_t p_lo = p * pg;
    const int64_t p_hi = (p + 1) * pg;
    mx::array sub = mx::array(0.0f);
    if (p_lo >= first_pos && p_hi <= new_n) {
      sub = rows(k, p_lo - first_pos, p_hi - first_pos);
    } else if (p_hi <= P && !prefix_.k.empty() && prefix_.kbits == 16 && p_hi <= prefix_.start + prefix_.len) {
      sub = rows(prefix_.k[0], p_lo - prefix_.start, p_hi - prefix_.start);
    } else if (!recent_.k.empty() && recent_.kbits == 16 && p_lo >= recent_.start && p_hi <= recent_.start + recent_.len) {
      sub = rows(recent_.k[0], p_lo - recent_.start, p_hi - recent_.start);
    } else {
      break;
    }
    page_k_min_.push_back(mx::min(sub, 2, /*keepdims=*/true));
    page_k_max_.push_back(mx::max(sub, 2, /*keepdims=*/true));
    indexed_to_ = p_hi;
  }
}

mx::array KVStore::update_and_attend_rel_sink(const mx::array& queries, const mx::array& keys,
                                              const mx::array& values, const mx::array& rel_logits,
                                              const mx::array& sink, float scale,
                                              const std::optional<mx::array>& rp_t_ed,
                                              bool defer_compact) {
  if (!init_) {
    prefix_.kbits = prefix_.vbits = policy_.prefix_bits;
    middle_.kbits = policy_.hist_k_bits;
    middle_.vbits = policy_.hist_v_bits;
    recent_.kbits = recent_.vbits = policy_.recent_bits;
    dense_recent_ = use_dense_recent();
    init_ = true;
  }
  const int64_t S = keys.shape(2);
  const int64_t n0 = n_;
  const int64_t SW = policy_.sliding_window;
  const int64_t P = SW > 0 ? 0 : policy_.prefix_tokens;
  const int64_t W = SW > 0 ? SW : policy_.recent_tokens;
  const bool use_middle = SW == 0 && W >= 0;
  const bool use_recent = SW > 0 || W != 0;
  const bool defer_middle = policy_.sparse_paged && SW == 0 && recent_.kbits == 16 && recent_.vbits == 16;

  const int64_t P_store = dense_recent_ ? 0 : P;
  const int64_t pre_end = std::clamp<int64_t>(P_store - n0, 0, S);
  if (pre_end > 0)
    append_seg(prefix_, rows(keys, 0, pre_end), rows(values, 0, pre_end), n0);
  if (pre_end < S) {
    mx::array k = pre_end > 0 ? rows(keys, pre_end, S) : keys;
    mx::array v = pre_end > 0 ? rows(values, pre_end, S) : values;
    if (use_middle && !defer_middle) append_seg(middle_, k, v, n0 + pre_end);
    if (use_recent) append_seg(recent_, k, v, n0 + pre_end);
  }
  if (policy_.sparse_paged && SW == 0) update_page_index(keys, n0);
  n_ = n0 + S;
  if (dense_recent_) split_dense_prefix();

  const int B = queries.shape(0), Hq = queries.shape(1), L = queries.shape(2), D = queries.shape(3);
  const int64_t H_compact = SW > 0 ? std::max<int64_t>(0, n_ - SW) : history_end();
  // During sliding-window prefill (L > SW), queries at [n_-L, n_) need keys back to n_-L-SW+1.
  const int64_t H = SW > 0 ? std::max<int64_t>(recent_.start, n_ - L - SW + 1) : H_compact;
  const int64_t pre_n = std::min<int64_t>(n_, P);
  const int Hkv = keys.shape(1);
  const int rep = Hq / Hkv;
  const int gs = policy_.group_size;
  const int E = rp_t_ed ? rp_t_ed->shape(0) : rel_logits.shape(3);

  auto view = [&](const std::vector<mx::array>& buf, int64_t a, int64_t b) {
    std::vector<mx::array> out;
    out.reserve(buf.size());
    for (const auto& x : buf) out.push_back(rows(x, a, b));
    return out;
  };

  struct Part {
    int kbits;
    int vbits;
    std::vector<mx::array> k;
    std::vector<mx::array> v;
    int len;
  };
  std::vector<Part> parts;
  std::vector<mx::array> abs_pos_parts;

  auto push_seg = [&](const KVSeg& seg, int64_t from, int64_t to, int64_t abs_start) {
    if (to <= from) return;
    Part p;
    p.kbits = seg.kbits;
    p.vbits = seg.vbits;
    p.k = view(seg.k, from, to);
    p.v = view(seg.v, from, to);
    if (rep > 1) {
      for (auto& a : p.k) a = mx::expand_dims(a, -3);
      for (auto& a : p.v) a = mx::expand_dims(a, -3);
    }
    p.len = static_cast<int>(to - from);
    parts.push_back(std::move(p));
    abs_pos_parts.push_back(mx::arange(static_cast<int>(abs_start),
                                       static_cast<int>(abs_start + (to - from)), mx::int32));
  };

  if (dense_recent_) {
    // One contiguous BF16 region [0, n): byte-for-byte the same SDPA input as
    // the [prefix, recent] layout the two-segment path used to concatenate.
    push_seg(recent_, 0, n_, 0);
  } else {
  if (pre_n > 0) push_seg(prefix_, 0, pre_n, 0);

  // Cold history: full dense scan when small, or Quest-style top-K page selection in sparse-paged mode.
  const int64_t H_read = defer_middle ? std::min<int64_t>(H, recent_.cap > 0 ? recent_.start : P) : H;
  if (use_middle && H_read > P) {
    const int64_t pg = policy_.page;
    const int64_t p_start = P / pg;
    const int64_t p_end = H_read / pg;
    const int64_t n_cold = p_end - p_start;
    if (policy_.sparse_paged && n_cold > policy_.sparse_top_pages &&
        static_cast<int64_t>(page_k_min_.size()) >= p_end) {
      std::vector<mx::array> cmin, cmax;
      cmin.reserve(static_cast<size_t>(n_cold));
      cmax.reserve(static_cast<size_t>(n_cold));
      for (int64_t p = p_start; p < p_end; ++p) {
        cmin.push_back(page_k_min_[p]);
        cmax.push_back(page_k_max_[p]);
      }
      mx::array kmin_all = mx::concatenate(cmin, 2);  // [B, Hkv, n_cold, D]
      mx::array kmax_all = mx::concatenate(cmax, 2);  // [B, Hkv, n_cold, D]
      mx::array q_sum = L == 1 ? queries : mx::mean(queries, 2, /*keepdims=*/true);  // [B, Hq, 1, D]
      if (rep > 1) {
        q_sum = mx::reshape(q_sum, {B, Hkv, rep, 1, D});
        kmin_all = mx::expand_dims(kmin_all, 2);
        kmax_all = mx::expand_dims(kmax_all, 2);
      }
      mx::array ub = mx::sum(mx::maximum(mx::multiply(q_sum, kmin_all),
                                         mx::multiply(q_sum, kmax_all)),
                             -1);  // [B, Hkv, (rep,) n_cold]
      std::vector<int> red_axes;
      for (int ax = 0; ax < ub.ndim() - 1; ++ax) red_axes.push_back(ax);
      mx::array pscores = mx::max(ub, red_axes);  // [n_cold]
      const int topk = policy_.sparse_top_pages;
      mx::array top_idx = mx::argpartition(pscores, static_cast<int>(n_cold - topk), 0);
      top_idx = mx::slice(top_idx, {static_cast<int>(n_cold - topk)}, {static_cast<int>(n_cold)});
      top_idx = mx::sort(top_idx, 0);
      mx::eval(top_idx);
      const uint32_t* pids = top_idx.data<uint32_t>();
      for (int i = 0; i < topk; ++i) {
        const int64_t cold_p = static_cast<int64_t>(pids[i]);
        const int64_t mid_lo = cold_p * pg;
        const int64_t mid_hi = std::min<int64_t>(mid_lo + pg, H_read - P);
        push_seg(middle_, mid_lo, mid_hi, P + mid_lo);
      }
    } else {
      push_seg(middle_, 0, H_read - P, P);
    }
  }

  if (use_recent && n_ > std::max(H_read, P)) {
    const int64_t abs_from = std::max(H_read, P);
    push_seg(recent_, abs_from - recent_.start, n_ - recent_.start, abs_from);
  }
  }
  if (parts.empty()) throw std::runtime_error("kv: empty rel_sink attention");

  mx::array k_abs_pos = abs_pos_parts.size() == 1 ? abs_pos_parts[0]
                                                  : mx::concatenate(abs_pos_parts, 0);
  int kL = 0;
  for (const auto& p : parts) kL += p.len;

  bool all_bf16 = !parts.empty();
  for (const auto& p : parts)
    if (p.kbits != 16 || p.vbits != 16) all_bf16 = false;
  // Compute r @ rp_t^T per query tile inside attend_rows (12-24 MiB vs 96 MiB upfront) for ALL KV modes.
  const bool tile_rel_matmul = (L > 1 && rp_t_ed.has_value());
  mx::array rel_mat = (!tile_rel_matmul && rp_t_ed) ? mx::matmul(rel_logits, mx::transpose(*rp_t_ed)) : rel_logits;
  mx::array q = mx::multiply(queries, weak(scale, queries.dtype()));
  mx::array rel_q = rel_mat;
  mx::array sink_q = mx::expand_dims(sink, -1);  // [B, Hq, L, 1]
  if (rep > 1) {
    q = mx::reshape(q, {B, Hkv, rep, L, D});
    if (!tile_rel_matmul) rel_q = mx::reshape(rel_q, {B, Hkv, rep, L, E});
    sink_q = mx::reshape(sink_q, {B, Hkv, rep, L, 1});
  }

  const int qb_kL = (SW > 0 && L > 1) ? std::min(kL, static_cast<int>(SW) + 512) : kL;
  int qb = attention_query_block(L, qb_kL, Hq);
  const int max_qb = all_bf16 ? 512 : 256;
  if (L > max_qb) qb = std::min(qb, max_qb);
  auto attend_rows = [&](int a, int b) -> mx::array {
    const int curL = b - a;
    mx::array qa = q;
    mx::array rela = rel_q;
    mx::array sinka = sink_q;
    if (curL != L) {
      auto slice_q = [&](const mx::array& t) {
        mx::Shape s0(t.ndim(), 0), s1(t.shape());
        s0[t.ndim() - 2] = a;
        s1[t.ndim() - 2] = b;
        return mx::slice(t, s0, s1);
      };
      qa = slice_q(q);
      if (!tile_rel_matmul) rela = slice_q(rel_q);
      sinka = slice_q(sink_q);
    }

    // When sliding_window > 0 and L > 1 (single recent segment), restrict K/V to only the keys
    // reachable by query tile [a, b): [max(0, q_start - SW + 1), q_end). Cuts prefill FLOPs & RAM ~2x.
    const bool sw_tile_k = (SW > 0 && L > 1 && parts.size() == 1);
    int k_lo = 0, k_hi = kL;
    if (sw_tile_k) {
      const int q_start_abs = static_cast<int>(n_ - L + a);
      const int q_end_abs = static_cast<int>(n_ - L + b);
      const int k_base_abs = static_cast<int>(n_ - kL);
      k_lo = std::max(0, q_start_abs - static_cast<int>(SW) + 1 - k_base_abs);
      k_hi = std::max(k_lo, std::min(kL, q_end_abs - k_base_abs));
    }
    const int eff_kL = k_hi - k_lo;
    auto slice_kv = [&](const mx::array& t) {
      if (k_lo == 0 && k_hi == kL) return t;
      mx::Shape s0(t.ndim(), 0), s1(t.shape());
      s0[t.ndim() - 2] = k_lo;
      s1[t.ndim() - 2] = k_hi;
      return mx::slice(t, s0, s1);
    };

    // Fast fused FlashAttention path for BF16 prefill (L >= 16), including sparse-paged's [prefix(bf16), recent(bf16)]:
    // build [rel_bias | sink] in a single custom Metal pass (fusing r @ rp_t when rp_t_ed is supplied).
    // Small verify batches (L = 2..8, e.g. DSpark) use the zero-copy compiled sink_sm path below so they
    // never copy 436-704 MB of K/V across 48 visits just to append one zero sink row.
    if (L >= 16 && all_bf16) {
      mx::array k_pos_cur = (k_lo == 0 && k_hi == kL) ? k_abs_pos : mx::slice(k_abs_pos, {k_lo}, {k_hi});
      mx::array qa_raw = (curL == L) ? queries : mx::slice(queries, {0, 0, a, 0}, {B, Hq, b, D});
      mx::array sinka_hq = mx::astype((curL == L) ? sink : mx::slice(sink, {0, 0, a}, {B, Hq, b}), qa_raw.dtype());
      const int cols = eff_kL + 1;
      const int rows_tot = B * Hq * curL;
      mx::array mask_full = mx::array(0.0f);
      {
        mx::array rela_hq = mx::array(0.0f);
        if (tile_rel_matmul) {
          const int Drel = rel_logits.shape(3);
          mx::array ra_hq = (curL == L) ? rel_logits : mx::slice(rel_logits, {0, 0, a, 0}, {B, Hq, b, Drel});
          rela_hq = mx::astype(mx::matmul(ra_hq, mx::transpose(*rp_t_ed)), qa_raw.dtype());
        } else {
          rela_hq = mx::astype((curL == L) ? rel_mat : mx::slice(rel_mat, {0, 0, a, 0}, {B, Hq, b, E}),
                               qa_raw.dtype());
        }
        static auto rel_sink_k = mx::fast::metal_kernel(
            "inkling_rel_sink_mask",
            {"rel", "sink", "k_pos", "params"},
            {"out"},
            R"(
              uint j = thread_position_in_grid.x;
              uint row = thread_position_in_grid.y;
              int q0 = params[0];
              int curL = params[1];
              int eff_kL = params[2];
              int E = params[3];
              int SW = params[4];
              int cols = eff_kL + 1;
              if ((int)j >= cols) return;
              uint out_idx = row * (uint)cols + j;
              if ((int)j == eff_kL) {
                out[out_idx] = sink[row];
                return;
              }
              int i = (int)(row % (uint)curL);
              int d = (q0 + i) - k_pos[j];
              bool ok = (d >= 0) && (SW <= 0 || d < SW);
              if (!ok) {
                out[out_idx] = (T)(-1e9f);
              } else if (d < E) {
                out[out_idx] = rel[row * (uint)E + (uint)d];
              } else {
                out[out_idx] = (T)(0.0f);
              }
            )",
            "", /*ensure_row_contiguous=*/true);
        const int32_t pdata[5] = {static_cast<int32_t>(n_ - L + a), curL, eff_kL, E, static_cast<int32_t>(SW)};
        mx::array params_arr(pdata, mx::Shape{5}, mx::int32);
        mask_full = rel_sink_k(
            {rela_hq, sinka_hq, k_pos_cur, params_arr},
            {mx::Shape{B, Hq, curL, cols}},
            {qa_raw.dtype()},
            std::make_tuple(cols, rows_tot, 1),
            std::make_tuple(std::min(256, cols), 1, 1),
            {{"T", qa_raw.dtype()}},
            std::nullopt, false, mx::StreamOrDevice{})[0];
      }

      mx::array k_base = (rep > 1) ? mx::squeeze(parts[0].k[0], 2) : parts[0].k[0];
      mx::array v_base = (rep > 1) ? mx::squeeze(parts[0].v[0], 2) : parts[0].v[0];
      if (parts.size() > 1) {
        std::vector<mx::array> ks, vs;
        ks.reserve(parts.size());
        vs.reserve(parts.size());
        for (const auto& p : parts) {
          ks.push_back((rep > 1) ? mx::squeeze(p.k[0], 2) : p.k[0]);
          vs.push_back((rep > 1) ? mx::squeeze(p.v[0], 2) : p.v[0]);
        }
        k_base = mx::concatenate(ks, 2);
        v_base = mx::concatenate(vs, 2);
      }
      mx::array k_tile = (k_lo == 0 && k_hi == kL) ? k_base : mx::slice(k_base, {0, 0, k_lo, 0}, {B, Hkv, k_hi, D});
      mx::array v_tile = (k_lo == 0 && k_hi == kL) ? v_base : mx::slice(v_base, {0, 0, k_lo, 0}, {B, Hkv, k_hi, D});
      mx::array z1 = mx::zeros({B, Hkv, 1, D}, k_tile.dtype());
      mx::array k_pad = mx::concatenate({k_tile, z1}, 2);
      mx::array v_pad = mx::concatenate({v_tile, z1}, 2);
      mx::array out_hq = mx::fast::scaled_dot_product_attention(qa_raw, k_pad, v_pad, scale, "", mask_full);
      return (rep > 1) ? mx::reshape(out_hq, {B, Hkv, rep, curL, D}) : out_hq;
    }

    // Fast zero-copy decode (L == 1) and DSpark verify (2 <= L <= 8) path for ANY context length
    // and ANY number of segments whenever the trailing tail_m = min(kL, E + L - 1) keys are
    // contiguous up to n_-1 (true for 100% of sliding-window visits and 100% of full-attention
    // visits across bf16/int8/k8v4/int4/tiered/sparse-paged). Zero K/V copies, zero z1 concat,
    // and compiled sink-softmax without sc_sink concatenation.
    const int tail_m = std::min(kL, E + curL - 1);
    if (L >= 1 && L <= 8 && !parts.empty() &&
        (parts.back().len >= tail_m || kL == static_cast<int>(n_)) &&
        (SW == 0 || kL <= tail_m)) {
      std::vector<mx::array> blk;
      blk.reserve(parts.size());
      for (const auto& p : parts) {
        if (p.kbits == 16)
          blk.push_back(mx::matmul(qa, mx::swapaxes(p.k[0], -1, -2)));
        else
          blk.push_back(mx::quantized_matmul(qa, p.k[0], p.k[1], p.k[2], true, gs, p.kbits, "affine"));
      }
      mx::array sc = blk.size() == 1 ? blk[0] : mx::concatenate(blk, -1);
      static auto sink_sm_c = mx::compile(
          [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
            mx::array s = mx::add(in[0], mx::astype(in[1], in[0].dtype()));
            mx::array s_f = mx::astype(s, mx::float32);
            mx::array lse = mx::logsumexp(s_f, -1, /*keepdims=*/true);
            mx::array p = mx::softmax(s_f, -1);
            mx::array scale_sink = mx::sigmoid(mx::subtract(lse, mx::astype(in[2], mx::float32)));
            return {mx::astype(mx::multiply(p, scale_sink), in[0].dtype())};
          },
          true);
      static auto sink_sm_only_c = mx::compile(
          [](const std::vector<mx::array>& in) -> std::vector<mx::array> {
            mx::array s_f = mx::astype(in[0], mx::float32);
            mx::array lse = mx::logsumexp(s_f, -1, /*keepdims=*/true);
            mx::array p = mx::softmax(s_f, -1);
            mx::array scale_sink = mx::sigmoid(mx::subtract(lse, mx::astype(in[1], mx::float32)));
            return {mx::astype(mx::multiply(p, scale_sink), in[0].dtype())};
          },
          true);
      static auto get_rev_idx = [](int m) -> mx::array {
        static std::map<int, mx::array> cache;
        auto it = cache.find(m);
        if (it != cache.end()) return it->second;
        mx::array arr = mx::arange(m - 1, -1, -1, mx::int32);
        mx::eval(arr);
        cache.emplace(m, arr);
        return arr;
      };

      struct VerifyIndexCache {
        mx::array safe_d;
        mx::array in_rel;
        mx::array causal_ok;
      };
      static auto get_verify_indices = [](int curL, int tail_m, int SW, int E) -> const VerifyIndexCache& {
        static std::map<std::tuple<int, int, int, int>, VerifyIndexCache> vcache;
        auto key = std::make_tuple(curL, tail_m, SW, E);
        auto it = vcache.find(key);
        if (it != vcache.end()) return it->second;
        mx::array q_rel = mx::arange(tail_m - curL, tail_m, mx::int32);
        mx::array k_rel = mx::arange(0, tail_m, mx::int32);
        mx::array dist = mx::subtract(mx::expand_dims(q_rel, 1), mx::expand_dims(k_rel, 0));
        mx::array causal_ok = mx::greater_equal(dist, mx::array(0, mx::int32));
        if (SW > 0)
          causal_ok = mx::logical_and(causal_ok, mx::less(dist, mx::array(SW, mx::int32)));
        mx::array in_rel = mx::logical_and(causal_ok, mx::less(dist, mx::array(E, mx::int32)));
        mx::array safe_d = mx::where(in_rel, dist, mx::array(0, mx::int32));
        mx::eval(std::vector<mx::array>{safe_d, in_rel, causal_ok});
        auto [ins, _] = vcache.emplace(key, VerifyIndexCache{safe_d, in_rel, causal_ok});
        return ins->second;
      };

      mx::array probs = mx::array(0.0f);
      if (curL == 1) {
        const int m = tail_m;
        mx::array rev_idx = get_rev_idx(m);
        mx::array rel_bias = mx::take(rela, rev_idx, -1);  // [B, Hkv, (rep,) 1, m]
        if (kL == m) {
          probs = sink_sm_c({sc, rel_bias, sinka})[0];
        } else {
          mx::Shape h0(sc.ndim(), 0), h1(sc.shape()), t0(sc.ndim(), 0), t1(sc.shape());
          h1.back() = kL - m;
          t0.back() = kL - m;
          mx::array sc_head = mx::slice(sc, h0, h1);
          mx::array sc_tail = mx::add(mx::slice(sc, t0, t1), mx::astype(rel_bias, sc.dtype()));
          probs = sink_sm_only_c({mx::concatenate({sc_head, sc_tail}, -1), sinka})[0];
        }
      } else {
        mx::array rela_use = rela;
        if (tile_rel_matmul) {
          const int Drel = rel_logits.shape(3);
          mx::array ra_hq = (curL == L) ? rel_logits : mx::slice(rel_logits, {0, 0, a, 0}, {B, Hq, b, Drel});
          mx::array rm = mx::matmul(ra_hq, mx::transpose(*rp_t_ed));
          rela_use = (rep > 1) ? mx::reshape(rm, {B, Hkv, rep, curL, E}) : rm;
        }
        const auto& vic = get_verify_indices(curL, tail_m, static_cast<int>(SW), E);
        mx::Shape lead_ones(sc.ndim(), 1);
        lead_ones[sc.ndim() - 2] = curL;
        lead_ones[sc.ndim() - 1] = tail_m;
        mx::Shape tail_sh(sc.shape());
        tail_sh.back() = tail_m;
        mx::array rel_bias = mx::take_along_axis(
            rela_use, mx::broadcast_to(mx::reshape(vic.safe_d, lead_ones), tail_sh), -1);
        rel_bias = mx::where(mx::reshape(vic.in_rel, lead_ones), rel_bias, mx::array(0.0f, rel_bias.dtype()));
        mx::array causal_nd = mx::reshape(vic.causal_ok, lead_ones);
        mx::array min_val(mx::finfo(sc.dtype()).min, sc.dtype());
        if (kL == tail_m) {
          mx::array sc_t = mx::where(causal_nd, mx::add(sc, mx::astype(rel_bias, sc.dtype())), min_val);
          probs = sink_sm_only_c({sc_t, sinka})[0];
        } else {
          mx::Shape h0(sc.ndim(), 0), h1(sc.shape()), t0(sc.ndim(), 0), t1(sc.shape());
          h1.back() = kL - tail_m;
          t0.back() = kL - tail_m;
          mx::array sc_head = mx::slice(sc, h0, h1);
          mx::array sc_tail = mx::where(causal_nd,
                                        mx::add(mx::slice(sc, t0, t1), mx::astype(rel_bias, sc.dtype())),
                                        min_val);
          probs = sink_sm_only_c({mx::concatenate({sc_head, sc_tail}, -1), sinka})[0];
        }
      }
      if (parts.size() == 1 && parts[0].vbits == 16) return mx::matmul(probs, parts[0].v[0]);
      std::optional<mx::array> out;
      int at = 0;
      for (const auto& p : parts) {
        mx::array pi = parts.size() == 1 ? probs : mx::slice(probs, [&] {
          mx::Shape s(probs.ndim(), 0); s.back() = at; return s; }(), [&] {
          mx::Shape s(probs.shape()); s.back() = at + p.len; return s; }());
        at += p.len;
        mx::array o = p.vbits == 16
                          ? mx::matmul(pi, p.v[0])
                          : mx::quantized_matmul(pi, p.v[0], p.v[1], p.v[2], false, gs, p.vbits, "affine");
        out = out ? mx::add(*out, o) : o;
      }
      return *out;
    }

    std::vector<mx::array> blk;
    blk.reserve(parts.size());
    for (const auto& p : parts) {
      if (p.kbits == 16)
        blk.push_back(mx::matmul(qa, mx::swapaxes(sw_tile_k ? slice_kv(p.k[0]) : p.k[0], -1, -2)));
      else
        blk.push_back(mx::quantized_matmul(qa, sw_tile_k ? slice_kv(p.k[0]) : p.k[0],
                                           sw_tile_k ? slice_kv(p.k[1]) : p.k[1],
                                           sw_tile_k ? slice_kv(p.k[2]) : p.k[2], true, gs, p.kbits, "affine"));
    }
    mx::array sc = blk.size() == 1 ? blk[0] : mx::concatenate(blk, -1);

    // Exact relative-position distance: dist[i, j] = pos(query i) - pos(key j).
    mx::array k_pos_cur = (k_lo == 0 && k_hi == kL) ? k_abs_pos : mx::slice(k_abs_pos, {k_lo}, {k_hi});
    mx::array q_abs = mx::arange(static_cast<int>(n_ - L + a), static_cast<int>(n_ - L + b), mx::int32);
    mx::array dist = mx::subtract(mx::expand_dims(q_abs, 1), mx::expand_dims(k_pos_cur, 0));  // [curL, eff_kL]
    mx::array causal_ok = mx::greater_equal(dist, mx::array(0, mx::int32));
    if (SW > 0 && L > 1)
      causal_ok = mx::logical_and(causal_ok, mx::less(dist, mx::array(static_cast<int>(SW), mx::int32)));
    mx::array in_rel = mx::logical_and(causal_ok, mx::less(dist, mx::array(E, mx::int32)));
    mx::array safe_d = mx::where(in_rel, dist, mx::array(0, mx::int32));

    mx::Shape lead_ones(sc.ndim(), 1);
    lead_ones[sc.ndim() - 2] = curL;
    lead_ones[sc.ndim() - 1] = eff_kL;
    mx::array in_rel_nd = mx::reshape(in_rel, lead_ones);
    mx::array rela_use = rela;
    if (tile_rel_matmul) {
      const int Drel = rel_logits.shape(3);
      mx::array ra_hq = (curL == L) ? rel_logits : mx::slice(rel_logits, {0, 0, a, 0}, {B, Hq, b, Drel});
      mx::array rm = mx::matmul(ra_hq, mx::transpose(*rp_t_ed));
      rela_use = (rep > 1) ? mx::reshape(rm, {B, Hkv, rep, curL, E}) : rm;
    }
    mx::array rel_bias = (curL == 1)
                             ? mx::take(rela_use, mx::reshape(safe_d, {eff_kL}), -1)
                             : mx::take_along_axis(
                                   rela_use, mx::broadcast_to(mx::reshape(safe_d, lead_ones), sc.shape()), -1);
    rel_bias = mx::where(in_rel_nd, rel_bias, mx::array(0.0f, rel_bias.dtype()));
    sc = mx::add(sc, mx::astype(rel_bias, sc.dtype()));

    if (L > 1) {
      mx::array causal_nd = mx::reshape(causal_ok, lead_ones);
      sc = mx::where(causal_nd, sc, mx::array(mx::finfo(sc.dtype()).min, sc.dtype()));
    }

    // Zero-value sink: append sink logit to softmax denominator, then drop the sink weight column.
    mx::array sc_sink = mx::concatenate({sc, mx::astype(sinka, sc.dtype())}, -1);
    mx::array probs_sink = mx::softmax(sc_sink, -1, /*precise=*/true);
    mx::Shape p0(probs_sink.ndim(), 0), p1(probs_sink.shape());
    p1.back() = eff_kL;
    mx::array probs = mx::slice(probs_sink, p0, p1);

    std::optional<mx::array> out;
    int at = 0;
    for (const auto& p : parts) {
      const int plen = sw_tile_k ? eff_kL : p.len;
      mx::array pi = parts.size() == 1 ? probs : mx::slice(probs, [&] {
        mx::Shape s(probs.ndim(), 0); s.back() = at; return s; }(), [&] {
        mx::Shape s(probs.shape()); s.back() = at + plen; return s; }());
      at += plen;
      mx::array o = p.vbits == 16
                        ? mx::matmul(pi, sw_tile_k ? slice_kv(p.v[0]) : p.v[0])
                        : mx::quantized_matmul(pi, sw_tile_k ? slice_kv(p.v[0]) : p.v[0],
                                               sw_tile_k ? slice_kv(p.v[1]) : p.v[1],
                                               sw_tile_k ? slice_kv(p.v[2]) : p.v[2], false, gs, p.vbits,
                                               "affine");
      out = out ? mx::add(*out, o) : o;
    }
    return *out;
  };

  const auto spans = query_spans(L, qb);
  mx::array o = attend_rows(spans[0].first, spans[0].second);
  if (spans.size() > 1) {
    std::vector<mx::array> outs{o};
    for (size_t i = 1; i < spans.size(); ++i) {
      mx::eval(outs.back());
      outs.push_back(attend_rows(spans[i].first, spans[i].second));
    }
    o = mx::concatenate(outs, -2);
  }
  if (rep > 1) o = mx::reshape(o, {B, Hq, L, D});

  if (!dense_recent_ && !defer_compact && use_recent && (use_middle || SW > 0))
    compact_recent(std::max(H_compact, P));
  return o;
}

bool KVStore::truncate(int64_t new_n) {
  if (new_n == n_) return true;
  if (new_n < 0 || new_n > n_) return false;
  if (new_n == 0) {
    dense_recent_ = use_dense_recent();
    prefix_.len = 0;
    prefix_.start = 0;
    middle_.len = 0;
    middle_.start = (policy_.sliding_window > 0 ? 0 : policy_.prefix_tokens);
    recent_.len = 0;
    recent_.start = (!dense_recent_ && policy_.sliding_window == 0 && policy_.prefix_tokens > 0 ? policy_.prefix_tokens : 0);
    n_ = 0;
    indexed_to_ = 0;
    page_k_min_.clear();
    page_k_max_.clear();
    return true;
  }
  const int64_t SW = policy_.sliding_window;
  const int64_t P = SW > 0 ? 0 : policy_.prefix_tokens;
  const bool defer_middle = policy_.sparse_paged && SW == 0 && recent_.kbits == 16 && recent_.vbits == 16;

  int64_t pre_len_new = prefix_.len;
  int64_t mid_len_new = middle_.len;
  int64_t rec_len_new = recent_.len;

  if (!prefix_.k.empty()) {
    const int64_t want = std::min<int64_t>(new_n, P);
    if (want < prefix_.start || want > prefix_.start + prefix_.len) return false;
    pre_len_new = want - prefix_.start;
  }
  if (!middle_.k.empty()) {
    if (defer_middle) {
      // In deferred sparse-paged mode, middle_ only holds aged pages < recent_.start.
      if (new_n < middle_.start + middle_.len) {
        const int64_t want = std::max<int64_t>(0, new_n - middle_.start);
        if (want > middle_.len) return false;
        mid_len_new = want;
      }
    } else {
      const int64_t want = std::max<int64_t>(0, new_n - P);
      if (middle_.start != P || want > middle_.len) return false;
      mid_len_new = want;
    }
  }
  if (!recent_.k.empty()) {
    if (dense_recent_) {
      if (new_n < recent_.start || new_n > recent_.start + recent_.len) return false;
      rec_len_new = new_n - recent_.start;
    } else if (SW == 0 && new_n <= P) {
      rec_len_new = 0;
      recent_.start = P;
    } else {
      if (new_n < recent_.start || new_n > recent_.start + recent_.len) return false;
      rec_len_new = new_n - recent_.start;
    }
  }

  prefix_.len = pre_len_new;
  middle_.len = mid_len_new;
  recent_.len = rec_len_new;
  n_ = new_n;

  if (policy_.sparse_paged && SW == 0 && new_n < indexed_to_) {
    const int64_t pg = policy_.page;
    const int64_t keep_pages = new_n / pg;
    if (static_cast<size_t>(keep_pages) < page_k_min_.size()) {
      page_k_min_.erase(page_k_min_.begin() + static_cast<std::ptrdiff_t>(keep_pages), page_k_min_.end());
      page_k_max_.erase(page_k_max_.begin() + static_cast<std::ptrdiff_t>(keep_pages), page_k_max_.end());
    }
    indexed_to_ = keep_pages * pg;
  }
  return true;
}

int64_t KVStore::nbytes() const {
  int64_t t = 0;
  for (const KVSeg* s : {&prefix_, &middle_, &recent_}) {
    for (const auto& a : s->k) t += a.nbytes();
    for (const auto& a : s->v) t += a.nbytes();
  }
  for (const auto& a : page_k_min_) t += a.nbytes();
  for (const auto& a : page_k_max_) t += a.nbytes();
  return t;
}

void KVStore::append_state_arrays(std::vector<mx::array>& out) const {
  for (const KVSeg* s : {&prefix_, &middle_, &recent_}) {
    for (const auto& a : s->k) out.push_back(a);
    for (const auto& a : s->v) out.push_back(a);
  }
  for (const auto& a : page_k_min_) out.push_back(a);
  for (const auto& a : page_k_max_) out.push_back(a);
}

std::vector<std::pair<std::string, mx::array>> KVStore::save(const std::string& pfx) const {
  std::vector<std::pair<std::string, mx::array>> out;
  const char* names[3] = {"prefix", "middle", "recent"};
  const KVSeg* segs[3] = {&prefix_, &middle_, &recent_};
  for (int s = 0; s < 3; ++s) {
    const KVSeg& g = *segs[s];
    if (g.k.empty()) continue;
    const std::string base = pfx + "." + names[s];
    out.emplace_back(base + ".meta",
                     mx::array({static_cast<int64_t>(g.start), static_cast<int64_t>(g.len)},
                               mx::Shape{2}, mx::int64));
    for (size_t i = 0; i < g.k.size(); ++i)
      out.emplace_back(base + ".k" + std::to_string(i), mx::contiguous(rows(g.k[i], 0, g.len)));
    for (size_t i = 0; i < g.v.size(); ++i)
      out.emplace_back(base + ".v" + std::to_string(i), mx::contiguous(rows(g.v[i], 0, g.len)));
  }
  return out;
}

void KVStore::load(const std::string& pfx,
                   const std::vector<std::pair<std::string, mx::array>>& arrays, int64_t n_tokens) {
  auto get = [&](const std::string& name) -> const mx::array* {
    for (const auto& kv : arrays)
      if (kv.first == name) return &kv.second;
    return nullptr;
  };
  prefix_ = KVSeg{}; middle_ = KVSeg{}; recent_ = KVSeg{};
  prefix_.kbits = prefix_.vbits = policy_.prefix_bits;
  middle_.kbits = policy_.hist_k_bits;
  middle_.vbits = policy_.hist_v_bits;
  recent_.kbits = recent_.vbits = policy_.recent_bits;
  init_ = true;
  const char* names[3] = {"prefix", "middle", "recent"};
  KVSeg* segs[3] = {&prefix_, &middle_, &recent_};
  for (int s = 0; s < 3; ++s) {
    KVSeg& g = *segs[s];
    const std::string base = pfx + "." + names[s];
    const mx::array* meta = get(base + ".meta");
    if (!meta) continue;
    mx::array m = mx::astype(*meta, mx::int64);
    mx::eval(m);
    g.start = m.data<int64_t>()[0];
    g.len = m.data<int64_t>()[1];
    g.cap = ((g.len + kStep - 1) / kStep) * kStep;
    const int nk = (g.kbits == 16) ? 1 : 3;
    const int nv = (g.vbits == 16) ? 1 : 3;
    for (int i = 0; i < nk; ++i) {
      const mx::array* a = get(base + ".k" + std::to_string(i));
      if (!a) throw std::runtime_error("kv: snapshot missing " + base + ".k" + std::to_string(i));
      g.k.push_back(*a);
    }
    for (int i = 0; i < nv; ++i) {
      const mx::array* a = get(base + ".v" + std::to_string(i));
      if (!a) throw std::runtime_error("kv: snapshot missing " + base + ".v" + std::to_string(i));
      g.v.push_back(*a);
    }
    if (g.cap > g.len) {
      for (auto* buf : {&g.k, &g.v})
        for (auto& a : *buf) {
          mx::Shape sh(a.shape());
          sh[2] = static_cast<int>(g.cap - g.len);
          a = mx::concatenate({a, mx::zeros(sh, a.dtype())}, 2);
        }
    }
  }
  n_ = n_tokens;
  // A restored snapshot that already split prefix_ (or has recent_.start > 0 or n_ > P+W)
  // must NOT re-enter dense_recent_ mode, otherwise the next append calls split_dense_prefix()
  // and tries to append pos 0 into a non-empty prefix_ (throwing non-contiguous segment append).
  dense_recent_ = use_dense_recent() && prefix_.k.empty() && recent_.start == 0 &&
                  n_ <= policy_.prefix_tokens + policy_.recent_tokens;
}

}  // namespace veystrix::engine::generic
