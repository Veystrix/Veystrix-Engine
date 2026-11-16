"""Experimental asymmetric/tiered affine KV. Only gather QSA may consume it.

No model weights, raw indexer keys or GDN states are transformed. History is
never dequantized by attention: only the selected rows are unpacked. INT4 is
written directly from original activations, avoiding an INT8-to-INT4 cascade.
Only the bounded recent window has a second representation.
"""
from __future__ import annotations

import os

import mlx.core as mx
from mlx_vlm.models.qwen4_exp.language import QSAKVCache

PAGE = 256
HISTORY_GROWTH = 4096
SCHEMA = 1


def _arrays(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _slice(value, start, end):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(a[..., start:end, :] for a in value)
    return value[..., start:end, :]


FLAG_FUSED_GATHER = "FLASHNEXT_QSA_FUSED_GATHER"   # opt-in: gather and dequantize INT8 rows in one kernel (bit-identical)


def fused_gather_enabled() -> bool:
    value = os.environ.get(FLAG_FUSED_GATHER, "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_FUSED_GATHER} must be 0 or 1")
    return value == "1"


_FUSED_SOURCE = """
    uint d4 = thread_position_in_grid.x;
    uint si = thread_position_in_grid.y;
    uint th = thread_position_in_grid.z;
    uint h = th % NH;
    int row = idx[(th / NH) * NS + si];
    uint base = (h * NCAP + (uint)row);
    uint word = w[base * DW + d4];
    for (uint k = 0; k < 4; k++) {
        uint q = (word >> (8 * k)) & 0xFFu;
        uint d = d4 * 4 + k;
        uint g = d / GSZ;
        float r = static_cast<float>(sc[base * NG + g]) * static_cast<float>(q) + static_cast<float>(bi[base * NG + g]);
        out[((th * NS) + si) * ND + d] = static_cast<bfloat16_t>(r);
    }
"""
_FUSED_KERNEL = [None]
_FUSED_TWO_TIER_SOURCE = """
    uint d4 = thread_position_in_grid.x;
    uint si = thread_position_in_grid.y;
    uint th = thread_position_in_grid.z;
    uint h = th % NH;
    int row = idx[(th / NH) * NS + si];
    if (row < (int)HIST) {
        // history tier: INT4, eight values per word; this thread handles four consecutive values
        uint base = (h * NCAPO + (uint)row);
        uint word = wo[base * DWO + d4 / 2];
        uint shift = (d4 % 2) * 16;
        for (uint k = 0; k < 4; k++) {
            uint q = (word >> (shift + 4 * k)) & 0xFu;
            uint d = d4 * 4 + k;
            uint g = d / GSZ;
            float r = static_cast<float>(so[base * NG + g]) * static_cast<float>(q) + static_cast<float>(bo[base * NG + g]);
            out[((th * NS) + si) * ND + d] = static_cast<bfloat16_t>(r);
        }
    } else {
        uint base = (h * NCAPR + (uint)(row - (int)HIST));
        uint word = wr[base * DWR + d4];
        for (uint k = 0; k < 4; k++) {
            uint q = (word >> (8 * k)) & 0xFFu;
            uint d = d4 * 4 + k;
            uint g = d / GSZ;
            float r = static_cast<float>(sr[base * NG + g]) * static_cast<float>(q) + static_cast<float>(br[base * NG + g]);
            out[((th * NS) + si) * ND + d] = static_cast<bfloat16_t>(r);
        }
    }
"""
_FUSED_TWO_TIER_KERNEL = [None]


def _fused_dequantize_two_tier(old, recent, indices, history, group):
    """The two-tier gather of ``gather_selected`` (INT4 rows below ``history``, INT8 rows from the recent tier above it) in one
    kernel, each element computed with the formula of the single-tier kernel (bit-identical to ``mx.dequantize``).
    Returns [T, H, S, D], or None for a layout it does not handle."""
    wo, so, bo = old
    wr, sr, br = recent
    if (wo.ndim != 4 or wo.shape[0] != 1 or wr.shape[0] != 1 or indices.ndim != 3 or indices.shape[0] != 1
            or wo.dtype != mx.uint32 or wr.dtype != mx.uint32
            or any(a.dtype != mx.bfloat16 for a in (so, bo, sr, br)) or so.shape != bo.shape or sr.shape != br.shape):
        return None
    heads, cap_old, words_old = wo.shape[1], wo.shape[2], wo.shape[3]
    cap_recent, words_recent = wr.shape[2], wr.shape[3]
    dim = words_recent * 4
    if (words_old * 8 != dim or dim % group or wr.shape[1] != heads or so.shape != (1, heads, cap_old, dim // group)
            or sr.shape != (1, heads, cap_recent, dim // group) or dim % 4 or history < 1):
        return None
    if _FUSED_TWO_TIER_KERNEL[0] is None:
        _FUSED_TWO_TIER_KERNEL[0] = mx.fast.metal_kernel(
            name="flashnext_gather_dequantize_two_tier", input_names=["wo", "so", "bo", "wr", "sr", "br", "idx"],
            output_names=["out"], source=_FUSED_TWO_TIER_SOURCE)
    queries, selected = indices.shape[1], indices.shape[2]
    out = _FUSED_TWO_TIER_KERNEL[0](
        inputs=[wo, so, bo, wr, sr, br, indices.astype(mx.int32)],
        template=[("NH", heads), ("NS", selected), ("NCAPO", cap_old), ("NCAPR", cap_recent), ("DWO", words_old),
                  ("DWR", words_recent), ("NG", dim // group), ("GSZ", group), ("ND", dim), ("HIST", int(history))],
        grid=(words_recent, selected, queries * heads), threadgroup=(words_recent, 1, 1),
        output_shapes=[(queries * heads, selected, dim)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(queries, heads, selected, dim)


FLAG_FUSED_SDPA = "FLASHNEXT_QSA_FUSED_SDPA"   # opt-in: one kernel gathers, dequantizes and attends (bit-identical to gather + MLX SDPA)


def fused_sdpa_enabled() -> bool:
    value = os.environ.get(FLAG_FUSED_SDPA, "0") or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG_FUSED_SDPA} must be 0 or 1")
    return value == "1"


# MLX's single-pass ``sdpa_vector`` kernel (mlx/backend/metal/kernels/sdpa_vector.h) with its K and V loads replaced by the
# gather and the affine dequantization of the cache rows. Same thread mapping, same per-thread order of operations, same
# simd reductions and the same final cross-simdgroup reduction, so the result is bit-identical to gathering bfloat16 K/V
# and calling ``mx.fast.scaled_dot_product_attention`` whenever MLX picks that single-pass kernel (checked bit for bit for
# one to 32 query rows and one to 2,052 keys, head dimension 256, 24 query heads, 2 key/value heads, boolean mask).
_FUSED_SDPA_SOURCE = """
    constexpr int BN = 32;
    constexpr int BD = 32;
    typedef float U;
    uint3 tid = threadgroup_position_in_grid;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];
    // One threadgroup per (query row, key/value head, group of HG query heads): it dequantizes each selected row once for the HG
    // heads. Every head keeps MLX's per-head order of operations.
    int hg = tid.x % NHG;
    int kvh = (tid.x / NHG) % HKV;
    int row_t = tid.x / (NHG * HKV);
    int qbh0 = row_t * HQ + kvh * GQ + hg * HG;
    U qv[HG][QPT];
    U o[HG][QPT];
    U max_score[HG];
    U sum_exp_score[HG];
    for (int h = 0; h < HG; h++) {
        auto qp = &q[0] + (qbh0 + h) * ND + simd_lid * QPT;
        for (int i = 0; i < QPT; i++) {
            qv[h][i] = static_cast<U>(scale[0]) * static_cast<U>(qp[i]);
            o[h][i] = 0;
        }
        max_score[h] = -3.4028234663852886e+38f;
        sum_exp_score[h] = 0;
    }
    int d0 = simd_lid * QPT;
    for (int i = simd_gid; i < NS; i += BN) {
        if (mask[row_t * NS + i]) {
            int row = idx[row_t * NS + i];
            U kf[QPT];
            U vf[QPT];
            if (row < HIST) {
                uint base = (uint)kvh * NCAPO + (uint)row;
                uint kword = ko_w[base * DWO + d0 / 8];
                uint vword = vo_w[base * DWO + d0 / 8];
                uint shift = (d0 % 8) * 4;
                for (int j = 0; j < QPT; j++) {
                    uint g = (d0 + j) / GSZ;
                    uint kq = (kword >> (shift + 4 * j)) & 0xFu;
                    uint vq = (vword >> (shift + 4 * j)) & 0xFu;
                    float kr = static_cast<float>(ko_s[base * NG + g]) * static_cast<float>(kq) + static_cast<float>(ko_b[base * NG + g]);
                    float vr = static_cast<float>(vo_s[base * NG + g]) * static_cast<float>(vq) + static_cast<float>(vo_b[base * NG + g]);
                    kf[j] = static_cast<float>(static_cast<bfloat16_t>(kr));
                    vf[j] = static_cast<float>(static_cast<bfloat16_t>(vr));
                }
            } else {
                uint base = (uint)kvh * NCAPR + (uint)(row - HIST);
                for (int w4 = 0; w4 < QPT / 4; w4++) {
                    uint kword = kr_w[base * DWR + d0 / 4 + w4];
                    uint vword = vr_w[base * DWR + d0 / 4 + w4];
                    for (int k = 0; k < 4; k++) {
                        int j = w4 * 4 + k;
                        uint g = (d0 + j) / GSZ;
                        uint kq = (kword >> (8 * k)) & 0xFFu;
                        uint vq = (vword >> (8 * k)) & 0xFFu;
                        float kr = static_cast<float>(kr_s[base * NG + g]) * static_cast<float>(kq) + static_cast<float>(kr_b[base * NG + g]);
                        float vr = static_cast<float>(vr_s[base * NG + g]) * static_cast<float>(vq) + static_cast<float>(vr_b[base * NG + g]);
                        kf[j] = static_cast<float>(static_cast<bfloat16_t>(kr));
                        vf[j] = static_cast<float>(static_cast<bfloat16_t>(vr));
                    }
                }
            }
            for (int h = 0; h < HG; h++) {
                U score = 0;
                for (int j = 0; j < QPT; j++) {
                    score += qv[h][j] * kf[j];
                }
                score = simd_sum(score);
                U new_max = max(max_score[h], score);
                U factor = fast::exp(max_score[h] - new_max);
                U exp_score = fast::exp(score - new_max);
                max_score[h] = new_max;
                sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;
                for (int j = 0; j < QPT; j++) {
                    o[h][j] = o[h][j] * factor + exp_score * vf[j];
                }
            }
        }
    }
    for (int h = 0; h < HG; h++) {
        if (simd_lid == 0) {
            max_scores[simd_gid] = max_score[h];
            sum_exp_scores[simd_gid] = sum_exp_score[h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U ms = max_scores[simd_lid];
        U new_max = simd_max(ms);
        U factor = fast::exp(ms - new_max);
        U total = simd_sum(sum_exp_scores[simd_lid] * factor);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < QPT; i++) {
            outputs[simd_lid * BD + simd_gid] = o[h][i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            U r = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
            r = total == 0 ? r : (r / total);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lid == 0) {
                out[(qbh0 + h) * ND + simd_gid * QPT + i] = static_cast<bfloat16_t>(r);
            }
        }
    }
"""
_FUSED_SDPA_KERNEL = [None]
_FUSED_SDPA_NAMES = ["q", "ko_w", "ko_s", "ko_b", "vo_w", "vo_s", "vo_b", "kr_w", "kr_s", "kr_b", "vr_w", "vr_s", "vr_b",
                     "idx", "mask", "scale"]


def fused_sdpa_selected(cache, queries, indices, valid, scale):
    """Attention of ``queries`` [T, Hq, 1, D] over the cache rows ``indices`` [1, T, S] (boolean ``valid`` [T, 1, 1, S]) in one
    kernel, bit-identical to ``gather_selected`` followed by ``mx.fast.scaled_dot_product_attention``. Returns [T, Hq, 1, D], or
    None when the cache layout, the shapes or the flag are not the supported ones (the caller then takes the reference path)."""
    if not fused_sdpa_enabled() or cache.mode != "tiered4" or not isinstance(cache.recent_keys, tuple):
        return None
    recent_n = cache.offset - cache.history_tokens
    history = int(cache.history_tokens)
    if recent_n <= 0 or (history and (not isinstance(cache.keys, tuple) or not isinstance(cache.values, tuple))):
        return None
    kr, vr = cache.recent_keys, cache.recent_values
    group = cache.group_size
    wr = kr[0]
    if (wr.ndim != 4 or wr.shape[0] != 1 or indices.ndim != 3 or indices.shape[0] != 1 or wr.dtype != mx.uint32
            or queries.ndim != 4 or queries.shape[2] != 1 or queries.dtype != mx.bfloat16 or valid.dtype != mx.bool_):
        return None
    kv_heads, cap_recent, words_recent = wr.shape[1], wr.shape[2], wr.shape[3]
    dim = words_recent * 4
    tiles, selected = indices.shape[1], indices.shape[2]
    q_heads = queries.shape[1]
    if (dim not in (128, 256) or dim % group or queries.shape[0] != tiles or queries.shape[3] != dim or q_heads % kv_heads
            or tiles > 32 or selected < 1 or valid.shape != (tiles, 1, 1, selected) or vr[0].shape != wr.shape
            or any(a.dtype != mx.bfloat16 for a in (kr[1], kr[2], vr[1], vr[2]))
            or kr[1].shape != (1, kv_heads, cap_recent, dim // group) or vr[1].shape != kr[1].shape
            or kr[2].shape != kr[1].shape or vr[2].shape != kr[1].shape):
        return None
    if history:
        ko, vo = cache.keys, cache.values
        wo = ko[0]
        words_old, cap_old = wo.shape[3], wo.shape[2]
        if (wo.ndim != 4 or wo.dtype != mx.uint32 or words_old * 8 != dim or wo.shape[1] != kv_heads or vo[0].shape != wo.shape
                or any(a.dtype != mx.bfloat16 for a in (ko[1], ko[2], vo[1], vo[2]))
                or ko[1].shape != (1, kv_heads, cap_old, dim // group) or vo[1].shape != ko[1].shape
                or ko[2].shape != ko[1].shape or vo[2].shape != ko[1].shape):
            return None
        at = indices
    else:
        ko, vo = kr, vr
        words_old, cap_old = words_recent, cap_recent
        at = mx.clip(indices, 0, recent_n - 1)
    # Query heads that share one dequantization of a key/value row. Two is the fastest on the M4 (one head redoes the dequantization
    # for every head, three or more spill registers and run several times slower); the bits do not depend on it.
    heads_per_group = int(os.environ.get("FLASHNEXT_QSA_FUSED_SDPA_HEADS", "2" if (q_heads // kv_heads) % 2 == 0 else "1"))
    if heads_per_group < 1 or (q_heads // kv_heads) % heads_per_group:
        return None
    if _FUSED_SDPA_KERNEL[0] is None:
        _FUSED_SDPA_KERNEL[0] = mx.fast.metal_kernel(
            name="flashnext_fused_sdpa", input_names=_FUSED_SDPA_NAMES, output_names=["out"], source=_FUSED_SDPA_SOURCE)
    out = _FUSED_SDPA_KERNEL[0](
        inputs=[queries, ko[0], ko[1], ko[2], vo[0], vo[1], vo[2], kr[0], kr[1], kr[2], vr[0], vr[1], vr[2],
                at.astype(mx.int32), valid, mx.array([scale], dtype=mx.float32)],
        template=[("HQ", q_heads), ("HKV", kv_heads), ("GQ", q_heads // kv_heads), ("HG", heads_per_group),
                  ("NHG", (q_heads // kv_heads) // heads_per_group), ("ND", dim), ("QPT", dim // 32), ("NS", selected),
                  ("NCAPO", cap_old), ("NCAPR", cap_recent), ("DWO", words_old), ("DWR", words_recent),
                  ("NG", dim // group), ("GSZ", group), ("HIST", history)],
        grid=(tiles * kv_heads * ((q_heads // kv_heads) // heads_per_group) * 1024, 1, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(tiles * q_heads, dim)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(tiles, q_heads, 1, dim)


def _fused_dequantize_rows(parts, indices, group):
    """``mx.dequantize(*gathered(parts, indices), group_size=group, bits=8)`` for one batch row without the packed copies.

    ``parts`` are the INT8 affine arrays (uint32 words, bfloat16 scales and biases) shaped [1, H, capacity, ...]; ``indices`` is
    [1, T, S]. The affine formula is the one MLX evaluates (a float product and sum rounded once to bfloat16; checked
    bit-for-bit against ``mx.dequantize`` in the unit tests and on real cache rows). Returns [T, H, S, D] or None when the
    layout is not the supported one (the caller then takes the reference path)."""
    w, scales, biases = parts
    if (w.ndim != 4 or w.shape[0] != 1 or indices.ndim != 3 or indices.shape[0] != 1 or w.dtype != mx.uint32
            or scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16 or scales.shape != biases.shape):
        return None
    heads, capacity, words = w.shape[1], w.shape[2], w.shape[3]
    dim = words * 4
    if scales.shape != (1, heads, capacity, dim // group) or dim % group:
        return None
    if _FUSED_KERNEL[0] is None:
        _FUSED_KERNEL[0] = mx.fast.metal_kernel(name="flashnext_gather_dequantize8", input_names=["w", "sc", "bi", "idx"],
                                                output_names=["out"], source=_FUSED_SOURCE)
    queries, selected = indices.shape[1], indices.shape[2]
    out = _FUSED_KERNEL[0](
        inputs=[w, scales, biases, indices.astype(mx.int32)],
        template=[("NH", heads), ("NS", selected), ("NCAP", capacity), ("DW", words), ("NG", dim // group), ("GSZ", group),
                  ("ND", dim)],
        grid=(words, selected, queries * heads), threadgroup=(words, 1, 1),
        output_shapes=[(queries * heads, selected, dim)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(queries, heads, selected, dim)


def _unpack(value, bits, group):
    if bits == 16:
        return value
    return mx.dequantize(*[mx.contiguous(a) for a in value], group_size=group, bits=bits)


def _append(storage, value, offset, bits, group, growth=PAGE):
    packed = value if bits == 16 else mx.quantize(value, group_size=group, bits=bits)
    end = offset + value.shape[-2]
    capacity = ((end + growth - 1) // growth) * growth
    parts, incoming = _arrays(storage), _arrays(packed)
    if not parts:
        parts = [mx.zeros((*a.shape[:-2], capacity, a.shape[-1]), a.dtype) for a in incoming]
    elif parts[0].shape[-2] < end:
        parts = [mx.concatenate((a[..., :offset, :],
                 mx.zeros((*a.shape[:-2], capacity - offset, a.shape[-1]), a.dtype)), axis=-2)
                 for a in parts]
    for a, new in zip(parts, incoming):
        a[..., offset:end, :] = new
    return parts[0] if bits == 16 else tuple(parts)


class MixedPrecisionQSAKVCache(QSAKVCache):
    """K16/V4, K8/V4, or history K4/V4 + recent K8/V8.

    Recent capacity is window + at most 255 tokens of page-alignment slack.
    There is no full-length shadow BF16 cache. All incoming chunks must be
    bounded by the prefill planner; attention only receives selected rows.
    """
    def __init__(self, mode="tiered4", group_size=64, recent_tokens=8192):
        super().__init__()
        if mode not in ("k8v4", "k16v4", "tiered4"):
            raise ValueError("unsupported mixed KV mode")
        if group_size not in (32, 64):
            raise ValueError("mixed KV group size must be 32 or 64")
        if recent_tokens < PAGE or recent_tokens % PAGE:
            raise ValueError("recent-token window must be a positive multiple of 256")
        self.mode, self.group_size, self.recent_tokens = mode, group_size, recent_tokens
        self.key_bits = {"k16v4": 16, "k8v4": 8, "tiered4": 4}[mode]
        self.value_bits = 4
        self.recent_keys = self.recent_values = None
        self.history_tokens = 0

    def update_and_fetch(self, keys, values):
        end = self.offset + keys.shape[-2]
        self.keys = _append(self.keys, keys, self.offset, self.key_bits, self.group_size, HISTORY_GROWTH)
        self.values = _append(self.values, values, self.offset, 4, self.group_size, HISTORY_GROWTH)
        if self.mode != "tiered4":
            self.history_tokens = end
        else:
            recent = self.offset - self.history_tokens
            self.recent_keys = _append(self.recent_keys, keys, recent, 8, self.group_size)
            self.recent_values = _append(self.recent_values, values, recent, 8, self.group_size)
            recent += keys.shape[-2]
            aged = max(0, (recent - self.recent_tokens) // PAGE * PAGE)
            if aged:
                # Drop the expired INT8 rows. Their already-written INT4
                # rows become visible; no historical row is requantized.
                self.history_tokens += aged
                keep = mx.arange(aged, recent)
                self.recent_keys = tuple(mx.take(a, keep, axis=-2) for a in self.recent_keys)
                self.recent_values = tuple(mx.take(a, keep, axis=-2) for a in self.recent_values)
        self.offset = end
        # Projection helper requires a pair, but only gather_selected consumes
        # this cache. Returning incoming tensors avoids a history-sized view.
        return keys, values

    def gather_selected(self, indices):
        from .qsa_chunk import _gather_rows

        def gather(storage, at, bits):
            if bits == 16:
                return _gather_rows(storage, at)
            return _unpack(tuple(_gather_rows(a, at) for a in storage), bits, self.group_size)

        if self.mode != "tiered4":
            return gather(self.keys, indices, self.key_bits), gather(self.values, indices, 4)
        recent_n = self.offset - self.history_tokens
        if recent_n == 0:
            return gather(self.keys, indices, 4), gather(self.values, indices, 4)
        at = mx.clip(indices - self.history_tokens, 0, recent_n - 1)
        if self.history_tokens == 0 and fused_gather_enabled():
            keys = _fused_dequantize_rows(self.recent_keys, at, self.group_size)
            values = _fused_dequantize_rows(self.recent_values, at, self.group_size) if keys is not None else None
            if keys is not None and values is not None:
                return keys, values
        rk, rv = gather(self.recent_keys, at, 8), gather(self.recent_values, at, 8)
        if self.history_tokens == 0:
            return rk, rv
        if fused_gather_enabled() and isinstance(self.keys, tuple) and isinstance(self.values, tuple):
            keys = _fused_dequantize_two_tier(self.keys, self.recent_keys, indices, self.history_tokens, self.group_size)
            values = (_fused_dequantize_two_tier(self.values, self.recent_values, indices, self.history_tokens, self.group_size)
                      if keys is not None else None)
            if keys is not None and values is not None:
                return keys, values
        old_at = mx.minimum(indices, self.history_tokens - 1)
        hk, hv = gather(self.keys, old_at, 4), gather(self.values, old_at, 4)
        old = (indices < self.history_tokens).reshape(-1, 1, indices.shape[-1], 1)
        return mx.where(old, hk, rk), mx.where(old, hv, rv)

    @property
    def state(self):
        # MLX evaluation / generation lifecycle can traverse this nested state.
        return (self.keys, self.values, self.recent_keys, self.recent_values,
                self.index_keys, self.index_position_ids)

    @state.setter
    def state(self, value):
        raise ValueError("mixed KV restoration requires its versioned snapshot")

    @property
    def nbytes(self):
        return sum(a.nbytes for item in self.state for a in _arrays(item))

    def trim(self, n):
        n = min(max(0, n), self.offset)
        self.offset -= n
        history = min(self.history_tokens, self.offset)
        self.keys = _slice(self.keys, 0, self.offset)
        self.values = _slice(self.values, 0, self.offset)
        self.recent_keys = _slice(self.recent_keys, 0, self.offset - history)
        self.recent_values = _slice(self.recent_values, 0, self.offset - history)
        self.history_tokens = history
        if self.index_keys is not None:
            self.index_keys = self.index_keys[:, :self.offset]
            self.index_position_ids = self.index_position_ids[..., :self.offset]
        return n

    def empty(self):
        return self.offset == 0

    def extract(self, idx):
        raise ValueError("mixed KV does not support batched cache extraction")

    def filter(self, batch_indices):
        raise ValueError("mixed KV does not support batched cache filtering")

    def to_quantized(self, *args, **kwargs):
        raise ValueError("mixed KV is already compressed")

    def snapshot(self, history_start=0, index_start=0, *, own=True):
        """Packed immutable primary delta, raw indexer delta, and full recent tier.

        The INT4 primary representation is written once, so its delta starts
        at the parent's end token. Only the leaf's recent tier is used.
        """
        if not 0 <= history_start <= self.offset or not 0 <= index_start <= self.offset:
            raise ValueError("invalid mixed KV snapshot delta start")
        result = {"config": mx.array([SCHEMA, self.key_bits, 4, self.group_size,
                   self.recent_tokens, self.offset, self.history_tokens,
                   int(self.mode == "tiered4")], dtype=mx.int64)}
        for name in ("keys", "values", "recent_keys", "recent_values"):
            n = self.offset if name in ("keys", "values") else self.offset - self.history_tokens
            start = history_start if name in ("keys", "values") else 0
            if start != n:
                for i, a in enumerate(_arrays(getattr(self, name))):
                    # take owns its small delta buffer even when a contiguous
                    # slice would retain a view of the whole live allocation.
                    result[f"{name}_{i}"] = (mx.take(a, mx.arange(start, n), axis=-2)
                                               if own else a[..., start:n, :])
        for name in ("index_keys", "index_position_ids"):
            a = getattr(self, name)
            if a is not None:
                if own:
                    result[name] = mx.take(a, mx.arange(index_start, self.offset),
                                           axis=1 if name == "index_keys" else -1)
                else:
                    result[name] = (a[:, index_start:self.offset] if name == "index_keys"
                                    else a[..., index_start:self.offset])
        return result

    def restore_snapshot(self, tensors, tokens):
        config = tensors.get("config")
        if config is None or config.dtype != mx.int64 or config.shape != (8,):
            raise ValueError("invalid mixed KV configuration")
        version, kb, vb, group, recent, offset, history, tiered = config.tolist()
        if (version, kb, vb, group, recent, tiered) != (
                SCHEMA, self.key_bits, 4, self.group_size, self.recent_tokens, int(self.mode == "tiered4")):
            raise ValueError("incompatible mixed KV configuration")
        if (offset != tokens or not 0 <= history <= offset or
                (not tiered and history != offset) or (tiered and offset-history >= recent+PAGE)):
            raise ValueError("invalid mixed KV offsets")
        expected = {"config", "index_keys", "index_position_ids"}
        restored = {}
        for name in ("keys", "values", "recent_keys", "recent_values"):
            bits = (kb if name == "keys" else 4) if name in ("keys", "values") else 8
            n = offset if name in ("keys", "values") else offset - history
            count = 1 if bits == 16 else 3
            names = [f"{name}_{i}" for i in range(count)]
            present = any(k in tensors for k in names)
            if not n and not present:
                restored[name] = None
                continue
            expected.update(names)
            parts = [tensors.get(k) for k in names]
            if any(a is None or a.ndim != 4 or a.shape[-2] != n for a in parts):
                raise ValueError(f"invalid mixed KV tensor: {name}")
            if bits != 16:
                data, scales, biases = parts
                if (data.dtype != mx.uint32 or scales.dtype != biases.dtype or
                    scales.shape != biases.shape or data.shape[:-1] != scales.shape[:-1] or
                    data.shape[-1] * (32 // bits) != scales.shape[-1] * group):
                    raise ValueError(f"invalid packed mixed KV layout: {name}")
            restored[name] = parts[0][...] if bits == 16 else tuple(a[...] for a in parts)
        if set(tensors) != expected:
            raise ValueError("unexpected or missing mixed KV fields")
        ik, positions = tensors["index_keys"], tensors["index_position_ids"]
        if ik.ndim != 3 or ik.shape[1] != offset or positions.shape[-1] != offset:
            raise ValueError("invalid mixed KV indexer lengths")
        for name, value in restored.items():
            setattr(self, name, value)
        self.offset, self.history_tokens = offset, history
        self.index_keys, self.index_position_ids = ik[...], positions[...]
