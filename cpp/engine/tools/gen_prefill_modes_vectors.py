#!/usr/bin/env python3
"""Synthetic BIT-EXACT oracle: batched (S>1) vs sequential (S=1) prefill.

QUESTION
--------
Does the tool's sequential S=1 prefill (what real_checkpoint_generate.cpp does:
one `run_step` per prompt token, GDN via `gdn_layer_forward`) differ from
production's batched whole-prompt prefill (one `Qwen4ExpGatedDeltaNet.__call__`
call with S>1)?  This script answers that with a small, seeded, synthetic model
run on the M4 GPU (the same MLX 0.32.2 build the C++ engine links against).

For a fixed random prompt of length S it computes, from the SAME initial cache
and weights:

  batched   : ONE call  model(x[S], mask=None, cache=cache0)
              -> full matmul, FULL depthwise conv1d, fused delta kernel (T=S)
  sequential: S calls   model(x[t:t+1], mask=None, cache=cache_t)
              -> per-token matmul, DECODE depthwise conv
                 (_causal_conv1d_decode), fused delta kernel (T=1 each)

It reports max-abs-diff / Pearson correlation / bit-identity of the final
hidden states (and of the final conv/recurrent cache), plus a stage-by-stage
attribution (projections / conv / delta kernel) so the source of any gap is
unambiguous.

It also dumps the BATCHED and SEQUENTIAL oracles as .npy + meta.json under
--out so `tests/test_prefill_modes.cpp` can replay both inside C++:
  * C++ gdn_prefill(S=S)       must be bit-exact to the Python BATCHED output
  * C++ gdn_layer_forward x S  must be bit-exact to the Python SEQUENTIAL output
  * C++ gdn_prefill(S=1) x S   is also checked (isolates the primitive itself)

Finally it performs the same batched-vs-sequential comparison for the
full-attention DENSE (causal, indexer=None) fallback and reports the numbers.

Usage:
    .venv/bin/python cpp/engine/tools/gen_prefill_modes_vectors.py \
        --out cpp/engine/tests/vectors_prefill_modes
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet  # noqa: E402
from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update  # noqa: E402


def np_of(a):
    mx.eval(a)
    return np.asarray(a)


def stats(a, b, label):
    """max abs diff / Pearson r / bit-identical for two float32 arrays."""
    na, nb = np_of(a).astype(np.float32), np_of(b).astype(np.float32)
    assert na.shape == nb.shape, (label, na.shape, nb.shape)
    d = np.abs(na - nb)
    maxd = float(d.max()) if d.size else 0.0
    if d.size > 1 and na.std() > 0 and nb.std() > 0:
        corr = float(np.corrcoef(na.ravel(), nb.ravel())[0, 1])
    else:
        corr = float("nan")
    # true bit identity via uint32 reinterpretation
    ba = na.view(np.uint32) if na.dtype == np.float32 else na.tobytes()
    bb = nb.view(np.uint32) if nb.dtype == np.float32 else nb.tobytes()
    bit = bool(np.array_equal(ba, bb))
    print(
        f"  {label:<28s} max_abs_diff={maxd:.9g}  corr={corr:+.9f}  "
        f"bit_identical={bit}",
        flush=True,
    )
    return {"max_abs_diff": maxd, "correlation": corr, "bit_identical": bit}


def make_config(args):
    Hk, Hv = args.hk, args.hv
    Dk, Dv = args.dk, args.dv
    return SimpleNamespace(
        hidden_size=args.hidden,
        linear_num_value_heads=Hv,
        linear_num_key_heads=Hk,
        linear_key_head_dim=Dk,
        linear_value_head_dim=Dv,
        linear_conv_kernel_dim=args.kernel,
        rms_norm_eps=1e-6,
        output_gate_type="sigmoid",
        hidden_act="silu",
        key_dim_total=Hk * Dk,
        value_dim_total=Hv * Dv,
        num_k_heads=Hk,
        num_v_heads=Hv,
        # full-attention config fields (small dense fallback)
        n_heads=args.n_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )


def make_model(cfg, seed):
    conv_dim = 2 * cfg.key_dim_total + cfg.value_dim_total
    rng = np.random.default_rng(seed)

    def rand(shape, scale=0.1, dtype=mx.float32):
        return mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(dtype)

    model = Qwen4ExpGatedDeltaNet(cfg)
    model.in_proj_qkv.weight = rand((conv_dim, cfg.hidden_size), dtype=mx.bfloat16)
    model.in_proj_z.weight = rand((cfg.value_dim_total, cfg.hidden_size), dtype=mx.bfloat16)
    model.in_proj_b.weight = rand((cfg.num_v_heads, cfg.hidden_size), dtype=mx.bfloat16)
    model.in_proj_a.weight = rand((cfg.num_v_heads, cfg.hidden_size), dtype=mx.bfloat16)
    model.conv1d.weight = rand((conv_dim, cfg.linear_conv_kernel_dim, 1), dtype=mx.bfloat16)
    model.A_log = rand((cfg.num_v_heads,), scale=1.0)     # f32
    model.dt_bias = rand((cfg.num_v_heads,), scale=0.5)   # f32
    model.norm.weight = mx.array(np.ones(cfg.linear_value_head_dim, dtype=np.float32))
    model.out_proj.weight = rand((cfg.hidden_size, cfg.value_dim_total), dtype=mx.bfloat16)
    model.eval()
    return model


def fresh_cache(rng, cfg, conv_dim, batch=1):
    conv = mx.array(
        (rng.standard_normal((batch, cfg.linear_conv_kernel_dim - 1, conv_dim)) * 0.1).astype(np.float32)
    )
    rec = mx.array(
        (rng.standard_normal((batch, cfg.num_v_heads, cfg.linear_value_head_dim,
                              cfg.linear_key_head_dim)) * 0.05).astype(np.float32)
    )
    return [conv, rec]


def run_gdn_modes(model, cfg, x, cache0, label, S):
    """Run batched once and S sequential single-token calls; compare + report."""
    B = x.shape[0]

    # ---- batched (production) ---------------------------------------------
    cb = [cache0[0], cache0[1]]  # layer reassigns slots
    y_batched = model(x, mask=None, cache=cb)
    mx.eval(y_batched, cb[0], cb[1])

    # ---- sequential (tool) --------------------------------------------------
    cs = [cache0[0], cache0[1]]
    steps = []
    for t in range(S):
        steps.append(model(x[:, t : t + 1], mask=None, cache=cs))
    y_seq = mx.concatenate(steps, axis=1)
    mx.eval(y_seq, cs[0], cs[1])

    print(f"\n[GDN {label} S={S}] batched(1x S={S}) vs sequential(S x 1):", flush=True)
    out = stats(y_batched, y_seq, "final hidden [B,S,H]")
    stats(y_batched[:, -1:, :], y_seq[:, -1:, :], "last-token hidden")
    stats(cb[0], cs[0], "final conv_state")
    stats(cb[1], cs[1], "final recurrent_state")

    # ---- stage attribution -------------------------------------------------
    K = cfg.linear_conv_kernel_dim
    mixed_b = model.in_proj_qkv(x)
    mixed_s = mx.concatenate([model.in_proj_qkv(x[:, t : t + 1]) for t in range(S)], axis=1)
    print("  stage attribution:", flush=True)
    stats(mixed_b, mixed_s, "in_proj_qkv (full vs per-tok)")

    # conv: batched uses full self.conv1d; sequential uses decode conv.
    conv_in_b = mx.concatenate([cache0[0], mixed_b], axis=1)
    conv_out_b = nn.silu(model.conv1d(conv_in_b))
    conv_in_s = mx.concatenate([cache0[0], mixed_s], axis=1)
    conv_out_s_list = []
    for t in range(S):
        cin = conv_in_s[:, t : t + K, :]
        conv_out_s_list.append(nn.silu(model._causal_conv1d_decode(cin)))
    conv_out_s = mx.concatenate(conv_out_s_list, axis=1)
    stats(conv_out_b, conv_out_s, "conv_out (full vs decode)")

    # delta kernel: same q/k/v fed both ways (T=S kernel vs S x T=1 kernel).
    conv_dim = 2 * cfg.key_dim_total + cfg.value_dim_total
    Hk, Hv = cfg.num_k_heads, cfg.num_v_heads
    Dk, Dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    Bsz = x.shape[0]
    qkv = mx.split(conv_out_b, [cfg.key_dim_total, 2 * cfg.key_dim_total], -1)
    q = qkv[0].reshape(Bsz, S, Hk, Dk)
    k = qkv[1].reshape(Bsz, S, Hk, Dk)
    v = qkv[2].reshape(Bsz, S, Hv, Dv)
    z = model.in_proj_z(x).reshape(Bsz, S, Hv, Dv)
    a = model.in_proj_a(x)
    b = model.in_proj_b(x)
    q, k = model._normalize_qk(q, k)
    yb, _ = gated_delta_update(q, k, v, a, b, model.A_log, model.dt_bias,
                               cache0[1], None, use_kernel=True)
    st = cache0[1]
    ys_list = []
    for t in range(S):
        yt, st = gated_delta_update(q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1],
                                    a[:, t : t + 1], b[:, t : t + 1], model.A_log,
                                    model.dt_bias, st, None, use_kernel=True)
        ys_list.append(yt)
    ys = mx.concatenate(ys_list, axis=1)
    stats(yb, ys, "delta y (T=S vs SxT=1)")

    return {
        "batched_output": np_of(y_batched),
        "sequential_output": np_of(y_seq),
        "batched_conv": np_of(cb[0]),
        "batched_rec": np_of(cb[1]),
        "sequential_conv": np_of(cs[0]),
        "sequential_rec": np_of(cs[1]),
        "comparison": out,
    }


# --------------------------------------------------------------------------- //
# Full-attention DENSE (causal, indexer=None) fallback oracle.                    //
# Mirrors real_checkpoint_generate.cpp::full_attention_dense with f32 weights.    //
# --------------------------------------------------------------------------- //
class DenseAttention(nn.Module):
    """Small standalone dense QSA-attention block (no indexer, causal dense)."""

    def __init__(self, cfg, seed):
        super().__init__()
        H, Hq, Hkv, D = (cfg.hidden_size, cfg.n_heads, cfg.kv_heads, cfg.head_dim)
        rng = np.random.default_rng(seed + 777)

        def rand(shape, scale=0.1, dtype=mx.float32):
            return mx.array((rng.standard_normal(shape) * scale).astype(np.float32)).astype(dtype)

        self.q_w = rand((Hq * 2 * D, H), dtype=mx.bfloat16)
        self.k_w = rand((Hkv * D, H), dtype=mx.bfloat16)
        self.v_w = rand((Hkv * D, H), dtype=mx.bfloat16)
        self.o_w = rand((H, Hq * D), dtype=mx.bfloat16)
        self.q_norm_w = rand((D,), scale=0.0)   # zero-centered gain (qsa adds 1.0)
        self.k_norm_w = rand((D,), scale=0.0)
        self.Hq, self.Hkv, self.D = Hq, Hkv, D

    def _rms(self, x, w, eps=1e-6):
        # engine.hpp qsa_rms_norm: (x / rms(x)) * (1 + w)
        y = mx.astype(x, mx.float32)
        y = y * mx.rsqrt(mx.mean(mx.square(y), -1, keepdims=True) + eps)
        return mx.astype(y * (1.0 + mx.astype(w, mx.float32)), x.dtype)

    def _qkv(self, x):
        B, S, _ = x.shape
        Hq, Hkv, D = self.Hq, self.Hkv, self.D
        qproj = mx.matmul(x, self.q_w.T)
        kproj = mx.matmul(x, self.k_w.T)
        vproj = mx.matmul(x, self.v_w.T)
        q4 = qproj.reshape(B, S, Hq, 2 * D)
        q, gate = mx.split(q4, 2, -1)
        gate = gate.reshape(B, S, Hq * D)
        q = mx.transpose(self._rms(q, self.q_norm_w), (0, 2, 1, 3))
        k = mx.transpose(self._rms(kproj.reshape(B, S, Hkv, D), self.k_norm_w), (0, 2, 1, 3))
        v = mx.transpose(vproj.reshape(B, S, Hkv, D), (0, 2, 1, 3))
        return q, k, v, gate

    def __call__(self, x, cache):
        """Mirrors full_attention_dense: append k/v to cache, SDPA causal, o_proj."""
        B, S, _ = x.shape
        q, k, v, gate = self._qkv(x)
        if cache["keys"] is None:
            cache["keys"], cache["values"] = k, v
        else:
            cache["keys"] = mx.concatenate([cache["keys"], k], axis=2)
            cache["values"] = mx.concatenate([cache["values"], v], axis=2)
        scale = 1.0 / math.sqrt(self.D)
        out = mx.fast.scaled_dot_product_attention(
            q, cache["keys"], cache["values"], scale=scale, mask="causal"
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, S, self.Hq * self.D)
        return mx.matmul(out * mx.sigmoid(gate), self.o_w.T)


def run_fa_modes(cfg, x, S, label):
    model = DenseAttention(cfg, seed=20261118)
    y_b = model(x, {"keys": None, "values": None})
    cache = {"keys": None, "values": None}
    y_s = mx.concatenate(
        [model(x[:, t : t + 1], cache) for t in range(S)], axis=1
    )
    mx.eval(y_b, y_s)
    print(f"\n[FullAttn dense {label} S={S}] batched vs sequential:", flush=True)
    return stats(y_b, y_s, "final hidden [B,S,H]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors_prefill_modes")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hk", type=int, default=2)
    ap.add_argument("--hv", type=int, default=4)
    ap.add_argument("--dk", type=int, default=32, help="must be multiple of 32 for Metal kernel")
    ap.add_argument("--dv", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=4)
    ap.add_argument("--S", type=int, default=8, help="prompt length for the comparison report")
    ap.add_argument("--dump-S", type=int, default=16, help="prompt length for the dumped C++ vectors")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=8)
    ap.add_argument("--no-full-attn", action="store_true")
    args = ap.parse_args()

    mx.set_default_device(mx.gpu)
    assert mx.metal.is_available(), "requires M4 GPU Metal kernel path"
    assert args.hv % args.hk == 0 and args.dk % 32 == 0

    cfg = make_config(args)
    conv_dim = 2 * cfg.key_dim_total + cfg.value_dim_total
    model = make_model(cfg, args.seed)
    rng = np.random.default_rng(args.seed + 1)

    print("=" * 78, flush=True)
    print("SYNTHETIC PREFILL MODES ORACLE (small config, M4 GPU, MLX 0.32.2)", flush=True)
    print(f"hidden={args.hidden} key_dim={cfg.key_dim_total} value_dim={cfg.value_dim_total} "
          f"Hk={args.hk} Hv={args.hv} Dk={args.dk} Dv={args.dv} K={args.kernel}", flush=True)
    print("=" * 78, flush=True)

    # ---- comparison report (S from --S) -----------------------------------
    x8 = mx.array((rng.standard_normal((1, args.S, args.hidden)) * 0.1).astype(np.float32))
    c8 = fresh_cache(rng, cfg, conv_dim)
    gdn_report = run_gdn_modes(model, cfg, x8, c8, "comparison", args.S)

    fa_report = None
    if not args.no_full_attn:
        fa_report = run_fa_modes(cfg, x8, args.S, "comparison")

    # ---- dump vectors for the C++ test (--dump-S) --------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shared = {
        "w_qkv": model.in_proj_qkv.weight,
        "w_z": model.in_proj_z.weight,
        "w_b": model.in_proj_b.weight,
        "w_a": model.in_proj_a.weight,
        "conv_w": model.conv1d.weight,
        "A_log": model.A_log,
        "dt_bias": model.dt_bias,
        "norm_w": model.norm.weight,
        "w_out": model.out_proj.weight,
    }
    for name, arr in shared.items():
        mx.save(str(out / f"{name}.npy"), arr)

    xd = mx.array((rng.standard_normal((1, args.dump_S, args.hidden)) * 0.1).astype(np.float32))
    cd = fresh_cache(rng, cfg, conv_dim)

    def dump_case(case, y, cache_after):
        cdir = out / case
        cdir.mkdir(parents=True, exist_ok=True)
        mx.save(str(cdir / "x.npy"), xd)
        mx.save(str(cdir / "i_conv_state.npy"), cd[0])
        mx.save(str(cdir / "i_rec_state.npy"), cd[1])
        mx.save(str(cdir / "expected_output.npy"), y)
        mx.save(str(cdir / "expected_conv_state.npy"), cache_after[0])
        mx.save(str(cdir / "expected_rec_state.npy"), cache_after[1])
        print(f"wrote {case}/: out={y.shape} conv={cache_after[0].shape} rec={cache_after[1].shape}",
              flush=True)

    # batched
    cbb = [cd[0], cd[1]]
    yb = model(xd, mask=None, cache=cbb)
    mx.eval(yb, cbb[0], cbb[1])
    dump_case("batched", yb, cbb)

    # sequential
    css = [cd[0], cd[1]]
    steps = [model(xd[:, t : t + 1], mask=None, cache=css) for t in range(args.dump_S)]
    ys = mx.concatenate(steps, axis=1)
    mx.eval(ys, css[0], css[1])
    dump_case("sequential", ys, css)

    # extra report at dump-S too (this is the number the C++ test will reproduce)
    print(f"\n[GDN dump S={args.dump_S}] batched vs sequential:", flush=True)
    stats(yb, ys, "final hidden [B,S,H]")

    # ---- dump full-attention DENSE oracle ---------------------------------
    fa = DenseAttention(cfg, seed=20261118)
    fa_dir = out / "full_attn"
    fa_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in {
        "fa_q_w": fa.q_w, "fa_k_w": fa.k_w, "fa_v_w": fa.v_w,
        "fa_o_w": fa.o_w, "fa_q_norm_w": fa.q_norm_w, "fa_k_norm_w": fa.k_norm_w,
    }.items():
        mx.save(str(fa_dir / f"{name}.npy"), arr)
    fa_yb = fa(xd, {"keys": None, "values": None})
    fcache = {"keys": None, "values": None}
    fa_ys = mx.concatenate(
        [fa(xd[:, t : t + 1], fcache) for t in range(args.dump_S)], axis=1
    )
    mx.eval(fa_yb, fa_ys)
    for case, y in (("batched", fa_yb), ("sequential", fa_ys)):
        cdir = fa_dir / case
        cdir.mkdir(parents=True, exist_ok=True)
        mx.save(str(cdir / "x.npy"), xd)
        mx.save(str(cdir / "expected_output.npy"), y)
    print(f"wrote full_attn/{{batched,sequential}}/: out={fa_yb.shape}", flush=True)
    print(f"\n[FullAttn dump S={args.dump_S}] batched vs sequential:", flush=True)
    stats(fa_yb, fa_ys, "final hidden [B,S,H]")

    meta = {
        "hidden_size": args.hidden,
        "key_dim": cfg.key_dim_total,
        "value_dim": cfg.value_dim_total,
        "num_k_heads": args.hk,
        "num_v_heads": args.hv,
        "head_k_dim": args.dk,
        "head_v_dim": args.dv,
        "conv_kernel": args.kernel,
        "eps": 1e-6,
        "output_activation": "sigmoid",
        "batch": 1,
        "comparison_S": args.S,
        "dump_S": args.dump_S,
        "seed": args.seed,
        "n_heads": args.n_heads,
        "kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "cases": ["batched", "sequential"],
        "gdn_comparison": gdn_report["comparison"],
        "full_attn_comparison": fa_report,
        "note": (
            "batched/ = ONE Python Qwen4ExpGatedDeltaNet.__call__ (full matmul + "
            "full conv1d + fused delta kernel T=S).  sequential/ = S Python "
            "single-token __call__s (per-token matmul + _causal_conv1d_decode + "
            "fused delta kernel T=1).  Both start from identical x/i_conv/i_rec."
        ),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nwrote shared weights and meta.json to {out}", flush=True)
    print("\n=== KEY NUMBER (GDN, prompt length {}) ===".format(args.S), flush=True)
    c = gdn_report["comparison"]
    print(f"max_abs_diff = {c['max_abs_diff']:.9g}   correlation = {c['correlation']:+.9f}   "
          f"bit_identical = {c['bit_identical']}", flush=True)


if __name__ == "__main__":
    main()
