#!/usr/bin/env python3
"""Generate bit-exact parity vectors for the C++ engine from the real Python ops.

Every vector is produced with the SAME MLX 0.32.2 ops the Python engine uses,
so the C++ test (which loads them with mx::load) must match byte-for-byte.

Emits several cases covering production-relevant shapes:
  case0: small G32 (E=16,H=64,I=64,K=4,T=2)
  case1: Qwen expert shape G32 (H=2560->I=640, reduced experts E=8, T=1 decode)
  case2: Nemotron expert shape G64 (H=2688->I=1856, E=8, T=1 decode)

Per case we save router_{logits,indices,scores}, x, route, scores,
gw/gs/gb, uw/us/ub, dw/ds/db, expected, and a meta.json.

Usage:
  .venv/bin/python cpp/engine/tools/gen_vectors.py --out cpp/engine/tests/vectors
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def ref_router(logits: mx.array, k: int):
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    topk_mass = scores.sum(axis=-1, keepdims=True)
    return inds, scores / topk_mass


def ref_projection(x, w, s, b, route, group_size, bits):
    return mx.gather_qmm(x, w, s, b, rhs_indices=route, transpose=True,
                         group_size=group_size, bits=bits, mode="affine")


def ref_moe_block(x, route, scores, weights, group_size, bits):
    x4 = mx.expand_dims(x, (-2, -3))
    g = ref_projection(x4, weights["gw"], weights["gs"], weights["gb"], route, group_size, bits)
    u = ref_projection(x4, weights["uw"], weights["us"], weights["ub"], route, group_size, bits)
    a = u * (g * mx.sigmoid(g))
    d = ref_projection(a, weights["dw"], weights["ds"], weights["db"], route, group_size, bits)
    d = mx.squeeze(d, axis=-2)
    return (d * scores[..., None]).sum(axis=-2)


def make_projection(E, out, inp, group, rng):
    codes = rng.integers(0, 16, size=(E, out, inp), dtype=np.uint8)
    packed = np.zeros((E, out, inp // 8), dtype=np.uint32)
    for n in range(8):
        packed |= codes[..., n::8].astype(np.uint32) << (n * 4)
    scales = mx.array(rng.random((E, out, inp // group)).astype(np.float32) * 0.5 + 0.5).astype(mx.bfloat16)
    biases = mx.array(rng.random((E, out, inp // group)).astype(np.float32) * 0.2 - 0.1).astype(mx.bfloat16)
    return mx.array(packed), scales, biases


def save(path, arr):
    mx.save(str(path), arr)


def emit_relu2_case(out: Path, name: str, E, H, I, G, K, T, seed):
    """Nemotron-style relu2 MoE (single up projection, no SwiGLU gate)."""
    rng = np.random.default_rng(seed)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    logits = mx.array(rng.normal(size=(T, E)).astype(np.float32))
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    route = mx.array(np.asarray(inds).astype(np.uint32))
    x = mx.array(rng.normal(size=(T, H)).astype(np.float32) * 0.3)
    up_w, up_s, up_b = make_projection(E, I, H, G, rng)
    dn_w, dn_s, dn_b = make_projection(E, H, I, G, rng)
    x4 = mx.expand_dims(x, (-2, -3))
    u = mx.gather_qmm(x4, up_w, up_s, up_b, rhs_indices=route, transpose=True,
                      group_size=G, bits=4, mode="affine")
    a = mx.square(mx.maximum(u, mx.array(0.0)))  # relu2 (same op order as C++)
    dn = mx.gather_qmm(a, dn_w, dn_s, dn_b, rhs_indices=route, transpose=True,
                       group_size=G, bits=4, mode="affine")
    dn = mx.squeeze(dn, axis=-2)
    expected = (dn * scores[..., None]).sum(axis=-2)
    arrays = {"x": x, "route": route, "scores": scores,
              "up_w": up_w, "up_s": up_s, "up_b": up_b,
              "dn_w": dn_w, "dn_s": dn_s, "dn_b": dn_b, "expected": expected}
    for k, v in arrays.items():
        save(d / f"{k}.npy", v)
    meta = {"E": E, "H": H, "I": I, "G": G, "K": K, "T": T, "bits": 4,
            "activation": "relu2"}
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def emit_case(out: Path, name: str, E, H, I, G, K, T, seed):
    rng = np.random.default_rng(seed)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)

    logits = mx.array(rng.normal(size=(T, E)).astype(np.float32))
    inds, scores = ref_router(logits, K)
    save(d / "router_logits.npy", logits)
    save(d / "router_indices.npy", inds)
    save(d / "router_scores.npy", scores)

    route = mx.array(np.asarray(inds).astype(np.uint32))
    x = mx.array(rng.normal(size=(T, H)).astype(np.float32) * 0.3)
    # SwiGLU: gate and up both project hidden->inter; down projects inter->hidden.
    gw, gs, gb = make_projection(E, I, H, G, rng)
    uw, us, ub = make_projection(E, I, H, G, rng)
    dw, ds, db = make_projection(E, H, I, G, rng)
    weights = {"gw": gw, "gs": gs, "gb": gb, "uw": uw, "us": us, "ub": ub,
               "dw": dw, "ds": ds, "db": db}
    expected = ref_moe_block(x, route, scores, weights, G, 4)
    # RMSNorm vectors (one-centered production convention, eps=1e-6).
    norm_x = mx.array(rng.normal(size=(T, H)).astype(np.float32))
    norm_w = mx.array(rng.random(H).astype(np.float32) + 0.5)  # one-centered gains
    ny = norm_x.astype(mx.float32)
    ny = ny * mx.rsqrt(mx.mean(mx.square(ny), axis=-1, keepdims=True) + 1e-6)
    norm_expected = (ny * norm_w).astype(norm_x.dtype)
    # Greedy sample vector: production takes ONE logit row and returns shape (1,).
    sample_logits = mx.array(rng.normal(size=(E,)).astype(np.float32))
    greedy = mx.astype(mx.reshape(mx.argmax(sample_logits.reshape(-1), axis=-1), (1,)), mx.uint32)
    # GDN chains (compile_gdn.py). q/k normalization + gates.
    q = mx.array(rng.normal(size=(T, 8, 64)).astype(np.float32))
    k = mx.array(rng.normal(size=(T, 8, 64)).astype(np.float32))
    scale = q.shape[-1] ** -0.5
    qn = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
    kn = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
    qn, kn = qn * scale, kn
    gy = mx.array(rng.normal(size=(T, 64)).astype(np.float32))
    gg = mx.array(rng.normal(size=(T, 64)).astype(np.float32))
    gate_sig = gy.astype(mx.float32) * mx.sigmoid(gg.astype(mx.float32))
    import mlx.nn as nn
    gate_silu = gy.astype(mx.float32) * nn.silu(gg.astype(mx.float32))
    # Gated residual (hyper-connection), one-centered grouped norm.
    HC, HD, LOW, EPS = 4, 8, 6, 1e-6
    hc_hidden = HC * HD
    hyp = mx.array(rng.normal(size=(1, hc_hidden)).astype(np.float32))
    nw = mx.array((rng.random(hc_hidden).astype(np.float32) + 0.5))
    nw2 = mx.reshape(nw, (HC, HD))
    # Manual grouped norm (patch_rmsnorm._rms_norm with group_size), one-centered.
    yg = mx.reshape(hyp, (1, HC, HD)).astype(mx.float32)
    yg = yg * mx.rsqrt(mx.mean(mx.square(yg), axis=-1, keepdims=True) + EPS)
    yg = yg * nw2.astype(mx.float32)   # one-centered gain (no +1)
    normed = mx.reshape(yg, (1, hc_hidden)).astype(hyp.dtype)
    dw = mx.array(rng.normal(size=(LOW, hc_hidden)).astype(np.float32) * 0.1)
    uw = mx.array(rng.normal(size=(hc_hidden, LOW)).astype(np.float32) * 0.1)
    iw = mx.array(rng.normal(size=(HC, hc_hidden)).astype(np.float32) * 0.1)
    low = mx.matmul(normed, dw.T) / HC
    mix = mx.sigmoid(mx.matmul(nn.silu(low), uw.T))
    mixg = mx.reshape(mix, (1, HC, HD))
    normg = mx.reshape(normed, (1, HC, HD))
    gr_mixed = mx.mean(mixg * normg, axis=-2)
    gr_inj = 2 * mx.sigmoid(mx.matmul(normed, iw.T) / HC)
    # Gated RMSNorm (sigmoid activation) at end of GDN.
    gx = mx.array(rng.normal(size=(1, 16)).astype(np.float32))
    gw = mx.array(np.ones(16).astype(np.float32))
    rgg = mx.array(rng.normal(size=(1, 16)).astype(np.float32))
    gy2 = mx.fast.rms_norm(gx, gw, 1e-6).astype(mx.float32)
    gated_sigmoid = (gy2 * mx.sigmoid(rgg.astype(mx.float32))).astype(gx.dtype)
    # GDN delta-rule step — EXACT kernel path (production decode), using the
    # real mlx_vlm kernel + compute_g_beta so C++ can be checked bit-for-bit.
    from mlx_vlm.models.qwen3_5.gated_delta import (
        gated_delta_kernel, _compute_g_beta)
    Bd, Td, Hkd, Hvd, Dkd, Dvd = 1, 1, 4, 8, 128, 128  # small but real dims
    dq = mx.array(rng.normal(size=(Bd, Td, Hkd, Dkd)).astype(np.float32) * 0.1)
    dk_ = mx.array(rng.normal(size=(Bd, Td, Hkd, Dkd)).astype(np.float32) * 0.1)
    dv = mx.array(rng.normal(size=(Bd, Td, Hvd, Dvd)).astype(np.float32) * 0.1)
    A_log = mx.array(rng.normal(size=(Hvd,)).astype(np.float32))
    da = mx.array(rng.normal(size=(Bd, Td, Hvd)).astype(np.float32) * 0.01)
    db = mx.array(rng.normal(size=(Bd, Td, Hvd)).astype(np.float32) * 0.01)
    dt_bias = mx.ones((Hvd,))
    dg, dbeta = _compute_g_beta(A_log, da, db, dt_bias)
    dstate = mx.array(rng.normal(size=(Bd, Hvd, Dvd, Dkd)).astype(np.float32) * 0.01)
    dy, dstate_out = gated_delta_kernel(dq, dk_, dv, dg, dbeta, dstate)
    arrays = {"x": x, "route": route, "scores": scores, **weights, "expected": expected,
              "norm_x": norm_x, "norm_w": norm_w, "norm_expected": norm_expected,
              "sample_logits": sample_logits, "greedy": greedy,
              "gdn_q": q, "gdn_k": k, "gdn_qn": qn, "gdn_kn": kn,
              "gdn_gy": gy, "gdn_gg": gg, "gdn_sig": gate_sig, "gdn_silu": gate_silu,
              "gr_hyp": hyp, "gr_nw": nw, "gr_dw": dw, "gr_uw": uw, "gr_iw": iw,
              "gr_mixed": gr_mixed, "gr_inj": gr_inj,
              "rng_gx": gx, "rng_gw": gw, "rng_gg": rgg, "rng_gated": gated_sigmoid,
              # GDN delta-rule step — EXACT kernel path reference (production decode).
              # Inputs (dq/dk_/dv/dstate) + g/beta gates (from A_log,da,db,dt_bias)
              # and the kernel outputs (dy, dstate_out) for byte-for-byte C++ parity.
              "gdn_delta_q": dq, "gdn_delta_k": dk_, "gdn_delta_v": dv,
              "gdn_delta_A_log": A_log, "gdn_delta_a": da, "gdn_delta_b": db,
              "gdn_delta_dt_bias": dt_bias,
              "gdn_delta_g": dg, "gdn_delta_beta": dbeta,
              "gdn_delta_state": dstate,
              "gdn_delta_y": dy, "gdn_delta_state_out": dstate_out}
    for k, v in arrays.items():
        save(d / f"{k}.npy", v)
    meta = {"E": E, "H": H, "I": I, "G": G, "K": K, "T": T, "bits": 4,
            "mlx_version": mx.__version__}
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/vectors")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    def case(name, E, H, I, G, K, T, seed):
        m = emit_case(out, name, E, H, I, G, K, T, seed)
        return {"dir": name, **m}

    cases = [
        case("case0_g32_small", 16, 64, 64, 32, 4, 2, 1234),
        case("case1_qwen_g32", 12, 2560, 640, 32, 10, 1, 77),
        case("case2_nemotron_g64", 8, 2688, 1856, 64, 6, 1, 99),
    ]
    relu2 = emit_relu2_case(out, "case3_nemotron_relu2", 16, 2688, 1856, 64, 6, 1, 555)
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    # Backward-compatible flat vectors for the existing flat parity test (case0).
    for f in (out / "case0_g32_small").glob("*.npy"):
        (out / f.name).write_bytes(f.read_bytes())
    (out / "meta.json").write_text(
        (out / "case0_g32_small" / "meta.json").read_text())
    print("wrote 3 parity cases:", ", ".join(c["dir"] for c in cases))


if __name__ == "__main__":
    main()
