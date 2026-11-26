#!/usr/bin/env python3
"""OWNER-RUN diagnostic: dump GDN intermediates at layer 20 from the REAL model.

Mirrors ``dump_python_layers.py`` (same backend, greedy setup, deterministic
16-token prompt) but hooks the *linear_attn* (Qwen4ExpGatedDeltaNet) of ONE
target layer (default 20) and records EVERY intermediate of its full-sequence
(S=16) forward, plus the weights the C++ probe needs to replay the same op.

The C++ counterpart ``probe_layer_intermediates.cpp`` reads these raw dumps and
recomputes each intermediate with the C++ primitives. Comparing the two files
localizes the first divergent op *inside* the GDN branch at REAL bf16 dims.

Intermediates (all raw little-endian f32, no header, unless dtype=uint32):
  mixed_input  [1,16,2560]  input to linear_attn (attn hyper-connection output)
  conv_state_in[1,3,10240]  GDN conv cache before the forward
  rec_state_in [1,48,128,128] f32 recurrent-state cache before the forward
  mixed_qkv    [1,16,10240] in_proj_qkv result
  z            [1,16,48,128] in_proj_z result (reshaped)
  a, b         [1,16,48]    in_proj_a / in_proj_b result
  conv_input   [1,19,10240] concat([conv_state, mixed_qkv])
  conv_out     [1,16,10240] silu(conv1d(conv_input))
  q_raw,k_raw  [1,16,16,128] pre-normalization q/k
  q_norm,k_norm[1,16,16,128] post-_normalize_qk q/k
  g            [1,16,48] f32 compute_g
  beta         [1,16,48] sigmoid(b)
  delta_y      [1,16,48,128] fused-kernel output y (bf16)
  state_out    [1,48,128,128] f32 fused-kernel recurrent state
  normed       [1,16,48,128] RMSNormGated output (bf16)
  out          [1,16,2560] out_proj output == GDN branch (bf16)
  real_out     [1,16,2560] output of the genuine (unreplicated) __call__

Weights needed by the C++ replay (only layers' linear_attn):
  w_conv1d.weight [10240,4,1] bf16 ; w_A_log [48] bf16 ; w_dt_bias [48] bf16
  w_norm.weight [128] bf16 ; w_outproj.weight/scales/biases (uint32/bf16)

The replica calls the layer's REAL submodules (in_proj_*, conv1d, _normalize_qk,
gated_delta_kernel, norm, out_proj) with the REAL checkpoint weights, so it is
the same computation as production; it only inserts capture points. A manifest
records shape/dtype/sha256/sample values for every file.

DO NOT run against the real checkpoint unless you are the owner and have the
memory guard in place. This script never writes to a shared source file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = "/Users/vincent/.local/share/opencode/worktree/dd9a0e/cpp-engine-qdense"
VENV_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
for _p in (REPO, VENV_SITE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet  # noqa: E402
from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet  # noqa: E402
from mlx_vlm.models.qwen3_5.gated_delta import (  # noqa: E402
    gated_delta_kernel, _compute_g_beta,
)

PROMPT_SEED = 20261117

_ORIG_CALL = [None]
_STATE = [None]  # dict with target id, capture sink


def _to_np(a) -> np.ndarray:
    mx.eval(a)
    if getattr(a, "dtype", None) in (mx.bfloat16, mx.float16):
        a = mx.astype(a, mx.float32)
    return np.asarray(a)


def _replica(module, hidden, mask, cache0, cache1):
    """Exact copy of Qwen3_5GatedDeltaNet.__call__ with capture points."""
    B, S, _ = hidden.shape
    Hv, Hk = module.num_v_heads, module.num_k_heads
    Dk, Dv = module.head_k_dim, module.head_v_dim
    K = module.conv_kernel_size
    conv_dim = module.conv_dim

    mixed_qkv = module.in_proj_qkv(hidden)
    z = module.in_proj_z(hidden).reshape(B, S, -1, Dv)
    b = module.in_proj_b(hidden)
    a = module.in_proj_a(hidden)

    m = mask
    if m is not None:
        if m.shape[0] != B:
            m = None
        else:
            mixed_qkv = mx.where(m[..., None], mixed_qkv, 0)

    conv_state = cache0
    if conv_state is None or conv_state.shape[0] != B:
        conv_state = mx.zeros((B, K - 1, conv_dim), dtype=hidden.dtype)
    conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)
    new_conv_state = mx.contiguous(conv_input[:, -(K - 1):, :])

    # S=16 -> full conv path (not the S==1 decode fast path)
    conv_out = nn.silu(module.conv1d(conv_input))

    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [module.key_dim, 2 * module.key_dim], -1),
            [Hk, Hk, Hv], [Dk, Dk, Dv],
        )
    ]
    q_raw, k_raw = q, k
    state = cache1
    if state is not None and state.shape[0] != B:
        state = None
    q, k = module._normalize_qk(q, k)
    q_norm, k_norm = q, k

    g, beta = _compute_g_beta(module.A_log, a, b, module.dt_bias)
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    # Production calls gated_delta_update(..., use_kernel=True) -> kernel path.
    # Call the kernel directly with the same g/beta to expose y + state.
    delta_y, state_out = gated_delta_kernel(
        q, k, v, g, beta, state, m
    )
    normed = module.norm(delta_y, z)
    out = module.out_proj(normed.reshape(B, S, -1))

    return dict(
        mixed_qkv=mixed_qkv, z=z, a=a, b=b,
        conv_input=conv_input, new_conv_state=new_conv_state, conv_out=conv_out,
        q_raw=q_raw, k_raw=k_raw, q_norm=q_norm, k_norm=k_norm,
        g=g, beta=beta, delta_y=delta_y, state_out=state_out,
        normed=normed, out=out,
    )


def install(target_layer_idx: int, out_dir: Path):
    # The backend wraps Qwen4ExpGatedDeltaNet.__call__ (linear_state.py) and
    # captures the ORIGINAL Qwen3_5GatedDeltaNet.__call__ at init, so patching
    # the base class is bypassed. Patch the concrete class's wrapper instead.
    _ORIG_CALL[0] = Qwen4ExpGatedDeltaNet.__call__

    def capturing(self, inputs, mask=None, cache=None):
        st = _STATE[0]
        target = st["target"]
        st.setdefault("call_log", []).append(
            (type(self).__name__, id(self) == id(target), int(inputs.shape[1]))
        )
        is_target = self is target and st["prefill_calls"] == 0 and inputs.shape[1] > 1
        pre = None
        if is_target:
            st["prefill_calls"] += 1
            # IMPORTANT: capture the PRE-forward cache. The real forward below
            # overwrites cache[0]/cache[1] with the updated state. MLX arrays are
            # immutable, so holding the old references is enough.
            c0 = cache[0] if cache is not None else None
            c1 = cache[1] if cache is not None else None
            pre = dict(mixed_input=inputs, mask=mask,
                       conv_state_in=c0, rec_state_in=c1)
        out = _ORIG_CALL[0](self, inputs, mask=mask, cache=cache)
        if is_target:
            try:
                cap = dict(pre)
                cap.update(_replica(self, inputs, mask,
                                    pre["conv_state_in"], pre["rec_state_in"]))
                cap["real_out"] = out
                st["capture"] = cap
                st["module"] = self
            except Exception as exc:  # pragma: no cover
                st["error"] = repr(exc)
                import traceback
                traceback.print_exc()
        return out

    Qwen4ExpGatedDeltaNet.__call__ = capturing


def uninstall():
    if _ORIG_CALL[0] is not None:
        Qwen4ExpGatedDeltaNet.__call__ = _ORIG_CALL[0]
        _ORIG_CALL[0] = None


def _dump_array(out_dir: Path, manifest: dict, name: str, a):
    a = mx.eval(a) or a
    dtype = getattr(a, "dtype", None)
    shape = [int(v) for v in a.shape]
    if dtype == mx.uint32:
        raw = np.asarray(a, dtype=np.uint32).tobytes()
        dtype_tag = "uint32"
    else:
        raw = _to_np(a).astype("<f4").tobytes()
        dtype_tag = str(dtype)
    (out_dir / (name + ".bin")).write_bytes(raw)
    manifest["files"][name] = {
        "file": name + ".bin", "shape": shape, "dtype": dtype_tag,
        "n": int(np.prod(shape)), "sha256": hashlib.sha256(raw).hexdigest(),
        "first": np.frombuffer(raw, dtype="<u4" if dtype_tag == "uint32" else "<f4")[:4].tolist(),
    }


def run_real(args):
    from veystrix.backends.flashnext import FlashNextBackend

    os.environ.setdefault("VEYSTRIX_SAMPLING_GREEDY", "1")
    backend = FlashNextBackend(model_path=args.checkpoint)
    if getattr(backend, "sampling", None) is not None:
        try:
            backend.sampling.greedy = True
        except Exception:
            pass

    layers = backend.language.model.layers
    assert len(layers) > args.layer, "target layer out of range"
    target = layers[args.layer].linear_attn
    assert isinstance(target, Qwen3_5GatedDeltaNet), "target layer not linear/GDN"
    _STATE[0] = {"target": target, "prefill_calls": 0, "capture": None,
                 "error": None, "module": None}

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    install(args.layer, out_dir)
    try:
        rng = np.random.default_rng(PROMPT_SEED)
        prompt = rng.integers(0, 200000, size=args.prompt_tokens).tolist()
        backend.append_tokens(prompt)
        backend.generate(args.tokens, on_decode_token=lambda *_: None)
    finally:
        uninstall()

    st = _STATE[0]
    if st["capture"] is None:
        cl = st.get("call_log", [])
        n_target = sum(1 for c in cl if c[1])
        print("HOOK call_log (first 40):", cl[:40], "... total", len(cl),
              "target_calls", n_target)
        raise RuntimeError(f"layer {args.layer} GDN not captured: {st['error']}")
    cap = st["capture"]
    module = st["module"]

    # A fresh prompt has no incoming GDN cache: materialize the zeros the real
    # forward uses so the C++ probe has concrete inputs to replay.
    K = module.conv_kernel_size
    conv_dim = module.conv_dim
    Hv, Dk, Dv = module.num_v_heads, module.head_k_dim, module.head_v_dim
    if cap["conv_state_in"] is None:
        cap["conv_state_in"] = mx.zeros((1, K - 1, conv_dim), dtype=mx.bfloat16)
    if cap["rec_state_in"] is None:
        cap["rec_state_in"] = mx.zeros((1, Hv, Dv, Dk), dtype=mx.float32)

    manifest = {
        "purpose": "GDN intermediates, layer %d, last S>1 prefill (REAL checkpoint)" % args.layer,
        "checkpoint": args.checkpoint, "layer": args.layer,
        "prompt_seed": PROMPT_SEED, "prompt_tokens": prompt,
        "mlx_version": mx.__version__,
        "config": {
            "hidden_size": module.hidden_size,
            "key_dim": module.key_dim, "value_dim": module.value_dim,
            "num_k_heads": module.num_k_heads, "num_v_heads": module.num_v_heads,
            "head_k_dim": module.head_k_dim, "head_v_dim": module.head_v_dim,
            "conv_kernel": module.conv_kernel_size,
            "eps": float(module.layer_norm_epsilon),
            "output_activation": getattr(module.norm, "activation", "sigmoid"),
            "seq_len": int(cap["mixed_input"].shape[1]),
            "mask_is_none": cap["mask"] is None,
        },
        "files": {},
    }

    for name in (
        "mixed_input", "conv_state_in", "rec_state_in",
        "mixed_qkv", "z", "a", "b", "conv_input", "new_conv_state", "conv_out",
        "q_raw", "k_raw", "q_norm", "k_norm",
        "g", "beta", "delta_y", "state_out", "normed", "out", "real_out",
    ):
        _dump_array(out_dir, manifest, name, cap[name])

    # Weights for the C++ replay.
    sd = module.parameters()
    _dump_array(out_dir, manifest, "w_conv1d_weight", module.conv1d.weight)
    _dump_array(out_dir, manifest, "w_A_log", module.A_log)
    _dump_array(out_dir, manifest, "w_dt_bias", module.dt_bias)
    _dump_array(out_dir, manifest, "w_norm_weight", module.norm.weight)
    op = module.out_proj
    _dump_array(out_dir, manifest, "w_outproj_weight", op.weight)
    _dump_array(out_dir, manifest, "w_outproj_scales", op.scales)
    _dump_array(out_dir, manifest, "w_outproj_biases", op.biases)

    # Sanity: replica output must equal the genuine forward output bit-for-bit.
    mx.eval(cap["out"], cap["real_out"])
    rd = np.abs(_to_np(cap["out"]) - _to_np(cap["real_out"]))
    manifest["replica_vs_real_maxdiff"] = float(rd.max())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cfg = manifest["config"]
    (out_dir / "config.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in cfg.items()) + "\n"
    )
    print(f"captured layer {args.layer} GDN intermediates: "
          f"{len(manifest['files'])} files -> {out_dir}")
    print(f"replica-vs-real maxdiff = {manifest['replica_vs_real_maxdiff']}")
    for name, d in manifest["files"].items():
        print(f"  {name:22s} shape={d['shape']} dtype={d['dtype']} "
              f"sha={d['sha256'][:12]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    args = ap.parse_args()
    run_real(args)


if __name__ == "__main__":
    main()
