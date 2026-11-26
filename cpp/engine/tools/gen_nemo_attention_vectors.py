#!/usr/bin/env python3
"""Generate bit-exact golden vectors for the Nemotron attention decode primitive.

PATH VALIDATED: **bf16 KV** (``VEYSTRIX_NEMOTRON_KV=bf16``).

Why bf16 and not the default INT8 block KV (FORMAT = "affine-int8-g64")?
  * In bf16 mode the backend keeps a plain ``mlx_lm.models.cache.KVCache``
    (see models/nemotron35/backend_core.py:53 -- AttentionCache is only
    substituted when int8_kv is True).  StreamedAttention.__call__ then sees a
    non-AttentionCache and falls back to the real NemotronHAttention.__call__,
    i.e. dense bf16 q/k/v projections + KV append + one
    mx.fast.scaled_dot_product_attention + o_proj.  There is no RoPE and no
    q_norm/k_norm in Nemotron-H attention.  This is a short, fully-exposed op
    chain that the C++ side can reproduce bit-for-bit with the same MLX build.
  * The default INT8 path quantizes each appended K/V with
    mx.quantize(group_size=64, bits=8) (packed uint32 + bf16 scales/biases) and
    must reproduce affine-INT8 packing + dequantization (and, for >1 block, a
    chunked quantized_matmul softmax).  numpy/MLX do not expose a clean tiny
    synthetic validation for the packed layout at head_dim<64, so INT8 is left
    as a documented TODO for this primitive.

No checkpoint is loaded: this uses the REAL attention module class with
deterministic synthetic bf16 weights and a tiny config scaled from config.py's
EXPECTED dims (H=4, Hkv=2, head_dim=16, hidden=64).  S=1 decode, 3 steps.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

# The worktree has no .venv; import the repo + its venv from the main checkout.
ROOT = Path(os.environ.get("VEYSTRIX_ROOT", "/Users/vincent/Documents/Veystrix"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.12/site-packages"))

from mlx_lm.models.cache import KVCache  # noqa: E402
from models.nemotron35.attention import StreamedAttention  # noqa: E402
from mlx_lm.models.nemotron_h import NemotronHAttention  # noqa: E402


def save(d: Path, n: str, a: mx.array) -> None:
    mx.save(str(d / f"{n}.npy"), a)


def bits(a: mx.array) -> np.ndarray:
    """Raw bits of a bf16 array (numpy cannot view a bfloat16 MLX buffer)."""
    return np.asarray(mx.view(mx.reshape(a, (-1,)), mx.uint16))


def proj(module, x, heads):
    """Exactly the reshape/transpose in NemotronHAttention.__call__."""
    B, L, _ = x.shape
    return module(x).reshape(B, L, heads, -1).transpose(0, 2, 1, 3)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               "cpp/engine/tests/vectors_nemo_attention")
    out.mkdir(parents=True, exist_ok=True)

    seed = 20261118
    rng = np.random.default_rng(seed)
    B = 1
    H, HKV, D = 4, 2, 16
    HIDDEN = H * D  # 64; no attention bias (config.py: attention_bias=False)
    STEPS = 3
    ST = B * HIDDEN

    cfg = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=H,
        num_key_value_heads=HKV,
        head_dim=D,
        attention_bias=False,
    )

    # Deterministic bf16 weights (no bias, matching attention_bias=False).
    def w(shape):
        return mx.array(rng.normal(size=shape).astype(np.float32) * 0.1).astype(mx.bfloat16)

    wq = w((H * D, HIDDEN))
    wk = w((HKV * D, HIDDEN))
    wv = w((HKV * D, HIDDEN))
    wo = w((HIDDEN, H * D))

    # ---- REAL Python Nemotron attention (bf16 KV fallback path) -------------
    attn = StreamedAttention(cfg)
    attn.q_proj.weight = wq
    attn.k_proj.weight = wk
    attn.v_proj.weight = wv
    attn.o_proj.weight = wo
    scale = attn.scale  # head_dim ** -0.5

    cache = KVCache()  # plain KVCache => bf16 fallback (no AttentionCache)
    xs = [mx.array(rng.normal(size=(B, 1, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
          for _ in range(STEPS)]

    for s, x in enumerate(xs):
        # Projection + cache append all happen inside the real module.
        expected = attn(x, cache=cache)
        mx.eval(expected)

        # Isolation intermediates: recompute with identical ops/weights.
        q = attn.q_proj(x).reshape(B, 1, H, D).transpose(0, 2, 1, 3)
        k = attn.k_proj(x).reshape(B, 1, HKV, D).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(B, 1, HKV, D).transpose(0, 2, 1, 3)
        ck, cv = cache.state  # logical keys/values after append: [B, HKV, s+1, D]
        attended = mx.fast.scaled_dot_product_attention(
            q, ck, cv, scale=scale, mask=None, sinks=None)
        # self-check: attended -> transpose/reshape -> o_proj reproduces the module output
        rebuilt = attn.o_proj(attended.transpose(0, 2, 1, 3).reshape(B, 1, -1))
        mx.eval(q, k, v, attended, ck, cv, rebuilt)
        if not np.array_equal(bits(rebuilt), bits(expected)):
            raise RuntimeError(f"step {s}: isolation intermediates do not reproduce module output")

        save(out, f"x_{s}", x)
        save(out, f"expected_{s}", expected)
        save(out, f"attn_q_{s}", q)
        save(out, f"attn_k_{s}", k)
        save(out, f"attn_v_{s}", v)
        save(out, f"attended_{s}", attended)

    ck, cv = cache.state
    mx.eval(ck, cv)

    save(out, "q_w", wq)
    save(out, "k_w", wk)
    save(out, "v_w", wv)
    save(out, "o_w", wo)
    save(out, "cache_keys_final", ck)
    save(out, "cache_values_final", cv)

    # Also validate against a direct NemotronHAttention call with the same cache
    # (confirms the bf16 fallback is exactly the stock module).
    real = NemotronHAttention(cfg)
    real.q_proj.weight, real.k_proj.weight = wq, wk
    real.v_proj.weight, real.o_proj.weight = wv, wo
    c2 = KVCache()
    outs_real = [real(x, cache=c2) for x in xs]
    mx.eval(*outs_real)
    for s in range(STEPS):
        assert np.array_equal(bits(outs_real[s]), bits(mx.load(
            str(out / f"expected_{s}.npy")))), f"StreamedAttention != NemotronHAttention at step {s}"

    (out / "meta.json").write_text(json.dumps({
        "kv_path": "bf16",
        "hidden_size": HIDDEN,
        "num_heads": H,
        "num_key_value_heads": HKV,
        "head_dim": D,
        "scale": float(scale),
        "steps": STEPS,
        "batch": B,
        "seed": seed,
    }, indent=2) + "\n")
    print(f"wrote Nemotron attention (bf16 KV) vectors to {out}")
    print(f"  H={H} Hkv={HKV} D={D} hidden={HIDDEN} steps={STEPS} scale={scale}")


if __name__ == "__main__":
    main()
