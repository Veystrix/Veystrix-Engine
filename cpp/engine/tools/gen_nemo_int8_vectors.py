#!/usr/bin/env python3
"""Generate bit-exact golden vectors for the Nemotron INT8 KV attention path.

PATH VALIDATED: **INT8 affine block KV** (``FORMAT = "affine-int8-g64"``,
``VEYSTRIX_NEMOTRON_KV=int8``).  This is the production default.

We use the REAL ``models.nemotron35.attention.StreamedAttention`` with a REAL
``AttentionCache`` (which owns ``mlx_lm.models.cache.QuantizedKVCache`` blocks).
No checkpoint is loaded: q/k/v/o are deterministic synthetic bf16 weights and
the tiny config is scaled from config.py's EXPECTED dims (H=4, Hkv=2,
head_dim=128, hidden=512).  head_dim=128 keeps two affine groups per head,
matching the real checkpoint (GROUP_SIZE=64).

Three cases:

  * ``decode``: 4 sequential S=1 steps, cache grows 1 -> 4 tokens, mask=None.
  * ``prompt``: one S=6 chunk (mask="causal"), then 2 S=1 decode steps.
  * ``multiblock``: the REAL module with only its global BLOCK_TOKENS
    monkeypatched to 4 (so >1 block is cheap); one S=5 chunk (causal masking
    across two blocks, PREFILL_SDPA=0) then 3 S=1 steps. This validates the
    multi-block quantized_matmul running-softmax loop algorithm bit-for-bit;
    production still uses BLOCK_TOKENS=4096 (see the engineering note).

The first two exercise the *single-block* branch of ``attend()`` -- the common
case for decode and short prompts, since 4096 tokens are needed for a second
block in production.

For each step we save x, the module output, q/k/v, and ``attended`` (the SDPA or
loop result before o_proj).  After each case we save the final cache state: six
arrays per block (packed/scales/biases for K, then for V) with the exact layout
of ``QuantizedKVCache.state``:

  packed  uint32 [B, Hkv, n, D//4]
  scales  bf16   [B, Hkv, n, D//GROUP_SIZE]
  biases  bf16   [B, Hkv, n, D//GROUP_SIZE]

Run with the repo venv, e.g.:
  VEYSTRIX_ROOT=/Users/vincent/Documents/Veystrix \
    /Users/vincent/Documents/Veystrix/.venv/bin/python \
    cpp/engine/tools/gen_nemo_int8_vectors.py cpp/engine/tests/vectors_nemo_int8
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

ROOT = Path(os.environ.get("VEYSTRIX_ROOT", "/Users/vincent/Documents/Veystrix"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.12/site-packages"))

import models.nemotron35.attention as attn_mod  # noqa: E402
from models.nemotron35.attention import AttentionCache, GROUP_SIZE, StreamedAttention  # noqa: E402


def save(d: Path, n: str, a: mx.array) -> None:
    mx.save(str(d / f"{n}.npy"), a)


def bits(a: mx.array) -> np.ndarray:
    return np.asarray(mx.view(mx.reshape(a, (-1,)), mx.uint16))


def proj(module, x, heads):
    B, L, _ = x.shape
    return module(x).reshape(B, L, heads, -1).transpose(0, 2, 1, 3)


def make_attention(seed):
    rng = np.random.default_rng(seed)
    B = 1
    H, HKV, D = 4, 2, 128
    HIDDEN = H * D  # 512
    cfg = SimpleNamespace(
        hidden_size=HIDDEN, num_attention_heads=H, num_key_value_heads=HKV,
        head_dim=D, attention_bias=False,
    )

    def w(shape):
        return mx.array(rng.normal(size=shape).astype(np.float32) * 0.1).astype(mx.bfloat16)

    attn = StreamedAttention(cfg)
    attn.q_proj.weight = w((H * D, HIDDEN))
    attn.k_proj.weight = w((HKV * D, HIDDEN))
    attn.v_proj.weight = w((HKV * D, HIDDEN))
    attn.o_proj.weight = w((HIDDEN, H * D))
    return rng, attn, cfg, (B, H, HKV, D, HIDDEN)


def dequant_pair(block):
    """Keys/values dequantized exactly as attend() does for a single block."""
    keys, values = block.state
    dk = mx.dequantize(*keys, group_size=GROUP_SIZE, bits=8)
    dv = mx.dequantize(*values, group_size=GROUP_SIZE, bits=8)
    return dk, dv


def save_case(out, name, seed, chunks, block_tokens=None):
    """chunks: list of S (tokens per call). One AttentionCache per case.

    block_tokens=None uses the production 4096. A smaller value monkeypatches
    the reference module's global BLOCK_TOKENS so the multi-block loop is
    exercised with a short (cheap) context; only the constant differs from
    production (documented in the engineering note).
    """
    cd = out / name
    cd.mkdir(parents=True, exist_ok=True)
    if block_tokens is not None:
        attn_mod.BLOCK_TOKENS = block_tokens
        # Force the quantized_matmul loop for S>1 multi-block chunks (production
        # diverts those to dequant+SDPA when this flag is on).
        os.environ["VEYSTRIX_NEMOTRON_ATTENTION_PREFILL_SDPA"] = "0"
    rng, attn, cfg, (B, H, HKV, D, HIDDEN) = make_attention(seed)
    scale = attn.scale

    cache = AttentionCache()
    step = 0
    dk = dv = None
    for S in chunks:
        x = mx.array(rng.normal(size=(B, S, HIDDEN)).astype(np.float32)).astype(mx.bfloat16)
        expected = attn(x, cache=cache)
        mx.eval(expected)

        # Isolation intermediates, recomputed with identical ops/weights.
        q = proj(attn.q_proj, x, H)
        k = proj(attn.k_proj, x, HKV)
        v = proj(attn.v_proj, x, HKV)
        if len(cache.blocks) == 1:
            dk, dv = dequant_pair(cache.blocks[0])
            attended = mx.fast.scaled_dot_product_attention(
                q, dk, dv, scale=scale, mask="causal" if S > 1 else None)
        else:
            # Exactly the multi-block running-softmax loop under test.
            attended = attn_mod.attend(q, cache, scale)
        rebuilt = attn.o_proj(attended.transpose(0, 2, 1, 3).reshape(B, S, -1))
        mx.eval(q, k, v, attended, rebuilt)
        if not np.array_equal(bits(rebuilt), bits(expected)):
            raise RuntimeError(f"{name} step {step}: isolation mismatch")

        save(cd, f"x_{step}", x)
        save(cd, f"expected_{step}", expected)
        save(cd, f"attn_q_{step}", q)
        save(cd, f"attn_k_{step}", k)
        save(cd, f"attn_v_{step}", v)
        save(cd, f"attended_{step}", attended)
        step += 1

    # Final cache state: six arrays per block, in AttentionCache.state order
    # (all keys-group arrays, then all values-group arrays per block).
    state = cache.state
    mx.eval(*state)
    for i, a in enumerate(state):
        save(cd, f"cache_block_state_{i}", a)
    assert len(state) == 6 * len(cache.blocks)

    # Self-check: dequantized state dtype is bf16 (what C++ dequantize produces).
    if dk is not None:
        assert dk.dtype == mx.bfloat16 and dv.dtype == mx.bfloat16, (dk.dtype, dv.dtype)

    meta = {
        "kv_path": "int8",
        "hidden_size": HIDDEN,
        "num_heads": H,
        "num_key_value_heads": HKV,
        "head_dim": D,
        "scale": float(scale),
        "group_size": GROUP_SIZE,
        "bits": 8,
        "steps": step,
        "chunks": chunks,
        "batch": B,
        "seed": seed,
        "blocks": len(cache.blocks),
        "block_tokens": block_tokens or 4096,
        "cache_state_arrays": len(state),
    }
    (cd / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {name}: {step} steps chunks={chunks} scale={scale} "
          f"final_offset={cache.offset} blocks={len(cache.blocks)} "
          f"block_tokens={meta['block_tokens']} state_arrays={len(state)}")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else
               "cpp/engine/tests/vectors_nemo_int8")
    out.mkdir(parents=True, exist_ok=True)
    (out / "common").mkdir(parents=True, exist_ok=True)

    # Shared synthetic q/k/v/o weights come from make_attention(seed) per case;
    # save one copy (all cases use the same seed) for the C++ test to load.
    rng, attn, cfg, (B, H, HKV, D, HIDDEN) = make_attention(20261118)
    for n, a in (("q_w", attn.q_proj.weight), ("k_w", attn.k_proj.weight),
                 ("v_w", attn.v_proj.weight), ("o_w", attn.o_proj.weight)):
        save(out / "common", n, a)

    save_case(out, "decode", 20261118, [1, 1, 1, 1])
    save_case(out, "prompt", 20261118, [6, 1, 1])
    # Multi-block loop: patched BLOCK_TOKENS=4 (real module, only the constant
    # differs) so >1 block is cheap. Includes an S=5 chunk with PREFILL_SDPA=0
    # (causal masking across blocks) followed by S=1 decode steps.
    save_case(out, "multiblock", 20261118, [5, 1, 1, 1], block_tokens=4)
    attn_mod.BLOCK_TOKENS = 4096  # restore production constant

    (out / "meta.json").write_text(json.dumps({
        "kv_path": "int8",
        "format": "affine-int8-g64",
        "group_size": GROUP_SIZE,
        "bits": 8,
        "hidden_size": HIDDEN,
        "num_heads": H,
        "num_key_value_heads": HKV,
        "head_dim": D,
        "scale": float(attn.scale),
        "seed": 20261118,
        "cases": ["decode", "prompt", "multiblock"],
        "block_tokens": 4096,
        "note": "decode/prompt use production 4096 (single block); multiblock "
                "patches only BLOCK_TOKENS=4 to validate the loop algorithm",
    }, indent=2) + "\n")
    print(f"wrote Nemotron INT8 attention vectors to {out}")


if __name__ == "__main__":
    main()
