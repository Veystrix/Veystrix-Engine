#!/usr/bin/env python3
"""Generate bit-exact fixtures for the production Qwen4ExpPLELayer (layer idx 1).

This builds a SMALL ``Qwen4ExpPLELayer`` directly from a TextConfig-like
namespace (no checkpoint, synthetic deterministic weights), runs the exact
Python compute path from
``mlx_vlm/models/qwen4_exp/language.py`` and dumps every input, weight and the
expected output as individual ``.npy`` files (C++ has no npz reader).

Key point: the n-gram embedding shards are ``mlx.nn.QuantizedEmbedding``
(affine Q4) because the real checkpoint stores them packed.  Their __call__ is
``mx.dequantize(weight[x], scales[x], biases[x], gs, bits, "affine")`` -- i.e.
gather the PACKED rows first, then dequantize.  The C++ port must do the same.
The key/value projections are also packed (``nn.QuantizedLinear``) so the
packed ``mx.quantized_matmul`` path is exercised end to end.

Three cases are produced:
  * decode_s1          -- S=1 decode, no mask
  * prefill_nomask     -- S>1 prefill, no mask
  * prefill_mask       -- S>1 prefill, 2-D bool mask

Each case's output is computed twice with a fresh cache and bit-compared.

Run:
  /Users/vincent/Documents/Veystrix/.venv/bin/python \
      cpp/engine/tools/gen_ple_vectors.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

# The Veystrix venv is NOT on sys.path when invoked from other environments.
MLX_VLM_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
if MLX_VLM_SITE not in sys.path:
    sys.path.insert(0, MLX_VLM_SITE)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpPLELayer  # noqa: E402

SEED = 20261118
DT = np.float32
Q_GROUP = 32
Q_BITS = 4
Q_MODE = "affine"

# Synthetic shape (small, but preserves the production structure).
HIDDEN = 64
HC_COUNT = 2
PLE_EMBED = 256
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 2
VOCAB_SIZE = 248320  # production value -> realistic layer multipliers
EOS_TOKEN_ID = 5     # deliberately small so it appears inside input_ids
NGRAM_BASE = 200
DIVISOR = 128
SPLIT_PARTS = 4
CONV_KERNEL = 3
EPS = 1e-6
PLE_LAYER_INDEX = 0   # ple_layer_ids=[1] (1-indexed) => layer idx 0
LAYER_IDX = 1

# The config.seed field is absent from the real checkpoint config.json; it
# resolves to the dataclass default 1234 in TextConfig.
CFG_SEED = 1234


def _np(a, dtype=DT):
    # numpy has no bfloat16 buffer format; promote through MLX explicitly.
    if isinstance(a, mx.array) and a.dtype in (mx.bfloat16, mx.float16):
        a = mx.astype(a, mx.float32)
    return np.array(a, dtype=dtype)


def _dump_shard_array(a):
    """BF16 shard rows are dumped as raw uint16 bits (numpy cannot hold BF16)."""
    if a.dtype == mx.bfloat16:
        return np.array(mx.view(a, mx.uint16), dtype=np.uint16)
    return np.array(a)


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _describe(name: str, arr: np.ndarray) -> dict:
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _digest(arr),
        "first": np.ascontiguousarray(arr).reshape(-1)[:4].tolist(),
    }


def build_weights(rng: np.random.Generator, bf16_shards: bool = False):
    """Construct a Qwen4ExpPLELayer with deterministic synthetic weights.

    bf16_shards makes the n-gram shard scales/biases BF16, matching the real
    checkpoint (mlx.nn.QuantizedEmbedding stores BF16 scales when its source
    weight is BF16).
    """
    cfg = types.SimpleNamespace(
        hidden_size=HIDDEN,
        hc_count=HC_COUNT,
        ple_embed_dim=PLE_EMBED,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
        vocab_size=VOCAB_SIZE,
        seed=CFG_SEED,
        eos_token_id=EOS_TOKEN_ID,
        ngram_vocab_size_base=NGRAM_BASE,
        make_ngram_vocab_size_divisible_by=DIVISOR,
        split_ngram_parts=SPLIT_PARTS,
        rms_norm_eps=EPS,
        ple_conv_kernel_size=CONV_KERNEL,
    )
    layer = Qwen4ExpPLELayer(cfg, LAYER_IDX, PLE_LAYER_INDEX)
    emb = layer.ple_embedding

    # --- n-gram embedding shards: dense nn.Embedding -> QuantizedEmbedding ---
    shards = []
    for i, sh in enumerate(emb.ngram_embedding.shards):
        rows, dims = sh.weight.shape
        w = mx.array((rng.standard_normal((rows, dims)) * 0.1).astype(DT))
        if bf16_shards:
            w = mx.astype(w, mx.bfloat16)
        sh.weight = w
        qsh = sh.to_quantized(group_size=Q_GROUP, bits=Q_BITS, mode=Q_MODE)
        emb.ngram_embedding.shards[i] = qsh
        shards.append(qsh)

    # --- key / value projections: dense nn.Linear -> QuantizedLinear -------
    for name in ("key_proj", "value_proj"):
        lin = getattr(layer, name)
        out_f, in_f = lin.weight.shape
        lin.weight = mx.array((rng.standard_normal((out_f, in_f)) * 0.05).astype(DT))
        setattr(layer, name, lin.to_quantized(group_size=Q_GROUP, bits=Q_BITS, mode=Q_MODE))

    # --- RMSNorm gains (one-centered convention) --------------------------
    for name in ("norm_key", "norm_query", "norm_conv"):
        n = getattr(layer, name)
        n.weight = mx.array((rng.standard_normal(n.weight.shape) * 0.1).astype(DT))

    # --- depthwise conv1d weight [C, K, 1] --------------------------------
    c, k, one = layer.conv1d.weight.shape
    layer.conv1d.weight = mx.array((rng.standard_normal((c, k, one)) * 0.05).astype(DT))

    mx.eval([p for p in layer.parameters()])

    arrays = {
        "key_w": _np(layer.key_proj.weight, np.uint32),
        "key_s": _np(layer.key_proj.scales),
        "key_b": _np(layer.key_proj.biases),
        "val_w": _np(layer.value_proj.weight, np.uint32),
        "val_s": _np(layer.value_proj.scales),
        "val_b": _np(layer.value_proj.biases),
        "norm_key": _np(layer.norm_key.weight),
        "norm_query": _np(layer.norm_query.weight),
        "norm_conv": _np(layer.norm_conv.weight),
        "conv_w": _np(layer.conv1d.weight),
        "layer_multipliers": _np(emb.layer_multipliers, np.int64),
        "head_offsets": _np(emb.ngram_heads_offsets, np.int64),
        "head_vocab_sizes": _np(emb.ngram_heads_vocab_sizes, np.int64),
        "shard_offsets": _np(emb.ngram_embedding.shard_offsets, np.int64),
    }
    for i, qsh in enumerate(shards):
        arrays[f"shard_{i}_w"] = _np(qsh.weight, np.uint32)
        arrays[f"shard_{i}_s"] = _dump_shard_array(qsh.scales)
        arrays[f"shard_{i}_b"] = _dump_shard_array(qsh.biases)
    return layer, arrays


def make_cache(B: int, state_len: int, context_len: int):
    return [
        None,
        None,
        mx.zeros((B, state_len, HIDDEN * HC_COUNT), dtype=mx.float32),
        mx.full((B, context_len), EOS_TOKEN_ID, dtype=mx.int64),
    ]


def build_case(layer, B: int, S: int, ids_np: np.ndarray,
               h_np: np.ndarray, mask_np: np.ndarray | None):
    state_len = (CONV_KERNEL - 1) * NGRAM_SIZE
    context_len = NGRAM_SIZE - 1
    expected = None
    expected_cache = None
    for _ in range(2):
        cache = make_cache(B, state_len, context_len)
        out = layer(mx.array(h_np), mx.array(ids_np.astype(np.int64)), cache,
                    mx.array(mask_np) if mask_np is not None else None)
        mx.eval(out, cache[2], cache[3])
        y = _np(out)
        c = (_np(cache[2]), _np(cache[3], np.int64))
        if expected is None:
            expected, expected_cache = y, c
        else:
            assert np.array_equal(expected.view(np.uint32), y.view(np.uint32)), \
                "NON-REPRODUCIBLE PLE output"
            assert np.array_equal(expected_cache[0], c[0])
            assert np.array_equal(expected_cache[1], c[1])
    return expected, expected_cache[0], expected_cache[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parent.parent / "tests" / "vectors_ple"))
    args = ap.parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    mx.random.seed(SEED)
    layer, weights = build_weights(rng)
    # BF16 shard variant: matches the real checkpoint's ngram scales/biases.
    rng_bf = np.random.default_rng(SEED + 1)
    layer_bf, weights_bf = build_weights(rng_bf, bf16_shards=True)
    state_len = (CONV_KERNEL - 1) * NGRAM_SIZE
    context_len = NGRAM_SIZE - 1

    cases: list[dict] = []

    case_specs = [
        ("decode_s1", 2, 1, False, False),
        ("prefill_nomask", 2, 5, False, False),
        ("prefill_mask", 2, 5, True, False),
        ("decode_s1_bf16", 2, 1, False, True),
    ]
    for cid, B, S, use_mask, bf16 in case_specs:
        active_layer = layer_bf if bf16 else layer
        ids = rng.integers(0, 60, size=(B, S), dtype=np.int64)
        # Force an EOS token into the prefill cases to exercise segmentation.
        if S > 1:
            ids[0, S // 2] = EOS_TOKEN_ID
            ids[1, 0] = EOS_TOKEN_ID
        ids = ids.astype(np.int64)
        h = (rng.standard_normal((B, S, HIDDEN * HC_COUNT)) * 0.5).astype(DT)
        mask = None
        if use_mask:
            mask = np.ones((B, S), dtype=np.bool_)
            mask[0, -2:] = False
            mask[1, -1] = False

        out, out_cache, out_ctx = build_case(active_layer, B, S, ids, h, mask)

        cd = out_dir / cid
        cd.mkdir(parents=True, exist_ok=True)
        arrays = dict(weights_bf if bf16 else weights)
        arrays.update({
            "hidden_states": h,
            "input_ids": ids,
            "cache_conv": np.zeros((B, state_len, HIDDEN * HC_COUNT), dtype=DT),
            "cache_context": np.full((B, context_len), EOS_TOKEN_ID, dtype=np.int64),
            "expected": out,
            "expected_cache_conv": out_cache,
            "expected_cache_context": out_ctx,
        })
        if mask is not None:
            arrays["mask"] = mask
        for name, arr in arrays.items():
            np.save(cd / f"{name}.npy", arr)
        if bf16:
            (cd / "shard_bf16.flag").write_text("ngram shard scales/biases are BF16 raw uint16 bits\n")

        desc = {
            "id": cid,
            "npy_dir": cid,
            "B": B, "S": S,
            "mask": use_mask,
            "hidden_size": HIDDEN,
            "hc_count": HC_COUNT,
            "hidden_dim": HIDDEN * HC_COUNT,
            "ple_embed_dim": PLE_EMBED,
            "ngram_size": NGRAM_SIZE,
            "heads_per_ngram": HEADS_PER_NGRAM,
            "ngram_heads": (NGRAM_SIZE - 1) * HEADS_PER_NGRAM,
            "head_dim": PLE_EMBED // ((NGRAM_SIZE - 1) * HEADS_PER_NGRAM),
            "vocab_size": VOCAB_SIZE,
            "eos_token_id": EOS_TOKEN_ID,
            "seed": CFG_SEED,
            "ple_layer_index": PLE_LAYER_INDEX,
            "ngram_vocab_size_base": NGRAM_BASE,
            "make_ngram_vocab_size_divisible_by": DIVISOR,
            "split_ngram_parts": SPLIT_PARTS,
            "num_shards": SPLIT_PARTS,
            "ple_conv_kernel_size": CONV_KERNEL,
            "conv_dilation": NGRAM_SIZE,
            "short_conv_state_len": state_len,
            "context_len": context_len,
            "eps": EPS,
            "quant_group_size": Q_GROUP,
            "quant_bits": Q_BITS,
            "quant_mode": Q_MODE,
            # The oracle builds the module straight from language.py, i.e.
            # WITHOUT models/flashnext/patch_rmsnorm.py, so its norms keep the
            # zero-centered "1.0 + weight" convention.
            "norm_one_centered": False,
            "shard_scale_dtype": "bf16" if bf16 else "f32",
            "reproducible": True,
            "arrays": [_describe(n, a) for n, a in arrays.items()],
        }
        cases.append(desc)
        print(f"{cid}: B={B} S={S} mask={use_mask} "
              f"out={out.shape} reproducible=True")

    meta = {
        "seed": SEED,          # RNG seed for the generator
        "config_seed": CFG_SEED,  # TextConfig.seed -> 1234 (dataclass default)
        "vocab_size": VOCAB_SIZE,
        "ngram_size": NGRAM_SIZE,
        "heads_per_ngram": HEADS_PER_NGRAM,
        "ple_layer_index": PLE_LAYER_INDEX,
        "ngram_vocab_size_base": NGRAM_BASE,
        "hidden_size": HIDDEN,
        "hc_count": HC_COUNT,
        "ple_embed_dim": PLE_EMBED,
        "eos_token_id": EOS_TOKEN_ID,
        "quant_group_size": Q_GROUP,
        "quant_bits": Q_BITS,
        "quant_mode": Q_MODE,
        "cases": cases,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {out_dir}")


if __name__ == "__main__":
    main()
