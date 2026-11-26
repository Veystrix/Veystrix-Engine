#!/usr/bin/env python3
"""Guarded REFERENCE capture for the C++ engine cross-check on the REAL checkpoint.

Loads the real Qwen Flash-Next backend, monkeypatches Sampler.__call__ at class
level so the internally-created sampler records every decode step's raw logit
SHA-256 (float32 little-endian, matching logit_reference.py), and runs a short
deterministic greedy decode. Token IDs come from the on_decode_token callback.

The emitted reference.json also records the EFFECTIVE runtime configuration
(prefill chunk size, KV mode, MTP, PLE, stream-embed, RoPE path, sampler) read
from the same env/backend defaults the backend itself uses, with file:line
citations. This makes a reference reproducible and tells the C++ parity gate
which of the documented gaps (docs/performance/CPP_PARITY_GAPS.md) actually
applied to that capture.

Run under the guard:
  scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048 \
    --compressor-growth-mib 4096 --thermal-level 99 --log <dir>/guard.jsonl -- \
    .venv/bin/python cpp/engine/tools/capture_real_reference.py \
      --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --out <dir>/reference.json --tokens 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np


# Citations for every effective-config field. Kept as data so a consumer of
# reference.json can see where each value came from without reading this file.
EFFECTIVE_CONFIG_SOURCES = {
    "prefill_chunk_tokens":
        "models/flashnext/prefill.py:26,75-93 (FLAG_CHUNK_TOKENS, default 0=off); "
        "models/flashnext/prefill_autotune.py:69 (autotune override)",
    "prefill_chunk_policy":
        "models/flashnext/prefill.py:36,96-105 (FLAG_CHUNK_POLICY default grid)",
    "kv_mode":
        "models/flashnext/quantized_kv.py:19,33-39 (FLAG_KV_DTYPE default bf16)",
    "kv_group_size":
        "models/flashnext/quantized_kv.py:20,86-94 (default 64)",
    "kv_bits":
        "models/flashnext/quantized_kv.py:21,97-109 (default by mode)",
    "mtp":
        "veystrix/backends/flashnext.py:62-79 (_resolve_startup_mtp: "
        "FLASHNEXT_NATIVE_MTP default off, FLASHNEXT_MTP_DEPTH default 3), :163",
    "ple_layer_ids":
        "mlx_vlm qwen4_exp/language.py:734-740,848 (layer_idx+1 in ple_layer_ids)",
    "stream_embed":
        "models/flashnext/loader.py:284,297 (FLAG_STREAM_EMBED default 0)",
    "rope_path":
        "models/flashnext/qsa_chunk.py:352,402 (indexer._apply_rope -> fused mx.fast.rope)",
    "sampler":
        "veystrix/backends/flashnext.py:211 (Sampling.greedy_settings); "
        "veystrix/sampling.py:47-62 (greedy iff temperature<=0)",
}


def _read_effective_config(backend=None):
    """Collect the effective runtime config with file:line citations.

    Every field is best-effort: a failure to read one must not prevent the
    reference (tokens + logit digests) from being written.
    """
    cfg = {"sources": EFFECTIVE_CONFIG_SOURCES, "env": {}, "read_errors": {}}

    def read(name, fn):
        try:
            return fn()
        except Exception as exc:  # keep the reference capture alive
            cfg["read_errors"][name] = f"{type(exc).__name__}: {exc}"
            return None

    # Prefill chunk (effective, autotune-aware, plus explicit/policy values).
    from models.flashnext import prefill
    cfg["env"]["FLASHNEXT_PREFILL_CHUNK_TOKENS"] = os.environ.get(
        "FLASHNEXT_PREFILL_CHUNK_TOKENS", "0")
    cfg["env"]["FLASHNEXT_PREFILL_CHUNK_POLICY"] = os.environ.get(
        "FLASHNEXT_PREFILL_CHUNK_POLICY", "grid")
    cfg["prefill_chunk_tokens"] = read("prefill_chunk_tokens", prefill.chunk_tokens)
    cfg["prefill_chunk_configured"] = read(
        "prefill_chunk_configured", prefill.configured_chunk_tokens)
    cfg["prefill_chunk_policy"] = read("prefill_chunk_policy", prefill.chunk_policy)

    # KV mode / group / bits (dense bf16 by default; INT4/INT8 are opt-in).
    from models.flashnext import quantized_kv
    cfg["env"]["FLASHNEXT_KV_CACHE_DTYPE"] = os.environ.get(
        "FLASHNEXT_KV_CACHE_DTYPE", "bf16")
    cfg["kv_mode"] = read("kv_mode", quantized_kv.kv_dtype)
    cfg["kv_group_size"] = read("kv_group_size", quantized_kv.group_size)
    cfg["kv_bits"] = read("kv_bits", quantized_kv.bits)

    # MTP: prefer the backend's resolved values; fall back to the same env
    # defaults used by _resolve_startup_mtp (flashnext.py:62-79).
    cfg["env"]["FLASHNEXT_NATIVE_MTP"] = os.environ.get(
        "FLASHNEXT_NATIVE_MTP", "off")
    cfg["env"]["FLASHNEXT_MTP_DEPTH"] = os.environ.get("FLASHNEXT_MTP_DEPTH", "3")
    cfg["mtp"] = {
        "native": getattr(backend, "native_mtp", None)
                  if backend is not None else os.environ.get("FLASHNEXT_NATIVE_MTP", "off"),
        "depth": getattr(backend, "mtp_depth", None)
                 if backend is not None else os.environ.get("FLASHNEXT_MTP_DEPTH", "3"),
        "active": getattr(backend, "_mtp_active", None) if backend is not None else None,
    }

    # PLE is a model-architecture list, not an env flag. Read it from the
    # language model args when exposed; None means "not exposed by this build".
    def ple_ids():
        language = getattr(backend, "language", None) if backend is not None else None
        args = getattr(language, "args", None)
        ids = getattr(args, "ple_layer_ids", None) if args is not None else None
        if ids is None and language is not None:
            ids = getattr(language, "ple_layer_ids", None)
        return list(ids) if ids is not None else None

    cfg["ple_layer_ids"] = read("ple_layer_ids", ple_ids)

    # Streamed input embedding (default off).
    cfg["env"]["FLASHNEXT_STREAM_EMBED"] = os.environ.get(
        "FLASHNEXT_STREAM_EMBED", "0")
    cfg["stream_embed"] = os.environ.get("FLASHNEXT_STREAM_EMBED", "0") == "1"

    # RoPE path: production uses the fused Metal mx.fast.rope via the indexer's
    # _apply_rope; there is no env switch for it.
    cfg["rope_path"] = "fused_mx_fast_rope (indexer._apply_rope)"

    # Sampler: the backend forces greedy in __init__ (flashnext.py:211) and this
    # script additionally sets the env default to greedy below.
    sampling = getattr(backend, "sampling", None) if backend is not None else None
    cfg["env"]["VEYSTRIX_SAMPLING_GREEDY"] = os.environ.get(
        "VEYSTRIX_SAMPLING_GREEDY", "1")
    cfg["sampler"] = {
        "greedy": bool(getattr(sampling, "greedy", True)) if sampling is not None else None,
        "temperature": getattr(sampling, "temperature", None) if sampling is not None else None,
        "top_p": getattr(sampling, "top_p", None) if sampling is not None else None,
        "top_k": getattr(sampling, "top_k", None) if sampling is not None else None,
        "min_p": getattr(sampling, "min_p", None) if sampling is not None else None,
        "presence_penalty": getattr(sampling, "presence_penalty", None)
                           if sampling is not None else None,
    }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    args = ap.parse_args()

    from veystrix.backends.flashnext import FlashNextBackend
    from veystrix.sampling import Sampler

    # Class-level capture: the backend creates Sampler(self.sampling) internally.
    original_call = Sampler.__call__
    logit_digests = []

    import os
    dump_prefix = os.environ.get("PYCAP_DUMP_LOGITS")

    def capturing_call(self, logits):
        row = np.asarray(logits.reshape(-1).astype(mx.float32), dtype="<f4")
        logit_digests.append(hashlib.sha256(row.tobytes()).hexdigest())
        if dump_prefix:
            step = len(logit_digests) - 1
            with open(f"{dump_prefix}.step{step}.f32", "wb") as fh:
                fh.write(row.tobytes())
        return original_call(self, logits)

    Sampler.__call__ = capturing_call

    # Force greedy in the startup preferences so the internal sampler is greedy.
    os.environ.setdefault("VEYSTRIX_SAMPLING_GREEDY", "1")

    backend = FlashNextBackend(model_path=args.checkpoint)
    # Greedy: set the backend's sampling settings directly if exposed.
    if hasattr(backend, "sampling") and backend.sampling is not None:
        try:
            backend.sampling.greedy = True
        except Exception:
            pass

    # Fixed prompt token ids (deterministic, no tokenizer dependence).
    rng = np.random.default_rng(20261117)
    prompt = rng.integers(0, 200000, size=args.prompt_tokens).tolist()
    backend.append_tokens(prompt)

    token_ids = []

    def emitted(token, _piece):
        token_ids.append(int(token))

    _text, stats = backend.generate(args.tokens, on_decode_token=emitted)

    # Pair token ids with logit digests. The sampler is called once per generated
    # token (and once for the first token via prefill seed when applicable).
    n = min(len(token_ids), len(logit_digests))
    steps = [
        {"token": token_ids[i], "logit_sha256": logit_digests[i]}
        for i in range(n)
    ]
    # Record the effective runtime config with citations so this reference is
    # reproducible and the C++ gate knows which parity gaps applied.
    effective_config = _read_effective_config(backend)

    out = {
        "checkpoint": args.checkpoint,
        "prompt_sha256": hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
        ).hexdigest(),
        "prompt_tokens": args.prompt_tokens,
        "requested_tokens": args.tokens,
        "token_ids": token_ids,
        "n_logit_digests": len(logit_digests),
        "steps": steps,
        "finish": getattr(stats, "finish", "?"),
        "mlx_version": mx.__version__,
        "effective_config": effective_config,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}: {len(token_ids)} tokens, {len(logit_digests)} logits")


if __name__ == "__main__":
    main()
