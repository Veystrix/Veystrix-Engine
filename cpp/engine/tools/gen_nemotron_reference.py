#!/usr/bin/env python3
"""Guarded REFERENCE capture for the C++ Nemotron 3.5 Lightning decode tool.

Loads the real streamed Nemotron backend (bf16 KV, single-token prefill and
decode, gather_qmm expert path), captures each generated token's raw logit
SHA-256 (float32 little-endian, the convention of capture_real_reference.py)
and writes a small reference.json with the exact prompt ids and tokens.

There is NO benchmark here: just enough to gate C++/Python bit-exactness.

The runtime knobs are pinned so the captured path is the S=1 decode path the
C++ primitives implement:

  VEYSTRIX_NEMOTRON_KV=bf16        plain KVCache (bf16), not INT8 AttentionCache
  chunk_tokens=1                   prefill feeds one token at a time (S=1)
  VEYSTRIX_NEMOTRON_GATHER=1       routed experts via gather_qmm (the primitive
                                   `moe_block_relu2` uses the same kernel)
  VEYSTRIX_NEMOTRON_FAST_SSM=0     stock S=1 ssm kernel
  VEYSTRIX_NEMOTRON_PARALLEL_MAMBA=0
  zero-seeded Mamba caches         first token uses ssm_update_kernel (not the
                                   S=1 ssm_attn fallback taken when state=None)

Run under real_run_guard.py with --floor-gib 1.5, --swap-growth-mib 5120,
--compressor-growth-mib 8192, --thermal-level 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Pin the effective path BEFORE importing any model code.
os.environ["VEYSTRIX_NEMOTRON_KV"] = "bf16"
os.environ["VEYSTRIX_NEMOTRON_GATHER"] = "1"
os.environ["VEYSTRIX_NEMOTRON_FAST_SSM"] = "0"
os.environ["VEYSTRIX_NEMOTRON_PARALLEL_MAMBA"] = "0"
os.environ["VEYSTRIX_NEMOTRON_NATIVE_MTP"] = "0"
os.environ["VEYSTRIX_NEMOTRON_PREFIX_CACHE"] = "0"
os.environ["VEYSTRIX_NEMOTRON_EXPERT_PACK"] = ""

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

EFFECTIVE_CONFIG_SOURCES = {
    "kv_mode": "VEYSTRIX_NEMOTRON_KV=bf16; models/nemotron35/backend_core.py:51-54 "
               "(_make_cache keeps plain KVCache when int8_kv=False)",
    "prefill_chunk_tokens": "NemotronBackend(chunk_tokens=1); "
                            "models/nemotron35/backend_core.py:288-299 (_prefill calls _step per chunk)",
    "expert_path": "VEYSTRIX_NEMOTRON_GATHER=1; models/nemotron35/streamed_experts.py:847-867 "
                   "(_gather -> gather_qmm, sorted_indices=False for 6 slots)",
    "decode_step": "models/nemotron35/backend_core.py:82-85,311-321 "
                   "(_step = lm_head(backbone(tokens)[:, -1:, :])[:, -1, :])",
    "mamba_state": "zero-seeded ArraysCache so ssm_update takes ssm_update_kernel; "
                   "mlx_lm/models/ssm.py:230-242",
    "sampler": "veystrix/sampling.py:172-180 (greedy argmax over flattened logits)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=6)
    args = ap.parse_args()

    from models.nemotron35.backend_core import EngineCore
    from models.nemotron35.config import validate_config
    from models.nemotron35.streamed_model import load_streamed

    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    rng = np.random.default_rng(20261118)
    prompt = [int(t) for t in rng.integers(3, config["vocab_size"] - 1,
                                           size=args.prompt_tokens)]
    prompt_sha = hashlib.sha256(
        b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
    ).hexdigest()

    pool_bytes = 256 * 2**20
    print(f"[ref] loading streamed model (pool={pool_bytes // (2**20)} MiB)",
          file=sys.stderr, flush=True)
    streamed = load_streamed(args.checkpoint, pool_bytes=pool_bytes,
                             config_check=validate_config, verify=True)
    core = EngineCore(streamed, chunk_tokens=1, int8_kv=False)

    # Prime the Mamba caches with explicit zero state so the first token uses
    # the same fused ssm_update_kernel the C++ primitive implements. With
    # state=None mlx-lm would take the (non-bit-identical) ssm_attn S=1 path.
    H = int(config["mamba_num_heads"])
    Dh = int(config["mamba_head_dim"])
    Ds = int(config["ssm_state_size"])
    G = int(config["n_groups"])
    K = int(config["conv_kernel"])
    conv_dim = H * Dh + 2 * G * Ds
    primed = 0
    for entry in core.cache:
        if hasattr(entry, "cache") and len(entry.cache) == 2 and entry.cache[1] is None:
            entry[0] = mx.zeros((1, K - 1, conv_dim), mx.bfloat16)
            entry[1] = mx.zeros((1, H, Dh, Ds), mx.float32)
            primed += 1
    print(f"[ref] primed {primed} Mamba caches with zero state", file=sys.stderr,
          flush=True)

    logit_digests: list[str] = []

    def digest(logits):
        row = np.asarray(logits.reshape(-1).astype(mx.float32), dtype="<f4")
        return hashlib.sha256(row.tobytes()).hexdigest()

    print(f"[ref] prefill {len(prompt)} tokens (chunk=1)", file=sys.stderr,
          flush=True)
    logits = core.prefill(prompt)
    token_ids: list[int] = []
    steps: list[dict] = []
    for i in range(args.tokens):
        d = digest(logits)
        token = int(mx.argmax(logits.reshape(-1)).item())
        logit_digests.append(d)
        token_ids.append(token)
        steps.append({"token": token, "logit_sha256": d})
        print(f"[ref] step {i}: token={token} sha={d[:12]}", file=sys.stderr,
              flush=True)
        if i + 1 < args.tokens:
            logits = core.decode(token)

    out = {
        "checkpoint": args.checkpoint,
        "prompt_ids": prompt,
        "prompt_sha256": prompt_sha,
        "prompt_tokens": len(prompt),
        "requested_tokens": args.tokens,
        "token_ids": token_ids,
        "n_logit_digests": len(logit_digests),
        "steps": steps,
        "mlx_version": mx.__version__,
        "effective_config": {
            "sources": EFFECTIVE_CONFIG_SOURCES,
            "env": {k: os.environ.get(k) for k in (
                "VEYSTRIX_NEMOTRON_KV", "VEYSTRIX_NEMOTRON_GATHER",
                "VEYSTRIX_NEMOTRON_FAST_SSM", "VEYSTRIX_NEMOTRON_PARALLEL_MAMBA",
                "VEYSTRIX_NEMOTRON_NATIVE_MTP", "VEYSTRIX_NEMOTRON_PREFIX_CACHE")},
            "pool_bytes": pool_bytes,
            "primed_mamba_caches": primed,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}: {len(token_ids)} tokens, "
          f"{len(logit_digests)} logit digests")
    streamed.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
