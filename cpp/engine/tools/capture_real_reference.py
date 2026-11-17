#!/usr/bin/env python3
"""Guarded REFERENCE capture for the C++ engine cross-check on the REAL checkpoint.

Loads the real Qwen Flash-Next backend, monkeypatches Sampler.__call__ at class
level so the internally-created sampler records every decode step's raw logit
SHA-256 (float32 little-endian, matching logit_reference.py), and runs a short
deterministic greedy decode. Token IDs come from the on_decode_token callback.

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
from pathlib import Path

import mlx.core as mx
import numpy as np


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

    def capturing_call(self, logits):
        row = np.asarray(logits.reshape(-1).astype(mx.float32), dtype="<f4")
        logit_digests.append(hashlib.sha256(row.tobytes()).hexdigest())
        return original_call(self, logits)

    Sampler.__call__ = capturing_call

    # Force greedy in the startup preferences so the internal sampler is greedy.
    import os
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
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}: {len(token_ids)} tokens, {len(logit_digests)} logits")


if __name__ == "__main__":
    main()
