#!/usr/bin/env python3
"""Generate bit-exact vectors for the Qwen4 final head (gated_residual
use_combine=False -> mixed_input -> lm_head) using the REAL mlx_vlm op path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedResidual  # noqa: E402


class Args:
    rms_norm_eps = 1e-6
    hc_count = 4
    hidden_size = 32
    hc_lowrank = 16


def save(d, n, a):
    mx.save(str(d / f"{n}.npy"), a)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "cpp/engine/tests/vectors_final_head")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4242)
    HC, HD, LOW, VOCAB = 4, 32, 16, 64
    hc_hidden = HC * HD

    hidden = mx.array(rng.normal(size=(1, hc_hidden)).astype(np.float32))
    nw = mx.array((rng.random(hc_hidden).astype(np.float32) + 0.5))  # one-centered

    # Real Qwen4ExpGatedResidual(use_combine=False).
    mixer = Qwen4ExpGatedResidual(Args(), use_combine=False)
    mixer.weight = nw
    mixer.input_mix_weight_down.weight = mx.array(rng.normal(size=(LOW, hc_hidden)).astype(np.float32) * 0.1)
    mixer.input_mix_weight_up.weight = mx.array(rng.normal(size=(hc_hidden, LOW)).astype(np.float32) * 0.1)
    mixed = mixer(hidden)

    lm_w = mx.array(rng.normal(size=(VOCAB, HD)).astype(np.float32) * 0.1)
    logits = mx.matmul(mixed, lm_w.T)

    save(out, "hidden", hidden)
    save(out, "norm_weight", nw)
    save(out, "down_w", mixer.input_mix_weight_down.weight)
    save(out, "up_w", mixer.input_mix_weight_up.weight)
    save(out, "lm_head_w", lm_w)
    save(out, "expected", logits)
    (out / "meta.json").write_text(json.dumps(
        {"hc_count": HC, "hidden_size": HD, "lowrank": LOW, "vocab": VOCAB, "eps": 1e-6},
        indent=2) + "\n")
    print(f"wrote final-head vectors to {out}")


if __name__ == "__main__":
    main()
