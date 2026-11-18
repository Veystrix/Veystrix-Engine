#!/usr/bin/env python3
"""Generate bit-exact parity vectors for the non-greedy sampling path.

Every expected value is produced by the REAL production ops in
veystrix/sampling.py (Sampler.distribution / Sampler._mask / Sampler.__call__),
using the same MLX 0.32.2 that the C++ engine links against. The C++ test
(cpp/engine/tests/test_sampling.cpp) loads these .npy files with mx::load and
compares raw bytes.

We store, per case:
  logits.npy  - the input row (float32, shape (vocab,))
  masked.npy  - Sampler.distribution(logits)[1]: the -inf-masked logit row the
                categorical draw reads (shape (vocab,) or (top_k,))
  kept.npy    - Sampler.distribution(logits)[0]: survivor token ids for the
                top-k path (empty int32 array on the full-vocabulary path)
  token.npy   - Sampler(logits) with mx.random.seed(seed): the final uint32
                token (shape (1,)), including the categorical draw and remap
  meta.json   - params, seen set, seed, shapes, has_ids

Note on RNG: MLX Python and MLX C++ share the same PRNG. Seeding with
mx.random.seed(seed) vs mlx::core::random::seed(seed) gives bit-identical
categorical draws for the same row, so the final token is reproducible too.

Usage:
  .venv/bin/python cpp/engine/tools/gen_sampling_vectors.py \
      --out cpp/engine/tests/sampling_vectors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Import the production sampler rather than reimplementing its ops.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from veystrix.sampling import Sampler, Sampling  # noqa: E402


# Each case exercises a distinct branch of distribution()/_mask().
CASES = [
    dict(name="case0_temp_only", vocab=32, temperature=1.0, top_p=1.0,
         top_k=0, min_p=0.0, presence_penalty=0.0, seen=[], seed=11,
         data_seed=101),
    dict(name="case1_temp_scaled", vocab=64, temperature=0.7, top_p=1.0,
         top_k=0, min_p=0.0, presence_penalty=0.0, seen=[], seed=12,
         data_seed=102),
    dict(name="case2_topk_only", vocab=64, temperature=1.0, top_p=1.0,
         top_k=20, min_p=0.0, presence_penalty=0.0, seen=[], seed=13,
         data_seed=103),
    dict(name="case3_minp_only", vocab=48, temperature=1.0, top_p=1.0,
         top_k=0, min_p=0.2, presence_penalty=0.0, seen=[], seed=14,
         data_seed=104),
    dict(name="case4_topp_only", vocab=48, temperature=1.0, top_p=0.8,
         top_k=0, min_p=0.0, presence_penalty=0.0, seen=[], seed=15,
         data_seed=105),
    dict(name="case5_thinking_full", vocab=128, temperature=1.0, top_p=0.95,
         top_k=20, min_p=0.0, presence_penalty=0.0, seen=[], seed=16,
         data_seed=106),
    dict(name="case6_presence_full", vocab=64, temperature=0.7, top_p=0.80,
         top_k=0, min_p=0.0, presence_penalty=1.5, seen=[3, 10, 27, 41],
         seed=17, data_seed=107),
    dict(name="case7_presence_topk", vocab=100, temperature=1.0, top_p=0.9,
         top_k=10, min_p=0.05, presence_penalty=1.0, seen=[0, 5, 19, 60, 88],
         seed=18, data_seed=108),
    dict(name="case8_topk_one", vocab=32, temperature=1.0, top_p=1.0,
         top_k=1, min_p=0.0, presence_penalty=0.0, seen=[], seed=19,
         data_seed=109),
    dict(name="case9_dominant_nucleus", vocab=32, temperature=1.0, top_p=0.5,
         top_k=0, min_p=0.0, presence_penalty=0.0, seen=[], seed=20,
         data_seed=110),
    # Exact INSTRUCT preset (0.7 / 0.80 / top_k=20 / min_p=0 / penalty=1.5)
    # exercised together with a seen-token history so presence penalty,
    # temperature scaling, top-k, top-p and the seeded draw all compose.
    dict(name="case10_instruct_topk", vocab=96, temperature=0.7, top_p=0.80,
         top_k=20, min_p=0.0, presence_penalty=1.5,
         seen=[2, 7, 13, 31, 55, 80], seed=21, data_seed=111),
]


def emit_case(out: Path, cfg: dict) -> dict:
    name = cfg["name"]
    d = out / name
    d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["data_seed"])
    vals = rng.normal(size=(cfg["vocab"],)).astype(np.float32)
    # A dominant token for the nucleus-first-crossing case.
    if name == "case9_dominant_nucleus":
        vals[7] = 20.0
    row = mx.array(vals, dtype=mx.float32)

    settings = Sampling(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        min_p=cfg["min_p"],
        presence_penalty=cfg["presence_penalty"],
    )
    sampler = Sampler(settings)
    for t in cfg["seen"]:
        sampler.observe(t)  # no-op when presence_penalty == 0

    kept, masked = sampler.distribution(row)
    mx.eval(masked)
    has_ids = kept is not None
    if has_ids:
        mx.eval(kept)
        mx.save(str(d / "kept.npy"), kept)
    else:
        np.save(str(d / "kept.npy"), np.array([], dtype=np.int32))

    # Seeded final draw: one categorical from the distribution, then remap.
    mx.random.seed(cfg["seed"])
    token = sampler(row)  # non-greedy branch -> uint32, shape (1,)
    mx.eval(token)

    mx.save(str(d / "logits.npy"), row)
    mx.save(str(d / "masked.npy"), masked)
    mx.save(str(d / "token.npy"), token)

    meta = {
        "name": name,
        "vocab": cfg["vocab"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "min_p": cfg["min_p"],
        "presence_penalty": cfg["presence_penalty"],
        "seen": list(cfg["seen"]),
        "seed": cfg["seed"],
        "has_ids": has_ids,
        "masked_len": int(masked.shape[0]),
        "mlx_version": mx.__version__,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cpp/engine/tests/sampling_vectors")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = []
    for cfg in CASES:
        emit_case(out, cfg)
        names.append(cfg["name"])

    (out / "cases.txt").write_text("\n".join(names) + "\n")
    # Also emit a flat copy of case0 at the top level for quick manual checks.
    flat = out / "case0_temp_only"
    for f in flat.glob("*.npy"):
        (out / f.name).write_bytes(f.read_bytes())
    (out / "meta.json").write_text((flat / "meta.json").read_text())
    print(f"wrote {len(names)} sampling parity cases to {out}")


if __name__ == "__main__":
    main()
