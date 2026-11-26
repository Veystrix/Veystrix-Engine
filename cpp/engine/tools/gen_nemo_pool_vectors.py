#!/usr/bin/env python3
"""Generate the tiny deterministic shard + manifest for test_nemo_pool.

NO real checkpoint, NO model run. The shard is pure bytes from a 64-bit LCG;
the C++ unit test writes the *same* bytes with the *same* LCG in a temp dir,
so this tool is a cross-language vector source / digest reference rather than
a runtime dependency. It also gives the owner a quick way to reproduce the
exact physical bytes a given route list should read.

Usage:
    .venv/bin/python cpp/engine/tools/gen_nemo_pool_vectors.py \
        --out cpp/engine/tests/vectors_nemo_pool

The LCG and byte order below MUST match test_nemo_pool.cpp's write_shard().
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RECORD_BYTES = 1024          # multiple of 4 so a uint32 view is byte-exact
LAYERS = 2
EXPERTS = 8
SEED = 0x9E3779B97F4A7C15
MUL = 6364136223846793005
ADD = 1442695040888963407
MASK = (1 << 64) - 1

# A representative route list (includes repeats) used for the gather digest.
GATHER_ROUTES = [(0, 0), (1, 7), (0, 4), (1, 1), (0, 0), (0, 3), (1, 7), (1, 2)]


def record_bytes(state: int, count: int) -> tuple[bytes, int]:
    """Advance the LCG `count` times and return (bytes, new_state)."""
    out = bytearray(count)
    for i in range(count):
        state = (state * MUL + ADD) & MASK
        out[i] = (state >> 33) & 0xFF
    return bytes(out), state


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = SEED
    records: dict[str, bytes] = {}
    shard = bytearray()
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            data, state = record_bytes(state, RECORD_BYTES)
            records[f"{layer}:{expert}"] = data
            shard += data

    shard_path = out_dir / "pool_shard.bin"
    shard_path.write_bytes(shard)

    gather = b"".join(records[f"{l}:{e}"] for l, e in GATHER_ROUTES)
    manifest = {
        "generator": "gen_nemo_pool_vectors.py",
        "shard": shard_path.name,
        "shard_bytes": len(shard),
        "shard_sha256": hashlib.sha256(shard).hexdigest(),
        "record_bytes": RECORD_BYTES,
        "layers": LAYERS,
        "experts": EXPERTS,
        "lcg": {"seed": hex(SEED), "mul": MUL, "add": ADD, "byte": "(state >> 33) & 0xff"},
        "record_offset": {key: i * RECORD_BYTES for i, key in enumerate(records)},
        "record_sha256": {key: hashlib.sha256(v).hexdigest() for key, v in records.items()},
        "gather_routes": GATHER_ROUTES,
        "gather_sha256": hashlib.sha256(gather).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="cpp/engine/tests/vectors_nemo_pool",
                    help="output directory (default: %(default)s)")
    args = ap.parse_args()
    m = build(Path(args.out))
    print(f"wrote {m['shard']} ({m['shard_bytes']} bytes) + manifest.json")
    print(f"shard sha256:  {m['shard_sha256']}")
    print(f"gather sha256: {m['gather_sha256']}")


if __name__ == "__main__":
    main()
