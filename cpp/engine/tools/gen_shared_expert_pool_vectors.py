#!/usr/bin/env python3
"""Generate tiny deterministic CPU-only golden payloads for test_shared_expert_pool.

NO real checkpoint, NO model run, NO SSD benchmark. Records come from a 64-bit
LCG; the C++ test builds the *same* bytes with the *same* LCG, so records.bin is
a cross-language byte-exactness reference (records.bin is compared directly by
the test when a vector dir is passed). There is no runtime dependency on this
tool.

Usage:
    python3 cpp/engine/tools/gen_shared_expert_pool_vectors.py \
        --out cpp/engine/tests/vectors_shared_expert_pool

The LCG and byte order MUST match test_shared_expert_pool.cpp's make_golden().
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RECORD_BYTES = 256          # multiple of 4 so a uint32 word view is byte-exact
LAYERS = 2
EXPERTS = 8
SEED = 0x9E3779B97F4A7C15
MUL = 6364136223846793005
ADD = 1442695040888963407
MASK = (1 << 64) - 1

# Layer-0 route multiset used by the union-fetch test (has repeats).
UNION_EXPERTS = [0, 1, 0, 2, 1, 3, 2]
UNION_DISTINCT = list(dict.fromkeys(UNION_EXPERTS))


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = SEED
    records: dict[str, bytes] = {}
    blob = bytearray()
    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            data = bytearray(RECORD_BYTES)
            for i in range(RECORD_BYTES):
                state = (state * MUL + ADD) & MASK
                data[i] = (state >> 33) & 0xFF
            records[f"{layer}:{expert}"] = bytes(data)
            blob += data

    records_path = out_dir / "records.bin"
    records_path.write_bytes(blob)
    union_gather = b"".join(records[f"0:{e}"] for e in UNION_DISTINCT)
    (out_dir / "union_gather.bin").write_bytes(union_gather)

    manifest = {
        "generator": "gen_shared_expert_pool_vectors.py",
        "note": "CPU-only synthetic LCG bytes; no checkpoint, no benchmark",
        "records_bin": records_path.name,
        "records_bytes": len(blob),
        "records_sha256": hashlib.sha256(blob).hexdigest(),
        "record_bytes": RECORD_BYTES,
        "layers": LAYERS,
        "experts": EXPERTS,
        "lcg": {"seed": hex(SEED), "mul": MUL, "add": ADD,
                "byte": "(state >> 33) & 0xff"},
        "union_layer": 0,
        "union_experts": UNION_EXPERTS,
        "union_distinct": UNION_DISTINCT,
        "union_gather_sha256": hashlib.sha256(union_gather).hexdigest(),
        "record_offset": {key: i * RECORD_BYTES for i, key in enumerate(records)},
        "record_sha256": {key: hashlib.sha256(v).hexdigest()
                          for key, v in records.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="cpp/engine/tests/vectors_shared_expert_pool",
                    help="output directory (default: %(default)s)")
    args = ap.parse_args()
    m = build(Path(args.out))
    print(f"wrote {m['records_bin']} ({m['records_bytes']} bytes) + "
          f"union_gather.bin + manifest.json")
    print(f"records sha256: {m['records_sha256']}")
    print(f"union gather sha256: {m['union_gather_sha256']}")


if __name__ == "__main__":
    main()
