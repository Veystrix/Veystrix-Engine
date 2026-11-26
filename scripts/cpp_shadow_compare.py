#!/usr/bin/env python3
"""Compare Python and C++ shadow-v2 records and report the first divergence.

Records are JSONL from VEYSTRIX_CPP_SHADOW_LOG (cpp) and the equivalent Python
engine hook. Pairs are matched on (prompt hash, seed, chunk boundaries, kv mode)
so a prefix-cache hit or a different chunk plan is never mistaken for a numeric
divergence.

Schema 2 records carry REAL float32-LE logit digests (step_digests, 64 hex
chars). A token_id-source record is reported as non-comparable, never as a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def key(rec):
    prompt = rec.get("prompt_token_ids") or []
    digest = hashlib.sha256(repr(prompt).encode()).hexdigest()[:16]
    return (digest, rec.get("seed"),
            repr(rec.get("chunk_boundaries")), rec.get("kv"),
            repr(rec.get("sampler")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("python_log")
    ap.add_argument("cpp_log")
    args = ap.parse_args()
    py, cpp = load(args.python_log), load(args.cpp_log)
    by_key = {}
    for r in py:
        by_key.setdefault(key(r), []).append(r)
    pairs = 0
    problems = []
    for c in cpp:
        matches = by_key.get(key(c), [])
        if not matches:
            problems.append(f"no Python record for key {key(c)}")
            continue
        p = matches[0]
        pairs += 1
        if c.get("schema") != 2 or p.get("schema") != 2:
            problems.append("non-schema-2 record (cannot compare logits)")
            continue
        if c.get("digest_source") != "logit" or p.get("digest_source") != "logit":
            problems.append("token-id digest source (cannot compare logits)")
            continue
        pt, ct = p.get("emitted_token_ids", []), c.get("emitted_token_ids", [])
        pd, cd = p.get("step_digests", []), c.get("step_digests", [])
        for i, (a, b) in enumerate(zip(pt, ct)):
            if a != b:
                problems.append(f"token divergence at step {i}: py={a} cpp={b}")
                break
        for i, (a, b) in enumerate(zip(pd, cd)):
            if a != b:
                problems.append(f"logit digest divergence at step {i}: py={a} cpp={b}")
                break
        else:
            if len(pt) != len(ct) or len(pd) != len(cd):
                problems.append("stream length mismatch")
    print(f"compared {pairs} pairs")
    for problem in problems:
        print("FAIL:", problem)
    if problems:
        return 1
    print("OK: all paired token + logit digest streams identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
