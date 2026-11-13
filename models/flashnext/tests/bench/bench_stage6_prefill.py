"""Paired Flash-Next prefill and exact prefix-restore measurements."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from collections import defaultdict
import statistics
from pathlib import Path
import tempfile
from unittest.mock import patch

from models.flashnext.tests.bench.bench_production import benchmark_provenance, write_evidence
from veystrix.bench.memory import MemorySampler
from veystrix.bench.provenance import collect
from veystrix.bench.roofline import arm_metrics, resolution
from veystrix.bench.telemetry import Trace
from veystrix.results import output_path, ensure_inside

ROOT = Path(__file__).resolve().parents[4]
SIZES = (2048, 8192, 16384, 32768)


def prompt_ids(backend, size, anchor, corpus):
    prefix = backend.encode("<|im_start|>system\n")
    marker = backend.encode("<|im_end|>\n")
    user = backend.encode("<|im_start|>user\nSummarize this code.\n")
    assistant = backend.encode("<|im_end|>\n<|im_start|>assistant\n")
    body = backend.encode(corpus)
    system_count = anchor - len(prefix) - len(marker)
    user_count = size - anchor - len(user) - len(assistant)
    if min(system_count, user_count) < 0 or not body:
        raise ValueError("prompt is too small or the source corpus is empty")
    padding = (body * (1 + (system_count + user_count) // len(body)))
    return prefix + padding[:system_count] + marker + user + padding[system_count:system_count + user_count] + assistant


def digest(ids):
    return hashlib.sha256(str(tuple(ids)).encode()).hexdigest()


def one(backend, ids, tokens, state_check):
    import mlx.core as mx
    from models.flashnext.diskio import ReadMeter
    from models.flashnext.tests.bench.layer_gate_probe import state_record

    backend.reset()
    meter = ReadMeter()
    trace = Trace()
    memory = MemorySampler(trace.origin)
    memory.prepare()
    mx.reset_peak_memory()

    def prefilled():
        nonlocal prefill_bytes
        trace.mark("prefill_callback")
        prefill_bytes = meter.bytes_since()
        meter.reset()
        trace.decode_read_meter = meter.bytes_since
        trace.mark("prefill_callback_end")

    prefill_bytes = None
    with memory, trace.activate():
        trace.mark("admission")
        backend.append_tokens(ids)
        _, stats = backend.generate(
            tokens, on_prefilled=prefilled,
            on_decode_token=lambda _token, _piece: trace.mark("token_emitted"),
        )
    prefix = backend.last_prefix
    choice = backend.last_prefill_choice
    row = {
        "prompt_tokens": len(ids),
        "prompt_sha256": digest(ids),
        "token_sha256": digest(backend.tape[len(ids):]),
        "output_tokens": stats.tokens,
        "cache_consistent": backend.check_invariant(),
        "prefix": prefix.as_dict() if prefix is not None else None,
        "choice": asdict(choice) if choice is not None else None,
        "metrics": arm_metrics(trace.events, trace.prefill_tokens, trace.decode_read_bytes, None),
        "prefill_physical_bytes": prefill_bytes,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "memory": memory.summary(),
    }
    if state_check:
        state = state_record(backend.cache, backend.tape, backend.language)
        row["state_sha256"] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    return row


def summarize(rows):
    groups = {}
    for size in sorted({row["size"] for row in rows}):
        for cache in ("cold", "prefix-hit"):
            controls = {r["round"]: r for r in rows if r["size"] == size and r["cache"] == cache and r["arm"] == "reference"}
            pairs = [(controls[r["round"]], r) for r in rows
                     if r["size"] == size and r["cache"] == cache and r["arm"] == "candidate" and r["round"] in controls]
            groups[f"{size}/{cache}"] = {
                "diagnostics": [{
                    "round": a["round"], "ttft_ns": [a["metrics"]["ttft_ns"], b["metrics"]["ttft_ns"]],
                    "computed_prefill_tok_s": [a["metrics"]["computed_prefill_tok_s"],
                                               b["metrics"]["computed_prefill_tok_s"]],
                } for a, b in pairs],
                "pairs": len(pairs),
                "token_parity": all(a["token_sha256"] == b["token_sha256"] for a, b in pairs) if pairs else None,
                "state_parity": (all(a["state_sha256"] == b["state_sha256"] for a, b in pairs)
                                 if pairs and all("state_sha256" in r for pair in pairs for r in pair) else None),
                "candidate_minus_reference": {
                    key: resolution([b["metrics"][key] - a["metrics"][key] for a, b in pairs
                                     if a["metrics"][key] is not None and b["metrics"][key] is not None])
                    for key in ("ttft_ns", "prefill_ns", "stream_decode_tok_s")
                },
            }
    return groups


def shape_profile(rows):
    samples = defaultdict(list)
    for row in rows:
        if row["cache"] != "cold" or row["arm"] != "reference":
            continue
        for chunk in row["prefix"]["chunk_trace"]:
            if chunk["end"] != row["prompt_tokens"] and chunk["seconds"] > 0:
                samples[chunk["tokens"]].append(chunk["seconds"])
    return {"schema_version": 1,
            "seconds_by_chunk": {width: statistics.median(times) for width, times in sorted(samples.items())},
            "samples_by_chunk": {width: len(times) for width, times in sorted(samples.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, choices=SIZES, default=SIZES)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--shape-profile", type=Path, help="optional measured shape profile for the autotuner")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--candidate", choices=("autotune", "layer", "kernels"), default="autotune")
    parser.add_argument("--chunk-log", action="store_true", help="write progress at completed chunk boundaries")
    parser.add_argument("--state-check", action="store_true", help="hash full continuation state outside timing")
    args = parser.parse_args()
    if not 64 <= args.chunk_tokens <= 2048:
        parser.error("--chunk-tokens must be between 64 and 2048")
    if args.rounds < 1 or args.tokens < 2:
        parser.error("rounds must be positive and tokens must be at least two")
    out = output_path("flashnext", "stage6-prefill", "stage6.json")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    evidence = {"schema_version": 1, "status": "running", "rows": [],
                "candidate": args.candidate, "sizes": args.sizes, "rounds": args.rounds,
                "output_limit": args.tokens, "state_check": args.state_check,
                "provenance": {**benchmark_provenance(checkpoint), **collect(ROOT, checkpoint)}}
    candidate = {
        "autotune": {"FLASHNEXT_PREFILL_AUTOTUNE": "1"},
        "layer": {"FLASHNEXT_PREFILL_ORDER": "layer", "FLASHNEXT_PREFILL_WINDOW_TOKENS": "4096",
                  "FLASHNEXT_LAYER_MAJOR_MIN_TOKENS": "0"},
        "kernels": {"FLASHNEXT_QSA_PREFILL_TILE_EVAL": "1"},
    }[args.candidate]
    if args.shape_profile is not None:
        candidate["FLASHNEXT_PREFILL_SHAPE_PROFILE"] = str(args.shape_profile.expanduser().resolve(strict=True))
    env = {
        "FLASHNEXT_PREFILL_AUTOTUNE": "0", "FLASHNEXT_PREFILL_ORDER": "chunk",
        "FLASHNEXT_QSA_PREFILL_TILE_EVAL": "0", "FLASHNEXT_QSA_KERNEL": "reference",
        "FLASHNEXT_AUTO_PURGE": "0", "FLASHNEXT_PREFIX_PROFILES": "0",
        "FLASHNEXT_PREFILL_ADAPTIVE_CHUNKS": "0", "FLASHNEXT_PREFILL_RELEASE_DECODE_POOL": "0",
        "FLASHNEXT_PREFIX_ASYNC_WRITE": "0", "FLASHNEXT_EXPERT_PREFETCH": "0",
        "FLASHNEXT_PREFILL_CHUNK_TOKENS": str(args.chunk_tokens), "FLASHNEXT_PREFILL_CHUNK_POLICY": "balanced",
        "FLASHNEXT_PREFILL_RING_EXPERTS": "0", "FLASHNEXT_UNCACHED_EXPERT_READS": "prefill",
        "FLASHNEXT_PREFIX_CACHE": "1", "FLASHNEXT_PREFIX_CACHE_RESTORE": "1",
        "FLASHNEXT_PREFIX_CACHE_BOUNDARIES": "system", "FLASHNEXT_PREFIX_CACHE_RAM_GB": "1",
        "FLASHNEXT_NATIVE_MTP": "off", "FLASHNEXT_PROMPT_LOOKUP": "0",
        "FLASHNEXT_EXPERT_POOL_GB": "0", "FLASHNEXT_SLAB_GLOBAL": "0", "FLASHNEXT_SLAB_PACK": "0",
        "FLASHNEXT_CHAT_GUARD": "0", "FLASHNEXT_WATCHDOG": "0", "FLASHNEXT_WATCHDOG_MEMORY": "0",
        "FLASHNEXT_REQUEST_ADMISSION": "0", "FLASHNEXT_MEMORY_MANAGER": "0",
        "FLASHNEXT_AUTO_MEMORY_RECLAIM": "0", "VEYSTRIX_MAX_COLD_PREFILL": "32768",
        "VEYSTRIX_MAX_INCREMENTAL_PREFILL": "32768",
    }
    if args.candidate == "kernels":
        # Same query/SDPA tile shape in both arms. Only evaluation
        # scheduling changes, which preserves the arithmetic.
        env["FLASHNEXT_QSA_QUERY_CHUNK"] = "512"
    evidence["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    evidence["environment"] = env
    evidence["candidate_environment"] = candidate
    write_evidence(out, evidence)
    try:
        with patch.dict(os.environ, env):
            from models.flashnext import linear_state, pool_lifecycle, gpu_sync
            from models.flashnext.settings.launch import apply_chat_environment
            from veystrix.backends.flashnext import FlashNextBackend

            apply_chat_environment(os.environ)
            evidence["environment"] = {**env, **{
                name: os.environ[name] for name in apply_chat_environment({})
            }}
            backend = FlashNextBackend(model_path=str(checkpoint))
            corpus = "\n".join((ROOT / path).read_text() for path in (
                "models/flashnext/prefill.py", "models/flashnext/prefix_session.py",
                "models/flashnext/layer_major.py", "models/flashnext/store.py",
            ))
            try:
                for size in args.sizes:
                    ids = prompt_ids(backend, size, size - 256, corpus)
                    for index in range(args.rounds):
                        order = ("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")
                        for arm in order:
                            with tempfile.TemporaryDirectory(dir=out.parent, prefix="prefix-") as directory:
                                flags = {"VEYSTRIX_CACHE_ROOT": directory}
                                if args.chunk_log:
                                    flags["FLASHNEXT_CHUNK_LOG"] = str(ensure_inside(
                                        out.with_name(f"{size}-{index + 1}-{arm}-chunks.jsonl")
                                    ))
                                if arm == "candidate":
                                    flags.update(candidate)
                                with patch.dict(os.environ, flags):
                                    linear_state.set_enabled(arm == "candidate" and args.candidate == "kernels")
                                    backend._prefix = None
                                    for cache in ("cold", "prefix-hit"):
                                        print(size, index + 1, arm, cache, "started", flush=True)
                                        row = one(backend, ids, args.tokens, args.state_check)
                                        evidence["rows"].append({**row, "size": size, "round": index + 1,
                                                                 "arm": arm, "cache": cache})
                                        write_evidence(out, evidence)
                                        print(size, index + 1, arm, cache, row["prefix"]["hit_tokens"], flush=True)
                                        if not row["cache_consistent"] or (
                                            row["prefix"]["hit_tokens"] + row["prefix"]["prefilled_tokens"] != size
                                        ):
                                            raise RuntimeError("prompt token accounting or continuation cache is inconsistent")
                                        if cache == "cold":
                                            if row["prefix"]["hit_tokens"]:
                                                raise RuntimeError("cold arm unexpectedly restored a prefix")
                                            first = row
                                        elif (not row["prefix"]["hit_tokens"] or row["token_sha256"] != first["token_sha256"]
                                              or row.get("state_sha256") != first.get("state_sha256")):
                                            raise RuntimeError("prefix restore missed or changed the exact continuation")
            finally:
                linear_state.set_enabled(False)
                pool_lifecycle.release_decode_pool(backend.store, synchronize=gpu_sync.synchronize)
                backend.store.close()
        evidence["summary"] = summarize(evidence["rows"])
        write_evidence(out.with_name("shape-profile.json"), shape_profile(evidence["rows"]))
        if any(row["token_parity"] is False or row["state_parity"] is False
               for row in evidence["summary"].values()):
            raise RuntimeError("candidate changed the reference continuation")
        evidence["status"] = "completed"
    except BaseException as error:
        evidence.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_evidence(out, evidence)
    print(out)


if __name__ == "__main__":
    main()
