"""Stage 11 single-request release matrix, with explicit evidence for each cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import tempfile
from unittest.mock import patch

from .provenance import checkpoint_metadata, collect
from .roofline import arm_metrics, resolution
from .telemetry import Trace
from veystrix.results import ROOT, output_path


MODELS = ("flashnext", "nemotron35")
SIZES = (512, 2048, 8192, 16384, 32768)
OUTPUTS = (128, 512)
MODES = ("cold-process", "warm-process-cold-prefix", "prefix-hit")
FLOOR = 3 * 2**30


def token_digest(ids):
    return hashlib.sha256(b"".join(int(token).to_bytes(4, "little", signed=True) for token in ids)).hexdigest()


def assess(rows, *, output_tokens, expected_digest):
    """A short, unsafe, or unverified run never contributes to the release gate."""
    valid = []
    for row in rows:
        rate = row.get("stream_decode_tok_s")
        if (row.get("output_tokens") == output_tokens and row.get("cache_consistent") is True
                and (row.get("available_floor_bytes") or 0) >= FLOOR
                and row.get("swap_growth_bytes") == 0 and row.get("swapout_growth_bytes") == 0
                and row.get("guard_passed") is True
                and expected_digest and row.get("token_sha256") == expected_digest
                and isinstance(rate, (int, float)) and math.isfinite(rate) and rate > 0):
            valid.append(rate)
    band = resolution(valid)
    failures = []
    if output_tokens != 512:
        failures.append("512 output tokens required")
    if not expected_digest:
        failures.append("reference digest required")
    if not rows or len(valid) != len(rows):
        failures.append("incomplete, unsafe or mismatched runs")
    if band is None:
        failures.append("five valid repetitions required")
    elif band["median"] < 20 or band["p10"] < 18:
        failures.append("decode below 20 tok/s median or 18 tok/s p10")
    return {"passed": not failures, "resolution": band, "valid_runs": len(valid), "reasons": failures}


def guard_passed(path):
    """The external guard covers setup and thermal/compressor behavior."""
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        start, end = rows[0], rows[-1]
        return (start["event"] == "start" and start["power"]["ac"] is True
                and start["power"]["low_power"] is False and not start["refusal"]
                and end["event"] == "end" and end["returncode"] == 0 and not end["stopped"]
                and end["available_floor_gib"] >= 3 and end["swapout_mib"] == 0
                and end["swap_peak_mib"] == end["swap_baseline_mib"]
                and all(int(level) < 2 for level in end["thermal_seconds"])
                and all(row["available"] >= FLOOR and row["swapouts"] == start["baseline"]["swapouts"]
                        and row["swap_used"] <= start["baseline"]["swap_used"] for row in rows[1:-1]))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def collect_cell(paths, expected_digest):
    from veystrix.results import ensure_inside

    paths = [ensure_inside(path) for path in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate qualification artifacts")
    rows, identity = [], None
    for path in paths:
        data = json.loads(path.read_text())
        signature = tuple(data.get(key) for key in (
            "schema_version", "model", "checkpoint_metadata", "checkpoint_shards",
            "harness_sha256", "runtime_environment", "mode", "prompt_tokens", "output_tokens",
            "checkpoint_content_sha256", "resident_experts", "nemotron_pool_gb", "overrides"))
        host = data["reference_host"]
        signature += (host["code"], host["dependencies"], host["host"]["model"], host["host"]["soc"],
                      host["host"]["macos"], host["host"]["ram_bytes"])
        if identity is not None and signature != identity:
            raise ValueError("qualification artifacts describe different cells or executables")
        identity = signature
        passed = (data["status"] == "completed" and data["concurrency"] == 1
                  and data["warmup_excluded_tokens"] == 0 and host["host"]["ram_bytes"] == 16 * 2**30
                  and bool(data.get("checkpoint_content_sha256"))
                  and guard_passed(path.with_name("guard.jsonl")))
        rows.extend(dict(row, guard_passed=passed) for row in data["rows"] or [{}])
    prompts = {row.get("prompt_sha256") for row in rows if row.get("prompt_sha256")}
    if len(prompts) > 1:
        raise ValueError("qualification artifacts use different prompt tokens")
    result = assess(rows, output_tokens=identity[8], expected_digest=expected_digest)
    return {"model": identity[1], "mode": identity[6], "prompt_tokens": identity[7],
            "output_tokens": identity[8], "artifacts": [str(path) for path in paths], "gate": result,
            "observed_rates": [row.get("stream_decode_tok_s") for row in rows]}


def prompt_ids(backend, size, variant=0):
    system = backend.encode("<|im_start|>system\n")
    end = backend.encode("<|im_end|>")
    tail = backend.encode("\n<|im_start|>user\nExplain the tradeoffs of a B-tree and show a Python implementation."
                          "<|im_end|>\n<|im_start|>assistant\n")
    body = backend.encode(("Review the buffered disk index.\n" if variant == 0 else
                           "Consider an in-memory lookup table.\n") +
                          "Keys are sorted inside each node. Splits propagate separators toward the root.\n")
    count = size - len(system) - len(end) - len(tail)
    if count < 0 or not body:
        raise ValueError("prompt too short for the fixture")
    return system + (body * (1 + count // len(body)))[:count] + end + tail


def once(backend, ids, output_tokens):
    from models.flashnext.diskio import ReadMeter
    from veystrix.backends.base import GenerationCancelled
    from .memory import MemorySampler
    import mlx.core as mx

    backend.reset()
    mx.reset_peak_memory()
    trace = Trace()
    memory = MemorySampler(trace.origin)
    memory.prepare()
    meter = ReadMeter()
    tokens = []

    prefill_bytes = []

    def prefilled():
        prefill_bytes.append(meter.bytes_since())
        meter.reset()

    def emitted(token, _piece):
        trace.mark("qualification_token")
        tokens.append(token)

    def cancelled():
        value = memory.samples[-1]["available_bytes"]
        return value is None or value < FLOOR

    if cancelled():
        raise RuntimeError("available memory below 3 GiB before inference")

    with memory, trace.activate():
        trace.mark("qualification_admission")
        backend.append_tokens(ids)
        try:
            _, stats = backend.generate(output_tokens, on_prefilled=prefilled, on_decode_token=emitted,
                                        should_cancel=cancelled)
        except GenerationCancelled as error:
            if cancelled():
                raise RuntimeError(f"memory floor crossed during inference ({memory.samples[-1]['available_bytes']} bytes)") from error
            raise
        read_bytes = meter.bytes_since()
    emitted_ns = [row["ns"] for row in trace.events if row["event"] == "qualification_token"]
    # Flash-Next already marks its internal emitted tokens. Use the explicit
    # callback timestamps once for both runtimes when calculating Stage 0 rates.
    events = [row for row in trace.events if row["event"] != "token_emitted"]
    events = [dict(row, event="token_emitted") if row["event"] == "qualification_token" else row
              for row in events]
    metrics = arm_metrics(events, trace.prefill_tokens, None, None)
    prefix = backend.last_prefix
    hits = prefix.hit_tokens if prefix is not None else 0
    mem = memory.summary()
    swaps = [row["swap_used_bytes"] for row in mem["samples"]]
    swapout = [row["swapout_bytes_total"] for row in mem["samples"]]
    measured = len(emitted_ns) > 1 and emitted_ns[-1] > emitted_ns[0]
    return {
        "prompt_sha256": token_digest(ids), "token_sha256": token_digest(tokens),
        "stage0_token_sha256": hashlib.sha256(str(tokens).encode()).hexdigest(),
        "output_tokens": len(tokens), "finish": stats.finish, "cache_consistent": backend.check_invariant(),
        "restored_tokens": hits, "computed_prefill_tokens": len(ids) - hits,
        "computed_prefill_tok_s": (metrics["computed_prefill_tok_s"] if metrics["computed_prefill_tok_s"] is not None else
                                  (len(ids) - hits) / stats.prefill_seconds if stats.prefill_seconds else None),
        "ttft_ms": (emitted_ns[0] - trace.events[0]["ns"]) / 1e6 if emitted_ns else None,
        "stream_decode_tok_s": (len(tokens) - 1) * 1e9 / (emitted_ns[-1] - emitted_ns[0]) if measured else None,
        "engine_decode_tok_s": stats.rate,
        "physical_mb_per_token": read_bytes / len(tokens) / 1e6 if tokens and read_bytes >= 0 else None,
        "physical_read_scope": "system-wide, from prefill callback through request return",
        "prefill_physical_bytes": prefill_bytes[0] if prefill_bytes and prefill_bytes[0] >= 0 else None,
        "prefill_reads": prefill_read_stats(),
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "available_floor_bytes": mem["sampled_available_floor_bytes"],
        "swap_growth_bytes": max(swaps) - swaps[0] if all(x is not None for x in swaps) else None,
        "swapout_growth_bytes": max(swapout) - swapout[0] if all(x is not None for x in swapout) else None,
        "target_rounds_per_token": metrics["target_rounds_per_token"] if metrics["target_rounds"] else None,
        "accepted_drafts_per_verification": metrics["accepted_tokens_per_verification"],
        "memory": mem,
    }


def prefill_read_stats():
    """Per-layer expert reads of the last Flash-Next prefill, when the engine recorded them."""
    try:
        from models.flashnext import prefill_reads
    except ImportError:
        return None
    return prefill_reads.summary()


def parse_overrides(pairs):
    """``NAME=VALUE`` engine settings a diagnostic arm changes. They are recorded with the run."""
    overrides = {}
    for pair in pairs or ():
        name, separator, value = pair.partition("=")
        if not separator or not name.startswith(("FLASHNEXT_", "VEYSTRIX_NEMOTRON_", "VEYSTRIX_MEMORY_")):
            raise ValueError(f"--set-env takes FLASHNEXT_*, VEYSTRIX_NEMOTRON_* or VEYSTRIX_MEMORY_* NAME=VALUE, not {pair!r}")
        overrides[name] = value
    return overrides


def load_backend(model, checkpoint, context, directory, *, resident_experts=0, pool_bytes=0):
    if model == "flashnext":
        from models.flashnext.settings.launch import apply_chat_environment
        from veystrix.backends.flashnext import FlashNextBackend

        apply_chat_environment(os.environ)
        return FlashNextBackend(model_path=str(checkpoint), resident_experts=resident_experts)
    from veystrix.backends.nemotron35 import NemotronBackend

    return NemotronBackend(str(checkpoint), context_tokens=context, pool_bytes=pool_bytes, prefix_dir=directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-matrix", action="store_true")
    parser.add_argument("--assess", type=Path, nargs="+", help="assess guarded repetitions of one cell")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt-tokens", type=int, choices=SIZES, default=512)
    parser.add_argument("--output-tokens", type=int, choices=OUTPUTS, default=128)
    parser.add_argument("--mode", choices=MODES, default=MODES[0])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--expect-token-sha256")
    parser.add_argument("--qwen-prefill-16gb", action="store_true")
    parser.add_argument("--seed-prefix-only", action="store_true",
                        help="seed the system prefix without decoding a token before a prefix-hit arm")
    parser.add_argument("--resident-experts", type=int, default=0,
                        help="Flash-Next experts pinned per layer (recorded; repetitions of one cell must match)")
    parser.add_argument("--nemotron-pool-gb", type=float, default=0.0,
                        help="Nemotron expert pool in GiB (recorded; repetitions of one cell must match)")
    parser.add_argument("--set-env", action="append", metavar="NAME=VALUE",
                        help="engine setting applied after the harness defaults and recorded with the run")
    parser.add_argument("--checkpoint-hashes", type=Path,
                        help="offline manifest mapping the checkpoint shard names to verified content SHA-256s")
    args = parser.parse_args(argv)
    if args.list_matrix:
        print(json.dumps([dict(model=model, prompt_tokens=size, output_tokens=count, mode=mode)
                          for model in MODELS for size in SIZES for count in OUTPUTS for mode in MODES], indent=2))
        return 0
    if args.assess:
        verdict = collect_cell(args.assess, args.expect_token_sha256)
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["gate"]["passed"] else 1
    if not args.model or not args.checkpoint or args.rounds < 1:
        parser.error("model, checkpoint and a positive round count are required")
    if args.mode == MODES[0] and args.rounds != 1:
        parser.error("run cold-process repetitions in separate invocations")
    if args.resident_experts < 0 or not 0 <= args.nemotron_pool_gb <= 4:
        parser.error("resident experts must be nonnegative and the Nemotron pool between 0 and 4 GiB")
    try:
        overrides = parse_overrides(args.set_env)
    except ValueError as error:
        parser.error(str(error))

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    out = output_path(args.model, "stage11-qualification", "qualification.json")
    from models.flashnext.tests.bench.bench_production import write_evidence

    index = checkpoint / "model.safetensors.index.json"
    shard_names = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
                   if index.is_file() else sorted(path.name for path in checkpoint.glob("*.safetensors")))
    shards = {name: {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "inode": info.st_ino}
              for name in shard_names for info in [(checkpoint / name).stat()]}
    content = json.loads(args.checkpoint_hashes.read_text()) if args.checkpoint_hashes else None
    if content is not None and (not isinstance(content, dict) or set(content) != set(shards)
                                or not all(isinstance(digest, str) and len(digest) == 64
                                           and all(char in "0123456789abcdef" for char in digest)
                                           for digest in content.values())):
        parser.error("checkpoint hashes must cover exactly the indexed shards with lowercase SHA-256s")
    report = {
        "schema_version": 1, "status": "running", "model": args.model, "checkpoint": str(checkpoint),
        "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens, "mode": args.mode,
        "concurrency": 1, "warmup_excluded_tokens": 0, "expected_token_sha256": args.expect_token_sha256,
        "reference_host": collect(ROOT, checkpoint), "checkpoint_metadata": checkpoint_metadata(checkpoint),
        "checkpoint_shards": shards, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_content_sha256": content,
        "resident_experts": args.resident_experts, "nemotron_pool_gb": args.nemotron_pool_gb,
        "overrides": overrides,
        "rows": [],
    }
    print(out, flush=True)
    write_evidence(out, report)
    try:
        with tempfile.TemporaryDirectory(dir=out.parent, prefix="prefix-") as directory:
            os.environ.update({
                "VEYSTRIX_CACHE_ROOT": directory, "FLASHNEXT_PREFIX_CACHE_DIR": directory,
                "FLASHNEXT_PREFIX_CACHE": "1", "FLASHNEXT_PREFIX_CACHE_BOUNDARIES": "system",
                "FLASHNEXT_PREFIX_CACHE_MIN_TOKENS": "256", "FLASHNEXT_PREFIX_CACHE_RAM_GB": "0",
                "VEYSTRIX_NEMOTRON_PREFIX_CACHE": "1", "FLASHNEXT_AUTO_PURGE": "0",
                "FLASHNEXT_EXPERT_POOL_GB": "0", "FLASHNEXT_SLAB_GLOBAL": "0",
                "FLASHNEXT_SLAB_PACK": "0", "FLASHNEXT_PROMPT_LOOKUP": "0",
                "FLASHNEXT_PREFIX_CACHE_RESTORE": "1", "FLASHNEXT_PREFIX_ASYNC_WRITE": "0",
                "VEYSTRIX_NEMOTRON_PROMPT_LOOKUP": "0",
                "FLASHNEXT_PREFILL_CACHE_LIMIT_MB": "512",
            })
            if args.model == "flashnext" and args.qwen_prefill_16gb:
                from models.flashnext.prefill_16gb import apply

                apply(os.environ)
            os.environ.update(overrides)
            backend = load_backend(args.model, checkpoint, args.prompt_tokens + args.output_tokens, directory,
                                   resident_experts=args.resident_experts,
                                   pool_bytes=int(args.nemotron_pool_gb * 2**30))
            report["runtime_environment"] = {key: value for key, value in sorted(os.environ.items())
                                             if key.startswith(("FLASHNEXT_", "VEYSTRIX_NEMOTRON_", "VEYSTRIX_MEMORY_"))
                                             and key not in ("FLASHNEXT_PREFIX_CACHE_DIR",)
                                             and not any(part in key for part in ("KEY", "SECRET", "PASSWORD"))}
            ids = prompt_ids(backend, args.prompt_tokens)
            if args.mode == MODES[1]:
                once(backend, prompt_ids(backend, args.prompt_tokens, variant=1), 1)
                if args.model == "flashnext":
                    os.environ["FLASHNEXT_PREFIX_CACHE_RESTORE"] = "0"
            elif args.mode == MODES[2]:
                once(backend, ids, 0 if args.seed_prefix_only else 1)
            for _ in range(args.rounds):
                if args.mode == MODES[1] and args.model == "nemotron35":
                    # Keep the cache's system anchor and exact chunk plan.
                    # Suppress only restoration so later repetitions stay cold.
                    with patch.object(backend.prefix_cache, "restore", return_value=(None, 0)):
                        row = once(backend, ids, args.output_tokens)
                else:
                    row = once(backend, ids, args.output_tokens)
                if bool(row["restored_tokens"]) != (args.mode == MODES[2]):
                    report["rejected_row"] = row
                    raise RuntimeError("observed prefix state differs from requested mode")
                if args.expect_token_sha256 and row["token_sha256"] != args.expect_token_sha256:
                    report["rejected_row"] = row
                    raise RuntimeError("continuation digest differs from the reference")
                report["rows"].append(row)
                write_evidence(out, report)
            report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "stopped"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["gate"] = {"passed": False, "reason": "Assess after the external guard has written its terminal record"}
        write_evidence(out, report)
    if report["status"] != "completed":
        print(report["error"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
