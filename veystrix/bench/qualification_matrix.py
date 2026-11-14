"""Execute the guarded Stage 11 matrix serially on the reference Mac."""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .qualification import MODELS, MODES, OUTPUTS, SIZES, collect_cell, guard_passed
from veystrix.results import ROOT, ensure_inside, new_run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for model in MODELS:
        parser.add_argument("--" + model, type=Path, required=True)
        parser.add_argument("--" + model + "-hashes", type=Path)
        parser.add_argument("--" + model + "-reference", type=Path,
                            help="JSON mapping prompt/output widths (such as 512/512) to independent digests")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--sizes", nargs="+", type=int, choices=SIZES, default=SIZES)
    parser.add_argument("--outputs", nargs="+", type=int, choices=OUTPUTS, default=OUTPUTS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--fail-fast", action="store_true", help="stop the matrix after the first failed cell")
    parser.add_argument("--rounds", type=int, default=1, help="independent guarded processes per cell")
    parser.add_argument("--seconds-per-cell", type=int, default=1800)
    parser.add_argument("--swap-growth-mib", type=float, default=16,
                        help="swap growth the guard tolerates before it stops a cell; zero stopped a healthy "
                             "launch on a 3 MiB tick (14 November 2026)")
    parser.add_argument("--cooldown-seconds", type=int, default=120,
                        help="maximum wait for nominal thermal pressure before each cell")
    parser.add_argument("--qwen-keepwarm", action="store_true", help="keep the GPU busy between Qwen expert reads")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.rounds, args.seconds_per_cell) < 1 or args.cooldown_seconds < 0:
        parser.error("rounds and deadline must be positive and cooldown must be nonnegative")
    from scripts.flashnext.real_run_guard import thermal

    out = (ensure_inside(args.resume) if args.resume else
           new_run_directory("shared", "stage11-matrix-manual") / "matrix.json")
    print(f"matrix: {out}", flush=True)
    with out.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("this matrix is already running")
        return run_matrix(args, out, thermal)


def run_matrix(args, out, thermal):
    results = json.loads(out.read_text()) if out.exists() else []
    def completed_cells():
        return {tuple(row["cell"]) for row in results if row["guard_passed"] and row["status"] == "completed"
                and row["completion_tokens"] == [row["cell"][2]]}

    done = completed_cells()
    checkpoints = {model: getattr(args, model).expanduser().resolve(strict=True) for model in args.models}
    references = {
        model: json.loads(path.expanduser().read_text())
        for model in args.models if (path := getattr(args, model + "_reference"))
    }
    for size, count, mode, repeat, model in itertools.product(
            args.sizes, args.outputs, args.modes, range(1, args.rounds + 1), args.models):
        cell = (model, size, count, mode, repeat)
        if cell in done:
            continue
        until = time.monotonic() + args.cooldown_seconds
        while thermal() > 0 and time.monotonic() < until:
            time.sleep(5)
        folder = new_run_directory(model, f"stage11-{size}-{count}-{mode}-r{repeat}")
        guard = folder / "guard.jsonl"
        env = dict(os.environ, VEYSTRIX_RESULTS_DIR=str(folder), VEYSTRIX_MEMORY_POLICY="legacy",
                   FLASHNEXT_NATIVE_MTP="off", VEYSTRIX_NEMOTRON_NATIVE_MTP="0", VEYSTRIX_NEMOTRON_KV="bf16",
                   FLASHNEXT_PREFILL_CHUNK_TOKENS="128", VEYSTRIX_NEMOTRON_PREFILL_CACHE_LIMIT_MB="512")
        command = [sys.executable, str(ROOT / "scripts/flashnext/real_run_guard.py"),
                   "--log", str(guard), "--floor-samples", "1", "--swap-growth-mib", str(args.swap_growth_mib),
                   "--max-seconds", str(args.seconds_per_cell), "--", sys.executable, "-m",
                   "veystrix.bench.qualification", "--model", model, "--checkpoint", str(checkpoints[model]),
                   "--prompt-tokens", str(size), "--output-tokens", str(count), "--mode", mode]
        if model == "flashnext":
            command.append("--qwen-prefill-16gb")
            env["FLASHNEXT_GPU_KEEPWARM"] = "1" if args.qwen_keepwarm else "0"
        if hashes := getattr(args, model + "_hashes"):
            command.extend(("--checkpoint-hashes", str(hashes.expanduser().resolve(strict=True))))
        if digest := references.get(model, {}).get(f"{size}/{count}"):
            command.extend(("--expect-token-sha256", digest))
        print(f"{cell}: {folder}", flush=True)
        run = subprocess.run(command, env=env, cwd=ROOT)
        evidence = folder / "qualification.json"
        artifact = json.loads(evidence.read_text()) if evidence.exists() else {}
        rows = artifact.get("rows", [])
        results.append({"cell": cell, "returncode": run.returncode, "guard_passed": guard_passed(guard),
                        "status": artifact.get("status", "refused"),
                        "token_sha256": [row["token_sha256"] for row in rows],
                        "completion_tokens": [row["output_tokens"] for row in rows],
                        "stream_tok_s": [row["stream_decode_tok_s"] for row in rows],
                        "restored_tokens": [row["restored_tokens"] for row in rows],
                        "guard": str(guard.relative_to(ROOT)),
                        "evidence": str(evidence.relative_to(ROOT)) if evidence.exists() else None})
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(out)
        if args.fail_fast and cell not in completed_cells():
            return 1
    wanted = set(itertools.product(args.models, args.sizes, args.outputs, args.modes, range(1, args.rounds + 1)))
    completed = completed_cells()
    matched = wanted & completed
    latest = {tuple(row["cell"]): row for row in results if tuple(row["cell"]) in matched
              and row["status"] == "completed" and row["guard_passed"]
              and row["completion_tokens"] == [row["cell"][2]]}
    gates = {}
    for model, size, output, mode in itertools.product(args.models, args.sizes, args.outputs, args.modes):
        if output != 512:
            continue
        paths = [ROOT / row["evidence"] for cell, row in latest.items() if cell[:4] == (model, size, output, mode)]
        if paths:
            key = f"{model}/{size}/{mode}"
            try:
                gates[key] = collect_cell(paths, references.get(model, {}).get(f"{size}/{output}"))["gate"]
            except ValueError as error:
                gates[key] = {"passed": False, "reason": str(error)}
    summary = {
        "planned": len(wanted), "completed": len(matched),
        "release_gate": (wanted <= completed and len(gates) == len(MODELS) * len(SIZES) * len(MODES)
                         and all(gate["passed"] for gate in gates.values())),
        "gates": gates,
    }
    summary_path = out.with_name("summary.json")
    temporary = summary_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(summary_path)
    return 0 if wanted <= completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
