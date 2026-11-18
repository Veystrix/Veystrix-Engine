#!/usr/bin/env python3
"""Multi-length cross-language parity harness: C++ vs Python, FRESH references.

Tests the reviewer's premise directly: does the integrated C++ real-checkpoint
generator diverge from Python on ARBITRARY real prompts, or does it match at any
prompt length?  For each prompt length N this harness:

  1. Captures a FRESH Python reference with capture_real_reference.py using
     --tokens T --prompt-tokens N (fixed seed 20261117, but a DIFFERENT prompt
     for every N), under real_run_guard.py.
  2. Generates a flat config from that reference with gen_real_config.py
     (JSON-only; no tensor data, no guard required).
  3. Runs the integrated real_checkpoint_generate under real_run_guard.py and
     captures its stdout step lines.
  4. Compares token ids AND per-step logit SHA-256 digests bit-for-bit and
     records the exact digests and the first divergent step.

The default comparison is a STRICT gate (count must match). With `--eos-aware`
the harness additionally captures ALL Python logit digests (via a derived,
results-local capture script) and classifies a Python early stop as
bit-exact-compute when Python's next sampled token is an EOS id: this is a stop
condition difference, not a numeric divergence. Both verdicts are always
recorded; `--eos-aware` only changes the harness exit code.

Hard rules enforced here: EVERY real-checkpoint run (steps 1 and 3) is wrapped
in real_run_guard.py with the owner's flags, `sysctl vm.swapusage` is sampled
immediately before each run, a cool-down separates consecutive runs, and the
whole harness ABORTS if a guard reports a non-null `stopped` (or any non-zero
exit).  The comparison in step 4 is done IN THIS PROCESS from the already
captured stdout -- check_real_parity.py is deliberately NOT invoked to re-run
the binary, because that would run a real checkpoint outside the guard.

Usage (from any cwd; the guarded commands run from the Veystrix repo):
  .venv/bin/python cpp/engine/tools/check_xlang_multilen.py \
    --worktree /Users/vincent/.local/share/opencode/worktree/dd9a0e/xlang-parity \
    --binary cpp/engine/build/real_checkpoint_generate \
    --lengths 16,64,128,256,512,1024 --tokens 4 --budget 420 \
    --results-dir results/flashnext/xlang_multilen_<timestamp>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

VEYSTRIX = Path("/Users/vincent/Documents/Veystrix")
GUARD = VEYSTRIX / "scripts/flashnext/real_run_guard.py"
VENV_PY = VEYSTRIX / ".venv/bin/python"
CHECKPOINT = Path("/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")

# Same strict line format as check_real_parity.py.
LINE_RE = re.compile(
    r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$"
)
GUARD_FLAGS = [
    "--floor-gib", "1.5",
    "--swap-growth-mib", "5120",
    "--compressor-growth-mib", "8192",
    "--thermal-level", "2",
]


def log(msg: str) -> None:
    print(f"[multilen {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def aggregate_name(eos_aware: bool) -> str:
    return "aggregate_eos_aware.json" if eos_aware else "aggregate.json"


def swap_used_mib() -> int | None:
    """Measured: `sysctl vm.swapusage` used value in MiB (None on parse error)."""
    try:
        out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True,
                             text=True, timeout=10).stdout
        m = re.search(r"used\s*=\s*([\d.]+)M", out)
        return int(float(m.group(1))) if m else None
    except Exception:
        return None


def read_guard_summary(path: Path) -> dict:
    """Last JSON line of a guard log is the end summary (measured)."""
    try:
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else {}
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}


def guarded_run(name: str, cmd: list[str], budget: int, log_path: Path,
                stdout_path: Path, stderr_path: Path) -> dict:
    """Run `cmd` under the guard from the Veystrix repo; return a result dict."""
    before = swap_used_mib()
    log(f"{name}: swap before = {before} MiB; guard log = {log_path}")
    full = [str(VENV_PY), str(GUARD), *GUARD_FLAGS,
            "--max-seconds", str(budget), "--log", str(log_path), "--", *cmd]
    with stdout_path.open("w") as so, stderr_path.open("w") as se:
        proc = subprocess.run(full, cwd=str(VEYSTRIX), stdout=so, stderr=se)
    summary = read_guard_summary(log_path)
    stopped = summary.get("stopped")
    result = {
        "name": name,
        "command": cmd,
        "swap_before_mib": before,
        "returncode": proc.returncode,
        "guard_stopped": stopped,
        "guard_summary": summary,
    }
    log(f"{name}: rc={proc.returncode} guard_stopped={stopped!r} "
        f"swap {before}->{summary.get('swap_peak_mib')} MiB "
        f"thermal={summary.get('thermal_seconds')}")
    return result


def parse_cpp_steps(stdout_path: Path) -> list[dict]:
    steps = []
    for line in stdout_path.read_text(errors="replace").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        steps.append({"step": int(m.group(1)), "in": int(m.group(2)),
                      "out": int(m.group(3)), "digest": m.group(4)})
    return steps


def make_all_digests_capture(capture_py: Path, out_path: Path) -> None:
    """Derived capture script: also writes every recorded logit digest.

    This is a results-local artifact, never a modification of the shared
    capture_real_reference.py primitive. It differs by exactly one output line.
    """
    src = capture_py.read_text()
    needle = '        "n_logit_digests": len(logit_digests),'
    if needle not in src:
        raise SystemExit("capture_real_reference.py layout changed; cannot patch")
    src = src.replace(needle, needle + '\n        "all_logit_digests": logit_digests,')
    out_path.write_text(src)


def read_eos_ids(checkpoint: Path) -> list[int]:
    """EOS ids from the checkpoint's generation_config.json (root), not the
    flat C++ config, which keeps only text_config.eos_token_id."""
    try:
        gc = json.loads((checkpoint / "generation_config.json").read_text())
        v = gc.get("eos_token_id", [])
        return [int(x) for x in (v if isinstance(v, list) else [v])]
    except Exception:
        return []


def compare(reference: dict, cpp_steps: list[dict], eos_ids: list[int] | None = None) -> dict:
    ref_steps = reference.get("steps", [])
    # All Python logit rows actually computed (may exceed emitted steps when
    # Python stops on EOS). Fall back to the paired digests if not captured.
    all_py_digests = reference.get("all_logit_digests") or \
        [s["logit_sha256"] for s in ref_steps]
    n = min(len(cpp_steps), len(ref_steps))
    rows = []
    first_token_div = first_digest_div = None
    token_ok = digest_ok = 0
    for i in range(n):
        e, g = ref_steps[i], cpp_steps[i]
        tok_ok = g["out"] == int(e["token"])
        dig_ok = g["digest"] == e["logit_sha256"]
        token_ok += int(tok_ok)
        digest_ok += int(dig_ok)
        if not tok_ok and first_token_div is None:
            first_token_div = i
        if not dig_ok and first_digest_div is None:
            first_digest_div = i
        rows.append({"step": i, "cpp_token": g["out"], "py_token": int(e["token"]),
                     "token_ok": tok_ok, "cpp_digest": g["digest"],
                     "py_digest": e["logit_sha256"], "digest_ok": dig_ok})

    # Digest rows Python computed but did not pair with an emitted token (EOS).
    extra_digest_rows = []
    nd = min(len(all_py_digests), len(cpp_steps))
    digests_ok_all = 0
    first_extra_div = None
    for i in range(nd):
        match = all_py_digests[i] == cpp_steps[i]["digest"]
        digests_ok_all += int(match)
        extra_digest_rows.append({"step": i, "cpp_digest": cpp_steps[i]["digest"],
                                  "py_digest": all_py_digests[i], "digest_ok": match})
        if not match and first_extra_div is None:
            first_extra_div = i

    count_ok = len(cpp_steps) == len(ref_steps)
    # Strict gate: exactly the reference's emitted steps, all tokens + digests.
    strict_pass = token_ok == len(ref_steps) and digest_ok == len(ref_steps) and count_ok
    # EOS-aware: every Python-computed logit row is bit-exact and any extra C++
    # step is the EOS token Python sampled but did not emit.
    eos_ids = eos_ids or []
    eos_early_stop = (
        len(cpp_steps) > len(ref_steps)
        and reference.get("finish") == "stop"
        and len(ref_steps) < len(cpp_steps)
        and cpp_steps[len(ref_steps)]["out"] in eos_ids
    )
    bit_exact = (
        token_ok == len(ref_steps)
        and digests_ok_all == len(all_py_digests)
        and len(all_py_digests) <= len(cpp_steps)
        and (count_ok or eos_early_stop)
    )
    return {
        "pass": strict_pass,
        "strict_pass": strict_pass,
        "bit_exact_compute_pass": bit_exact,
        "eos_early_stop": eos_early_stop,
        "eos_ids": eos_ids,
        "n_ref_steps": len(ref_steps), "n_cpp_steps": len(cpp_steps),
        "tokens_ok": token_ok, "digests_ok": digest_ok,
        "all_python_digests": len(all_py_digests), "digests_ok_all": digests_ok_all,
        "count_ok": count_ok,
        "first_token_divergence_step": first_token_div,
        "first_digest_divergence_step": first_digest_div,
        "first_unpaired_digest_divergence_step": first_extra_div,
        "rows": rows, "all_digest_rows": extra_digest_rows,
        "python_token_ids": reference.get("token_ids", []),
        "cpp_token_ids": [s["out"] for s in cpp_steps],
        "python_digests": [s["logit_sha256"] for s in ref_steps],
        "python_all_digests": all_py_digests,
        "cpp_digests": [s["digest"] for s in cpp_steps],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--binary", default="cpp/engine/build/real_checkpoint_generate")
    ap.add_argument("--lengths", default="16,64,128,256,512,1024")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--budget", type=int, default=420, help="per-run --max-seconds")
    ap.add_argument("--cooldown", type=int, default=18, help="seconds between guarded runs")
    ap.add_argument("--eos-aware", action="store_true",
                    help="capture all Python logit digests and treat EOS early-stop "
                         "as bit-exact-compute PASS (strict verdict still recorded)")
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()

    wt = Path(args.worktree).resolve()
    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = (wt / binary).resolve()
    capture_py = wt / "cpp/engine/tools/capture_real_reference.py"
    config_py = wt / "cpp/engine/tools/gen_real_config.py"
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    results = Path(args.results_dir)
    if not results.is_absolute():
        results = (wt / results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    # results-local derived capture: adds `all_logit_digests` to the reference.
    capture_all_py = results / "_capture_all_digests.py"
    if args.eos_aware:
        make_all_digests_capture(capture_py, capture_all_py)
    eos_ids = read_eos_ids(CHECKPOINT)

    log(f"binary={binary}")
    log(f"checkpoint={CHECKPOINT}")
    log(f"lengths={lengths} tokens={args.tokens} budget={args.budget}s cooldown={args.cooldown}s")
    log(f"results-dir={results}")

    aggregate = {
        "purpose": "Test reviewer premise: C++ vs Python bit-exactness on fresh "
                   "prompts at multiple lengths (fixed seed 20261117, distinct prompt per N).",
        "binary": str(binary), "checkpoint": str(CHECKPOINT),
        "tokens": args.tokens, "budget_seconds": args.budget,
        "lengths": lengths, "runs": [], "aborted": None,
        "claim": "bit-exact = identical greedy token ids AND identical per-step "
                 "float32-LE logit SHA-256 digests",
    }
    aborted = False
    for idx, n in enumerate(lengths):
        d = results / f"len_{n:04d}"
        d.mkdir(parents=True, exist_ok=True)
        record: dict = {"prompt_tokens": n, "dir": str(d)}
        log(f"=== length N={n} ({idx + 1}/{len(lengths)}) ===")

        # 1) Fresh Python reference (guarded real-checkpoint run).
        ref_path = d / "reference.json"
        active_capture = capture_all_py if args.eos_aware else capture_py
        py = guarded_run(
            f"N={n} python-capture",
            [str(VENV_PY), str(active_capture), "--checkpoint", str(CHECKPOINT),
             "--out", str(ref_path), "--tokens", str(args.tokens),
             "--prompt-tokens", str(n)],
            args.budget, d / "py_guard.jsonl", d / "py_stdout.txt", d / "py_stderr.txt")
        record["python_run"] = py
        if py["returncode"] != 0 or py["guard_stopped"] is not None:
            aggregate["aborted"] = f"N={n}: python-capture guard/command did not complete cleanly"
            aggregate["runs"].append(record)
            log(f"ABORT: {aggregate['aborted']}")
            aborted = True
            break
        reference = json.loads(ref_path.read_text())
        record["prompt_sha256"] = reference.get("prompt_sha256")
        record["mlx_version"] = reference.get("mlx_version")
        log(f"N={n}: python prompt_sha256={reference.get('prompt_sha256')}")

        time.sleep(args.cooldown)

        # 2) Flat config from the fresh reference (JSON-only, no guard needed).
        cfg_path = d / "real_config.txt"
        cfg = subprocess.run(
            [str(VENV_PY), str(config_py), "--checkpoint", str(CHECKPOINT),
             "--reference", str(ref_path), "--out", str(cfg_path)],
            cwd=str(VEYSTRIX), capture_output=True, text=True)
        if cfg.returncode != 0:
            aggregate["aborted"] = f"N={n}: gen_real_config failed: {cfg.stderr[-500:]}"
            aggregate["runs"].append(record)
            log(f"ABORT: {aggregate['aborted']}")
            aborted = True
            break
        record["config_gen"] = {"returncode": cfg.returncode, "stdout": cfg.stdout.strip()}

        # 3) C++ generation (guarded real-checkpoint run).
        cpp = guarded_run(
            f"N={n} cpp-generate",
            [str(binary), str(CHECKPOINT), str(ref_path), str(args.tokens), str(cfg_path)],
            args.budget, d / "cpp_guard.jsonl", d / "cpp_stdout.txt", d / "cpp_stderr.txt")
        record["cpp_run"] = cpp
        if cpp["returncode"] != 0 or cpp["guard_stopped"] is not None:
            aggregate["aborted"] = f"N={n}: cpp-generate guard/command did not complete cleanly"
            aggregate["runs"].append(record)
            log(f"ABORT: {aggregate['aborted']}")
            aborted = True
            break

        # 4) Bit-for-bit comparison, in-process (never re-runs the binary).
        cmp = compare(reference, parse_cpp_steps(d / "cpp_stdout.txt"), eos_ids)
        record["comparison"] = cmp
        verdict = ("PASS" if cmp["bit_exact_compute_pass"] else "FAIL") if args.eos_aware \
            else ("PASS" if cmp["strict_pass"] else "FAIL")
        log(f"N={n}: {verdict} strict={cmp['strict_pass']} "
            f"bit_exact={cmp['bit_exact_compute_pass']} "
            f"eos_early_stop={cmp['eos_early_stop']} "
            f"tokens {cmp['tokens_ok']}/{cmp['n_ref_steps']} "
            f"digests {cmp['digests_ok']}/{cmp['n_ref_steps']} "
            f"(all py digests {cmp['digests_ok_all']}/{cmp['all_python_digests']}) "
            f"first_digest_div={cmp['first_digest_divergence_step']}")
        aggregate["runs"].append(record)
        (d / "verdict.json").write_text(json.dumps(record, indent=2) + "\n")

        if idx < len(lengths) - 1:
            time.sleep(args.cooldown)

    aggregate_path = results / aggregate_name(args.eos_aware)
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    key = "bit_exact_compute_pass" if args.eos_aware else "strict_pass"
    passed_lengths = [r["prompt_tokens"] for r in aggregate["runs"]
                      if r.get("comparison", {}).get(key)]
    failed_lengths = [r["prompt_tokens"] for r in aggregate["runs"]
                      if r.get("comparison") and not r["comparison"][key]]
    log(f"DONE aborted={aborted} pass={passed_lengths} fail={failed_lengths} "
        f"aggregate={aggregate_path}")
    return 0 if not aborted and not failed_lengths and len(passed_lengths) == len(lengths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
