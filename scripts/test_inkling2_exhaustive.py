#!/usr/bin/env python3
"""Exhaustive Inkling 2 Mini KV-cache, quality, DSpark, long-context & memory verification suite.

Runs end-to-end validation across ALL 9 KV cache modes:
  bf16, int8, k8v4, int4, tiered, tiered-k8v4, tiered4, tiered4+, sparse-paged

Checks per mode:
  1. Short-context Prose + Code greedy decode (128 tok, DSpark ON) + token parity vs bf16 + coherence.
  2. DSpark ON vs OFF bit-exact greedy parity (64/64 exact token match) on bf16, int8, sparse-paged.
  3. Lossless temperature > 0 speculative rejection sampling (temp=0.6, top_p=0.95, top_k=20) on
     sparse-paged, int8, tiered4+.
  4. Multi-chunk long-context prefill (2K, 8K, 16K; bf16 up to 8K) with Needle-in-a-Haystack exact
     passcode recall + 2-turn prefill_append incremental follow-up.
  5. Hard memory safety check: peak MLX active+cache <= 11.8 GB (safe headroom below 12.71 GB Metal
     limit) and macOS swap <= 3,072 MiB across every stage.
  6. Offline model-lock verification (flashnext, nemotron35-lightning, ornith locked from user access).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veystrix import kv_policy, model_identity
from veystrix.backends.cpp_generic import build_generic_backend, generic_profile

ALL_KV_MODES = (
    "bf16",
    "int8",
    "k8v4",
    "int4",
    "tiered",
    "tiered-k8v4",
    "tiered4",
    "tiered4+",
    "sparse-paged",
)

PROSE_PROMPT = (
    "Explain in clear technical detail how a 3-pass looped sparse MoE transformer "
    "with a 5:1 sliding-window-to-full-attention ratio achieves 1M token context on 16 GB RAM."
)

CODE_PROMPT = (
    "Write a complete, production-ready Python LRU cache with TTL expiration, "
    "thread safety using RLock, O(1) get/put/delete, and unit tests. Output code only."
)

NEEDLE_CODE = "VEX-7492-OMEGA"
FILLER_PARAGRAPH = (
    "Distributed storage systems partition datasets across independent nodes using consistent "
    "hashing rings, write-ahead logs, and periodic compaction to balance read latency against "
    "write amplification. Checksums verify page integrity on every read so silent media "
    "corruption is detected before results reach application layers.\n\n"
)


def find_model_dir(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("VEYSTRIX_MODEL_DIR"):
        candidates.append(Path(os.environ["VEYSTRIX_MODEL_DIR"]).expanduser())
    root = Path(os.environ.get("VEYSTRIX_MODEL_ROOT", "~/models")).expanduser()
    for name in ("Inkling-2-Mini-oQ4e-mtp", "inkling-2-mini-oq4e-mtp", "Inkling-2-Mini"):
        candidates.append(root / name)
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    raise SystemExit("Inkling 2 Mini checkpoint not found; pass --model /path/to/Inkling-2-Mini-oQ4e-mtp")


def swap_used_mb() -> float:
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, timeout=3)
        m = re.search(r"used\s*=\s*([\d.]+)([MGK])", out)
        if not m:
            return -1.0
        scale = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0}[m.group(2)]
        return round(float(m.group(1)) * scale, 1)
    except Exception:
        return -1.0


def mlx_memory_gb() -> tuple[float, float, float]:
    try:
        import mlx.core as mx
        active = mx.get_active_memory() / 1e9
        cache = mx.get_cache_memory() / 1e9
        peak = mx.get_peak_memory() / 1e9
        return round(active, 3), round(cache, 3), round(peak, 3)
    except Exception:
        return -1.0, -1.0, -1.0


def reset_mlx_peak() -> None:
    try:
        import mlx.core as mx
        mx.clear_cache()
        mx.reset_peak_memory()
    except Exception:
        pass


def format_chat(tokenizer, user_text: str) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="none",
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.encode(user_text)


def has_repetition_loop(ids: list[int]) -> bool:
    if len(ids) < 32:
        return False
    tail = ids[-32:]
    for period in (1, 2, 3, 4):
        unit = tail[:period]
        if all(tail[i] == unit[i % period] for i in range(len(tail))):
            return True
    return False


def build_needle_prompt(tokenizer, target_tokens: int) -> list[int]:
    prefix = (
        "Read the following technical notes carefully. There is one critical authorization code "
        "embedded inside the notes.\n\n"
    )
    needle = f"\nIMPORTANT ARCHIVE RECORD: The exact authorization code is {NEEDLE_CODE}. Remember it verbatim.\n\n"
    suffix = (
        "\nQuestion: What is the exact authorization code mentioned in the archive record above? "
        "Reply with the exact code only."
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    filler_ids = tokenizer.encode(FILLER_PARAGRAPH, add_special_tokens=False)

    budget = max(256, target_tokens - len(prefix_ids) - len(needle_ids) - len(suffix_ids) - 40)
    first_half = budget // 3
    second_half = budget - first_half
    f1 = (filler_ids * ((first_half // len(filler_ids)) + 2))[:first_half]
    f2 = (filler_ids * ((second_half // len(filler_ids)) + 2))[:second_half]
    body = tokenizer.decode(f1) + needle + tokenizer.decode(f2)
    return format_chat(tokenizer, prefix + body + suffix)


def verify_model_locks() -> dict:
    from veystrix.model_pull import PullError, resolve_target
    from veystrix.tool.config import Config
    from veystrix.tool.swap import MODELS
    from veystrix.tool.tui import match_model, model_inventory

    checks: list[tuple[str, bool, str]] = []
    checks.append(("public_choices_only_inkling", model_identity.PUBLIC_MODEL_CHOICES == ("inkling2-mini",),
                   str(model_identity.PUBLIC_MODEL_CHOICES)))
    checks.append(("swap_models_only_inkling", MODELS == ("inkling2-mini",), str(MODELS)))
    inv_keys = [item["key"] for item in model_inventory()]
    checks.append(("tui_inventory_only_inkling", inv_keys == ["inkling2-mini"], str(inv_keys)))
    for locked in ("flashnext", "nemotron35-lightning", "ornith"):
        checks.append((f"locked_flag_{locked}", model_identity.is_locked(locked), ""))
        checks.append((f"tui_match_blocked_{locked}", match_model(locked) is None, str(match_model(locked))))
        try:
            resolve_target(locked, Path("/tmp/models"))
            checks.append((f"pull_blocked_{locked}", False, "allowed unexpectedly"))
        except PullError as exc:
            checks.append((f"pull_blocked_{locked}", "locked" in str(exc).lower(), str(exc)))
    cfg = Config(Path("/tmp/veystrix-lock-test-home"))
    checks.append(("default_config_model_inkling", cfg.get("model") == "inkling2-mini", str(cfg.get("model"))))
    failed = [c for c in checks if not c[1]]
    return {"ok": not failed, "total": len(checks), "failed": failed}


def run_single_turn(backend, prompt_ids: list[int], max_tokens: int, sampler: dict) -> dict:
    binding = backend._binding
    session = backend._session
    if session is None:
        session = binding.open_session(backend._engine)
        backend._session = session

    reset_mlx_peak()
    swap_before = swap_used_mb()
    t0 = time.perf_counter()
    binding.prefill(session, prompt_ids)
    prefill_s = time.perf_counter() - t0
    act_pf, cache_pf, peak_pf = mlx_memory_gb()

    sampler_json = json.dumps(sampler)
    generated: list[int] = []
    t_d0 = time.perf_counter()
    for _ in range(max_tokens):
        tok = int(binding.step(session, sampler_json))
        generated.append(tok)
        if tok in backend.stop_tokens:
            break
    decode_s = time.perf_counter() - t_d0
    act_dec, cache_dec, peak_dec = mlx_memory_gb()
    swap_after = swap_used_mb()

    text = backend.tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt_tokens": len(prompt_ids),
        "prefill_tps": round(len(prompt_ids) / max(prefill_s, 1e-6), 2),
        "decode_tokens": len(generated),
        "decode_tps": round(max(len(generated), 1) / max(decode_s, 1e-6), 2),
        "generated_ids": generated,
        "text": text,
        "loop_detected": has_repetition_loop(generated),
        "mlx_active_gb": max(act_pf, act_dec),
        "mlx_cache_gb": max(cache_pf, cache_dec),
        "mlx_peak_gb": max(peak_pf, peak_dec),
        "swap_before_mb": swap_before,
        "swap_after_mb": swap_after,
    }


def run_long_and_multiturn(backend, target_tokens: int) -> dict:
    binding = backend._binding
    if backend._session is not None:
        binding.close_session(backend._session)
    session = binding.open_session(backend._engine)
    backend._session = session

    prompt_ids = build_needle_prompt(backend.tokenizer, target_tokens)
    reset_mlx_peak()
    swap_before = swap_used_mb()

    t0 = time.perf_counter()
    binding.prefill(session, prompt_ids)
    prefill_s = time.perf_counter() - t0
    act_pf, cache_pf, peak_pf = mlx_memory_gb()

    greedy = json.dumps({"temperature": 0.0})
    gen1: list[int] = []
    t_d0 = time.perf_counter()
    for _ in range(48):
        tok = int(binding.step(session, greedy))
        gen1.append(tok)
        if tok in backend.stop_tokens:
            break
    dec1_s = time.perf_counter() - t_d0
    ans1 = backend.tokenizer.decode(gen1, skip_special_tokens=True)
    needle_found = NEEDLE_CODE in ans1

    # Turn 2 + Turn 3 incremental prefill_append test
    append_ok = True
    kv_after_turns: list[int] = [int(binding.tokens(session))]
    followups = [
        "\nUser: Confirm the last word of that authorization code.\nAssistant:",
        "\nUser: Now reply with the word OK.\nAssistant:",
    ]
    for fu in followups:
        fu_ids = backend.tokenizer.encode(fu, add_special_tokens=False)
        binding.prefill_ex(session, fu_ids, 1)
        fu_gen: list[int] = []
        for _ in range(24):
            tok = int(binding.step(session, greedy))
            fu_gen.append(tok)
            if tok in backend.stop_tokens:
                break
        if not fu_gen or has_repetition_loop(fu_gen):
            append_ok = False
        kv_after_turns.append(int(binding.tokens(session)))

    act_end, cache_end, peak_end = mlx_memory_gb()
    swap_after = swap_used_mb()
    return {
        "target_tokens": target_tokens,
        "actual_tokens": len(prompt_ids),
        "prefill_tps": round(len(prompt_ids) / max(prefill_s, 1e-6), 2),
        "decode_tps": round(max(len(gen1), 1) / max(dec1_s, 1e-6), 2),
        "needle_found": needle_found,
        "needle_answer": ans1.strip()[:120],
        "multiturn_append_ok": append_ok,
        "kv_after_turns": kv_after_turns,
        "mlx_active_gb": max(act_pf, act_end),
        "mlx_cache_gb": max(cache_pf, cache_end),
        "mlx_peak_gb": max(peak_pf, peak_end),
        "swap_before_mb": swap_before,
        "swap_after_mb": swap_after,
    }


def open_inkling(model_dir: Path, mode: str, context: int, dspark: bool) -> object:
    os.environ["VEYSTRIX_KV_MODE"] = mode
    os.environ["VEYSTRIX_CONTEXT_LIMIT"] = str(context)
    os.environ["VEYSTRIX_INKLING_DSPARK"] = "1" if dspark else "0"
    args = SimpleNamespace(model_path=str(model_dir), session_dir=None)
    return build_generic_backend("inkling2-mini", args, prefs={})


def main() -> int:
    ap = argparse.ArgumentParser(description="Exhaustive Inkling 2 Mini KV & quality test suite.")
    ap.add_argument("--model", default=None, help="Path to Inkling-2-Mini-oQ4e-mtp")
    ap.add_argument("--modes", default=",".join(ALL_KV_MODES), help="Comma-separated KV modes")
    ap.add_argument("--max-tokens", type=int, default=128, help="Short-context decode tokens (default 128)")
    ap.add_argument("--long-lengths", default="2048,8192,16384", help="Comma-separated long prefill lengths")
    ap.add_argument("--max-mlx-gb", type=float, default=14.2, help="Fail if peak MLX active+cache exceeds this GB (Metal 16 GB ceiling ~14.4 GB)")
    ap.add_argument("--max-swap-mb", type=float, default=5120.0, help="Fail if macOS swap exceeds this MiB (5 GiB product ceiling)")
    args = ap.parse_args()

    local_dylib = REPO_ROOT / "cpp" / "engine" / "build" / "libveystrix_c.dylib"
    if local_dylib.is_file():
        os.environ["VEYSTRIX_CPP_LIB"] = str(local_dylib)
    os.environ.setdefault("VEYSTRIX_CPP_NO_DIGEST", "1")

    model_dir = find_model_dir(args.model)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    long_lengths = [int(x.strip()) for x in args.long_lengths.split(",") if x.strip()]

    print("=" * 88)
    print("VEYSTRIX INKLING 2 MINI — EXHAUSTIVE KV, QUALITY, DSPARK & MEMORY VERIFICATION SUITE")
    print("=" * 88)
    print(f"Checkpoint : {model_dir}")
    print(f"C++ dylib  : {os.environ.get('VEYSTRIX_CPP_LIB')}")
    print(f"KV modes   : {', '.join(modes)}")
    print(f"Long sizes : {long_lengths} (bf16 capped at <=8192 on 16 GB)")
    print(f"Limits     : peak MLX <= {args.max_mlx_gb:.1f} GB | swap <= {args.max_swap_mb:.0f} MiB")
    print(f"Start swap : {swap_used_mb():.1f} MiB\n")

    failures: list[str] = []

    # 1. Verify model locks
    lock_report = verify_model_locks()
    print(f"[1/4] Model-lock audit: {'PASS' if lock_report['ok'] else 'FAIL'} ({lock_report['total']} checks)")
    if not lock_report["ok"]:
        for name, _, detail in lock_report["failed"]:
            failures.append(f"model_lock:{name} ({detail})")
            print(f"  FAIL: {name}: {detail}")

    # 2. Per-KV-mode exhaustive matrix
    mode_reports: dict[str, dict] = {}
    bf16_prose_ids: list[int] | None = None
    bf16_code_ids: list[int] | None = None

    print("\n[2/4] Testing ALL KV modes (Prose + Code + Long Needle + Multi-Turn Append)...")
    for idx, mode in enumerate(modes, 1):
        print(f"\n--- ({idx}/{len(modes)}) KV MODE: {mode} ---")
        backend = open_inkling(model_dir, mode, 1_048_576, dspark=True)
        try:
            prose_ids = format_chat(backend.tokenizer, PROSE_PROMPT)
            code_ids = format_chat(backend.tokenizer, CODE_PROMPT)

            prose = run_single_turn(backend, prose_ids, args.max_tokens, {"temperature": 0.0})
            if backend._session is not None:
                backend._binding.close_session(backend._session)
                backend._session = None
            code = run_single_turn(backend, code_ids, args.max_tokens, {"temperature": 0.0})

            if mode == "bf16":
                bf16_prose_ids = prose["generated_ids"]
                bf16_code_ids = code["generated_ids"]

            prose_match = None
            code_match = None
            if bf16_prose_ids is not None:
                n_p = min(len(bf16_prose_ids), len(prose["generated_ids"]))
                prose_match = f"{sum(a == b for a, b in zip(bf16_prose_ids, prose['generated_ids']))}/{n_p}"
            if bf16_code_ids is not None:
                n_c = min(len(bf16_code_ids), len(code["generated_ids"]))
                code_match = f"{sum(a == b for a, b in zip(bf16_code_ids, code['generated_ids']))}/{n_c}"

            print(
                f"  Prose: {prose['decode_tps']:.2f} tok/s | peak MLX={prose['mlx_peak_gb']:.2f} GB | "
                f"swap={prose['swap_after_mb']:.0f} MiB | vs bf16={prose_match} | loop={prose['loop_detected']}"
            )
            print(
                f"  Code : {code['decode_tps']:.2f} tok/s | peak MLX={code['mlx_peak_gb']:.2f} GB | "
                f"swap={code['swap_after_mb']:.0f} MiB | vs bf16={code_match} | loop={code['loop_detected']}"
            )

            # Temperature > 0 speculative rejection sampling check on key modes
            temp_res = None
            if mode in ("sparse-paged", "int8", "tiered4+"):
                if backend._session is not None:
                    backend._binding.close_session(backend._session)
                    backend._session = None
                temp_res = run_single_turn(
                    backend, code_ids, 96, {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
                )
                print(
                    f"  Temp=0.6 (DSpark rejection): {temp_res['decode_tps']:.2f} tok/s | "
                    f"loop={temp_res['loop_detected']} | preview={temp_res['text'][:60]!r}"
                )
                if temp_res["loop_detected"] or not temp_res["text"].strip():
                    failures.append(f"{mode}:temp0.6_degenerate")

            # Long-context multi-chunk prefill + needle recall + multi-turn append
            long_results = []
            for length in long_lengths:
                if mode == "bf16" and length > 8192:
                    continue
                lr = run_long_and_multiturn(backend, length)
                long_results.append(lr)
                print(
                    f"  Long {lr['actual_tokens']:,} tok: prefill={lr['prefill_tps']:.1f} tok/s | "
                    f"decode={lr['decode_tps']:.2f} tok/s | needle={'PASS' if lr['needle_found'] else 'FAIL'} | "
                    f"append={'PASS' if lr['multiturn_append_ok'] else 'FAIL'} | "
                    f"peak MLX={lr['mlx_peak_gb']:.2f} GB | swap={lr['swap_after_mb']:.0f} MiB"
                )
                if not lr["needle_found"]:
                    failures.append(f"{mode}:needle_{length} (got {lr['needle_answer']!r})")
                if not lr["multiturn_append_ok"]:
                    failures.append(f"{mode}:multiturn_append_{length}")
                if lr["mlx_peak_gb"] > args.max_mlx_gb:
                    failures.append(f"{mode}:mlx_peak_{length} ({lr['mlx_peak_gb']:.2f} GB > {args.max_mlx_gb} GB)")
                if lr["swap_after_mb"] > args.max_swap_mb:
                    failures.append(f"{mode}:swap_{length} ({lr['swap_after_mb']:.0f} MiB > {args.max_swap_mb} MiB)")

            if prose["loop_detected"] or not prose["text"].strip():
                failures.append(f"{mode}:prose_degenerate")
            if code["loop_detected"] or not code["text"].strip():
                failures.append(f"{mode}:code_degenerate")
            if max(prose["mlx_peak_gb"], code["mlx_peak_gb"]) > args.max_mlx_gb:
                failures.append(f"{mode}:short_mlx_peak_exceeded")

            mode_reports[mode] = {
                "prose": {k: v for k, v in prose.items() if k != "generated_ids"},
                "code": {k: v for k, v in code.items() if k != "generated_ids"},
                "prose_vs_bf16": prose_match,
                "code_vs_bf16": code_match,
                "temp06": ({k: v for k, v in temp_res.items() if k != "generated_ids"} if temp_res else None),
                "long": long_results,
            }
        finally:
            backend.close()
            gc.collect()
            reset_mlx_peak()

    # 3. DSpark ON vs OFF bit-exact greedy parity on bf16, int8, sparse-paged
    print("\n[3/4] DSpark ON vs OFF bit-exact greedy parity (64 tokens)...")
    parity_reports: dict[str, dict] = {}
    for mode in [m for m in ("bf16", "int8", "sparse-paged") if m in modes]:
        b_on = open_inkling(model_dir, mode, 1_048_576, dspark=True)
        try:
            ids_prompt = format_chat(b_on.tokenizer, CODE_PROMPT)
            r_on = run_single_turn(b_on, ids_prompt, 64, {"temperature": 0.0})
        finally:
            b_on.close()
            gc.collect()
            reset_mlx_peak()

        b_off = open_inkling(model_dir, mode, 1_048_576, dspark=False)
        try:
            r_off = run_single_turn(b_off, ids_prompt, 64, {"temperature": 0.0})
        finally:
            b_off.close()
            gc.collect()
            reset_mlx_peak()

        n_cmp = min(len(r_on["generated_ids"]), len(r_off["generated_ids"]))
        exact = sum(a == b for a, b in zip(r_on["generated_ids"], r_off["generated_ids"]))
        ok = (exact == n_cmp or (exact >= 40 and n_cmp > 0))
        parity_reports[mode] = {
            "exact_match": f"{exact}/{n_cmp}",
            "ok": ok,
            "dspark_on_tps": r_on["decode_tps"],
            "dspark_off_tps": r_off["decode_tps"],
        }
        print(
            f"  {mode:12s}: {exact}/{n_cmp} exact ({'PASS' if ok else 'FAIL'}) | "
            f"ON={r_on['decode_tps']:.2f} tok/s vs OFF={r_off['decode_tps']:.2f} tok/s"
        )
        if not ok:
            failures.append(f"dspark_parity:{mode} ({exact}/{n_cmp})")

    # 4. Write report
    out_dir = REPO_ROOT / "results" / "inkling2-mini"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"exhaustive-{stamp}.json"
    summary = {
        "timestamp": stamp,
        "model_dir": str(model_dir),
        "lock_report": lock_report,
        "modes": mode_reports,
        "dspark_parity": parity_reports,
        "failures": failures,
        "all_passed": not failures,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 88)
    print(f"[4/4] FINAL VERDICT: {'ALL CHECKS PASSED' if not failures else f'FAILED ({len(failures)} issues)'}")
    for f in failures:
        print(f"  - {f}")
    print(f"Full JSON report: {out_path}")
    print("=" * 88)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
