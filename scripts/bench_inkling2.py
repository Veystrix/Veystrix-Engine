#!/usr/bin/env python3
"""Live Metal GPU benchmark & KV-cache verification for Inkling 2 Mini on Veystrix C++ generic engine.

Run from native macOS Terminal (requires Metal GPU and local checkpoint):
  .venv/bin/python scripts/bench_inkling2.py
  .venv/bin/python scripts/bench_inkling2.py --model ~/models/Inkling-2-Mini-oQ4e-mtp --modes bf16,tiered4+,sparse-paged
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veystrix import kv_policy, model_identity
from veystrix.backends.cpp_generic import build_generic_backend, generic_profile


DEFAULT_PROMPT = (
    "Explain in clear technical detail how a 3-pass looped sparse MoE transformer "
    "with a 5:1 sliding-window-to-full-attention ratio achieves 1M token context on 16 GB RAM."
)

CODE_PROMPT = (
    "Write a complete, production-ready Python LRU cache with TTL expiration, "
    "thread safety using RLock, O(1) get/put/delete, and unit tests. Output code only."
)


def find_model_dir(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_dir = os.environ.get("VEYSTRIX_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    root = Path(os.environ.get("VEYSTRIX_MODEL_ROOT", "~/models")).expanduser()
    for name in (
        "Inkling-2-Mini-oQ4e-mtp",
        "inkling-2-mini-oq4e-mtp",
        "Inkling-2-Mini",
        "inkling2-mini",
    ):
        candidates.append(root / name)
    candidates.append(REPO_ROOT / "models" / "Inkling-2-Mini-oQ4e-mtp")
    for path in candidates:
        if (path / "config.json").is_file():
            return path
    tried = "\n  - ".join(str(p) for p in candidates)
    raise SystemExit(
        f"Inkling 2 Mini checkpoint not found. Pass --model /path/to/Inkling-2-Mini-oQ4e-mtp.\nTried:\n  - {tried}"
    )


def check_system_health() -> dict:
    info: dict[str, object] = {}
    try:
        pm = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=3)
        info["on_ac_power"] = "AC Power" in pm
    except Exception:
        info["on_ac_power"] = None
    try:
        vm = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, timeout=3).strip()
        info["swap"] = vm
    except Exception:
        info["swap"] = None
    return info


def peak_rss_mib() -> float:
    # macOS ru_maxrss is in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run_mode(model_dir: Path, mode: str, context: int, max_tokens: int, prompt: str, long_prefill: int) -> dict:
    env = dict(os.environ)
    env["VEYSTRIX_KV_MODE"] = mode
    env["VEYSTRIX_CONTEXT_LIMIT"] = str(context)
    profile, plan, recipe_path = generic_profile(str(model_dir), "inkling2-mini", environ=env)

    # Apply env for build_generic_backend.
    os.environ["VEYSTRIX_KV_MODE"] = mode
    os.environ["VEYSTRIX_CONTEXT_LIMIT"] = str(context)
    args = SimpleNamespace(model_path=str(model_dir), session_dir=None)
    t_open = time.perf_counter()
    backend = build_generic_backend("inkling2-mini", args, prefs={})
    open_s = time.perf_counter() - t_open

    try:
        runtime = backend.runtime_info()
        tokenizer = backend.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort="none",
                )
            except TypeError:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
        else:
            prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise RuntimeError("Tokenizer returned empty prompt_ids")

        # 1. Greedy decode benchmark (prefill + timed decode).
        binding = backend._binding
        session = backend._session
        if session is None:
            session = binding.open_session(backend._engine)
            backend._session = session

        greedy_sampler = json.dumps({"temperature": 0.0})

        t0 = time.perf_counter()
        binding.prefill(session, prompt_ids)
        prefill_s = time.perf_counter() - t0

        generated: list[int] = []
        t_dec0 = time.perf_counter()
        for _ in range(max_tokens):
            tok = int(binding.step(session, greedy_sampler))
            generated.append(tok)
            if tok in backend.stop_tokens:
                break
        decode_s = time.perf_counter() - t_dec0
        decode_steps = max(len(generated), 1)
        decode_tps = decode_steps / max(decode_s, 1e-6)
        prefill_tps = len(prompt_ids) / max(prefill_s, 1e-6)

        stats_after = binding.stats(backend._engine) or {}
        kv_after_short = int(binding.tokens(session))

        # 2. Optional longer prefill KV-cache growth check (e.g., 2,048 tokens > 1,024 sliding window).
        long_prefill_tps = None
        kv_after_long = None
        if long_prefill > 0:
            binding.close_session(session)
            session = binding.open_session(backend._engine)
            backend._session = session
            unit = prompt_ids
            long_ids = (unit * ((long_prefill // len(unit)) + 1))[:long_prefill]
            tl0 = time.perf_counter()
            binding.prefill(session, long_ids)
            tl_s = time.perf_counter() - tl0
            long_prefill_tps = len(long_ids) / max(tl_s, 1e-6)
            kv_after_long = int(binding.tokens(session))

        text_preview = tokenizer.decode(generated[:32], skip_special_tokens=False).replace("\n", "\\n")
        return {
            "mode": mode,
            "context_ceiling": context,
            "open_s": round(open_s, 2),
            "kv_plan": plan.as_dict(),
            "kv_bytes_per_token": runtime.get("kv_bytes_per_token"),
            "prompt_tokens": len(prompt_ids),
            "prefill_tps": round(prefill_tps, 2),
            "decode_tokens": decode_steps,
            "decode_tps": round(decode_tps, 2),
            "ms_per_tok": round(1000.0 / max(decode_tps, 1e-6), 2),
            "meets_12_tps": decode_tps >= 12.0,
            "meets_14_tps_drop_ornith": decode_tps >= 14.0,
            "long_prefill_tokens": long_prefill,
            "long_prefill_tps": round(long_prefill_tps, 2) if long_prefill_tps is not None else None,
            "kv_tokens_short": kv_after_short,
            "kv_tokens_long": kv_after_long,
            "expert_bytes_read_mb": round((stats_after.get("expert_bytes_read", 0) or 0) / (1024 * 1024), 1),
            "resident_mb": round((stats_after.get("resident_bytes", 0) or 0) / (1024 * 1024), 1),
            "streamed_mb": round((stats_after.get("streamed_bytes", 0) or 0) / (1024 * 1024), 1),
            "spec_rounds": int(stats_after.get("spec_rounds", 0) or 0),
            "spec_drafted": int(stats_after.get("spec_drafted", 0) or 0),
            "spec_accepted": int(stats_after.get("spec_accepted", 0) or 0),
            "peak_rss_mib": round(peak_rss_mib(), 1),
            "generated_ids": generated,
            "preview": text_preview,
        }
    finally:
        backend.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Inkling 2 Mini on Veystrix C++ generic engine.")
    ap.add_argument("--model", default=None, help="Path to Inkling-2-Mini-oQ4e-mtp checkpoint directory")
    ap.add_argument("--modes", default="bf16,tiered4+,sparse-paged", help="Comma-separated KV modes to test")
    ap.add_argument("--context", type=int, default=1_048_576, help="Target context ceiling (default 1M)")
    ap.add_argument("--max-tokens", type=int, default=64, help="Decode tokens per mode (default 64)")
    ap.add_argument("--long-prefill", type=int, default=2048, help="Long-context prefill test tokens (0 to skip)")
    ap.add_argument("--dspark", action="store_true", help="Enable DSpark speculative decoding (mtp.safetensors)")
    ap.add_argument("--draft-tokens", type=int, default=4, help="DSpark draft tokens per round (1..7, default 4)")
    ap.add_argument("--preset", choices=("default", "code"), default="default", help="Built-in prompt preset")
    ap.add_argument("--prompt", default=None, help="Custom prompt text (overrides --preset)")
    ap.add_argument("--json-out", default=None, help="Optional path to write JSON benchmark report")
    args = ap.parse_args()

    prompt_text = args.prompt if args.prompt is not None else (CODE_PROMPT if args.preset == "code" else DEFAULT_PROMPT)
    model_dir = find_model_dir(args.model)
    local_dylib = REPO_ROOT / "cpp" / "engine" / "build" / "libveystrix_c.dylib"
    if local_dylib.is_file():
        os.environ.setdefault("VEYSTRIX_CPP_LIB", str(local_dylib))
    os.environ.setdefault("VEYSTRIX_CPP_NO_DIGEST", "1")
    if args.dspark:
        os.environ["VEYSTRIX_INKLING_DSPARK"] = "1"
        os.environ["VEYSTRIX_INKLING_DSPARK_DRAFT"] = str(args.draft_tokens)
    health = check_system_health()
    print(f"Checkpoint : {model_dir}")
    print(f"C++ dylib  : {os.environ.get('VEYSTRIX_CPP_LIB', '(default)')}")
    print(f"Health     : AC={health['on_ac_power']} | swap={health['swap']}")
    if health["on_ac_power"] is False:
        print("WARNING: Running on battery — decode tok/s may be lower than AC baseline.")

    # Print static 1M KV memory budget from kv_policy before launching Metal.
    _, plan_1m, recipe_path = generic_profile(
        str(model_dir),
        "inkling2-mini",
        environ={**os.environ, "VEYSTRIX_KV_MODE": "auto", "VEYSTRIX_CONTEXT_LIMIT": str(args.context)},
    )
    geom = kv_policy.KVGeometry.from_recipe(json.loads(Path(recipe_path).read_text()))
    total_visits = geom.attention_layers + geom.sliding_layers
    kv_gib = plan_1m.bytes_at_context / (1024 ** 3)
    print(
        f"Recipe     : {recipe_path} (visits={total_visits}, "
        f"sliding={geom.sliding_layers}x{geom.sliding_window}, full={geom.attention_layers})"
    )
    print(f"1M auto KV : mode={plan_1m.mode}, kv_budget={kv_gib:.2f} GiB @ {args.context:,} tokens\n")

    results: list[dict] = []
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        print(f"--- Running mode={mode} (context={args.context:,}, max_tokens={args.max_tokens}) ---")
        res = run_mode(model_dir, mode, args.context, args.max_tokens, prompt_text, args.long_prefill)
        results.append(res)
        verdict = "PASS >=14 (drop Ornith)" if res["meets_14_tps_drop_ornith"] else (
            "PASS >=12" if res["meets_12_tps"] else "BELOW 12 tok/s"
        )
        spec_str = ""
        if res["spec_rounds"] > 0:
            acc_rate = 100.0 * res["spec_accepted"] / max(res["spec_drafted"], 1)
            avg_len = (res["spec_accepted"] + res["spec_rounds"]) / res["spec_rounds"]
            spec_str = f" | dspark={res['spec_accepted']}/{res['spec_drafted']} ({acc_rate:.1f}%, {avg_len:.2f} tok/round)"
        print(
            f"  prefill={res['prefill_tps']:.1f} tok/s | decode={res['decode_tps']:.2f} tok/s "
            f"({res['ms_per_tok']:.1f} ms/tok) [{verdict}] | RSS={res['peak_rss_mib']:.0f} MiB | "
            f"kv={res['kv_bytes_per_token']} B/tok{spec_str}"
        )
        if res["long_prefill_tps"] is not None:
            print(f"  long prefill ({res['long_prefill_tokens']} tok): {res['long_prefill_tps']:.1f} tok/s")
        post_health = check_system_health()
        print(f"  post-run swap: {post_health['swap']}")
        print(f"  preview: {res['preview']!r}\n")

    # Greedy token consistency check across KV modes vs first mode.
    if len(results) >= 2:
        ref = results[0]["generated_ids"]
        for r in results[1:]:
            match = sum(1 for a, b in zip(ref, r["generated_ids"]) if a == b)
            total = min(len(ref), len(r["generated_ids"]))
            print(f"Greedy match {results[0]['mode']} vs {r['mode']}: {match}/{total} tokens identical")

    if args.json_out:
        out_path = Path(args.json_out).expanduser()
        out_path.write_text(json.dumps({"model_dir": str(model_dir), "health": health, "results": results}, indent=2))
        print(f"Wrote JSON report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
