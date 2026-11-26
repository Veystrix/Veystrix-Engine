#!/usr/bin/env python3
"""Real-Metal decode/prefill measurement helper for Inkling 2 Mini.

Benchmarks the C++ engine through the same bindings as scripts/bench_inkling2.py,
but supports sampled decoding and arbitrary prompt lengths. It writes a JSON
report; it does not assert, grade, or modify any product state.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from veystrix.backends.cpp_generic import build_generic_backend  # noqa: E402

PROMPT = (
    "Write a complete, production-ready Python LRU cache with TTL expiration, "
    "thread safety using RLock, O(1) get/put/delete, and unit tests. Output code only."
)


class _Swap(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("avail", ctypes.c_uint64),
                ("used", ctypes.c_uint64), ("page", ctypes.c_uint32),
                ("encrypted", ctypes.c_bool)]


def swap_mb() -> float:
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    u = _Swap()
    n = ctypes.c_size_t(ctypes.sizeof(u))
    if lib.sysctlbyname(b"vm.swapusage", ctypes.byref(u), ctypes.byref(n), None, 0) == 0:
        return u.used / 1048576.0
    return -1.0


def pressure() -> int:
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    v = ctypes.c_int(0)
    n = ctypes.c_size_t(ctypes.sizeof(v))
    if lib.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(v), ctypes.byref(n), None, 0) == 0:
        return int(v.value)
    return -1


def mlx_gb() -> dict:
    try:
        import mlx.core as mx
        m = mx.metal
        return {"active": m.get_active_memory() / 2**30, "cache": m.get_cache_memory() / 2**30,
                "peak": m.get_peak_memory() / 2**30}
    except Exception:
        return {"active": -1, "cache": -1, "peak": -1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.path.expanduser("~/models/Inkling-2-Mini-oQ4e-mtp"))
    ap.add_argument("--mode", default="sparse-paged")
    ap.add_argument("--dspark", type=int, default=1)
    ap.add_argument("--draft-tokens", type=int, default=5)
    ap.add_argument("--prompt-tokens", type=int, default=54)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--sampler", default='{"temperature":0.0}')
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    os.environ["VEYSTRIX_KV_MODE"] = args.mode
    os.environ["VEYSTRIX_CONTEXT_LIMIT"] = str(1048576)
    os.environ["VEYSTRIX_INKLING_DSPARK"] = str(args.dspark)
    os.environ["VEYSTRIX_INKLING_DSPARK_DRAFT"] = str(args.draft_tokens)

    backend = build_generic_backend("inkling2-mini",
                                    SimpleNamespace(model_path=args.model, session_dir=None), prefs={})
    try:
        tokenizer = backend.tokenizer
        text = PROMPT
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": PROMPT}], tokenize=False,
                    add_generation_prompt=True, reasoning_effort="none")
            except TypeError:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
        unit = tokenizer.encode(text, add_special_tokens=False) or tokenizer.encode(PROMPT)
        ids = (unit * ((args.prompt_tokens // len(unit)) + 1))[:args.prompt_tokens]

        binding = backend._binding
        session = binding.open_session(backend._engine)
        backend._session = session
        sw0, pr0 = swap_mb(), pressure()
        t0 = time.perf_counter()
        binding.prefill(session, ids)
        prefill_s = time.perf_counter() - t0
        mlx_pf = mlx_gb()

        sampled = json.loads(args.sampler)
        out = []
        digests = []
        with_digests = bool(os.environ.get("VEYSTRIX_CPP_DIGESTS") == "1" and
                            getattr(binding, "has_v2", False))
        t1 = time.perf_counter()
        for _ in range(args.max_tokens):
            tok = int(binding.step(session, json.dumps(sampled)))
            out.append(tok)
            if with_digests:
                digests.append(binding.last_digest(session))
            if tok in backend.stop_tokens:
                break
        decode_s = time.perf_counter() - t1
        stats = binding.stats(backend._engine) or {}
        report = {
            "mode": args.mode, "dspark": args.dspark, "draft": args.draft_tokens,
            "prompt_tokens": len(ids), "decode_tokens": len(out),
            "prefill_tps": round(len(ids) / max(prefill_s, 1e-6), 2),
            "decode_tps": round(len(out) / max(decode_s, 1e-6), 2),
            "sampler": sampled,
            "generated_ids": out,
            "step_digests": digests,
            "digest_source": "logit" if with_digests else "none",
            "preview": tokenizer.decode(out[:24], skip_special_tokens=False).replace("\n", "\\n"),
            "stats": {k: stats.get(k) for k in (
                "spec_rounds", "spec_drafted", "spec_accepted", "expert_bytes_read",
                "resident_bytes", "streamed_bytes")},
            "mlx_prefill": mlx_pf, "mlx_end": mlx_gb(),
            "swap_before_mb": round(sw0, 1), "swap_after_mb": round(swap_mb(), 1),
            "pressure_before": pr0, "pressure_after": pressure(),
            "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 1),
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        print("prefill=%.1f tok/s decode=%.2f tok/s tokens=%d spec=%s/%s" % (
            report["prefill_tps"], report["decode_tps"], len(out),
            report["stats"].get("spec_accepted"), report["stats"].get("spec_drafted")))
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
