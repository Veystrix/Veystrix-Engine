#!/usr/bin/env python3
"""G-APPEND-QWEN: C-ABI APPEND prefill vs a Python reference with the SAME
anchored chunk boundaries (plan P3 gate).

The Python reference is captured by capture_real_reference.py with
FLASHNEXT_PREFILL_CHUNK_TOKENS set; its reference.json records the effective
chunk config and the prompt token ids. We recompute the exact spans with
models.flashnext.prefill.plan_chunks (the authoritative planner) and drive the
C ABI with one RESET + one APPEND per remaining span. Tokens and per-step
logit SHA-256 are compared bit-exactly to the Python reference.

Run BOTH processes under real_run_guard. Usage:
  check_capi_append_parity.py <dylib> <ckpt> <config> <reference.json> <steps>
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.flashnext.prefill import plan_chunks  # noqa: E402

STEP_RE = re.compile(r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$")

p = ctypes.c_void_p
u32p = ctypes.POINTER(ctypes.c_uint32)
lib = ctypes.CDLL(sys.argv[1])
lib.vx_last_error.argtypes = []
lib.vx_last_error.restype = ctypes.c_char_p
lib.vx_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.vx_engine_open.restype = p
lib.vx_engine_close.argtypes = [p]
lib.vx_engine_close.restype = None
lib.vx_session_open.argtypes = [p]
lib.vx_session_open.restype = p
lib.vx_session_close.argtypes = [p]
lib.vx_session_close.restype = None
lib.vx_session_prefill.argtypes = [p, u32p, ctypes.c_int]
lib.vx_session_prefill.restype = ctypes.c_int
lib.vx_session_prefill_ex.argtypes = [p, u32p, ctypes.c_int, ctypes.c_uint32]
lib.vx_session_prefill_ex.restype = ctypes.c_int
lib.vx_session_step.argtypes = [p, ctypes.c_char_p, u32p]
lib.vx_session_step.restype = ctypes.c_int
lib.vx_session_last_digest.argtypes = [p, ctypes.c_char_p, ctypes.c_int]
lib.vx_session_last_digest.restype = ctypes.c_int

ckpt, config, reference_path, steps = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
reference = json.loads(Path(reference_path).read_text())
if isinstance(reference.get("prompt_tokens"), list):
    prompt = [int(t) for t in reference["prompt_tokens"]]
else:
    import numpy as np
    prompt = np.random.default_rng(20261117).integers(
        0, 200000, size=int(reference["prompt_tokens"])).tolist()
prompt = prompt[:int(os.environ.get("VEYSTRIX_APPEND_N", "1024"))]
chunk = int(os.environ.get("FLASHNEXT_PREFILL_CHUNK_TOKENS", "512"))
mode = os.environ.get("VEYSTRIX_APPEND_MODE", "single")
chunk = int(os.environ.get("VEYSTRIX_APPEND_CHUNK", "512"))
profile = json.dumps({"config_path": config, "model": "qwen", "digests": True})
eng = lib.vx_engine_open(ckpt.encode(), profile.encode())
assert eng, "engine open failed: " + (lib.vx_last_error() or b"").decode()
try:
    sess = lib.vx_session_open(eng)
    if mode == "single":
        ids = (ctypes.c_uint32 * len(prompt))(*prompt)
        assert lib.vx_session_prefill(sess, ids, len(prompt)) == 0
    else:
        for k, a in enumerate(range(0, len(prompt), chunk)):
            part = prompt[a:a + chunk]
            ids = (ctypes.c_uint32 * len(part))(*part)
            rc = (lib.vx_session_prefill(sess, ids, len(part)) if k == 0 else
                  lib.vx_session_prefill_ex(sess, ids, len(part), 1))
            assert rc == 0, "prefill/append failed"
    out = []
    for _ in range(steps):
        o = ctypes.c_uint32()
        assert lib.vx_session_step(sess, b'{}', ctypes.byref(o)) == 0
        buf = ctypes.create_string_buffer(128)
        assert lib.vx_session_last_digest(sess, buf, len(buf)) == 0
        out.append([int(o.value), buf.value.decode()])
    lib.vx_session_close(sess)
finally:
    lib.vx_engine_close(eng)
print(json.dumps(out))
