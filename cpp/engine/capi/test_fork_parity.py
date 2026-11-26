#!/usr/bin/env python3
"""G-PC-MULTI: vx_session_fork (COW) == fresh solo session, bit-exact.

One engine, one process (MLX stream is process-wide). A is prefilled 12 tokens
and forked into B; both append 4 tokens and decode 4 steps. C is a fresh
session prefilled with all 16 tokens and decoded 4 steps. All three streams
must match in tokens and logit digests.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

p = ctypes.c_void_p
u32p = ctypes.POINTER(ctypes.c_uint32)
lib = ctypes.CDLL(sys.argv[1])
lib.vx_last_error.argtypes = []; lib.vx_last_error.restype = ctypes.c_char_p
lib.vx_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]; lib.vx_engine_open.restype = p
lib.vx_engine_close.argtypes = [p]
lib.vx_session_open.argtypes = [p]; lib.vx_session_open.restype = p
lib.vx_session_close.argtypes = [p]
lib.vx_session_prefill.argtypes = [p, u32p, ctypes.c_int]; lib.vx_session_prefill.restype = ctypes.c_int
lib.vx_session_prefill_ex.argtypes = [p, u32p, ctypes.c_int, ctypes.c_uint32]; lib.vx_session_prefill_ex.restype = ctypes.c_int
lib.vx_session_fork.argtypes = [p, p]; lib.vx_session_fork.restype = ctypes.c_int
lib.vx_session_step.argtypes = [p, ctypes.c_char_p, u32p]; lib.vx_session_step.restype = ctypes.c_int
lib.vx_session_last_digest.argtypes = [p, ctypes.c_char_p, ctypes.c_int]; lib.vx_session_last_digest.restype = ctypes.c_int

ckpt, config, ref_path = sys.argv[2:5]
ref = json.loads(Path(ref_path).read_text())
prompt = [int(t) for t in (ref["prompt_ids"] if isinstance(ref.get("prompt_ids"), list)
                           else __import__("numpy").default_rng(20261117).integers(
                               0, 200000, size=int(ref["prompt_tokens"])).tolist())][:16]

def ck(code, what):
    if code: raise RuntimeError(what + ": " + (lib.vx_last_error() or b"").decode())

def prefill(s, ids):
    a = (ctypes.c_uint32 * len(ids))(*ids); ck(lib.vx_session_prefill(s, a, len(ids)), "prefill")
def append(s, ids):
    a = (ctypes.c_uint32 * len(ids))(*ids); ck(lib.vx_session_prefill_ex(s, a, len(ids), 1), "append")
def gen(s, n=4):
    out = []
    for _ in range(n):
        o = ctypes.c_uint32(); ck(lib.vx_session_step(s, b'{}', ctypes.byref(o)), "step")
        b = ctypes.create_string_buffer(128); ck(lib.vx_session_last_digest(s, b, len(b)), "digest")
        out.append([int(o.value), b.value.decode()])
    return out

eng = lib.vx_engine_open(ckpt.encode(), json.dumps(
    {"config_path": config, "model": "nemotron", "digests": True}).encode())
assert eng
try:
    a = lib.vx_session_open(eng); b = lib.vx_session_open(eng); c = lib.vx_session_open(eng)
    prefill(a, prompt[:12])
    ck(lib.vx_session_fork(a, b), "fork")
    append(a, prompt[12:]); append(b, prompt[12:])
    prefill(c, prompt[:12]); append(c, prompt[12:])  # same 12+4 chunk plan
    ra, rb, rc = gen(a), gen(b), gen(c)
    for s in (a, b, c): lib.vx_session_close(s)
finally:
    lib.vx_engine_close(eng)
print("fork-A:", ra)
print("fork-B:", rb)
print("solo-C:", rc)
print("G-PC-MULTI:", "MATCH" if ra == rb == rc else "DIFF")
sys.exit(0 if ra == rb == rc else 1)
