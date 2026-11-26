#!/usr/bin/env python3
"""G-PC-COLDWARM for Nemotron: cold prefill == restore + APPEND, bit-exact.

Run under real_run_guard. Uses the live C ABI dylib and a real checkpoint.
Session A: prefill 16 tokens, greedy 4 steps, record tokens+logit digests.
Session B: prefill 13 tokens, restore a snapshot captured after 12 tokens,
APPEND the remaining 4 tokens, greedy 4 steps; compare to A.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

p = ctypes.c_void_p
u32p = ctypes.POINTER(ctypes.c_uint32)
lib = ctypes.CDLL(sys.argv[1])
lib.vx_last_error.argtypes = []
lib.vx_last_error.restype = ctypes.c_char_p
lib.vx_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.vx_engine_open.restype = p
lib.vx_engine_close.argtypes = [p]
lib.vx_session_open.argtypes = [p]
lib.vx_session_open.restype = p
lib.vx_session_close.argtypes = [p]
lib.vx_session_prefill.argtypes = [p, u32p, ctypes.c_int]
lib.vx_session_prefill.restype = ctypes.c_int
lib.vx_session_prefill_ex.argtypes = [p, u32p, ctypes.c_int, ctypes.c_uint32]
lib.vx_session_prefill_ex.restype = ctypes.c_int
lib.vx_snapshot_save.argtypes = [p, ctypes.c_char_p]
lib.vx_snapshot_save.restype = ctypes.c_int
lib.vx_snapshot_restore.argtypes = [p, ctypes.c_char_p]
lib.vx_snapshot_restore.restype = ctypes.c_int
lib.vx_session_step.argtypes = [p, ctypes.c_char_p, u32p]
lib.vx_session_step.restype = ctypes.c_int
lib.vx_session_last_digest.argtypes = [p, ctypes.c_char_p, ctypes.c_int]
lib.vx_session_last_digest.restype = ctypes.c_int

ckpt, config, reference_path, snapdir = sys.argv[2:6]
steps = 4
ref = json.loads(Path(reference_path).read_text())
prompt = [int(t) for t in ref["prompt_ids"]]
profile = json.dumps({"config_path": config, "model": "nemotron", "digests": True})


def ck(code, what):
    if code != 0:
        raise RuntimeError(what + ": " + (lib.vx_last_error() or b"").decode())


def prefill(s, ids):
    a = (ctypes.c_uint32 * len(ids))(*ids)
    ck(lib.vx_session_prefill(s, a, len(ids)), "prefill")


def append(s, ids):
    a = (ctypes.c_uint32 * len(ids))(*ids)
    ck(lib.vx_session_prefill_ex(s, a, len(ids), 1), "append")


def generate(s, n):
    out = []
    for _ in range(n):
        o = ctypes.c_uint32()
        ck(lib.vx_session_step(s, b'{}', ctypes.byref(o)), "step")
        buf = ctypes.create_string_buffer(128)
        ck(lib.vx_session_last_digest(s, buf, len(buf)), "digest")
        out.append((int(o.value), buf.value.decode()))
    return out


import os
mode = os.environ.get("VEYSTRIX_GPC_MODE", "cold")
NP = int(os.environ.get("VEYSTRIX_GPC_N", "16"))
SNAP = int(os.environ.get("VEYSTRIX_GPC_SNAP", "12"))
prompt = prompt[:NP]
eng = lib.vx_engine_open(ckpt.encode(), profile.encode())
assert eng, (lib.vx_last_error() or b"").decode()
try:
    if mode == "cold":
        cold = lib.vx_session_open(eng)
        prefill(cold, prompt[:SNAP])
        ck(lib.vx_snapshot_save(cold, snapdir.encode()), "snapshot save")
        append(cold, prompt[SNAP:])
        cold_out = generate(cold, steps)
        lib.vx_session_close(cold)
        print(json.dumps(cold_out))
    else:
        warm = lib.vx_session_open(eng)
        # Fresh session: restore carries its own token history, so the 512
        # prefix tokens are NOT re-prefilled (this is the TTFT saving).
        ck(lib.vx_snapshot_restore(warm, snapdir.encode()), "snapshot restore")
        append(warm, prompt[SNAP:])
        warm_out = generate(warm, steps)
        lib.vx_session_close(warm)
        print(json.dumps(warm_out))
finally:
    lib.vx_engine_close(eng)
