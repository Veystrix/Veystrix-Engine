#!/usr/bin/env python3
"""Bit-exact gate for the overlap path: N=2 pipelined == N=2 serial.

Run under real_run_guard with the owner's flags. Uses the live C ABI dylib and
a real Nemotron checkpoint. Two sessions with distinct prompt prefixes are
driven with VEYSTRIX_CPP_OVERLAP=0 (serial) and =1 (pipelined); tokens and
per-step logit SHA-256 must match for BOTH sessions.

Usage: test_overlap_parity.py <dylib> <ckpt> <config> <reference.json> <steps>
"""
from __future__ import annotations

import ctypes
import json
import os
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
lib.vx_engine_close.restype = None
lib.vx_session_open.argtypes = [p]
lib.vx_session_open.restype = p
lib.vx_session_close.argtypes = [p]
lib.vx_session_close.restype = None
lib.vx_session_prefill.argtypes = [p, u32p, ctypes.c_int]
lib.vx_session_prefill.restype = ctypes.c_int
lib.vx_session_step_batch.argtypes = [p, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_char_p), u32p]
lib.vx_session_step_batch.restype = ctypes.c_int
lib.vx_session_last_digest.argtypes = [p, ctypes.c_char_p, ctypes.c_int]
lib.vx_session_last_digest.restype = ctypes.c_int

ckpt, config, reference_path, steps = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
ref = json.loads(Path(reference_path).read_text())
prompt = [int(t) for t in ref["prompt_ids"][:16]]
profile = json.dumps({"config_path": config, "model": "nemotron", "digests": True})


def run(overlap: str):
    os.environ["VEYSTRIX_CPP_OVERLAP"] = overlap
    eng = lib.vx_engine_open(ckpt.encode(), profile.encode())
    assert eng, (lib.vx_last_error() or b"").decode()
    sessions = [lib.vx_session_open(eng) for _ in range(2)]
    for i, s in enumerate(sessions):
        ids = prompt[: 8 + 8 * i]  # distinct prefixes
        arr = (ctypes.c_uint32 * len(ids))(*ids)
        assert lib.vx_session_prefill(s, arr, len(ids)) == 0, (lib.vx_last_error() or b"").decode()
    sess_arr = (p * 2)(*sessions)
    samplers = (ctypes.c_char_p * 2)(b'{}', b'{}')
    result = [[], []]
    for _ in range(steps):
        out = (ctypes.c_uint32 * 2)()
        rc = lib.vx_session_step_batch(sess_arr, 2, samplers, out)
        assert rc == 0, (lib.vx_last_error() or b"").decode()
        for i, s in enumerate(sessions):
            buf = ctypes.create_string_buffer(128)
            assert lib.vx_session_last_digest(s, buf, len(buf)) == 0
            result[i].append((int(out[i]), buf.value.decode()))
    for s in sessions:
        lib.vx_session_close(s)
    lib.vx_engine_close(eng)
    return result


# One process per mode: MLX's driver-thread stream is process-wide state and a
# second engine in the same process hits "no Stream(gpu,0) in current thread".
mode = os.environ.get("VEYSTRIX_CPP_OVERLAP", "0")
print(json.dumps(run(mode)))
