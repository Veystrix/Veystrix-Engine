#!/usr/bin/env python3
"""Segmented long-context prefill: one logical session across guarded segments.

Each segment restores the previous snapshot, APPENDs one bounded chunk, then
saves a new snapshot. Running one segment per process keeps the guard's swap
ceiling honest (swap is measured from each process launch) while the snapshot
chain preserves the exact computation across the whole prompt.

Modes:
  init  - prefill the first chunk, save snapshot (no restore)
  step  - restore snapshot at (seg-1)*CHUNK, append CHUNK tokens, save snapshot
  final - restore the last snapshot, append remainder, generate N steps,
          emit tokens+digests for parity
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
lib.vx_last_error.argtypes = []; lib.vx_last_error.restype = ctypes.c_char_p
lib.vx_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]; lib.vx_engine_open.restype = p
lib.vx_engine_close.argtypes = [p]
lib.vx_session_open.argtypes = [p]; lib.vx_session_open.restype = p
lib.vx_session_close.argtypes = [p]
lib.vx_session_prefill.argtypes = [p, u32p, ctypes.c_int]; lib.vx_session_prefill.restype = ctypes.c_int
lib.vx_session_prefill_ex.argtypes = [p, u32p, ctypes.c_int, ctypes.c_uint32]; lib.vx_session_prefill_ex.restype = ctypes.c_int
lib.vx_snapshot_save.argtypes = [p, ctypes.c_char_p]; lib.vx_snapshot_save.restype = ctypes.c_int
lib.vx_snapshot_restore.argtypes = [p, ctypes.c_char_p]; lib.vx_snapshot_restore.restype = ctypes.c_int
lib.vx_session_step.argtypes = [p, ctypes.c_char_p, u32p]; lib.vx_session_step.restype = ctypes.c_int
lib.vx_session_last_digest.argtypes = [p, ctypes.c_char_p, ctypes.c_int]; lib.vx_session_last_digest.restype = ctypes.c_int

ckpt, cfg, prompt_path, snapdir = sys.argv[2:6]
mode = os.environ.get("LC_MODE", "init")
seg = int(os.environ.get("LC_SEG", "1"))
chunk = int(os.environ.get("LC_CHUNK", "1024"))
prompt = [int(t) for t in json.loads(Path(prompt_path).read_text())["prompt_ids"]]

def ck(code, what):
    if code: raise RuntimeError(what + ": " + (lib.vx_last_error() or b"").decode())

profile = {"config_path": cfg, "model": "nemotron", "digests": True,
           "context_window": int(os.environ.get("LC_CONTEXT", "1048576")),
           "max_cold_prefill": int(os.environ.get("LC_COLD", "65536")),
           "max_incremental_prefill": int(os.environ.get("LC_INC", "65536"))}
eng = lib.vx_engine_open(ckpt.encode(), json.dumps(profile).encode())
assert eng, (lib.vx_last_error() or b"").decode()
try:
    s = lib.vx_session_open(eng)
    if mode == "init":
        part = prompt[:chunk]
        arr = (ctypes.c_uint32 * len(part))(*part)
        ck(lib.vx_session_prefill(s, arr, len(part)), "init prefill")
        ck(lib.vx_snapshot_save(s, snapdir.encode()), "init snapshot")
        print(json.dumps({"segment": 0, "tokens": len(part)}))
    elif mode == "step":
        ck(lib.vx_snapshot_restore(s, snapdir.encode()), "restore")
        start = (seg - 1) * chunk
        part = prompt[start:start + chunk]
        arr = (ctypes.c_uint32 * len(part))(*part)
        ck(lib.vx_session_prefill_ex(s, arr, len(part), 1), "append")
        ck(lib.vx_snapshot_save(s, snapdir.encode()), "snapshot")
        print(json.dumps({"segment": seg, "tokens": start + len(part)}))
    elif mode == "final":
        ck(lib.vx_snapshot_restore(s, snapdir.encode()), "restore")
        start = (seg - 1) * chunk
        part = prompt[start:start + chunk]
        if part:
            arr = (ctypes.c_uint32 * len(part))(*part)
            ck(lib.vx_session_prefill_ex(s, arr, len(part), 1), "final append")
        out = []
        for _ in range(4):
            o = ctypes.c_uint32()
            ck(lib.vx_session_step(s, b'{}', ctypes.byref(o)), "step")
            b = ctypes.create_string_buffer(128)
            ck(lib.vx_session_last_digest(s, b, len(b)), "digest")
            out.append([int(o.value), b.value.decode()])
        print(json.dumps({"segment": seg, "total_tokens": len(prompt), "output": out}))
    lib.vx_session_close(s)
finally:
    lib.vx_engine_close(eng)
