#!/usr/bin/env python3
"""Phase G: real guarded single-request generation through the C ABI dylib.

Usage (under real_run_guard with the owner's flags):
  python3 cpp/engine/capi/test_capi_live.py <dylib> <ckpt> <config> <prompt_ids_csv> <steps> [engine] [reference.json]

`engine` is optional (qwen|nemotron); when present it is passed as
{"model": engine} so model selection is explicit rather than inferred from the
checkpoint directory name. When `reference.json` is supplied, the C++ step log
(tokens + per-step logit SHA-256) is compared bit-exactly to its `steps` array
and the script exits non-zero on any divergence.

Prints one `python step=... token=...` line per vx_session_step. The C++ runtime
appends CLI-format `step=.. in=.. out=.. sha256=..` lines to the file named by
VEYSTRIX_CPP_CAPI_STEP_LOG (a private temp file when the variable is unset).
"""
import ctypes
import json
import os
import re
import sys
import tempfile
from pathlib import Path

STEP_RE = re.compile(r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$")

lib = ctypes.CDLL(sys.argv[1])
lib.vx_abi_version.argtypes = []
lib.vx_abi_version.restype = ctypes.c_int
lib.vx_mode.argtypes = []
lib.vx_mode.restype = ctypes.c_char_p
lib.vx_last_error.argtypes = []
lib.vx_last_error.restype = ctypes.c_char_p
lib.vx_engine_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.vx_engine_open.restype = ctypes.c_void_p
lib.vx_engine_close.argtypes = [ctypes.c_void_p]
lib.vx_engine_close.restype = None
lib.vx_session_open.argtypes = [ctypes.c_void_p]
lib.vx_session_open.restype = ctypes.c_void_p
lib.vx_session_close.argtypes = [ctypes.c_void_p]
lib.vx_session_close.restype = None
lib.vx_session_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
lib.vx_session_prefill.restype = ctypes.c_int
lib.vx_session_step.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
lib.vx_session_step.restype = ctypes.c_int
lib.vx_engine_stats.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.vx_engine_stats.restype = ctypes.c_int

ckpt, config, prompt_csv, steps = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
engine = sys.argv[6] if len(sys.argv) > 6 else ""
reference_path = sys.argv[7] if len(sys.argv) > 7 else ""
if prompt_csv.startswith("@"):
    # Long prompts exceed ARG_MAX; read the CSV from a file instead.
    ids = [int(x) for x in Path(prompt_csv[1:]).read_text().strip().split(",") if x]
else:
    ids = [int(x) for x in prompt_csv.split(",")]
profile_obj = {"config_path": config}
if engine:
    profile_obj["model"] = engine
profile = json.dumps(profile_obj)


def error():
    v = lib.vx_last_error()
    return v.decode("utf-8", "replace") if v else ""


# Capture only the records this run appends. If the caller did not set the step
# log path, use a private temp file so the comparison is self-contained.
own_log = False
log_path = os.environ.get("VEYSTRIX_CPP_CAPI_STEP_LOG", "")
if not log_path:
    fd, log_path = tempfile.mkstemp(prefix="vx-capi-step-")
    os.close(fd)
    os.environ["VEYSTRIX_CPP_CAPI_STEP_LOG"] = log_path
    own_log = True
try:
    offset = os.path.getsize(log_path)
except OSError:
    offset = 0

print("[phase-g] abi=%d mode=%s engine=%s" %
      (lib.vx_abi_version(), lib.vx_mode().decode(), engine or "(inferred)"), flush=True)
engine_handle = lib.vx_engine_open(ckpt.encode(), profile.encode())
assert engine_handle, "engine open failed: " + error()
session = lib.vx_session_open(engine_handle)
assert session, "session open failed: " + error()
try:
    arr = (ctypes.c_uint32 * len(ids))(*ids)
    rc = lib.vx_session_prefill(session, arr, len(ids))
    assert rc == 0, "prefill rc=%d: %s" % (rc, error())
    out = ctypes.c_uint32()
    for i in range(steps):
        rc = lib.vx_session_step(session, b"{}", ctypes.byref(out))
        assert rc == 0, "step %d rc=%d: %s" % (i, rc, error())
        print("python step=%d token=%d" % (i, out.value), flush=True)
    buf = ctypes.create_string_buffer(4096)
    rc = lib.vx_engine_stats(engine_handle, buf, len(buf))
    assert rc == 0, "stats rc=%d: %s" % (rc, error())
    print("[phase-g] stats=" + buf.value.decode(), flush=True)
finally:
    lib.vx_session_close(session)
    lib.vx_engine_close(engine_handle)

records = []
with open(log_path, "rb") as f:
    f.seek(offset)
    for line in f.read().decode("utf-8", "replace").splitlines():
        m = STEP_RE.match(line.strip())
        if m:
            records.append({"step": int(m.group(1)), "in": int(m.group(2)),
                            "out": int(m.group(3)), "digest": m.group(4)})
        else:
            print("[phase-g] unparsed step-log line: %r" % line, flush=True)
for r in records:
    print("cpp step=%d in=%d out=%d sha256=%s" %
          (r["step"], r["in"], r["out"], r["digest"]), flush=True)
if own_log:
    try:
        os.unlink(log_path)
    except OSError:
        pass

if reference_path:
    ref = json.load(open(reference_path))
    expected = ref.get("steps", [])
    ok = len(records) == len(expected)
    print("[phase-g] comparison: %d dylib steps vs %d reference steps" %
          (len(records), len(expected)), flush=True)
    for i in range(min(len(records), len(expected))):
        g, e = records[i], expected[i]
        tok_ok = g["out"] == int(e["token"])
        dig_ok = g["digest"] == e["logit_sha256"]
        print("[phase-g] step %2d: token %s digest %s" %
              (i, "OK" if tok_ok else "DIFF", "OK" if dig_ok else "DIFF"), flush=True)
        ok = ok and tok_ok and dig_ok
    print("[phase-g] GATE %s (tokens+digests bit-exact)" %
          ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)
