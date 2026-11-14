#!/usr/bin/env python3
"""Stage 4 kernel laboratory: what one decode kernel costs, shape by shape.

One decode token spends about 72 ms waiting for the GPU and issues about 720
tiny `quantized_matmul` calls and 140 custom Metal MoE launches (the shape
corpus, `kernel_corpus.py`). The hypothesis is that launch overhead, not
arithmetic, dominates at M=1, and that verification rounds (M = 2 to 8 rows)
scale differently per shape. This laboratory prices every corpus shape, the
custom MoE executor, and the GatedDeltaNet recurrence on synthetic weights,
with no checkpoint and no model load.

Timing method. This project twice published wrong per-op GPU numbers by timing
one `mx.eval` per operation: each eval pays a full submit and wait round trip,
so summing them inflates totals (see bench_gather_qmm, bench_layer_split,
bench_glue). Here every figure is a chain of many repetitions in one lazy
graph, evaluated once, in two forms:

- independent: distinct inputs per repetition, so nothing is folded and the GPU
  is free to overlap neighbours (a throughput figure; an optimistic bound);
- dependent: each repetition consumes the previous result, as the layer chain
  of a real token does. Ops whose output is not their input shape use a one
  kernel link (`x0 + y.reshape(-1)[:1] * 0`); the link is priced on its own and
  subtracted, and both numbers are stored.

Each chain is also split into graph-build time (Python, host) and eval time
(encode, submit, GPU, wait). The headline microseconds are eval time: the
cost the GPU timeline sees. The eval total includes the host's C++ encode
cost per launch, so "launch floor" below is host encode plus GPU execution,
not GPU alone; timestamps would be needed to split those. The one-eval-per-op
figure is reported separately and labelled round trip; the gap to the chained
figure is itself a result. Linearity is checked by timing chains at R and 2R
repetitions (fastest chain of each, so interference spikes do not decide it):
the ratio must be 2 within 10%, and the intercept is the fixed submit and wait
cost of a chain.

The token model at the end sums calls per step times chained microseconds
over the corpus. It omits everything not in the corpus (norms, elementwise
glue, attention, the router matmul, hyper-connection elementwise work,
sampling, n-gram, eval round trips), so it is a lower bound on the kernels it
covers, not a token time.

The block section times the row-batched MoE kernels (`block=` of the executor)
against the per-pair kernels at M = 2, 4, 8 on same, disjoint and recorded
("trace") routing, checks every block output bit for bit against the per-pair
output, and prices the host-side plan separately; the token model then states
a verification round both ways. A section that raises is recorded under
`skipped` and the run still writes its artifact, then exits non-zero.

This benchmark changes no model behavior. It loads no checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import mmap
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import mlx.core as mx

from models.flashnext.tests.bench import kernel_corpus

SCHEMA_VERSION = 1
ROWS = (1, 2, 4, 8)
HIDDEN, INTER, TOP_K, GROUP = 2560, 640, 10, 32
LINEARITY_TOLERANCE = 0.10
SYNTHETIC_WEIGHT_LIMIT = 512 * 2**20
RETAINED_LIMIT = 96 * 2**20
SECTIONS = ("floor", "dense", "moe", "block", "gdn")
# Block (row-batched) MoE section: top_k 8 as in the recorded route windows, bfloat16 activations, pool path.
BLOCK_TOP_K = 8
BLOCK_ROWS = (2, 4, 8)
BLOCK_M = (2, 4, 8)
BLOCK_PATTERNS = ("same", "disjoint", "trace")
# A multiple of the 24 trace windows, so every chain (and both lengths of a linearity check) covers each window equally.
BLOCK_REPS = 48
POOL_EXPERTS = 80
WINDOWS_FIXTURE = kernel_corpus.REPO_ROOT / "models/flashnext/tests/inputs/stage4-route-windows.json"
DRIFT_WARN_PCT = 5.0
# Tolerances of MLXQ4G32SmokeTests.assert_mlx_close in test_metal_runtime.py.
RTOL, ATOL, MIN_COSINE = 2e-2, 3e-3, 0.9999
BF16_TOLERANCE = (RTOL, ATOL)
TINY_OUT = 64
# Activation scale of the MoE inputs: with weights of std 0.02 the intermediates stay near 0.1, so one
# bfloat16 rounding step is below the unit tests' atol and the elementwise comparison is meaningful.
X_STD = 0.25
# GDN geometry. Hv, Dv and Hk*Dk follow from the corpus (in_proj_a 48 outputs,
# in_proj_z 6144, in_proj_qkv 10240 = 2*Hk*Dk + 6144); the Hk/Dk split does
# not, and comes from checkpoint config.json (linear_num_key_heads=16,
# linear_key_head_dim=128). `gdn_geometry` re-checks the sums against the corpus.
GDN_HK, GDN_DK, GDN_HV, GDN_DV = 16, 128, 48, 128
# Kernel launches per call, used only for the launch-floor-only model.
# The GDN figure is an assumption: one compiled g/beta kernel plus the
# recurrence kernel.
LAUNCHES = {"dense": 1, "gather": 1, "moe_fused": 3, "moe_unfused": 4, "gdn": 2}


# ----------------------------------------------------------------- system

def _run(command, timeout=10):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as error:  # pragma: no cover - tool missing
        return f"unavailable: {error}"


def thermal_level():
    """`notifyutil` thermal pressure level, or None when it cannot be read."""
    text = _run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"])
    match = re.search(r"(\d+)\s*$", text)
    return int(match.group(1)) if match else None


def power_state():
    batt = _run(["pmset", "-g", "batt"])
    settings = _run(["pmset", "-g"])
    low = re.search(r"lowpowermode\s+(\d+)", settings)
    return {"batt": batt, "low_power_mode": int(low.group(1)) if low else None}


def mlx_provenance():
    from importlib import metadata

    def version(name):
        try:
            return metadata.version(name)
        except Exception:
            return None

    info = {}
    try:
        info = dict(mx.device_info())
    except Exception:
        try:
            info = dict(mx.metal.device_info())
        except Exception:
            pass
    return {
        "version": mx.__version__, "mlx_core_file": mx.__file__,
        "mlx_package_path": list(getattr(sys.modules.get("mlx"), "__path__", [])),
        "distributions": {"mlx": version("mlx"), "mlx-metal": version("mlx-metal"),
                          "mlx-vlm": version("mlx-vlm"), "mlx-lm": version("mlx-lm")},
        "device_info": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in info.items()},
        "python": sys.version.split()[0], "sys_path_head": sys.path[:4],
    }


def host_provenance():
    root = Path(__file__).resolve().parents[4]

    def git(*args):
        return _run(["git", "-C", str(root), *args])

    return {
        "macos": _run(["sw_vers", "-productVersion"]), "platform": platform.platform(),
        "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ram_bytes": int(_run(["sysctl", "-n", "hw.memsize"]) or 0),
        "power": power_state(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_paths": len([line for line in git("status", "--porcelain").splitlines() if line]),
        "lab_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


class Thermal:
    """Thermal pressure samples around the run; any non-zero level is reported loudly."""

    def __init__(self):
        self.samples = []

    def sample(self, label):
        self.samples.append({"label": label, "level": thermal_level(), "t": time.time()})

    def summary(self):
        levels = [s["level"] for s in self.samples if s["level"] is not None]
        first = levels[0] if levels else None
        rose = next((s["label"] for s in self.samples if s["level"] and s["level"] > (first or 0)), None)
        return {"samples": self.samples, "start": first, "end": levels[-1] if levels else None,
                "first_rise_at": rose,
                "max": max(levels) if levels else None,
                "nonzero_at_start": bool(first), "rose_during_run": bool(levels and max(levels) > (first or 0))}


class Budget:
    def __init__(self, seconds):
        self.started = time.perf_counter()
        self.seconds = seconds
        self.skipped = []

    def elapsed(self):
        return time.perf_counter() - self.started

    def exhausted(self, what):
        if self.elapsed() > self.seconds:
            self.skipped.append(what)
            return True
        return False


# ----------------------------------------------------------------- statistics

def stat(values):
    values = [float(v) for v in values]
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values}


def linearity_check(t_small, reps_small, t_large, reps_large, tolerance=LINEARITY_TOLERANCE):
    """Does chain time scale with repetitions?

    `t_*` are chain totals in any one unit. A chain costs a + R*c: a fixed
    submit and wait plus c per repetition. Returns the observed ratio against
    the expected reps_large/reps_small, the fitted per-repetition cost `c`
    (marginal), the intercept `a`, and whether the ratio is within `tolerance`.
    """
    if reps_large <= reps_small or t_small <= 0:
        raise ValueError("need two increasing repetition counts and a positive time")
    expected = reps_large / reps_small
    ratio = t_large / t_small
    marginal = (t_large - t_small) / (reps_large - reps_small)
    intercept = t_small - marginal * reps_small
    deviation = ratio / expected - 1.0
    return {"reps": [reps_small, reps_large], "ratio": ratio, "expected": expected,
            "deviation": deviation, "marginal": marginal, "average": t_small / reps_small,
            "intercept": intercept, "ok": abs(deviation) <= tolerance}


# ----------------------------------------------------------------- measurement core

class Case:
    """One thing to time.

    `make_inputs(n)` builds `n` distinct, already evaluated inputs. `op(inp)`
    returns an array or tuple. `link(inp0, out)` builds the next input from the
    previous output (None: no dependent chain). `link_free` says the link
    launches nothing. `keep(out)` picks what a chain evaluates and retains.
    `bytes_per_rep` bounds the repetitions by retained memory.
    """

    def __init__(self, name, make_inputs, op, link=None, link_free=False, keep=None, bytes_per_rep=0,
                 reps_max=4096):
        self.name, self.make_inputs, self.op, self.link = name, make_inputs, op, link
        self.link_free = link_free
        self.keep = keep or (lambda out: out)
        self.bytes_per_rep, self.reps_max = int(bytes_per_rep), reps_max


def _now():
    return time.perf_counter_ns()


def run_independent(case, inputs):
    began = _now()
    outs = [case.keep(case.op(x)) for x in inputs]
    built = _now()
    mx.eval(outs)
    return built - began, _now() - built


def run_dependent(case, first, reps):
    began = _now()
    x = first
    for _ in range(reps):
        out = case.op(x)
        x = case.link(first, out)
    built = _now()
    mx.eval(case.keep(out), x)
    return built - began, _now() - built


def run_link_only(case, first, reps, sample):
    """The link alone, chained on itself, with a fixed stand-in for the op output."""
    if case.link_free:
        return 0, 0
    began = _now()
    x = first
    for _ in range(reps):
        x = case.link(x, sample)
    built = _now()
    mx.eval(x)
    return built - began, _now() - built


def _us(ns, reps):
    return ns / 1000.0 / reps


def choose_reps(case, target_ms, fixed=None):
    """Repetitions so one chain lasts about `target_ms`, bounded by memory."""
    cap = case.reps_max
    if case.bytes_per_rep:
        # a linearity chain holds 2R inputs and 2R outputs at once
        cap = max(8, min(cap, RETAINED_LIMIT // (2 * case.bytes_per_rep)))
    if fixed:
        return max(2, min(int(fixed), cap))
    probe = case.make_inputs(8)
    run_independent(case, probe)
    _, ns = run_independent(case, probe)
    per_call = max(ns / 8, 500.0)
    return int(max(8, min(cap, target_ms * 1e6 / per_call)))


def assert_finite(case, first, steps=8):
    """A dependent chain of `steps` must stay finite, or its timings describe NaN arithmetic."""
    if case.link is None:
        return
    x = first
    for _ in range(steps):
        out = case.op(x)
        x = case.link(first, out)
    check = case.keep(out)
    mx.eval(check)
    if not bool(mx.all(mx.isfinite(check.astype(mx.float32))).item()):
        raise FloatingPointError(f"{case.name}: dependent chain produced non-finite values")


def measure(case, target_ms=25.0, chains=5, fixed_reps=None, roundtrip_samples=30):
    """All timings of one case. Returns a JSON-ready dict.

    Raises when the case produces non-finite output in a short dependent chain.
    """
    warm = case.make_inputs(4)
    for x in warm:
        mx.eval(case.keep(case.op(x)))
    assert_finite(case, warm[0])
    reps = choose_reps(case, target_ms, fixed_reps)
    inputs = case.make_inputs(2 * reps)
    first = inputs[0]
    mx.eval(case.keep(case.op(first)))

    run_independent(case, inputs[:reps])      # clocks ramp during the first chain; it is not timed
    ind, ind_large = {"build": [], "eval": []}, []
    for _ in range(chains):
        build, ev = run_independent(case, inputs[:reps])
        ind["build"].append(_us(build, reps)); ind["eval"].append(_us(ev, reps))
    for _ in range(3):
        ind_large.append(run_independent(case, inputs)[1] / 1000.0)
    out = {"reps": reps, "chains": chains,
           "independent": {
               "eval_us": stat(ind["eval"]), "build_us": stat(ind["build"]),
               "total_us": stat([a + b for a, b in zip(ind["eval"], ind["build"])]),
               "linearity": linearity_check(min(ind["eval"]) * reps, reps, min(ind_large), 2 * reps)}}

    if case.link is not None:
        sample = case.keep(case.op(first))
        mx.eval(sample)
        mx.eval(case.link(first, case.op(first)))
        run_dependent(case, first, reps)
        dep = {"build": [], "eval": []}
        for _ in range(chains):
            build, ev = run_dependent(case, first, reps)
            dep["build"].append(_us(build, reps)); dep["eval"].append(_us(ev, reps))
        large = [run_dependent(case, first, 2 * reps)[1] / 1000.0 for _ in range(3)]
        link_eval, link_build = [], []
        if not case.link_free:
            op_out = case.op(first)
            mx.eval(case.keep(op_out))
            link_sample = op_out
            for _ in range(chains):
                build, ev = run_link_only(case, first, reps, link_sample)
                link_eval.append(_us(ev, reps)); link_build.append(_us(build, reps))
        link_med = statistics.median(link_eval) if link_eval else 0.0
        link_build_med = statistics.median(link_build) if link_build else 0.0
        net = [max(0.0, v - link_med) for v in dep["eval"]]
        out["dependent"] = {
            "gross_eval_us": stat(dep["eval"]), "gross_build_us": stat(dep["build"]),
            "link_eval_us": stat(link_eval) if link_eval else {"median": 0.0, "min": 0.0, "max": 0.0, "values": []},
            "net_eval_us": stat(net),
            "net_total_us": stat([max(0.0, e + b - link_med - link_build_med)
                                  for e, b in zip(dep["eval"], dep["build"])]),
            "link_is_free": case.link_free,
            "linearity": linearity_check(min(dep["eval"]) * reps, reps, min(large), 2 * reps)}

    trips = []
    for x in inputs[:min(roundtrip_samples, len(inputs))]:
        began = _now()
        mx.eval(case.keep(case.op(x)))
        trips.append((_now() - began) / 1000.0)
    out["roundtrip_us"] = stat(trips)
    out["roundtrip_note"] = ("one mx.eval per op, build included: submit and wait paid on every call; "
                             "not a per-op GPU cost and never to be summed")
    return out


# ----------------------------------------------------------------- launch floor

def floor_cases():
    """Trivial kernels: a 4-float metal copy and a 4-element built-in add."""
    kernel = mx.fast.metal_kernel(
        name="kernel_lab_copy4", input_names=["inp"], output_names=["out"],
        source="uint i = thread_position_in_grid.x; out[i] = inp[i];",
    )

    def copy(x):
        return kernel(inputs=[x], template=[("T", mx.float32)], grid=(4, 1, 1), threadgroup=(4, 1, 1),
                      output_shapes=[(4,)], output_dtypes=[mx.float32])[0]

    def inputs(n):
        xs = [mx.random.normal((4,)) for _ in range(n)]
        mx.eval(xs)
        return xs

    return [
        Case("metal_copy4", inputs, copy, link=lambda first, out: out, link_free=True),
        Case("builtin_add4", inputs, lambda x: x + 1.0, link=lambda first, out: out, link_free=True),
    ]


# ----------------------------------------------------------------- dense

def largest_divisor_up_to(n, limit):
    for d in range(min(n, limit), 0, -1):
        if n % d == 0:
            return d
    return 1


def synthetic_q4(out_width, in_width, seed=0, scale_dtype=None):
    """Q4 G32 affine weights of shape [out, in] built by `mx.quantize` from random matrices.

    Wide matrices are built from one divisor-sized block and tiled, so the
    float32 source never exceeds a few MB; identical rows do not change a
    bandwidth-bound kernel. Raises when the result would exceed the synthetic
    weight limit.
    """
    scale_dtype = scale_dtype or mx.bfloat16
    block = largest_divisor_up_to(out_width, 4096)
    key = mx.random.key(1000003 + seed)
    w, s, b = mx.quantize(mx.random.normal((block, in_width), key=key) * 0.02, group_size=GROUP, bits=4)
    s, b = s.astype(scale_dtype), b.astype(scale_dtype)
    repeats = out_width // block
    if repeats > 1:
        w, s, b = (mx.tile(a, (repeats, 1)) for a in (w, s, b))
    total = w.nbytes + s.nbytes + b.nbytes
    if total > SYNTHETIC_WEIGHT_LIMIT:
        raise MemoryError(f"synthetic weights of {total} bytes exceed {SYNTHETIC_WEIGHT_LIMIT}")
    mx.eval(w, s, b)
    return w, s, b


def link_add(first, out):
    """A shape-preserving one-kernel dependency: `first` plus a zero taken from `out`."""
    return first + out.reshape(-1)[:1] * 0


_LINK = None


def dense_link():
    global _LINK
    if _LINK is None:
        _LINK = mx.compile(link_add)
    return _LINK


def dense_case(entry, weights, rows, dtype):
    in_width, out_width = entry["in_width"], entry["out_width"]
    w, s, b = weights
    link = dense_link()

    def make(n):
        xs = [mx.random.normal((1, rows, in_width)).astype(dtype) for _ in range(n)]
        mx.eval(xs)
        return xs

    def op(x):
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GROUP, bits=4, mode="affine")

    item = mx.array(0, dtype).itemsize
    # float32 activations make MLX cast the bfloat16 scales and biases per call; the lazy graph holds those casts
    convert = 2 * out_width * (in_width // GROUP) * 4 if dtype == mx.float32 else 0
    return Case(f"dense {in_width}->{out_width} M={rows}", make, op, link=lambda first, out: link(first, out),
                bytes_per_rep=rows * (in_width + out_width) * item + convert)


def gather_case(entry, weights, dtype):
    """Layer-0 decode `gather_qmm` exactly as the corpus row shows it."""
    w, s, b = weights
    shape = entry["operand_shapes"][0]
    slots = entry["slots"]
    indices = mx.arange(slots, dtype=mx.uint32).reshape(entry["operand_shapes"][-1])
    link = dense_link()

    def make(n):
        xs = [mx.random.normal(tuple(shape)).astype(dtype) for _ in range(n)]
        mx.eval(xs)
        return xs

    def op(x):
        return mx.gather_qmm(x, w, s, b, rhs_indices=indices, transpose=True, group_size=GROUP,
                             bits=4, mode="affine", sorted_indices=False)

    return Case(f"gather {entry['in_width']}->{entry['out_width']}", make, op,
                link=lambda first, out: link(first, out), bytes_per_rep=math.prod(shape) * 2 * 2)


def synthetic_experts(experts, out_width, in_width, seed=0):
    w, s, b = mx.quantize(
        mx.random.normal((experts, out_width, in_width), key=mx.random.key(1000003 + seed)) * 0.02,
        group_size=GROUP, bits=4)
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    mx.eval(w, s, b)
    return w, s, b


def entry_id(entry):
    return f"{entry['kind']}_{entry['in_width']}x{entry['out_width']}"


def f32_skip_reason(entry):
    """float32 activations make MLX cast scales and biases per call; refuse when 2R of those copies would not fit."""
    convert = 2 * entry["out_width"] * (entry["in_width"] // GROUP) * 4
    if convert * 16 > RETAINED_LIMIT:
        return (f"per-call float32 copies of scales and biases are {convert / 2**20:.1f} MiB; "
                f"a chain of at least 8 repetitions would exceed the {RETAINED_LIMIT // 2**20} MiB bound")
    return None


DENSE_DTYPES = (("bfloat16", mx.bfloat16), ("float32", mx.float32))


def bench_dense(corpus, args, budget, results):
    """Every dense decode shape at M = 1, 2, 4, 8, then the layer-0 gathers at M=1.

    bfloat16 activations are what the code produces; float32 is measured as
    well, labelled, because metal_runtime's header comment claims float32.
    """
    dtypes = DENSE_DTYPES[:1] if args.no_dense_f32 else DENSE_DTYPES
    entries = [e for e in corpus["decode"] if e["kind"] == "dense"]
    entries.sort(key=lambda e: e["weight_bytes"])
    gathers = [e for e in corpus["decode"] if e["kind"] == "gather"]
    for entry in entries:
        ident = entry_id(entry)
        if budget.exhausted(f"dense {ident}"):
            continue
        weights = synthetic_q4(entry["out_width"], entry["in_width"], seed=entry["out_width"] % 97)
        record = {"entry": {k: v for k, v in entry.items() if k != "calls_by_arm"}, "activation_dtype": "bfloat16"}
        for name, dtype in dtypes:
            reason = f32_skip_reason(entry) if dtype == mx.float32 else None
            if reason:
                record["by_rows_float32_skipped"] = reason
                continue
            per_m = {}
            for rows in ROWS:
                if budget.exhausted(f"dense {ident} {name} M={rows}"):
                    break
                per_m[rows] = measure(dense_case(entry, weights, rows, dtype), args.ms, args.chains, args.reps)
            record["by_rows" if name == "bfloat16" else f"by_rows_{name}"] = per_m
        results[ident] = record
        print_dense_row(entry, record["by_rows"])
        args.thermal.sample(f"dense {ident}")
        del weights
        _release()
    for entry in gathers:
        ident = entry_id(entry)
        if budget.exhausted(ident):
            continue
        shape = entry["operand_shapes"][1]
        weights = synthetic_experts(shape[0], shape[1], shape[2] * 8, seed=shape[1])
        per_m = {1: measure(gather_case(entry, weights, mx.bfloat16), args.ms, args.chains, args.reps)}
        results[ident] = {"entry": {k: v for k, v in entry.items() if k != "calls_by_arm"},
                          "activation_dtype": "bfloat16", "by_rows": per_m,
                          "note": "layer-0 decode gather_qmm; measured at M=1 only"}
        print_dense_row(entry, per_m)
        del weights
        _release()


def annotate(results, floor):
    """Derived figures per measured case: weight GB/s and multiples of the launch floor.

    Dense and gather records get them per M; MoE records get GB/s over the
    distinct and the addressed bytes. Ratios use the matching mode of the
    trivial-kernel floor.
    """
    def floor_us(mode):
        return unit_us(floor, mode) if floor else None

    for ident, record in results.get("dense", {}).items():
        for key in [k for k in record if k.startswith("by_rows") and isinstance(record[k], dict)]:
            for rows, measured in record[key].items():
                derived = {}
                for mode in MODES:
                    us = unit_us(measured, mode)
                    derived[mode] = {"us": us, "gb_per_s": record["entry"]["weight_bytes"] / us / 1e3 if us else None,
                                     "x_floor": us / floor_us(mode) if floor_us(mode) else None}
                measured["derived"] = derived
    for key, record in results.get("moe", {}).items():
        if key.startswith("_") or "independent" not in record:
            continue
        derived = {}
        for mode in MODES:
            us = unit_us(record, mode)
            derived[mode] = {"us": us, "gb_per_s_distinct": record["bytes_distinct"] / us / 1e3,
                             "gb_per_s_addressed": record["bytes_addressed"] / us / 1e3,
                             "x_floor": us / floor_us(mode) if floor_us(mode) else None,
                             "x_floor_per_launch": us / record["launches_per_call"] / floor_us(mode)
                             if floor_us(mode) else None}
        record["derived"] = derived


def moe_scaling(results, variant="pool bf16 fused"):
    """Per-row cost and growth against M=1 for each routing case of one MoE variant."""
    mode, name, fused = variant.split()
    out = {}
    base = results.get("moe", {}).get(MoEConfig(mode, mx.bfloat16 if name == "bf16" else mx.float32, 1, "single",
                                              fused == "fused").label)
    if not base or "independent" not in base:
        return out
    for routing in ("same", "disjoint"):
        rows_out = {}
        for rows in ROWS:
            label = MoEConfig(mode, mx.bfloat16 if name == "bf16" else mx.float32, rows,
                              "single" if rows == 1 else routing, fused == "fused").label
            record = results["moe"].get(label)
            if not record or "independent" not in record:
                continue
            rows_out[rows] = {m: {"us": unit_us(record, m), "x_m1": unit_us(record, m) / unit_us(base, m),
                                  "us_per_row": unit_us(record, m) / rows} for m in MODES}
        out[routing] = rows_out
    return out


def linearity_summary(results):
    """Every linearity check in the tree: how many, and which fall outside tolerance."""
    checks = []

    def walk(node, path):
        if isinstance(node, dict):
            if "ratio" in node and "expected" in node and "ok" in node:
                checks.append({"path": "/".join(path), "ratio": node["ratio"], "ok": node["ok"],
                               "intercept_us": node["intercept"]})
                return
            for key, child in node.items():
                walk(child, path + [str(key)])

    walk(results, [])
    bad = [c for c in checks if not c["ok"]]
    return {"checks": len(checks), "outside_tolerance": len(bad), "tolerance": LINEARITY_TOLERANCE,
            "worst": sorted(bad, key=lambda c: abs(c["ratio"] - 2.0), reverse=True)[:12]}


CANARY_SHAPE = (2560, 10240)


def canary(args):
    """A bandwidth-bound dense matvec measured at fixed repetitions, to expose throttling drift.

    Measured at the start and again at the end of a run. Both figures are
    microseconds per call, so a higher end figure means the GPU ran slower
    late in the run, whatever the thermal level says.
    """
    entry_ = {"in_width": CANARY_SHAPE[0], "out_width": CANARY_SHAPE[1]}
    weights = synthetic_q4(CANARY_SHAPE[1], CANARY_SHAPE[0], seed=5)
    result = measure(dense_case(entry_, weights, 1, mx.bfloat16), args.ms, 7, 128, roundtrip_samples=3)
    value = result["independent"]["eval_us"]["median"]
    del weights
    _release()
    return value


def drift_wording(drift_pct, threshold=DRIFT_WARN_PCT):
    """The canary warning for a drift in microseconds per call (positive: slower at the end), or ''."""
    if drift_pct > threshold:
        return (f"   *** END OF RUN {drift_pct:.1f}% SLOWER THAN THE START: "
                "later numbers are throttle-suspect ***")
    if drift_pct < -threshold:
        return (f"   *** END OF RUN {-drift_pct:.1f}% FASTER THAN THE START: "
                "early numbers are suspect (clocks were still ramping or throttled) ***")
    return ""


def settle(thermal, baseline, max_wait):
    """Wait, up to `max_wait` seconds, for thermal pressure to fall back to `baseline`; returns seconds waited."""
    waited, began = 0.0, time.perf_counter()
    while waited < max_wait:
        level = thermal_level()
        if level is None or level <= (baseline or 0):
            break
        time.sleep(3.0)
        waited = time.perf_counter() - began
    return waited


def _release():
    gc.collect()
    clear = getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache", None)
    if clear:
        clear()


# ----------------------------------------------------------------- MoE

LAYOUT = None
POOL_HEADER = 4096
SLOT_BIT = 0x80000000
PROJECTIONS = {"gate_proj": (INTER, HIDDEN), "up_proj": (INTER, HIDDEN), "down_proj": (HIDDEN, INTER)}
_PROJECTION_SEED = {"gate_proj": 1, "up_proj": 2, "down_proj": 3}


def slab_layout():
    global LAYOUT
    if LAYOUT is None:
        from models.flashnext.slab_pack import get_slab_layout

        LAYOUT = get_slab_layout(GROUP)
    return LAYOUT


def expert_projection(expert, projection):
    """Deterministic synthetic Q4 G32 parts (weight u32, scales bf16, biases bf16) of one expert."""
    out_width, in_width = PROJECTIONS[projection]
    key = mx.random.key((1000003 * (expert + 1) + _PROJECTION_SEED[projection]) % (2**31))
    w, s, b = mx.quantize(mx.random.normal((out_width, in_width), key=key) * 0.02, group_size=GROUP, bits=4)
    s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
    mx.eval(w, s, b)        # eager, so the float32 source is freed per expert and peak memory stays bounded
    return w, s, b


class SyntheticPool:
    """A buffer in the expert pool's record layout, wrapped zero-copy as the pool does.

    Same construction as `ExpertPool`: an anonymous mmap viewed as uint32 and
    handed to MLX by DLPack, so the kernel reads the same kind of memory it
    reads in production. Record `e` lives at header + e * stride.
    """

    def __init__(self, experts):
        layout = slab_layout()
        self.experts, self.stride, self.layout = experts, layout.record_stride, layout
        length = -(-(POOL_HEADER + experts * self.stride) // mmap.PAGESIZE) * mmap.PAGESIZE
        self._mm = mmap.mmap(-1, length)
        self.array = np.frombuffer(self._mm, dtype=np.uint8)
        for expert in range(experts):
            self.write(expert)
        self.zero_copy = True
        try:
            self.buffer_mx = mx.from_dlpack(np.frombuffer(self._mm, dtype=np.uint32), copy=False)
        except Exception:
            self.zero_copy = False
            self.buffer_mx = mx.array(np.frombuffer(self._mm, dtype=np.uint32))
        mx.eval(self.buffer_mx)

    def write(self, expert):
        base = POOL_HEADER + expert * self.stride
        for projection in PROJECTIONS:
            w, s, b = expert_projection(expert, projection)
            for part, array in (("weight", w), ("scales", s), ("biases", b)):
                raw = array if part == "weight" else array.view(mx.uint16)
                data = np.asarray(raw).view(np.uint8).reshape(-1)
                offset = base + self.layout.offset(projection, part)
                self.array[offset:offset + data.size] = data

    def read(self, expert, projection, part):
        """Bytes of one part of one record, as a numpy uint8 array."""
        out_width, in_width = PROJECTIONS[projection]
        count = out_width * (in_width // 8) * 4 if part == "weight" else out_width * (in_width // GROUP) * 2
        offset = POOL_HEADER + expert * self.stride + self.layout.offset(projection, part)
        return np.array(self.array[offset:offset + count])

    def close(self):
        self.buffer_mx = None
        self.array = None
        try:
            self._mm.close()
        except BufferError:
            pass


def routing_for(rows, kind, top_k=TOP_K):
    """Expert ids per row. `same`: every row the first top_k experts. `disjoint`: top_k * rows distinct."""
    if kind in ("same", "single"):
        return [list(range(top_k)) for _ in range(rows)]
    if kind == "disjoint":
        return [list(range(r * top_k, (r + 1) * top_k)) for r in range(rows)]
    raise ValueError(f"unknown routing {kind!r}")


def streamed_stack(experts):
    """[experts, ...] stacked projections in `_one_pass` layout, built one part at a time to bound memory."""
    stacks = {}
    for projection in PROJECTIONS:
        parts = [[], [], []]
        for expert in experts:
            for index, array in enumerate(expert_projection(expert, projection)):
                parts[index].append(array)
        stacked = tuple(mx.stack(p) for p in parts)
        mx.eval(stacked)
        parts = None
        stacks[projection] = stacked
    return stacks


def dummy_projections():
    """What `_get_dummy_streamed_weights` hands the executor when the slab pack holds the experts."""
    zeros = {}
    for projection, (out_width, in_width) in PROJECTIONS.items():
        zeros[projection] = (mx.zeros((1, out_width, in_width // 8), dtype=mx.uint32),
                             mx.zeros((1, out_width, in_width // GROUP), dtype=mx.bfloat16),
                             mx.zeros((1, out_width, in_width // GROUP), dtype=mx.bfloat16))
    return zeros


def compare(actual, expected, rtol=RTOL, atol=ATOL, min_cosine=MIN_COSINE):
    """The unit tests' assert_mlx_close as a value: elementwise allclose plus cosine similarity.

    Adds a scale-free check, relative RMS error within `rtol`, so the atol
    cannot hide an error when outputs are small.
    """
    a = np.asarray(actual.astype(mx.float32))
    e = np.asarray(expected.astype(mx.float32))
    if a.shape != e.shape:
        return {"ok": False, "reason": f"shape {a.shape} != {e.shape}"}
    delta = np.abs(a - e)
    close = bool(np.all(delta <= atol + rtol * np.abs(e)))
    norm = float(np.linalg.norm(a) * np.linalg.norm(e))
    cosine = float(np.sum(a * e) / norm) if norm > 0 else 1.0
    rel_rms = float(np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-12))
    return {"ok": bool(close and cosine >= min_cosine and rel_rms <= rtol and np.isfinite(a).all()),
            "rel_rms": rel_rms,
            "max_abs": float(delta.max()) if delta.size else 0.0,
            "max_rel": float((delta / np.maximum(np.abs(e), 1e-6)).max()) if delta.size else 0.0,
            "cosine": cosine, "rtol": rtol, "atol": atol, "min_cosine": min_cosine}


def routed_projections(experts):
    """The routed experts of one row stacked in `[experts, ...]` layout, built from the generator."""
    packs = {}
    for projection in PROJECTIONS:
        parts = [expert_projection(e, projection) for e in experts]
        packs[projection] = tuple(mx.stack([p[i] for p in parts]) for i in range(3))
    return packs


def moe_reference(x, rows_experts, scores, shared_y):
    """The MLX reference the executor's unit tests use, one token row at a time.

    `gather_qmm` for gate and up, `swiglu`, `gather_qmm` for down, then
    `weighted_combine` and the shared add, as in test_metal_runtime. Experts
    are regenerated per row so no second copy of the pool lives. The
    executor's own dequantize reference (`_reference_all`) is checked in the
    unit tests; it rounds weights to bfloat16 and differs from the kernel by
    about 1% RMS, which would hide real errors.
    """
    from mlx_vlm.models.activations import swiglu
    from models.flashnext.metal_runtime import weighted_combine

    out = []
    for row, experts in enumerate(rows_experts):
        packs = routed_projections(experts)
        routes = mx.arange(len(experts), dtype=mx.uint32).reshape(1, -1)
        qmm = dict(rhs_indices=routes, transpose=True, group_size=GROUP, bits=4, mode="affine",
                   sorted_indices=False)
        xr = x[row:row + 1]
        gate = mx.gather_qmm(xr, *packs["gate_proj"], **qmm).squeeze(-2)
        up = mx.gather_qmm(xr, *packs["up_proj"], **qmm).squeeze(-2)
        down = mx.gather_qmm(mx.expand_dims(swiglu(gate, up), -2), *packs["down_proj"], **qmm).squeeze(-2)
        result = weighted_combine(down, routes, scores[row:row + 1]) + shared_y[row:row + 1]
        mx.eval(result)
        out.append(result)
        packs = None
    return mx.concatenate(out, axis=0)


class MoEConfig:
    """One executor configuration: how the experts are addressed, dtypes, rows, routing."""

    def __init__(self, mode, dtype, rows, routing, fused=True):
        self.mode, self.dtype, self.rows, self.routing, self.fused = mode, dtype, rows, routing, fused

    @property
    def label(self):
        name = "bf16" if self.dtype == mx.bfloat16 else "f32"
        return (f"{self.mode} {name} {'fused' if self.fused else 'unfused'} "
                f"M={self.rows} {self.routing if self.rows > 1 else 'single'}")


def build_moe_inputs(config, seed=7):
    """Constant per-configuration tensors: routes, scores, shared_y."""
    mx.random.seed(seed)
    experts = routing_for(config.rows, config.routing)
    scores_dtype = config.dtype
    scores = mx.softmax(mx.random.normal((config.rows, TOP_K)), axis=-1).astype(scores_dtype)
    shared = (mx.random.normal((config.rows, HIDDEN)) * 0.05).astype(config.dtype)
    mx.eval(scores, shared)
    return experts, scores, shared


def moe_call_factory(executor, config, experts, scores, shared, pool=None, stack=None):
    """Returns (call(x), routes) with routes encoded as `_pool_pass` or `_one_pass` encode them."""
    if config.mode == "pool":
        encoded = [[SLOT_BIT | e for e in row] for row in experts]
        weights, slab = dummy_projections(), pool.buffer_mx
    else:
        wanted = list(dict.fromkeys(e for row in experts for e in row))
        order = {e: i for i, e in enumerate(wanted)}
        encoded = [[order[e] for e in row] for row in experts]
        weights, slab = stack, None
    routes = mx.array(encoded, dtype=mx.uint32).reshape(config.rows, TOP_K)
    mx.eval(routes)

    def call(x):
        return executor.execute(x, routes, weights, scores=scores, slab_pack=slab, stream_pack=None,
                                shared_y=shared)

    return call, routes


def verify_moe(call, config, experts, scores, shared, seed=11):
    """Run the executor once and compare with the reference path. Returns the comparison dict."""
    mx.random.seed(seed)
    x = (mx.random.normal((config.rows, HIDDEN)) * X_STD).astype(config.dtype)
    mx.eval(x)
    actual = call(x)
    mx.eval(actual)
    expected = moe_reference(x, experts, scores, shared)
    # bfloat16 uses the tolerance of the production-output test, float32 that of the exact mixed-dtype tests.
    tolerance = (RTOL, ATOL) if config.dtype == mx.bfloat16 else (1e-3, 1e-4)
    result = compare(actual, expected, *tolerance)
    result["dtype_out"] = str(actual.dtype)
    return result


def launch_audit(call, config):
    """Metal launches of one executor call, as the production shape telemetry records them.

    Runs `call` under a `Trace(capture_shapes=True)`, the mechanism that
    produced the corpus, and returns its rows: the kernel names, operand
    shapes, options, grid and threadgroup. Gives the launch count per layer
    call, and lets the streamed configuration be compared with the corpus.
    """
    from veystrix.bench.telemetry import Trace

    x = (mx.random.normal((config.rows, HIDDEN)) * X_STD).astype(config.dtype)
    mx.eval(x)
    trace = Trace(capture_shapes=True)
    with trace.activate():
        mx.eval(call(x))
    rows = trace.shape_rows()
    return {"metal_launches": int(trace.counts["custom_metal_dispatches"]),
            "kernels": sorted(r["operation"] for r in rows), "rows": rows}


def audit_against_corpus(audit, corpus):
    """Do the streamed M=1 launch rows equal the corpus metal rows (shapes, options, grid)?"""
    frozen = lambda r: kernel_corpus._freeze([r["operation"], r["operand_shapes"], r["options"]])
    mine = sorted(frozen({"operation": r["operation"], "operand_shapes": r["operand_shapes"],
                          "options": r["options"]}) for r in audit["rows"])
    theirs = sorted(frozen(e) for e in corpus["decode"] if e["kind"] == "metal")
    return {"matches": mine == theirs, "lab_rows": len(mine), "corpus_rows": len(theirs)}


def moe_case(call, config):
    def make(n):
        xs = [(mx.random.normal((config.rows, HIDDEN)) * X_STD).astype(config.dtype) for _ in range(n)]
        mx.eval(xs)
        return xs

    item = mx.array(0, config.dtype).itemsize
    link = dense_link()
    return Case(config.label, make, call, link=lambda first, out: link(first, out),
                bytes_per_rep=2 * config.rows * HIDDEN * item)


def moe_configs(include_f32=True, include_unfused=True):
    """Ordered by priority: the production configuration first."""
    def block(mode, dtype, fused):
        configs = [MoEConfig(mode, dtype, 1, "single", fused)]
        for rows in ROWS[1:]:
            configs += [MoEConfig(mode, dtype, rows, "same", fused), MoEConfig(mode, dtype, rows, "disjoint", fused)]
        return configs

    out = block("pool", mx.bfloat16, True)
    if include_f32:
        out += block("pool", mx.float32, True)
    if include_unfused:
        out += block("pool", mx.bfloat16, False)
    out += block("streamed", mx.bfloat16, True)
    if include_f32:
        out += block("streamed", mx.float32, True)
    return out


def moe_record(config, call_measure, comparison, experts):
    distinct = len({e for row in experts for e in row})
    stride = slab_layout().record_stride
    record = dict(call_measure)
    record.update({
        "config": {"mode": config.mode, "dtype": "bfloat16" if config.dtype == mx.bfloat16 else "float32",
                   "rows": config.rows, "routing": config.routing, "fused_up_swiglu": config.fused},
        "verification": comparison, "bytes_distinct": distinct * stride,
        "bytes_addressed": config.rows * TOP_K * stride, "distinct_experts": distinct,
        "launches_per_call": LAUNCHES["moe_fused" if config.fused else "moe_unfused"],
        "per_row_us_hint": None,
    })
    record.pop("per_row_us_hint")
    return record


def bench_moe(args, budget, results):
    from models.flashnext.metal_runtime import MetalMoEExecutor

    probe = MetalMoEExecutor(TOP_K, HIDDEN, TOP_K, group_size=GROUP)
    if not probe.available:
        raise SystemExit(f"custom Metal is unavailable ({probe.fallback_reason}); the MoE section cannot run. "
                         "Run with --only floor,dense,gdn or on a GPU build of MLX.")
    configs = moe_configs(not args.no_f32, not args.no_unfused)
    executors = {}

    def executor_for(config, expert_count):
        key = (config.mode, config.fused)
        if key not in executors:
            executors[key] = MetalMoEExecutor(expert_count, HIDDEN, TOP_K, group_size=GROUP,
                                              fused_up_swiglu=config.fused)
        return executors[key]

    pool = None
    if any(c.mode == "pool" for c in configs):
        began = time.perf_counter()
        pool = SyntheticPool(80)
        results["_pool"] = {"experts": 80, "bytes": 80 * pool.stride, "header": POOL_HEADER,
                            "zero_copy_dlpack": pool.zero_copy, "build_s": time.perf_counter() - began}
        print(f"  synthetic pool: 80 records, {80 * pool.stride / 2**20:.0f} MiB, zero-copy={pool.zero_copy}, "
              f"built in {time.perf_counter() - began:.1f} s")
    try:
        for config in (c for c in configs if c.mode == "pool"):
            run_moe_config(config, executor_for(config, TOP_K), pool, None, args, budget, results)
        pool.close() if pool else None
        pool = None
        _release()
        by_shape = {}
        for config in (c for c in configs if c.mode == "streamed"):
            by_shape.setdefault((config.rows, config.routing), []).append(config)
        for (rows, routing), group in by_shape.items():
            experts = routing_for(rows, routing)
            wanted = list(dict.fromkeys(e for row in experts for e in row))
            if budget.exhausted(f"moe streamed M={rows} {routing}"):
                continue
            stack = streamed_stack(wanted)
            for config in group:
                run_moe_config(config, executor_for(config, max(len(wanted), TOP_K)), None, stack, args,
                               budget, results)
            stack = None
            _release()
    finally:
        if pool is not None:
            pool.close()


def run_moe_config(config, executor, pool, stack, args, budget, results):
    key = config.label
    if budget.exhausted(f"moe {key}"):
        return
    experts, scores, shared = build_moe_inputs(config)
    call, _ = moe_call_factory(executor, config, experts, scores, shared, pool, stack)
    comparison = verify_moe(call, config, experts, scores, shared)
    if not comparison["ok"]:
        results[key] = {"config": {"mode": config.mode, "rows": config.rows, "routing": config.routing},
                        "refused": "executor output does not match the reference path",
                        "verification": comparison}
        print(f"  REFUSED {key}: {comparison}")
        return
    audit = launch_audit(call, config)
    measured = measure(moe_case(call, config), args.ms, args.chains, args.reps)
    results[key] = moe_record(config, measured, comparison, experts)
    args.thermal.sample(f"moe {key}")
    results[key]["launch_audit"] = audit
    if config.mode == "streamed" and config.rows == 1 and config.fused and config.dtype == mx.bfloat16:
        results[key]["launch_audit"]["corpus_match"] = audit_against_corpus(audit, args.corpus_data)
    print_moe_row(key, results[key])


# ----------------------------------------------------------------- block MoE

def load_route_windows(path=WINDOWS_FIXTURE):
    """The recorded expert-overlap windows: ({M: [window, ...]}, metadata).

    A window is M rows of `BLOCK_TOP_K` relabelled expert ids (relabelled per
    window in order of first appearance). Raises ValueError for a fixture that
    is not schema 1, lacks an M, has a row of the wrong width, or whose
    `unique` disagrees with its ids.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported route-window schema {data.get('schema_version')!r}")
    windows, meta = {}, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "source": data.get("source"), "windows": {}, "trace_unique_per_pair": {},
                         "fixture_unique_per_pair": {}}
    for rows in BLOCK_M:
        entry = data["by_rows"].get(str(rows))
        if not entry or not entry.get("windows"):
            raise ValueError(f"{path}: no windows for M={rows}")
        listed = []
        for index, window in enumerate(entry["windows"]):
            routes = window["routes"]
            if len(routes) != rows or any(len(row) != BLOCK_TOP_K for row in routes):
                raise ValueError(f"{path}: M={rows} window {index} is not {rows} rows of {BLOCK_TOP_K} ids")
            if len({e for row in routes for e in row}) != window["unique"]:
                raise ValueError(f"{path}: M={rows} window {index} disagrees with its unique count")
            listed.append([list(row) for row in routes])
        windows[rows] = listed
        meta["windows"][rows] = len(listed)
        meta["trace_unique_per_pair"][rows] = entry.get("trace_unique_per_pair")
        meta["fixture_unique_per_pair"][rows] = entry.get("fixture_unique_per_pair")
    return windows, meta


def map_to_slots(routes, experts=POOL_EXPERTS):
    """Relabelled expert ids onto synthetic pool records: id i is record i. ValueError past the pool."""
    top = max(e for row in routes for e in row)
    if top >= experts or min(e for row in routes for e in row) < 0:
        raise ValueError(f"expert id {top} does not fit a pool of {experts} records")
    return [list(row) for row in routes]


def block_windows(rows, pattern, fixture):
    """Expert ids per row, for every window the pattern cycles through."""
    if pattern == "trace":
        return [map_to_slots(window) for window in fixture[rows]]
    if pattern in ("same", "disjoint"):
        return [routing_for(rows, pattern, BLOCK_TOP_K)]
    raise ValueError(f"unknown block routing {pattern!r}")


def unique_per_pair(windows):
    """Mean over windows of distinct experts per (row, expert) pair."""
    return statistics.fmean(len({e for row in w for e in row}) / (len(w) * len(w[0])) for w in windows)


def pool_routes(experts):
    """Raw pool routes of one window in pair order, as `_pool_pass` builds them."""
    return [SLOT_BIT | e for row in experts for e in row]


def plan_arrays(host_routes, rows):
    """The host plan and its two uint32 arrays: what production pays per layer-call."""
    from models.flashnext.metal_runtime import block_plan

    group_routes, group_pairs = block_plan(host_routes, rows)
    return mx.array(group_routes, dtype=mx.uint32), mx.array(group_pairs, dtype=mx.uint32)


def plan_cost(windows, rows, batches=5, rounds=8):
    """Microseconds per plan (`block_plan` plus two `mx.array`), batches of `rounds` passes over the windows."""
    hosts = [pool_routes(w) for w in windows]
    plan_arrays(hosts[0], rows)
    per_plan = []
    for _ in range(batches):
        began = _now()
        for _ in range(rounds):
            for host in hosts:
                plan_arrays(host, rows)
        per_plan.append((_now() - began) / 1000.0 / (rounds * len(hosts)))
    return stat(per_plan)


def bit_identical(a, b):
    """True when two arrays have the same dtype, shape and bytes."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    unsigned = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}.get(a.dtype)
    if unsigned is None:
        raise TypeError(f"bit comparison is not defined for {a.dtype}")
    mx.eval(a, b)
    return bool(np.array_equal(np.asarray(mx.view(a.reshape(-1), unsigned)),
                               np.asarray(mx.view(b.reshape(-1), unsigned))))


class BlockWindows:
    """The windows of one configuration with their routes and block plans, uploaded once."""

    def __init__(self, windows, rows_list=BLOCK_ROWS):
        self.windows = windows
        self.variants = []
        for experts in windows:
            host = pool_routes(experts)
            routes = mx.array(host, dtype=mx.uint32).reshape(len(experts), BLOCK_TOP_K)
            blocks = {rows: plan_arrays(host, rows) for rows in rows_list}
            mx.eval(routes, *[a for pair in blocks.values() for a in pair])
            self.variants.append((routes, blocks))

    def __len__(self):
        return len(self.variants)


def block_executor_call(executor, pool, scores, shared, routes, block):
    dummy = dummy_projections()
    return lambda x: executor.execute(x, routes, dummy, scores=scores, slab_pack=pool.buffer_mx,
                                      stream_pack=None, shared_y=shared, block=block)


def cycling_op(executor, pool, scores, shared, variants, block_rows):
    """An op that takes the next window on every call; the plan is built beforehand."""
    counter = itertools.count()
    dummy = dummy_projections()

    def op(x):
        routes, blocks = variants.variants[next(counter) % len(variants)]
        return executor.execute(x, routes, dummy, scores=scores, slab_pack=pool.buffer_mx, stream_pack=None,
                                shared_y=shared, block=blocks[block_rows] if block_rows else None)

    return op


def block_pair_check(executor, pool, scores, shared, variants, x):
    """Per block row count: is the block output bit-identical to the per-pair output on every window?"""
    identical = {rows: True for rows in BLOCK_ROWS}
    dummy = dummy_projections()
    for routes, blocks in variants.variants:
        def run(block):
            return executor.execute(x, routes, dummy, scores=scores, slab_pack=pool.buffer_mx,
                                    stream_pack=None, shared_y=shared, block=block)

        expected = run(None)
        for rows in BLOCK_ROWS:
            identical[rows] = identical[rows] and bit_identical(run(blocks[rows]), expected)
    return identical


def groups_mean(windows, rows):
    from models.flashnext.metal_runtime import block_plan

    return statistics.fmean(len(block_plan(pool_routes(w), rows)[0]) for w in windows)


def run_block_config(executor, pool, rows, pattern, fixture, args, results):
    """Per-pair against block rows 2, 4, 8 for one (M, routing); verified, bit-compared, timed."""
    key = f"M={rows} {pattern}"
    windows = block_windows(rows, pattern, fixture)
    variants = BlockWindows(windows)
    mx.random.seed(100 + rows)
    scores = mx.softmax(mx.random.normal((rows, BLOCK_TOP_K)), axis=-1).astype(mx.bfloat16)
    shared = (mx.random.normal((rows, HIDDEN)) * 0.05).astype(mx.bfloat16)
    x = (mx.random.normal((rows, HIDDEN)) * X_STD).astype(mx.bfloat16)
    mx.eval(scores, shared, x)
    config = MoEConfig("pool", mx.bfloat16, rows, pattern)
    pair_call = block_executor_call(executor, pool, scores, shared, variants.variants[0][0], None)
    comparison = compare(pair_call(x), moe_reference(x, windows[0], scores, shared), *BF16_TOLERANCE)
    record = {"config": {"mode": "pool", "dtype": "bfloat16", "rows": rows, "routing": pattern,
                         "top_k": BLOCK_TOP_K, "windows": len(windows), "fused_up_swiglu": True,
                         "group_size": GROUP,
                         "block_applies_to": "gate and up+swiglu; the fused down+combine kernel is unchanged at G32"},
              "verification": comparison, "unique_per_pair": unique_per_pair(windows)}
    if not comparison["ok"]:
        record["refused"] = "per-pair output does not match the reference path"
        results[key] = record
        args.failures.append(f"block {key}: per-pair output does not match the reference ({comparison})")
        print(f"  REFUSED {key}: {comparison}")
        return
    identical = block_pair_check(executor, pool, scores, shared, variants, x)
    record["per_pair"] = measure(moe_case(cycling_op(executor, pool, scores, shared, variants, None), config),
                                 args.ms, args.chains, BLOCK_REPS)
    record["block"] = {}
    for block_rows in BLOCK_ROWS:
        measured = measure(moe_case(cycling_op(executor, pool, scores, shared, variants, block_rows), config),
                           args.ms, args.chains, BLOCK_REPS)
        measured.update({"bit_identical": identical[block_rows], "groups_mean": groups_mean(windows, block_rows),
                         "plan_us": plan_cost(windows, block_rows)})
        record["block"][str(block_rows)] = measured
        if not identical[block_rows]:
            args.failures.append(f"block {key} rows {block_rows}: output is not bit-identical to per-pair")
    results[key] = record
    args.thermal.sample(f"block {key}")
    print_block_row(key, record)


def bench_block(args, budget, results):
    from models.flashnext.metal_runtime import MetalMoEExecutor

    executor = MetalMoEExecutor(BLOCK_TOP_K, HIDDEN, BLOCK_TOP_K, group_size=GROUP, fused_up_swiglu=True)
    if not executor.available:
        raise SystemExit(f"custom Metal is unavailable ({executor.fallback_reason}); the block section cannot run.")
    fixture, meta = load_route_windows(args.windows)
    results["_fixture"] = meta
    began = time.perf_counter()
    pool = SyntheticPool(POOL_EXPERTS)
    results["_pool"] = {"experts": POOL_EXPERTS, "bytes": POOL_EXPERTS * pool.stride, "zero_copy_dlpack": pool.zero_copy,
                        "build_s": time.perf_counter() - began}
    try:
        print_block_header()
        for rows in BLOCK_M:
            for pattern in BLOCK_PATTERNS:
                if budget.exhausted(f"block M={rows} {pattern}"):
                    continue
                run_block_config(executor, pool, rows, pattern, fixture, args, results)
    finally:
        pool.close()


# ----------------------------------------------------------------- GDN

def gdn_geometry(corpus):
    """Hv, Dv, Hk, Dk, layers, with sums checked against the corpus; raises ValueError when they disagree."""
    by_shape = {(e["in_width"], e["out_width"]): e for e in corpus["decode"] if e["kind"] == "dense"}
    need = [(HIDDEN, 10240), (HIDDEN, 6144), (HIDDEN, GDN_HV)]
    missing = [shape for shape in need if shape not in by_shape]
    if missing:
        raise ValueError(f"corpus lacks the GDN projections {missing}")
    if 6144 != GDN_HV * GDN_DV or 10240 != 2 * GDN_HK * GDN_DK + GDN_HV * GDN_DV:
        raise ValueError("GDN geometry does not reproduce the corpus projection widths")
    layers = by_shape[(HIDDEN, 6144)]["calls_per_step"]
    return {"Hk": GDN_HK, "Dk": GDN_DK, "Hv": GDN_HV, "Dv": GDN_DV, "layers_per_step": layers,
            "from_corpus": "Hv=48 (in_proj_a out), Dv=128 (in_proj_z 6144/48), Hk*Dk=2048 (in_proj_qkv 10240 - 6144)/2; "
                           "layers = in_proj_z calls per step",
            "not_in_corpus": "Hk/Dk split (16 x 128) is from checkpoint config.json"}


def gdn_case(rows):
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update, gated_delta_update_with_states

    a_log = (mx.random.normal((GDN_HV,)) * 0.5).astype(mx.bfloat16)
    dt_bias = (mx.random.normal((GDN_HV,)) * 0.5).astype(mx.bfloat16)
    mx.eval(a_log, dt_bias)

    def make(n):
        xs = []
        for _ in range(n):
            q = mx.random.normal((1, rows, GDN_HK, GDN_DK)).astype(mx.bfloat16) * 0.1
            k = mx.random.normal((1, rows, GDN_HK, GDN_DK)).astype(mx.bfloat16) * 0.1
            v = mx.random.normal((1, rows, GDN_HV, GDN_DV)).astype(mx.bfloat16)
            a = mx.random.normal((1, rows, GDN_HV)).astype(mx.bfloat16)
            b = mx.random.normal((1, rows, GDN_HV)).astype(mx.bfloat16)
            xs.append((q, k, v, a, b))
        mx.eval([t for x in xs for t in x])
        return xs

    state0 = mx.zeros((1, GDN_HV, GDN_DV, GDN_DK), dtype=mx.float32)
    mx.eval(state0)

    if rows == 1:
        def op(inp):
            return gated_delta_update(*inp[:5], a_log, dt_bias, inp[5] if len(inp) > 5 else state0)
        name = "gated_delta_update"
    else:
        def op(inp):
            y, state, _ = gated_delta_update_with_states(
                *inp[:5], a_log, dt_bias, inp[5] if len(inp) > 5 else state0, state_steps=rows - 1)
            return y, state
        name = "gated_delta_update_with_states"

    state_bytes = state0.nbytes
    inter = (rows - 1) * state_bytes if rows > 1 else 0
    case = Case(f"gdn {name} M={rows}", make, op,
                link=lambda first, out: (*first[:5], out[1]), link_free=True,
                keep=lambda out: out[0], bytes_per_rep=max(state_bytes + inter, 1) * 2, reps_max=256)
    case.function = name
    return case


def bench_gdn(corpus, args, budget, results):
    try:
        geometry = gdn_geometry(corpus)
        from mlx_vlm.models.qwen3_5 import gated_delta  # noqa: F401
    except (ValueError, ImportError) as error:
        results["_skipped"] = f"GDN not measured: {error}"
        print(f"  {results['_skipped']}")
        return
    results["_geometry"] = geometry
    for rows in ROWS:
        if budget.exhausted(f"gdn M={rows}"):
            continue
        case = gdn_case(rows)
        measured = measure(case, args.ms, args.chains, args.reps)
        measured["function"] = case.function
        results[f"M={rows}"] = measured
        gross = measured["dependent"]["net_eval_us"]["median"]
        print(f"  M={rows}  {case.function:34s} independent {measured['independent']['eval_us']['median']:8.1f} us  "
              f"dependent {gross:8.1f} us  round trip {measured['roundtrip_us']['median']:8.1f} us")
        _release()


# ----------------------------------------------------------------- token model

MODES = ("independent", "dependent")


def unit_us(measured, mode, metric="eval"):
    """Median microseconds of one measured case. Dependent figures are link-corrected."""
    if mode == "independent":
        return measured["independent"][f"{metric}_us"]["median"]
    return measured["dependent"][f"net_{metric}_us"]["median"]


def moe_layer_count(corpus):
    """MoE layer-calls per step: the common call count of the corpus' custom Metal kernels, or None."""
    metal = [e for e in corpus["decode"] if e["kind"] == "metal"]
    if not metal:
        return None
    counts = {e["operation"]: e["calls_per_step"] for e in metal}
    if len(set(counts.values())) != 1:
        raise ValueError(f"metal kernels are not called equally often per step: {counts}")
    return next(iter(counts.values()))


def token_model(corpus, results, rows_list=ROWS, metric="eval", moe_variant="pool bf16 fused"):
    """Modelled kernel milliseconds per decode step and per verification round.

    Sums `calls_per_step * microseconds` over the corpus by class, for each
    timing mode, and the same sums at the launch floor (every kernel free
    apart from its launch). MoE is priced per layer-call (the corpus counts
    each kernel 47 times per step, so 47 layer-calls); `moe_variant` is
    "<mode> <dtype> <fused|unfused>". Gather at layer 0 is measured at M=1
    only and is left out of the M>1 sums. Returns a JSON-ready dict.
    """
    dense = {k: v for k, v in results.get("dense", {}).items() if not k.startswith("gather_")}
    gather = {k: v for k, v in results.get("gather", {}).items()}
    gather.update({k: v for k, v in results.get("dense", {}).items() if k.startswith("gather_")})
    floor_runs = results.get("floor", {}).get("metal_copy4")
    moe = results.get("moe", {})
    gdn = results.get("gdn", {})
    moe_layers = moe_layer_count(corpus)
    gdn_layers = None
    if "_geometry" in gdn:
        gdn_layers = gdn["_geometry"]["layers_per_step"]

    def moe_key(rows, routing):
        name = moe_variant.split()
        label = MoEConfig(name[0], mx.bfloat16 if name[1] == "bf16" else mx.float32, rows, routing,
                          name[2] == "fused").label
        return moe.get(label)

    out = {"metric": metric, "moe_variant": moe_variant, "modes": {}, "omitted": [
        "norms", "elementwise glue and hyper-connection mixing", "attention and QSA indexer arithmetic",
        "router matmul (unquantized, not in the corpus)", "sampling", "n-gram embedding", "eval round trips",
        "GDN conv, norms and gating glue (only the recurrence is priced)"]}
    for mode in MODES:
        floor_us = unit_us(floor_runs, mode, metric) if floor_runs else None
        by_rows = {}
        for rows in rows_list:
            routings = ["single"] if rows == 1 else ["same", "disjoint"]
            variants = {}
            for routing in routings:
                classes, excluded = {}, []

                def add(name, calls, per_call_us, launches):
                    entry = classes.setdefault(name, {"calls": 0.0, "launches": 0.0, "us": 0.0})
                    entry["calls"] += calls
                    entry["launches"] += calls * launches
                    entry["us"] += calls * per_call_us

                complete = True
                for ident, record in dense.items():
                    measured = record["by_rows"].get(rows) or record["by_rows"].get(str(rows))
                    if measured is None:
                        complete = False
                        excluded.append(f"{ident} M={rows}")
                        continue
                    klass = "dense_tiny" if record["entry"]["out_width"] <= TINY_OUT else "dense_other"
                    add(klass, record["entry"]["calls_per_step"], unit_us(measured, mode, metric), LAUNCHES["dense"])
                for ident, record in gather.items():
                    measured = record["by_rows"].get(rows) or record["by_rows"].get(str(rows))
                    if measured is None:
                        excluded.append(f"{ident} (measured at M=1 only)")
                        continue
                    add("gather_layer0", record["entry"]["calls_per_step"], unit_us(measured, mode, metric),
                        LAUNCHES["gather"])
                moe_record_ = moe_key(rows, routing) if moe_layers else None
                if moe_layers and moe_record_ and "independent" in moe_record_:
                    fused = moe_variant.endswith("fused") and not moe_variant.endswith("unfused")
                    add("moe", moe_layers, unit_us(moe_record_, mode, metric),
                        LAUNCHES["moe_fused" if fused else "moe_unfused"])
                elif moe_layers:
                    complete = False
                    excluded.append(f"moe {moe_variant} M={rows} {routing}")
                if gdn_layers:
                    measured = gdn.get(f"M={rows}")
                    if measured:
                        add("gdn", gdn_layers, unit_us(measured, mode, metric), LAUNCHES["gdn"])
                    else:
                        excluded.append(f"gdn M={rows}")
                total_us = sum(c["us"] for c in classes.values())
                floor_total = sum(c["launches"] for c in classes.values()) * floor_us if floor_us else None
                for c in classes.values():
                    c["ms"] = c["us"] / 1000.0
                    c["floor_ms"] = c["launches"] * floor_us / 1000.0 if floor_us else None
                variants[routing] = {"classes": classes, "total_ms": total_us / 1000.0,
                                     "floor_only_ms": floor_total / 1000.0 if floor_total is not None else None,
                                     "launches": sum(c["launches"] for c in classes.values()),
                                     "excluded": excluded, "complete": complete}
            by_rows[rows] = variants
        out["modes"][mode] = {"floor_us_per_launch": floor_us, "by_rows": by_rows}
    return out


def block_rounds(corpus, results, model, block_rows=2, pattern="trace", rows_list=BLOCK_M):
    """One verification round priced with per-pair and with block MoE kernels, per timing mode.

    The non-MoE classes come from `model` (a `token_model` result, whose
    metric is used throughout); the MoE class is the corpus' layer-call count
    times the measured figure on `pattern`. The block figure adds the host
    plan cost per layer-call. `difference_*` is block minus per-pair, so a
    negative value is a saving.
    """
    layers = moe_layer_count(corpus)
    metric = model["metric"]
    out = {}
    if not layers:
        return out
    for mode, body in model["modes"].items():
        per_m = {}
        for rows in rows_list:
            record = results.get("block", {}).get(f"M={rows} {pattern}")
            variants = body["by_rows"].get(rows)
            if not record or "per_pair" not in record or str(block_rows) not in record["block"] or not variants:
                continue
            classes = next(iter(variants.values()))["classes"]
            other_ms = sum(c["ms"] for name, c in classes.items() if name != "moe")
            block = record["block"][str(block_rows)]
            pair_ms = layers * unit_us(record["per_pair"], mode, metric) / 1000.0
            kernel_ms = layers * unit_us(block, mode, metric) / 1000.0
            plan_ms = layers * block["plan_us"]["median"] / 1000.0
            round_pair, round_block = other_ms + pair_ms, other_ms + kernel_ms + plan_ms
            per_m[rows] = {"moe_layers": layers, "block_rows": block_rows, "pattern": pattern,
                           "other_classes_ms": other_ms, "moe_per_pair_ms": pair_ms, "moe_block_kernels_ms": kernel_ms,
                           "host_plan_ms": plan_ms, "round_per_pair_ms": round_pair, "round_block_ms": round_block,
                           "difference_ms": round_block - round_pair,
                           "difference_pct": (round_block / round_pair - 1.0) * 100.0}
        out[mode] = per_m
    return out


# ----------------------------------------------------------------- printing

def _fmt(value, width=8, digits=1):
    return f"{value:{width}.{digits}f}" if value is not None else " " * (width - 1) + "-"


def print_floor(name, m):
    print(f"  {name:14s} independent {m['independent']['eval_us']['median']:7.2f} us "
          f"(min {m['independent']['eval_us']['min']:6.2f}, build {m['independent']['build_us']['median']:6.2f})  "
          f"dependent {m['dependent']['net_eval_us']['median']:7.2f} us "
          f"(min {m['dependent']['net_eval_us']['min']:6.2f})  round trip {m['roundtrip_us']['median']:8.1f} us  "
          f"linearity {m['independent']['linearity']['ratio']:.3f}/{m['dependent']['linearity']['ratio']:.3f}")


def print_dense_header():
    head = " ".join(f"{m:>8d}" for m in ROWS)
    print(f"  {'shape (in->out)':20s} {'MB':>7s} |  independent us/call, M = {head}  |  dependent-net us/call, M = {head}")


def print_dense_row(entry, per_m):
    ind = " ".join(_fmt(per_m[m]["independent"]["eval_us"]["median"]) if m in per_m else _fmt(None) for m in ROWS)
    dep = " ".join(_fmt(per_m[m]["dependent"]["net_eval_us"]["median"]) if m in per_m else _fmt(None) for m in ROWS)
    label = f"{entry['in_width']}->{entry['out_width']}" + (" gather" if entry["kind"] == "gather" else "")
    pad = " " * 26
    print(f"  {label:20s} {entry['weight_bytes'] / 2**20:7.2f} | {pad}{ind}  | {pad}{dep}")


def print_dense_detail(results):
    """M=1 detail: weight bandwidth, multiples of the launch floor, round trip, and the float32 comparison."""
    print("\n  M=1 detail   (GB/s = weight bytes / chained time; x floor = multiples of one trivial-kernel launch;"
          " rt = one eval per op)")
    print(f"  {'shape':20s} {'calls/step':>10s} | {'ind us':>8s} {'GB/s':>7s} {'x floor':>8s} | "
          f"{'dep us':>8s} {'GB/s':>7s} {'x floor':>8s} | {'rt us':>8s} | f32/bf16 ind  dep")
    for ident, record in sorted(results.items(), key=lambda kv: kv[1]["entry"]["weight_bytes"]):
        m1 = record["by_rows"].get(1)
        if not m1 or "derived" not in m1:
            continue
        entry, d = record["entry"], m1["derived"]
        label = f"{entry['in_width']}->{entry['out_width']}" + (" gather" if entry["kind"] == "gather" else "")
        f32 = (record.get("by_rows_float32") or {}).get(1)
        ratio = (f"{unit_us(f32, 'independent') / unit_us(m1, 'independent'):5.2f} "
                 f"{unit_us(f32, 'dependent') / unit_us(m1, 'dependent'):5.2f}") if f32 else "    -     -"
        print(f"  {label:20s} {entry['calls_per_step']:>10g} | {d['independent']['us']:8.1f} "
              f"{d['independent']['gb_per_s']:7.1f} {_fmt(d['independent']['x_floor'], 8)} | "
              f"{d['dependent']['us']:8.1f} {d['dependent']['gb_per_s']:7.1f} {_fmt(d['dependent']['x_floor'], 8)} | "
              f"{m1['roundtrip_us']['median']:8.1f} |   {ratio}")


def print_moe_row(key, record):
    print(f"  {key:44s} independent {record['independent']['eval_us']['median']:8.1f} us  "
          f"dependent {record['dependent']['net_eval_us']['median']:8.1f} us  "
          f"round trip {record['roundtrip_us']['median']:8.1f} us  "
          f"{record['bytes_distinct'] / record['independent']['eval_us']['median'] / 1e3:6.1f} GB/s distinct")


def print_block_header():
    print(f"  {'M routing':14s} {'uniq/pair':>9s} | {'per-pair':>9s} {'block2':>8s} {'block4':>8s} {'block8':>8s} us/call "
          f"(independent, eval) | groups 2/4/8 | plan us 2/4/8 | bit-identical 2/4/8 | "
          f"dependent-net per-pair {'b2':>7s} {'b4':>7s} {'b8':>7s}")


def print_block_row(key, record):
    blocks = [record["block"][str(r)] for r in BLOCK_ROWS]
    ind = " ".join(f"{b['independent']['eval_us']['median']:8.1f}" for b in blocks)
    dep = " ".join(f"{b['dependent']['net_eval_us']['median']:7.1f}" for b in blocks)
    print(f"  {key:14s} {record['unique_per_pair']:9.3f} | {record['per_pair']['independent']['eval_us']['median']:9.1f} "
          f"{ind} | {'/'.join(f'{b['groups_mean']:.1f}' for b in blocks):>12s} | "
          f"{'/'.join(f'{b['plan_us']['median']:.0f}' for b in blocks):>13s} | "
          f"{'/'.join('yes' if b['bit_identical'] else 'NO' for b in blocks):>19s} | "
          f"{record['per_pair']['dependent']['net_eval_us']['median']:21.1f} {dep}")


def print_block_rounds(rounds, metric):
    if not rounds:
        return
    print(f"\n  Verification round, MoE per-pair versus block rows 2 on the recorded routes "
          f"({metric} metric; block includes the host plan cost; difference = block - per-pair)")
    for mode, per_m in rounds.items():
        for rows, r in per_m.items():
            print(f"    {mode:11s} M={rows}: per-pair {r['round_per_pair_ms']:7.2f} ms  block {r['round_block_ms']:7.2f} ms  "
                  f"difference {r['difference_ms']:+6.2f} ms ({r['difference_pct']:+5.1f}%)   "
                  f"[moe {r['moe_per_pair_ms']:.2f} -> {r['moe_block_kernels_ms']:.2f} + plan {r['host_plan_ms']:.2f}; "
                  f"other classes {r['other_classes_ms']:.2f}]")


def print_model(model):
    print("\nToken model (kernels in the corpus only; a lower bound on those kernels, not a token time)")
    print("  omitted: " + "; ".join(model["omitted"]))
    for mode, body in model["modes"].items():
        print(f"\n  timing mode: {mode}   launch floor {body['floor_us_per_launch']:.2f} us"
              if body["floor_us_per_launch"] else f"\n  timing mode: {mode}")
        for rows, variants in body["by_rows"].items():
            for routing, v in variants.items():
                parts = "  ".join(f"{name} {c['ms']:.2f}" for name, c in v["classes"].items())
                floor = f"{v['floor_only_ms']:.2f}" if v["floor_only_ms"] is not None else "-"
                tag = f"M={rows}" + ("" if routing == "single" else f" {routing}")
                print(f"    {tag:18s} total {v['total_ms']:7.2f} ms   floor-only {floor:>6s} ms   "
                      f"launches {v['launches']:6.0f}   [{parts}]")
                if v["excluded"] and rows == 2 and routing == "same":
                    print(f"      not in the M>1 sums: {'; '.join(v['excluded'])}")


# ----------------------------------------------------------------- main

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", type=Path, default=kernel_corpus.DEFAULT_FIXTURE,
                        help="committed shape corpus (default: the Stage 4 fixture)")
    parser.add_argument("--ms", type=float, default=25.0,
                        help="target wall time of one timed chain; repetitions are chosen from it")
    parser.add_argument("--only", action="append", choices=SECTIONS,
                        help="run only this section (repeatable)")
    parser.add_argument("--reps", type=int, default=None, help="fixed repetitions per chain (overrides --ms)")
    parser.add_argument("--json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--label", default="", help="free text stored in the artifact, e.g. the MLX build")
    parser.add_argument("--chains", type=int, default=5, help="timed chains per figure (at least 5)")
    parser.add_argument("--windows", type=Path, default=WINDOWS_FIXTURE,
                        help="recorded expert-overlap windows for the block section")
    parser.add_argument("--budget-s", type=float, default=480.0,
                        help="wall-time budget; remaining configurations are skipped and recorded")
    parser.add_argument("--no-f32", action="store_true", help="skip the float32-activation MoE variants")
    parser.add_argument("--no-unfused", action="store_true", help="skip the unfused up/swiglu MoE variant")
    parser.add_argument("--settle-s", type=float, default=45.0,
                        help="before the MoE and GDN sections, wait up to this long for thermal pressure to fall to its starting level")
    parser.add_argument("--no-dense-f32", action="store_true", help="skip the float32-activation dense variants")
    args = parser.parse_args(argv)
    if args.chains < 5:
        parser.error("--chains must be at least 5")
    if args.ms <= 0:
        parser.error("--ms must be positive")
    args.sections = tuple(args.only) if args.only else SECTIONS
    args.thermal = Thermal()
    args.failures = []
    return args


def run_section(name, work, budget, failures):
    """Run one section. An exception (or SystemExit) is recorded under `skipped` and in `failures`.

    Returns whether the section completed. The tables measured so far stay in
    the results, so a foreign MLX that breaks one section still yields an
    artifact.
    """
    try:
        work()
        return True
    except (Exception, SystemExit) as error:
        text = f"{type(error).__name__}: {error}"
        budget.skipped.append({"section": name, "exception": text})
        failures.append(f"section {name} raised {text}")
        print(f"  SECTION {name} FAILED: {text}")
        return False


def print_moe_scaling(scaling, variant):
    if not scaling:
        return
    print(f"\n  MoE scaling with rows ({variant}; x M=1 and microseconds per row)")
    print(f"  {'routing':10s} {'M':>3s} | {'independent us':>14s} {'x M=1':>7s} {'us/row':>8s} | "
          f"{'dependent us':>12s} {'x M=1':>7s} {'us/row':>8s}")
    shown_single = False
    for routing, rows in scaling.items():
        for m, v in rows.items():
            if m == 1 and shown_single:
                continue
            shown_single = shown_single or m == 1
            i, d = v["independent"], v["dependent"]
            print(f"  {routing if m > 1 else 'single':10s} {m:>3d} | {i['us']:14.1f} {i['x_m1']:7.2f} {i['us_per_row']:8.1f} | "
                  f"{d['us']:12.1f} {d['x_m1']:7.2f} {d['us_per_row']:8.1f}")


def main(argv=None) -> int:
    args = parse_args(argv)
    corpus = kernel_corpus.load(args.corpus)
    args.corpus_data = corpus
    for line in _live_process_warning():
        print(line)
    thermal = args.thermal = Thermal()
    thermal.sample("start")
    budget = Budget(args.budget_s)
    provenance = {"mlx": mlx_provenance(), "host": host_provenance(),
                  "corpus": {"path": str(args.corpus), "sha256": kernel_corpus.sha256_file(args.corpus),
                             "source_artifact": corpus["source"]["artifact"],
                             "decode_steps": corpus["decode_steps"]["steps"]},
                  "argv": list(argv if argv is not None else sys.argv[1:]), "label": args.label}
    print(f"kernel lab  mlx {mx.__version__}  {provenance['mlx']['mlx_core_file']}")
    print(f"label       {args.label or '(none)'}")
    print(f"host        {provenance['host']['chip']}  macOS {provenance['host']['macos']}  "
          f"{provenance['host']['power']['batt'].splitlines()[0] if provenance['host']['power']['batt'] else ''}"
          f"  low power mode {provenance['host']['power']['low_power_mode']}")
    print(f"thermal     pressure level {thermal.samples[0]['level']} at start"
          + ("   *** NON-ZERO AT START: numbers are throttling-suspect ***" if thermal.samples[0]["level"] else ""))
    print(f"corpus      {args.corpus}  ({len(corpus['decode'])} decode shapes, {corpus['decode_steps']['steps']} steps)")
    print(f"timing      ~{args.ms:g} ms per chain, {args.chains} chains per figure, headline = eval time "
          "(graph build excluded)\n")

    results = {"floor": {}, "dense": {}, "moe": {}, "block": {}, "gdn": {}}
    mx.random.seed(0)
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    timings = {}
    began = time.perf_counter()
    print("launch floor (trivial kernels, chained)"
          + ("" if "floor" in args.sections else "   [reference for the ratios below]"))
    for case in floor_cases():
        # The floor divides every ratio below, so it gets longer chains than the cases it normalises.
        results["floor"][case.name] = measure(case, 2 * args.ms, max(args.chains, 7), args.reps)
        print_floor(case.name, results["floor"][case.name])
    timings["floor"] = time.perf_counter() - began
    thermal.sample("after floor")
    canary_start = canary(args)
    waits = {}
    if "dense" in args.sections:
        print("\ndense quantized_matmul and layer-0 gather_qmm, every decode shape (bfloat16 activations)")
        print_dense_header()
        began = time.perf_counter()
        run_section("dense", lambda: bench_dense(corpus, args, budget, results["dense"]), budget, args.failures)
        timings["dense"] = time.perf_counter() - began
        thermal.sample("after dense")
    if "moe" in args.sections:
        waits["before moe"] = settle(thermal, thermal.samples[0]["level"], args.settle_s)
        thermal.sample("before moe, after settle")
        print("\ncustom MoE executor (per layer-call: gate, up+swiglu, down+combine+shared)")
        began = time.perf_counter()
        run_section("moe", lambda: bench_moe(args, budget, results["moe"]), budget, args.failures)
        timings["moe"] = time.perf_counter() - began
        thermal.sample("after moe")
    if "block" in args.sections:
        waits["before block"] = settle(thermal, thermal.samples[0]["level"], args.settle_s)
        thermal.sample("before block, after settle")
        print("\nblock MoE kernels against per-pair (top_k 8, bfloat16, pool path; block plan priced separately)")
        began = time.perf_counter()
        run_section("block", lambda: bench_block(args, budget, results["block"]), budget, args.failures)
        timings["block"] = time.perf_counter() - began
        thermal.sample("after block")
    if "gdn" in args.sections:
        waits["before gdn"] = settle(thermal, thermal.samples[0]["level"], args.settle_s)
        print("\nGatedDeltaNet recurrence (gated_delta_update; with_states for M > 1)")
        began = time.perf_counter()
        run_section("gdn", lambda: bench_gdn(corpus, args, budget, results["gdn"]), budget, args.failures)
        timings["gdn"] = time.perf_counter() - began
        thermal.sample("after gdn")
    thermal.sample("end")

    canary_end = canary(args)
    drift = (canary_end / canary_start - 1.0) * 100.0
    print(f"canary      dense 2560->10240 M=1: start {canary_start:.1f} us, end {canary_end:.1f} us, "
          f"drift {drift:+.1f}% (positive: slower at the end)" + drift_wording(drift))
    floor = results["floor"]["metal_copy4"]
    scaling, model = {}, None

    def summarise():
        nonlocal scaling, model
        annotate(results, floor)
        if results["dense"]:
            print_dense_detail(results["dense"])
        scaling = moe_scaling(results)
        print_moe_scaling(scaling, "pool bf16 fused")
        if results["dense"] or results["moe"] or results["gdn"]:
            model = token_model(corpus, results)
            print_model(model)
            for other in ("pool bf16 unfused", "streamed bf16 fused", "pool f32 fused"):
                if any(k.startswith(other) for k in results["moe"]):
                    model.setdefault("alternatives", {})[other] = token_model(corpus, results, moe_variant=other)
            model["total_metric_alternative"] = token_model(corpus, results, metric="total")
            if any(k.startswith("M=") for k in results["block"]):
                model["verification_rounds"] = block_rounds(corpus, results, model)
                model["total_metric_alternative"]["verification_rounds"] = block_rounds(
                    corpus, results, model["total_metric_alternative"])
                print_block_rounds(model["verification_rounds"], model["metric"])

    run_section("summary", summarise, budget, args.failures)
    linearity = linearity_summary(results)
    print(f"\nlinearity   {linearity['checks']} chain-scaling checks (R vs 2R), {linearity['outside_tolerance']} outside "
          f"+/-{LINEARITY_TOLERANCE:.0%}")
    for c in linearity["worst"][:5]:
        print(f"            ratio {c['ratio']:.3f}  {c['path']}")
    audits = [(k, r["launch_audit"]) for k, r in results["moe"].items()
              if isinstance(r, dict) and "launch_audit" in r and "corpus_match" in r["launch_audit"]]
    for key, audit in audits:
        print(f"launch audit {key}: {audit['metal_launches']} metal launches; rows match corpus metal rows: "
              f"{audit['corpus_match']['matches']}")
    summary = thermal.summary()
    print(f"\nthermal     start {summary['start']}  end {summary['end']}  max {summary['max']}"
          + (f"   *** THERMAL PRESSURE ROSE DURING THE RUN (first seen after: {summary['first_rise_at']}) ***"
             if summary["rose_during_run"] else "")
          + ("   *** NON-ZERO AT START ***" if summary["nonzero_at_start"] else ""))
    peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else None
    print(f"wall        {budget.elapsed():.0f} s   mlx peak memory {peak / 2**20:.0f} MiB" if peak else
          f"wall        {budget.elapsed():.0f} s")
    unmeasured = [item for item in budget.skipped if isinstance(item, str)]
    if unmeasured:
        print(f"skipped (wall budget {args.budget_s:.0f} s): {len(unmeasured)} configurations, first {unmeasured[:3]}")
    for failure in args.failures:
        print(f"FAILED      {failure}")

    stride = slab_layout().record_stride if "moe" in args.sections else None
    artifact = {"schema_version": SCHEMA_VERSION, "provenance": provenance, "thermal": summary,
                "wall_s": budget.elapsed(), "section_wall_s": timings, "settle_waits_s": waits,
                "canary": {"shape": list(CANARY_SHAPE), "start_us": canary_start, "end_us": canary_end, "drift_pct": drift,
                           "wording": drift_wording(drift).strip(" *")},
                "skipped": budget.skipped, "failures": args.failures,
                "memory": {"mlx_peak_bytes": peak, "synthetic_weight_limit_bytes": SYNTHETIC_WEIGHT_LIMIT,
                           "pool_bytes": 80 * stride if stride else None,
                           "note": "the pool (80 records) and a streamed stack (up to 80 experts) are never alive together; "
                                   "mlx peak includes reference-path and chain temporaries, and excludes the mmap pool"},
                "linearity": linearity, "linearity_tolerance": LINEARITY_TOLERANCE,
                "headline_metric": "eval_us: chain eval time per call, graph build excluded",
                "results": results, "moe_scaling": scaling, "model": model,
                "caveats": ["activation dtype is derived from code, not observed (see corpus dtypes)",
                            "launch floor includes host encode cost per launch",
                            "round-trip figures are one mx.eval per op and must not be summed",
                            "the model omits everything outside the corpus"]}
    if args.json:
        from veystrix.results import output_path

        path = output_path("flashnext", f"kernel-lab-{args.label}" if args.label else "kernel-lab", "kernel-lab.json")
        path.write_text(json.dumps(artifact, indent=1, default=str))
        print(f"artifact    {path}")
    return 1 if args.failures else 0


def _live_process_warning():
    out = _run(["pgrep", "-fl", "veystrix|flashnext|session.py"])
    mine = str(os.getpid())
    live = [line for line in out.splitlines() if line and line.split()[0] != mine and "bench_kernel_lab" not in line
            and "pgrep" not in line]
    return [f"WARNING: other processes match inference patterns: {line}" for line in live]


if __name__ == "__main__":
    raise SystemExit(main())
