#!/usr/bin/env python3
"""Opus E0/E1 — does decode I/O actually overlap with compute on this M4?

This is the decisive pair of experiments from Opus's miracle report
(docs/performance/OPUS_MIRACLE_RESPONSE.md). The "impossible on 16 GB" ceilings
assume decode time is ``compute + bytes/BW``. If I/O and GPU work truly overlap,
the floor is ``max(compute, bytes/BW)`` instead, which is far higher.

Our earlier "perfect predictor + overlap tied" result is ambiguous: it may be
byte-bound (total bytes unchanged, so total time unchanged) OR the I/O and
compute never ran at the same time. E0 and E1 separate those two explanations.

  E0 (concurrency proof): pin a fixed expert route so a decode step is
      compute-only (zero physical reads), then run a background thread doing
      cold F_NOCACHE reads at full speed. If a decode step inflates >25% or
      read bandwidth drops >20%, GPU and SSD contend and overlap is capped.
      Prediction: step inflates <10%, reads stay 3.0-3.7 GB/s.

  E1 (oracle pipeline): pass 1 records the REAL greedy route for every MoE
      call in execution order. Pass 2 replays those exact routes as prefetch
      hints 1-3 calls ahead (buffered pread fills the page cache) while the
      REAL router still decides -- so routing is untouched and output is
      bit-exact by construction. Measures tok/s and critical-path I/O wait.
      Prediction: Qwen >=5.5 tok/s, critical-path io_await <=30 ms.
      Kill: oracle below 4.5 tok/s means no predictor can save this path.

Bit-exact gate (both experiments): greedy decode, token IDs + SHA-256 of raw
logits at every step, per-layer residual checksums for the first 64 steps, and
a chaos mode that injects random 0-5 ms delays into I/O threads. Output must be
identical with and without the mechanism.

Run under the guard, e.g.:
  scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048 \
    --compressor-growth-mib 4096 --log results/.../guard.jsonl -- \
    .venv/bin/python -m models.flashnext.tests.bench.bench_overlap_oracle \
      --experiment e1 --tokens 1200 --warmup 100 --prompt chat

This measures time and bytes, not a chat reply. It writes one JSON result.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

from veystrix.checkpoints import resolve_flashnext
from models.flashnext.adaptive_topk import set_threshold
from models.flashnext.loader import load_streaming
from models.flashnext.diskio import disk_bytes_read, vm_counters, vm_delta
from veystrix.bench.qualification import token_digest
from veystrix.results import output_path

MODEL = str(resolve_flashnext())
PARTS = ("weight", "scales", "biases")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
MAIN_THREAD = threading.main_thread()

PROMPTS = {
    "chat": "Explique a fotossintese em duas frases.",
    "code": "Write a Python function that merges two sorted lists. Return only the code.",
    "agentic": "You are an agent. Plan the steps to book a flight to Paris, then list the tools you would call and why.",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha(values) -> str:
    """Canonical release-gate token digest (signed int32 little-endian)."""
    return token_digest(values)


def _logit_sha(logits) -> str:
    """Raw logit SHA-256, matching models/shared/logit_reference.py: bf16 row
    cast to little-endian float32, then hashed byte-for-byte."""
    row = np.asarray(logits.reshape(-1).astype(mx.float32), dtype="<f4")
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()


def _to_list(inds) -> list[int]:
    """Router indices (mx array or list) to a plain python int list."""
    try:
        return np.asarray(inds).reshape(-1).astype(np.int64).tolist()
    except Exception:
        return [int(v) for v in inds]


def _first_array(obj):
    """Best-effort first MLX array out of a layer return value."""
    if obj is None:
        return None
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = _first_array(item)
            if found is not None:
                return found
        return None
    for attr in ("hidden_states", "last_hidden_state", "output", "x"):
        if hasattr(obj, attr):
            found = _first_array(getattr(obj, attr))
            if found is not None:
                return found
    return None


def _array_sha(arr) -> str | None:
    if arr is None:
        return None
    try:
        if isinstance(arr, mx.array):
            arr = arr.astype(mx.float32)
        return hashlib.sha256(
            np.ascontiguousarray(np.asarray(arr, dtype="<f4")).tobytes()
        ).hexdigest()
    except Exception:
        return None


def _switch_blocks(language):
    return [
        layer.mlp.switch_mlp
        for layer in language.model.layers
        if getattr(layer.mlp, "switch_mlp", None) is not None
    ]


def _row_bytes(store, name: str) -> int:
    try:
        return int(store.empty_rows(name, 1).nbytes)
    except Exception:
        return 0


def new_state() -> dict:
    return {
        "token": 0, "exec": 0,
        "collect": False,        # record route + layer checksums
        "prefetch_on": False,    # issue oracle prefetch hints
        "layer_digests": {},
    }


def reset_state(state: dict) -> None:
    state["token"] = 0
    state["exec"] = 0
    state["layer_digests"] = {}


def new_stats() -> dict:
    return {
        "main_rows": 0, "main_bytes": 0, "main_secs": 0.0,
        "prefetch_rows": 0, "prefetch_bytes": 0, "prefetch_secs": 0.0,
    }


def zero_stats(stats: dict) -> None:
    for key in stats:
        stats[key] = 0


# ---------------------------------------------------------------------------
# I/O wrapper: critical-path vs background accounting + chaos delays
# ---------------------------------------------------------------------------
def wrap_rows_np(store, stats: dict, chaos: dict):
    """Time store.rows_np on the main thread (critical path) vs workers.

    A prefetch issued from a pool thread counts as background; the model's own
    gather on the main thread counts as critical path. Chaos delays are injected
    only into non-main (I/O) threads.
    """
    original = store.rows_np
    lock = threading.Lock()

    def wrapped(name, indices, read_mode=None):
        if chaos["enabled"] and threading.current_thread() is not MAIN_THREAD:
            time.sleep(random.uniform(0.0, chaos["max_ms"] / 1000.0))
        started = time.perf_counter()
        out = original(name, indices, read_mode)
        dt = time.perf_counter() - started
        n = len(indices)
        nb = n * _row_bytes(store, name)
        is_main = threading.current_thread() is MAIN_THREAD
        with lock:
            bucket = "main" if is_main else "prefetch"
            stats[bucket + "_rows"] += n
            stats[bucket + "_bytes"] += nb
            stats[bucket + "_secs"] += dt
        return out

    store.rows_np = wrapped
    return original


def _prefetch_rows(store, prefix, experts):
    """Buffered pread of one expert set; fills the page cache, changes nothing."""
    for projection in PROJECTIONS:
        for part in PARTS:
            try:
                store.rows_np(f"{prefix}.{projection}.{part}", experts, "pread")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Forward patches (installed ONCE; behaviour switched by state flags)
# ---------------------------------------------------------------------------
def install_switch_patch(language, state: dict, cfg: dict) -> None:
    """Patch SwitchMLP.__call__ once. cfg may be mutated between passes.

    cfg keys (all optional):
      fixed_route: {layer_id: [expert, ...]}  -> force a fixed route (E0)
      route_log:   list                        -> append (exec_index, layer, experts)
      prefetch:    dict(routes, depth, width, pool, store, prefixes)
    """
    blocks = _switch_blocks(language)
    if not blocks:
        raise RuntimeError("no SwitchMLP blocks found")
    cls = type(blocks[0])
    for block in blocks:
        if type(block) is not cls:
            raise RuntimeError("switch_mlp classes differ across layers")
    original = cls.__call__
    order = [b.layer_id for b in blocks]
    prefixes = {b.layer_id: b.gate_proj.cache.prefix.rsplit(".", 1)[0]
                for b in blocks}
    first_layer = min(order)

    def issue_prefetch():
        prefetch = cfg.get("prefetch")
        if not prefetch:
            return
        recorded = prefetch["routes"]
        for delta in range(1, prefetch["depth"] + 1):
            k = state["exec"] + delta - 1
            entry = recorded.get(k)
            if not entry:
                continue
            future_layer, experts = entry
            pool = prefetch["pool"]
            pool.submit(_prefetch_rows, prefetch["store"],
                        prefixes[future_layer], experts[:prefetch["width"]])

    def call(self, x, inds, *args, **kwargs):
        # The one-sync path caches the original route on this object. A forced
        # or recorded route must not consume that stale host list.
        self._routed_host = None
        layer = self.layer_id
        if layer == first_layer:
            state["token"] += 1
        state["exec"] += 1
        real = _to_list(inds)
        if state["collect"] and cfg.get("route_log") is not None:
            cfg["route_log"].append((state["exec"] - 1, layer, real))
        if state["prefetch_on"]:
            issue_prefetch()
        fixed = cfg.get("fixed_route")
        if fixed is not None:
            width = inds.shape[-1]
            vals = list(fixed[layer])
            # Repeat from the pinned pool to fill every top-k slot. Prefill has
            # a token axis (shape ...,width); decode is a single row. A 1-D
            # array of `width` indices broadcasts to either. Duplicate rows are
            # summed bit-exactly, as in bench_read_ceiling.draw().
            if len(vals) < width:
                vals = (vals * (width // len(vals) + 1))[:width]
            inds = mx.broadcast_to(
                mx.array(vals[:width], dtype=inds.dtype), inds.shape
            )
        return original(self, x, inds, *args, **kwargs)

    cls.__call__ = call


def install_layer_patch(language, state: dict) -> None:
    """Hash each transformer layer's output for the first 64 decode steps."""
    layers = language.model.layers
    if not layers:
        return
    cls = type(layers[0])
    index_by_id = {id(layer): i for i, layer in enumerate(layers)}
    original = cls.__call__

    def call(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        idx = index_by_id.get(id(self))
        if state["collect"] and 0 <= state["token"] < 64 and idx is not None:
            digest = _array_sha(_first_array(out))
            if digest is not None:
                state["layer_digests"].setdefault(idx, []).append(
                    (state["token"], digest)
                )
        return out

    cls.__call__ = call


# ---------------------------------------------------------------------------
# decode passes + gate
# ---------------------------------------------------------------------------
def run_decode(language, token, cache, *, tokens, state, stats, chaos,
               collect=True):
    """Run a fixed-length greedy decode. Returns digests + timing + bytes."""
    vm_before = vm_counters()
    bytes_before = disk_bytes_read()
    produced: list[int] = []
    logits_digests: list[str] = []
    started = time.perf_counter()
    for _ in range(tokens):
        logits = language(token[None], cache=cache).logits
        if collect:
            logits_digests.append(_logit_sha(logits[0, -1, :]))
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        produced.append(int(token.item()))
    elapsed = time.perf_counter() - started
    physical = disk_bytes_read() - bytes_before
    vm_after = vm_counters()
    layer = sorted(state["layer_digests"].items())
    return {
        "tokens": produced,
        "token_sha256": _sha(produced),
        "logits_digests": logits_digests,
        "layer_digests": {str(idx): vals for idx, vals in layer},
        "elapsed_s": elapsed,
        "tok_s": tokens / elapsed,
        "physical_bytes": physical,
        "vm": vm_delta(vm_before, vm_after, tokens),
        "stats": dict(stats),
    }


def gate(reference: dict, candidate: dict) -> dict:
    """Bit-exact gate: token IDs + per-step logits SHA-256 + layer checksums."""
    mismatches = []
    if reference["tokens"] != candidate["tokens"]:
        first = next(
            (i for i, (a, b) in enumerate(zip(reference["tokens"],
                                               candidate["tokens"]))
             if a != b), min(len(reference["tokens"]), len(candidate["tokens"]))
        )
        mismatches.append(f"token divergence at step {first}")
    for i, (a, b) in enumerate(zip(reference["logits_digests"],
                                    candidate["logits_digests"])):
        if a != b:
            mismatches.append(f"logit digest divergence at step {i}")
            break
    for layer_id, vals in reference["layer_digests"].items():
        other = candidate["layer_digests"].get(layer_id, [])
        if vals != other:
            mismatches.append(f"layer {layer_id} residual checksum divergence")
            break
    return {"pass": not mismatches, "mismatches": mismatches[:20]}


def fresh_cache(language):
    from mlx_vlm.models.cache import make_prompt_cache
    return make_prompt_cache(language)


def prefill(language, tokenizer, prompt, cache, state):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="np")["input_ids"]
    token = mx.array(ids)
    logits = language(token, cache=cache).logits
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    return token


def _largest_shard(store) -> str:
    shards = {ref.shard for ref in store.refs.values()}
    return max(shards, key=lambda s: os.path.getsize(os.path.join(store.dir, s)))


# ---------------------------------------------------------------------------
# E0 — concurrency proof
# ---------------------------------------------------------------------------
def run_e0(args, language, tokenizer, store):
    blocks = _switch_blocks(language)
    per_layer = {b.layer_id: list(range(args.experts)) for b in blocks}
    for block in blocks:
        prefix = block.gate_proj.cache.prefix.rsplit(".", 1)[0]
        for projection in PROJECTIONS:
            for part in PARTS:
                store.pin_rows(f"{prefix}.{projection}.{part}",
                               list(range(args.experts)))

    state = new_state()
    stats = new_stats()
    chaos = {"enabled": args.chaos, "max_ms": 5.0}
    original_rows = wrap_rows_np(store, stats, chaos)
    cfg = {"fixed_route": per_layer}
    install_switch_patch(language, state, cfg)
    install_layer_patch(language, state)
    prompt = PROMPTS[args.prompt]

    def sweep():
        reset_state(state)
        cache = fresh_cache(language)
        token = prefill(language, tokenizer, prompt, cache, state)
        reset_state(state)  # prefill incremented token/exec; decode starts clean
        zero_stats(stats)   # exclude prefill reads from the critical-path metric
        state["collect"] = True
        result = run_decode(language, token, cache, tokens=args.tokens,
                            state=state, stats=stats, chaos=chaos)
        state["collect"] = False
        return result

    base = sweep()  # no background reader

    # The M4 Air is fanless: let it cool between the two loaded sweeps so the
    # guard's thermal stop does not truncate the measurement.
    if args.cooldown:
        time.sleep(args.cooldown)

    # Realistic cold reader: F_NOCACHE, small reusable buffers (no per-read
    # allocation), scanning a region larger than the SSD's DRAM cache. A prior
    # version cycled a 2 GB region with 64 MB Python bytes allocations and hit
    # 24.79 GB/s -- that is a memcpy storm, not SSD I/O. Physical bytes come
    # from disk_bytes_read (proc_pid_rusage), not from bytes requested.
    shards = sorted({ref.shard for ref in store.refs.values()})
    fds = []
    for shard in shards:
        fd = os.open(os.path.join(store.dir, shard), os.O_RDONLY)
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        fds.append(fd)
    sizes = [os.fstat(fd).st_size for fd in fds]
    total_bytes = sum(sizes)
    chunk = args.read_chunk_mib << 20
    stop = threading.Event()
    counter = [0]
    lock = threading.Lock()

    def reader(tid):
        # Each worker scans its own stride over the whole checkpoint so the
        # working set is far larger than any SSD DRAM cache.
        buf = bytearray(chunk)
        view = memoryview(buf)
        fd_idx = tid % len(fds)
        off = (tid * chunk) % max(1, sizes[fd_idx])
        while not stop.is_set():
            try:
                n = os.preadv(fds[fd_idx], [view], off)
            except OSError:
                break  # fd closed during guard teardown
            if not n:
                fd_idx = (fd_idx + 1) % len(fds)
                off = 0
                continue
            with lock:
                counter[0] += n
            off = (off + n) % max(1, sizes[fd_idx])

    threads = [threading.Thread(target=reader, args=(i,), daemon=True)
               for i in range(args.io_threads)]
    for t in threads:
        t.start()
    time.sleep(0.2)  # let the queue fill before timing
    counter[0] = 0   # ignore the warm-up reads
    physical_start = disk_bytes_read()
    t0 = time.perf_counter()
    contention = sweep()
    read_dt = time.perf_counter() - t0
    physical_read = max(0, disk_bytes_read() - physical_start)
    stop.set()
    for t in threads:
        t.join(timeout=2)
    for fd in fds:
        os.close(fd)

    # Use PHYSICAL bytes: this is the real SSD bandwidth under contention.
    bandwidth = physical_read / read_dt / 1e9 if read_dt else 0.0
    inflation = ((contention["elapsed_s"] - base["elapsed_s"])
                 / base["elapsed_s"] * 100 if base["elapsed_s"] else 0.0)
    g = gate(base, contention)
    verdict = "KILL" if (inflation > 25 or bandwidth < 2.4) else "KEEP"
    store.unpin_all()
    store.rows_np = original_rows
    return {
        "experiment": "E0", "verdict_hint": verdict, "gate": g,
        "baseline_ms_per_token": base["elapsed_s"] / args.tokens * 1000,
        "contention_ms_per_token": contention["elapsed_s"] / args.tokens * 1000,
        "step_inflation_pct": round(inflation, 2),
        "read_bandwidth_gbs_physical": round(bandwidth, 2),
        "physical_read_mb_during_sweep": round(physical_read / 1e6, 1),
        "requested_read_mb_during_sweep": round(counter[0] / 1e6, 1),
        "checkpoint_total_gb": round(total_bytes / 1e9, 1),
        "read_chunk_mib": args.read_chunk_mib,
        "model_physical_mb_per_token": base["physical_bytes"] / args.tokens / 1e6,
        "baseline_tok_s": base["tok_s"], "contention_tok_s": contention["tok_s"],
        "critical_path_main_io_ms_per_token": (
            contention["stats"]["main_secs"] / args.tokens * 1000),
        "chaos": args.chaos, "io_threads": args.io_threads,
        "experts": args.experts, "tokens": args.tokens, "prompt": args.prompt,
        "gate_baseline_sha": base["token_sha256"],
        "gate_contention_sha": contention["token_sha256"],
    }


# ---------------------------------------------------------------------------
# E1 — oracle pipeline
# ---------------------------------------------------------------------------
def run_e1(args, language, tokenizer, store):
    if os.environ.get("FLASHNEXT_UNCACHED_EXPERT_READS") == "1":
        raise SystemExit("E1 requires buffered reads; unset FLASHNEXT_UNCACHED_EXPERT_READS")
    store.set_uncached_reads(False)

    state = new_state()
    stats = new_stats()
    chaos = {"enabled": args.chaos, "max_ms": 5.0}
    original_rows = wrap_rows_np(store, stats, chaos)
    prompt = PROMPTS[args.prompt]

    blocks = _switch_blocks(language)
    prefixes = {b.layer_id: b.gate_proj.cache.prefix.rsplit(".", 1)[0]
                for b in blocks}
    route_log: list[tuple[int, int, list[int]]] = []
    cfg = {"route_log": route_log}
    install_switch_patch(language, state, cfg)
    install_layer_patch(language, state)

    # pass 1: reference + record the real route in execution order.
    cache = fresh_cache(language)
    token = prefill(language, tokenizer, prompt, cache, state)
    reset_state(state)
    zero_stats(stats)  # exclude prefill reads from the critical-path metric
    state["collect"] = True
    reference = run_decode(language, token, cache, tokens=args.tokens,
                           state=state, stats=stats, chaos=chaos)
    state["collect"] = False
    recorded = {idx: (layer, experts) for idx, layer, experts in route_log}

    # The M4 Air is fanless: cool between the two loaded decode passes.
    if args.cooldown:
        time.sleep(args.cooldown)

    # pass 2: replay as prefetch hints; real router still decides.
    with ThreadPoolExecutor(max_workers=args.io_threads,
                            thread_name_prefix="flashnext-oracle") as pool:
        cfg["route_log"] = None  # pass 1 routes already captured in `recorded`
        cfg["prefetch"] = {
            "routes": recorded, "depth": args.ahead_depth,
            "width": args.ahead_width, "pool": pool,
            "store": store, "prefixes": prefixes,
        }
        cache = fresh_cache(language)
        token = prefill(language, tokenizer, prompt, cache, state)
        reset_state(state)
        zero_stats(stats)
        state["collect"] = True
        state["prefetch_on"] = True
        candidate = run_decode(language, token, cache, tokens=args.tokens,
                               state=state, stats=stats, chaos=chaos)
        state["collect"] = False
        state["prefetch_on"] = False
        cfg["prefetch"] = None

    g = gate(reference, candidate)
    verdict = "KEEP" if (g["pass"] and candidate["tok_s"] >= 4.5) else "KILL"
    store.rows_np = original_rows
    return {
        "experiment": "E1", "verdict_hint": verdict, "gate": g,
        "reference_tok_s": reference["tok_s"],
        "oracle_tok_s": candidate["tok_s"],
        "reference_ms_per_token": reference["elapsed_s"] / args.tokens * 1000,
        "oracle_ms_per_token": candidate["elapsed_s"] / args.tokens * 1000,
        "critical_path_main_io_ms_per_token": (
            candidate["stats"]["main_secs"] / args.tokens * 1000),
        "reference_main_io_ms_per_token": (
            reference["stats"]["main_secs"] / args.tokens * 1000),
        "physical_mb_per_token_ref": reference["physical_bytes"] / args.tokens / 1e6,
        "physical_mb_per_token_oracle": candidate["physical_bytes"] / args.tokens / 1e6,
        "prefetch_rows_issued": candidate["stats"]["prefetch_rows"],
        "prefetch_mb_issued": candidate["stats"]["prefetch_bytes"] / 1e6,
        "warmup": args.warmup, "tokens": args.tokens, "prompt": args.prompt,
        "ahead_depth": args.ahead_depth, "ahead_width": args.ahead_width,
        "chaos": args.chaos,
        "gate_reference_sha": reference["token_sha256"],
        "gate_candidate_sha": candidate["token_sha256"],
        "vm_reference": reference["vm"], "vm_oracle": candidate["vm"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e0", "e1"), required=True)
    parser.add_argument("--prompt", choices=tuple(PROMPTS), default="chat")
    parser.add_argument("--tokens", type=int, default=1200)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--experts", type=int, default=4,
                        help="E0: routed experts per layer (fixed route, pinned)")
    parser.add_argument("--ahead-depth", type=int, default=3,
                        help="E1: how many future MoE calls to prefetch")
    parser.add_argument("--ahead-width", type=int, default=10,
                        help="E1: experts to prefetch per future call")
    parser.add_argument("--io-threads", type=int, default=8)
    parser.add_argument("--read-chunk-mib", type=int, default=4,
                        help="E0 reader chunk size (small, reusable buffer)")
    parser.add_argument("--cooldown", type=int, default=20,
                        help="seconds to cool between the two E0 sweeps (fanless Air)")
    parser.add_argument("--chaos", action="store_true",
                        help="inject random 0-5 ms delays into I/O threads")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    os.environ.setdefault("FLASHNEXT_TOPK_THRESHOLD", "1.0")
    set_threshold(1.0)  # route all ten slots so the oracle width matches exactly
    model, _, _ = load_streaming(MODEL, verbose=False, keep_vision=False,
                                 use_mtp=False)
    language = model.language_model
    store = language.model.layers[0].mlp.switch_mlp.gate_proj.cache.store
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    if args.experiment == "e0":
        result = run_e0(args, language, tokenizer, store)
    else:
        result = run_e1(args, language, tokenizer, store)

    target = output_path("flashnext", "overlap-oracle", f"{args.experiment}.json",
                         args.json)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {target}", flush=True)


if __name__ == "__main__":
    main()
