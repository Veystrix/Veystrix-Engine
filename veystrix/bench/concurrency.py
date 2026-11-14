"""Stage 10 in-process concurrency harness for the cooperative (decode-first) scheduler.

No HTTP and one backend per process. Arms run one after another with ``backend.reset()`` between them:

  dedicated       one request through the backend's normal ``generate`` (what ``qualification.once`` runs)
  cooperative-1   the same request through ``backend.fork_for_batch()``, ``cooperative.generate`` and a
                  ``BatchScheduler`` whose ``serve_jobs`` loop runs on the main thread
  cooperative-2/4 two or four simultaneous requests with different prompts, started together

MLX work stays on the main thread: request threads only call ``scheduler.reserve / submit_quantum / submit_decode /
release`` (as ``server.py`` does); ``serve_jobs`` executes every forward here.

What one arm records, per stream: token digest, whether that digest equals the one the same prompt produced alone
(``digests_match_single``), TTFT, stream tok/s ((n - 1) / (t_last - t_first) from the token-callback timestamps) and the
inter-token gap p50/p95/max. Per arm: aggregate tok/s (all tokens of the arm over the wall time from the arm's first token
to its last), physical MB read per emitted token (``ReadMeter``: bytes this process read between the arm's first and last
emitted token, divided by the tokens emitted after the first; it includes any prefill of other streams that overlaps the
interval), the scheduler's ``decode_groups`` / ``max_decode_group``, the memory floor and swap growth. An arm is cancelled
through ``should_cancel`` when the sampled available memory is below 3 GiB and the evidence says so; the run then ends.

TTFT is measured from the instant the arm releases its request(s), so for several streams it includes waiting behind the
other streams' work. Run it under ``scripts/flashnext/real_run_guard.py``. Evidence is rewritten after every arm into
``results/<model>/<stamp>-stage10-concurrency-manual/concurrency.json``.

Exit status: 0 clean; 1 stopped (memory floor, interrupt, error); 2 usage or a configuration that is not eligible for
cooperative batching; 3 completed but some stream's tokens differ from its single-stream reference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import threading
import time
from time import perf_counter_ns
import uuid

from .provenance import checkpoint_metadata, collect
from .qualification import FLOOR, load_backend, prompt_ids, token_digest
from veystrix.results import ROOT, ensure_inside, output_path

MODELS = ("flashnext", "nemotron35")
ARMS = ("dedicated", "cooperative-1", "cooperative-2", "cooperative-4")
STREAMS = {"dedicated": 1, "cooperative-1": 1, "cooperative-2": 2, "cooperative-4": 4}
SINGLE_ARMS = ("dedicated", "cooperative-1")
SCHEMA_VERSION = 1
ENGINE_PREFIXES = ("FLASHNEXT_", "VEYSTRIX_")
EXTRA_BODIES = ("Consider a skip list with probabilistic levels.\n",
                "Review the write-ahead log and its checkpoint cadence.\n")
UNLIMITED_HEADROOM = 1 << 62

EXIT_CLEAN, EXIT_STOPPED, EXIT_USAGE, EXIT_MISMATCH = 0, 1, 2, 3


class Ineligible(RuntimeError):
    """``fork_for_batch()`` returned None: the configuration is not eligible for cooperative batching."""


# -- pure statistics (tested with injected timestamps) -------------------------------------------------------------------

def percentile(values, q):
    """Linear-interpolation percentile (the numpy default) of ``values``; None when empty."""
    ordered = sorted(values)
    if not ordered:
        return None
    rank = q * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def stream_metrics(times_ns, start_ns):
    """TTFT, stream tok/s and gap statistics of one stream from its token-callback timestamps."""
    count = len(times_ns)
    gaps = [(later - earlier) / 1e6 for earlier, later in zip(times_ns, times_ns[1:])]
    span = times_ns[-1] - times_ns[0] if count > 1 else 0
    return {
        "output_tokens": count,
        "ttft_ms": (times_ns[0] - start_ns) / 1e6 if count else None,
        "stream_tok_s": (count - 1) * 1e9 / span if span > 0 else None,
        "gap_p50_ms": percentile(gaps, 0.50),
        "gap_p95_ms": percentile(gaps, 0.95),
        "gap_max_ms": max(gaps) if gaps else None,
    }


def aggregate_metrics(streams_times_ns):
    """All tokens of the arm over the wall time from the arm's first token to its last."""
    emitted = [times for times in streams_times_ns if times]
    total = sum(len(times) for times in emitted)
    if not emitted:
        return {"aggregate_tokens": 0, "aggregate_span_s": None, "aggregate_tok_s": None}
    span = max(times[-1] for times in emitted) - min(times[0] for times in emitted)
    return {"aggregate_tokens": total, "aggregate_span_s": span / 1e9 if span > 0 else None,
            "aggregate_tok_s": total * 1e9 / span if span > 0 else None}


def overhead_pct(dedicated_tok_s, cooperative_tok_s):
    """(dedicated - cooperative-1) / dedicated * 100; None unless both rates are usable."""
    if not dedicated_tok_s or cooperative_tok_s is None or dedicated_tok_s <= 0:
        return None
    return (dedicated_tok_s - cooperative_tok_s) / dedicated_tok_s * 100


# -- prompts -------------------------------------------------------------------------------------------------------------

def prompt_variants(backend, size, count):
    """``count`` different prompts of exactly ``size`` tokens.

    Variants 0 and 1 are ``qualification.prompt_ids``; further ones put a distinct short body after the system header
    of one of those, so every arm shares the same length and chat framing.
    """
    if not 1 <= count <= 2 + len(EXTRA_BODIES):
        raise ValueError(f"between 1 and {2 + len(EXTRA_BODIES)} prompt variants are supported")
    system = len(backend.encode("<|im_start|>system\n"))
    prompts = []
    for variant in range(count):
        if variant < 2:
            ids = prompt_ids(backend, size, variant=variant)
        else:
            extra = backend.encode(EXTRA_BODIES[variant - 2])
            base = prompt_ids(backend, size - len(extra), variant=variant % 2)
            ids = base[:system] + extra + base[system:]
        prompts.append(list(ids))
    if any(len(ids) != size for ids in prompts) or len({token_digest(ids) for ids in prompts}) != count:
        raise ValueError("prompt variants must have the requested length and differ from each other")
    return [{"variant": index, "ids": ids, "prompt_sha256": token_digest(ids)} for index, ids in enumerate(prompts)]


# -- evidence ------------------------------------------------------------------------------------------------------------

def write_evidence(path, payload):
    """Atomically rewrite the evidence file; it must live under ``results/``."""
    target = ensure_inside(Path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def engine_environment():
    return {key: value for key, value in sorted(os.environ.items())
            if key.startswith(ENGINE_PREFIXES) and key not in ("FLASHNEXT_PREFIX_CACHE_DIR", "VEYSTRIX_RESULTS_DIR",
                                                               "VEYSTRIX_CACHE_ROOT")
            and not any(part in key for part in ("KEY", "SECRET", "PASSWORD"))}


def harness_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def apply_harness_environment(directory, *, prefix_cache):
    """The environment block of ``qualification.main`` (kept in step by a test), then this harness's prefix choice.

    ``prefix_cache=False`` keeps every arm cold: otherwise a later arm restores the system prefix an earlier one saved and
    its TTFT is not comparable.
    """
    os.environ.update({
        "VEYSTRIX_CACHE_ROOT": str(directory), "FLASHNEXT_PREFIX_CACHE_DIR": str(directory),
        "FLASHNEXT_PREFIX_CACHE": "1", "FLASHNEXT_PREFIX_CACHE_BOUNDARIES": "system",
        "FLASHNEXT_PREFIX_CACHE_MIN_TOKENS": "256", "FLASHNEXT_PREFIX_CACHE_RAM_GB": "0",
        "VEYSTRIX_NEMOTRON_PREFIX_CACHE": "1", "FLASHNEXT_AUTO_PURGE": "0",
        "FLASHNEXT_EXPERT_POOL_GB": "0", "FLASHNEXT_SLAB_GLOBAL": "0",
        "FLASHNEXT_SLAB_PACK": "0", "FLASHNEXT_PROMPT_LOOKUP": "0",
        "FLASHNEXT_PREFIX_CACHE_RESTORE": "1", "FLASHNEXT_PREFIX_ASYNC_WRITE": "0",
        "VEYSTRIX_NEMOTRON_PROMPT_LOOKUP": "0",
        "FLASHNEXT_PREFILL_CACHE_LIMIT_MB": "512",
    })
    if not prefix_cache:
        os.environ.update({"FLASHNEXT_PREFIX_CACHE": "0", "VEYSTRIX_NEMOTRON_PREFIX_CACHE": "0",
                           "FLASHNEXT_PREFIX_CACHE_RESTORE": "0"})


# -- one arm -------------------------------------------------------------------------------------------------------------

class Stream:
    """Everything one request reports while it runs."""

    def __init__(self, index, prompt, arm):
        self.index, self.prompt, self.arm = index, prompt, arm
        self.times, self.tokens = [], []
        self.prefilled_ns = None
        self.stats = None
        self.error = None
        self.restored_tokens = 0
        self.cache_consistent = None

    def token(self, value, _piece):
        self.arm.mark_first()
        self.times.append(self.arm.clock())
        self.tokens.append(int(value))

    def prefilled(self):
        self.prefilled_ns = self.arm.clock()


class Arm:
    """The shared instruments of one arm: clock, floor check, read meter and memory sampler."""

    def __init__(self, name, *, clock, floor, read_meter, memory_read, memory_interval, interrupted):
        from .memory import MemorySampler, sample

        self.name, self.clock, self.floor = name, clock, floor
        self.interrupted = interrupted
        self.meter = read_meter()
        self.origin = perf_counter_ns()
        self.memory = MemorySampler(self.origin, memory_interval, read=memory_read or sample)
        self.lock = threading.Lock()
        self.first_ns = None
        self.trigger = None
        self.start_ns = None

    def mark_first(self):
        with self.lock:
            if self.first_ns is None:
                self.first_ns = self.clock()
                self.meter.reset()

    def cancelled(self):
        if self.interrupted.is_set():
            return True
        value = self.memory.samples[-1]["available_bytes"]
        if value is None or value < self.floor:
            with self.lock:
                if self.trigger is None:
                    self.trigger = {"reason": "available memory below the floor" if value is not None
                                    else "available memory could not be sampled",
                                    "available_bytes": value, "floor_bytes": self.floor}
            return True
        return False


def _record_stream(stream, start_ns):
    from veystrix.backends.base import GenerationCancelled

    stats = stream.stats
    record = {
        "stream": stream.index, "variant": stream.prompt["variant"], "prompt_sha256": stream.prompt["prompt_sha256"],
        "token_sha256": token_digest(stream.tokens),
        **stream_metrics(stream.times, start_ns),
        "finish": getattr(stats, "finish", None),
        "engine_decode_tok_s": getattr(stats, "rate", None),
        "prefill_seconds": getattr(stats, "prefill_seconds", None),
        "prefill_done_ms": (stream.prefilled_ns - start_ns) / 1e6 if stream.prefilled_ns is not None else None,
        "restored_tokens": stream.restored_tokens,
        "cache_consistent": stream.cache_consistent,
        "completed": stream.error is None and stats is not None,
        "error": None if stream.error is None else f"{type(stream.error).__name__}: {stream.error}",
        "cancelled": isinstance(stream.error, GenerationCancelled),
        "token_times_ms": [(moment - start_ns) / 1e6 for moment in stream.times],
        "digests_match_single": None, "reference_source": None,
    }
    return record


def _dedicated(backend, streams, output_tokens, arm):
    stream = streams[0]
    backend.reset()
    backend.append_tokens(stream.prompt["ids"])
    arm.start_ns = arm.clock()
    try:
        _, stream.stats = backend.generate(output_tokens, on_prefilled=stream.prefilled,
                                           on_decode_token=stream.token, should_cancel=arm.cancelled)
        stream.cache_consistent = backend.check_invariant() if hasattr(backend, "check_invariant") else None
    except BaseException as error:
        stream.error = error
        if not isinstance(error, Exception):
            raise
    prefix = getattr(backend, "last_prefix", None)
    stream.restored_tokens = int(getattr(prefix, "hit_tokens", 0) or 0)
    return None


def _cooperative(backend, streams, output_tokens, arm, headroom):
    from veystrix.backends import cooperative
    from veystrix.batch_scheduler import BatchScheduler

    forks = []
    for _ in streams:
        fork = backend.fork_for_batch()
        if fork is None:
            raise Ineligible("fork_for_batch() returned None: this configuration cannot run cooperatively")
        forks.append(fork)
    scheduler = BatchScheduler(max_active=len(streams), headroom=headroom)
    gate = threading.Event()

    def request(stream, fork):
        gate.wait()
        slot = False
        try:
            scheduler.acquire_request()
            slot = True
            fork.append_tokens(stream.prompt["ids"])
            state = scheduler.reserve(uuid.uuid4().hex, fork.batch_state_bytes(len(stream.prompt["ids"]) + output_tokens))
            try:
                _, stream.stats = cooperative.generate(
                    fork, output_tokens,
                    dispatch=lambda work, decode: scheduler.submit_quantum(state, work, decode),
                    dispatch_decode=lambda args, cancel: scheduler.submit_decode(state, args, cancel),
                    on_prefilled=stream.prefilled, on_decode_token=stream.token, should_cancel=arm.cancelled)
            finally:
                scheduler.release(state)
        except BaseException as error:
            stream.error = error
        finally:
            if slot:
                scheduler.release_request()
        prefix = getattr(fork, "last_prefix", None)
        stream.restored_tokens = int(getattr(prefix, "hit_tokens", 0) or 0)

    threads = [threading.Thread(target=request, args=(stream, fork), name=f"request-{stream.index}", daemon=True)
               for stream, fork in zip(streams, forks, strict=True)]
    stop = threading.Event()

    def watch():
        for thread in threads:
            thread.join()
        stop.set()

    def open_when_serving():
        # Requests are refused until the loop below is running; release them together the moment it is.
        for _ in range(30000):
            if scheduler.loop_running:
                break
            time.sleep(0.001)
        arm.start_ns = arm.clock()
        gate.set()

    watcher = threading.Thread(target=watch, daemon=True)
    opener = threading.Thread(target=open_when_serving, daemon=True)
    for thread in threads:
        thread.start()
    watcher.start()
    opener.start()
    try:
        scheduler.serve_jobs(stop)               # every forward runs here, on the main thread
    except BaseException:
        arm.interrupted.set()                    # unwinds the request threads at their next check
        stop.set()
        for thread in threads:
            thread.join(30)
        raise
    watcher.join(30)
    for thread in threads:
        thread.join(30)
    batch = scheduler.snapshot()["batch"]
    return {key: batch.get(key) for key in ("decode_groups", "max_decode_group", "max_active", "refused")}


def run_arm(backend, name, prompts, output_tokens, *, floor=FLOOR, clock=perf_counter_ns, read_meter=None,
            memory_read=None, memory_interval=1.0, headroom=None, interrupted=None):
    """Run one arm and return its evidence record. Raises ``Ineligible`` for a non-eligible configuration."""
    import mlx.core as mx

    if read_meter is None:
        from models.flashnext.diskio import ReadMeter as read_meter
    interrupted = interrupted or threading.Event()
    arm = Arm(name, clock=clock, floor=floor, read_meter=read_meter, memory_read=memory_read,
              memory_interval=memory_interval, interrupted=interrupted)
    streams = [Stream(index, prompt, arm) for index, prompt in enumerate(prompts)]
    mx.reset_peak_memory()
    arm.memory.prepare()
    if arm.cancelled():
        return {"arm": name, "status": "refused", "streams": [], "stopped": arm.trigger,
                "reason": "available memory below the floor before inference"}
    scheduler_report = None
    with arm.memory:
        if name == "dedicated":
            _dedicated(backend, streams, output_tokens, arm)
        else:
            scheduler_report = _cooperative(backend, streams, output_tokens, arm, headroom)
        read_bytes = arm.meter.bytes_since() if arm.first_ns is not None else None
    records = [_record_stream(stream, arm.start_ns) for stream in streams]
    memory = arm.memory.summary()
    swaps = [row["swap_used_bytes"] for row in memory["samples"]]
    swapouts = [row["swapout_bytes_total"] for row in memory["samples"]]
    times = [stream.times for stream in streams]
    emitted = sum(len(stream.times) for stream in streams)
    after_first = emitted - 1
    cancelled = arm.trigger is not None or any(record["cancelled"] for record in records)
    other_error = next((record["error"] for record in records if record["error"] and not record["cancelled"]), None)
    result = {
        "arm": name, "streams_requested": len(streams), "output_tokens_requested": output_tokens,
        "status": "stopped" if cancelled else ("failed" if other_error else "completed"),
        "stopped": arm.trigger if arm.trigger is not None else (
            {"reason": "interrupted"} if interrupted.is_set() else None),
        "error": other_error,
        "streams": records,
        **aggregate_metrics(times),
        "physical_mb_per_token": (read_bytes / after_first / 1e6
                                  if read_bytes is not None and read_bytes >= 0 and after_first > 0 else None),
        "physical_read_scope": "this process (proc_pid_rusage) from the arm's first emitted token to the arm's end, "
                               "per token emitted after the first",
        "scheduler": scheduler_report,
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "available_floor_bytes": memory["sampled_available_floor_bytes"],
        "swap_growth_bytes": max(swaps) - swaps[0] if all(x is not None for x in swaps) else None,
        "swapout_growth_bytes": max(swapouts) - swapouts[0] if all(x is not None for x in swapouts) else None,
        "memory": memory,
    }
    return result


# -- references and the run ----------------------------------------------------------------------------------------------

def settle(report):
    """Fill the digest comparisons and the derived figures from whatever has been recorded so far."""
    references = {}
    for source in ("cooperative-1", "dedicated"):                 # dedicated wins when both exist
        for arm in report["arms"]:
            if arm["arm"] == source and arm.get("status") == "completed":
                stream = arm["streams"][0]
                references[stream["prompt_sha256"]] = {"token_sha256": stream["token_sha256"], "source": source}
    for run in report.get("reference_runs", ()):
        if run.get("status") == "completed":
            stream = run["streams"][0]
            references.setdefault(stream["prompt_sha256"], {"token_sha256": stream["token_sha256"],
                                                            "source": "reference-run"})
    report["references"] = references
    checked = mismatched = 0
    for arm in report["arms"]:
        for stream in arm.get("streams", ()):
            reference = references.get(stream["prompt_sha256"])
            if reference is None or not stream["completed"]:
                stream["digests_match_single"], stream["reference_source"] = None, None
                continue
            own = arm["arm"] == reference["source"] and arm["arm"] in SINGLE_ARMS
            stream["digests_match_single"] = True if own else stream["token_sha256"] == reference["token_sha256"]
            stream["reference_source"] = "self" if own else reference["source"]
            checked += 1
            mismatched += not stream["digests_match_single"]
    report["exactness"] = {"streams_checked": checked, "mismatches": mismatched,
                           "all_match": checked > 0 and mismatched == 0}
    by_name = {arm["arm"]: arm for arm in report["arms"] if arm.get("status") == "completed"}
    first = by_name.get("dedicated"), by_name.get("cooperative-1")
    report["cooperative_1_overhead_pct"] = (
        overhead_pct(first[0]["streams"][0]["stream_tok_s"], first[1]["streams"][0]["stream_tok_s"])
        if all(first) else None)
    if all(first):
        report["cooperative_1_matches_dedicated"] = (
            first[0]["streams"][0]["token_sha256"] == first[1]["streams"][0]["token_sha256"])


def run_benchmark(backend, report, out, *, arms, prompts, output_tokens, single_references=True, headroom=None,
                  floor=FLOOR, clock=perf_counter_ns, read_meter=None, memory_read=None, memory_interval=1.0,
                  interrupted=None):
    """Run ``arms`` in the given order, rewriting the evidence after each; return the process exit status."""
    interrupted = interrupted or threading.Event()
    options = dict(floor=floor, clock=clock, read_meter=read_meter, memory_read=memory_read,
                   memory_interval=memory_interval, headroom=headroom, interrupted=interrupted)
    report.setdefault("arms", [])
    report.setdefault("reference_runs", [])
    report["status"] = "running"
    write_evidence(out, report)

    def attempt(name, chosen, destination):
        record = run_arm(backend, name, chosen, output_tokens, **options)
        destination.append(record)
        settle(report)
        write_evidence(out, report)
        backend.reset()                                  # drop the arm's conversation before the next one
        gc.collect()
        return record

    try:
        for name in arms:
            record = attempt(name, prompts[:STREAMS[name]], report["arms"])
            if record["status"] != "completed":
                report["status"] = "stopped"
                report["error"] = record.get("error") or (record.get("stopped") or {}).get("reason") or record["status"]
                return EXIT_STOPPED
        if single_references:
            known = {stream["prompt_sha256"] for arm in report["arms"] if arm["arm"] in SINGLE_ARMS
                     and arm["status"] == "completed" for stream in arm["streams"]}
            for prompt in prompts:
                needed = any(prompt["prompt_sha256"] == stream["prompt_sha256"]
                             for arm in report["arms"] for stream in arm["streams"])
                if prompt["prompt_sha256"] in known or not needed:
                    continue
                record = attempt("dedicated", [prompt], report["reference_runs"])
                record["purpose"] = "single-stream reference"
                if record["status"] != "completed":
                    report["status"] = "stopped"
                    report["error"] = "single-stream reference run did not complete"
                    return EXIT_STOPPED
        report["status"] = "completed"
    except Ineligible as error:
        report["status"] = "ineligible"
        report["error"] = str(error)
        return EXIT_USAGE
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "stopped"
        report["error"] = f"{type(error).__name__}: {error}"
        return EXIT_STOPPED
    finally:
        settle(report)
        write_evidence(out, report)
    return EXIT_CLEAN if report["exactness"]["mismatches"] == 0 else EXIT_MISMATCH


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["dedicated", "cooperative-1"])
    parser.add_argument("--order", choices=("given", "reversed"), default="given")
    parser.add_argument("--single-references", action=argparse.BooleanOptionalAction, default=True,
                        help="run each distinct prompt once alone when no single-stream arm produced it (default on)")
    parser.add_argument("--qwen-prefill-16gb", action="store_true")
    parser.add_argument("--resident-experts", type=int, default=0)
    parser.add_argument("--nemotron-pool-gb", type=float, default=0.0)
    parser.add_argument("--prefix-cache", choices=("on", "off"), default="off",
                        help="off (default) keeps every arm cold; on mirrors qualification's prefix-cache environment")
    parser.add_argument("--ignore-admission", action="store_true",
                        help="replace the scheduler's host-headroom admission (available - 4 GiB) with no limit; "
                             "the harness's own 3 GiB floor still stops the arm")
    args = parser.parse_args(argv)
    if args.prompt_tokens < 64 or args.output_tokens < 2:
        parser.error("prompt tokens must be at least 64 and output tokens at least 2")
    if args.resident_experts < 0 or not 0 <= args.nemotron_pool_gb <= 4:
        parser.error("resident experts must be nonnegative and the Nemotron pool between 0 and 4 GiB")
    arms = list(dict.fromkeys(args.arms))
    args.arms = arms[::-1] if args.order == "reversed" else arms
    return args


def main(argv=None):
    args = parse_arguments(argv)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    out = output_path(args.model, "stage10-concurrency", "concurrency.json")
    report = {
        "schema_version": SCHEMA_VERSION, "status": "running", "model": args.model, "checkpoint": str(checkpoint),
        "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
        "arms_requested": args.arms, "order": args.order, "single_references": args.single_references,
        "prefix_cache": args.prefix_cache == "on",
        "admission": "disabled by --ignore-admission" if args.ignore_admission else "production headroom",
        "floor_bytes": FLOOR, "reference_host": collect(ROOT, checkpoint),
        "checkpoint_metadata": checkpoint_metadata(checkpoint), "harness_sha256": harness_sha256(),
        "resident_experts": args.resident_experts, "nemotron_pool_gb": args.nemotron_pool_gb,
        "qwen_prefill_16gb": args.qwen_prefill_16gb, "arms": [], "reference_runs": [],
    }
    print(out, flush=True)
    write_evidence(out, report)
    status = EXIT_STOPPED
    try:
        with tempfile.TemporaryDirectory(dir=out.parent, prefix="prefix-") as directory:
            apply_harness_environment(directory, prefix_cache=args.prefix_cache == "on")
            if args.model == "flashnext" and args.qwen_prefill_16gb:
                from models.flashnext.prefill_16gb import apply

                apply(os.environ)
            backend = load_backend(args.model, checkpoint, args.prompt_tokens + args.output_tokens, directory,
                                   resident_experts=args.resident_experts,
                                   pool_bytes=int(args.nemotron_pool_gb * 2**30))
            report["runtime_environment"] = engine_environment()
            streams = max(STREAMS[name] for name in args.arms)
            if any(name != "dedicated" for name in args.arms):
                probe = getattr(backend, "fork_for_batch", lambda: None)()
                if probe is None:
                    report["status"] = "ineligible"
                    report["error"] = ("fork_for_batch() returned None: this configuration is not eligible for "
                                       "cooperative batching (greedy sampling, no MTP/prompt-lookup/swap routing)")
                    write_evidence(out, report)
                    print(report["error"], file=sys.stderr, flush=True)
                    return EXIT_USAGE
                del probe
            prompts = prompt_variants(backend, args.prompt_tokens, streams)
            report["prompts"] = [{"variant": row["variant"], "prompt_sha256": row["prompt_sha256"],
                                  "tokens": len(row["ids"])} for row in prompts]
            status = run_benchmark(
                backend, report, out, arms=args.arms, prompts=prompts, output_tokens=args.output_tokens,
                single_references=args.single_references,
                headroom=(lambda: UNLIMITED_HEADROOM) if args.ignore_admission else None)
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "stopped"
        report["error"] = f"{type(error).__name__}: {error}"
        status = EXIT_STOPPED
    finally:
        write_evidence(out, report)
    if report["status"] != "completed":
        print(report.get("error") or report["status"], file=sys.stderr, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
