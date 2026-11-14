"""Sweep buffered, uncached, mmap and vectored reads with optional GPU traffic."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import mmap
import os
from pathlib import Path
import statistics
import sys
import threading
import time

from veystrix.measurement import MeasurementRun, source_hash
from veystrix.results import output_path

ALIGNMENT = 65536
MODES = ("pread", "uncached", "mmap", "preadv")


def percentile(samples, fraction):
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def read_arm(path, mode, depth, block, total, start, *, gpu):
    from models.flashnext.diskio import disk_bytes_read, vm_counters

    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    mapping = None
    stop = threading.Event()
    ready = threading.Event()
    gpu_thread = None
    gpu_errors = []
    gpu_iterations = []

    def gpu_stream():
        try:
            import mlx.core as mx

            with mx.stream(mx.new_stream(mx.gpu)):
                a = mx.ones((8 * 2**20,), dtype=mx.float32)
                b = mx.full(a.shape, 1.0001)
                mx.eval(a, b)
                count = 0
                while not stop.is_set():
                    a = a * b + 0.0001
                    mx.eval(a)
                    count += 1
                    ready.set()
                gpu_iterations.append(count)
        except Exception as error:
            gpu_errors.append(error)
            ready.set()

    def read(offset):
        began = time.perf_counter()
        if mode == "mmap":
            data = mapping[offset - start:offset - start + block]
            size = len(data)
        elif mode == "preadv":
            buffer = bytearray(block)
            half = block // 2
            view = memoryview(buffer)
            size = os.preadv(fd, [view[:half], view[half:]], offset)
        else:
            size = len(os.pread(fd, block, offset))
        if size != block:
            raise OSError(f"short read at {offset}")
        return (time.perf_counter() - began) * 1000

    try:
        if mode == "uncached":
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        if mode == "mmap":
            mapping = mmap.mmap(fd, total, access=mmap.ACCESS_READ, offset=start)
        if gpu:
            gpu_thread = threading.Thread(target=gpu_stream, name="nvme-gpu-load")
            gpu_thread.start()
            if not ready.wait(timeout=30):
                raise TimeoutError("GPU traffic did not start")
            if gpu_errors:
                raise RuntimeError("GPU traffic failed") from gpu_errors[0]
        before, vm_before = disk_bytes_read(), vm_counters()
        with ThreadPoolExecutor(depth) as workers:
            began = time.perf_counter()
            latency = list(workers.map(read, range(start, start + total, block)))
            seconds = time.perf_counter() - began
        after = disk_bytes_read()
        metrics = {"wall_s": seconds, "read_gb_s": total / seconds / 1e9,
                   "physical_bytes": after - before if before >= 0 and after >= before else None,
                   "p50_ms": statistics.median(latency), "p99_ms": percentile(latency, 0.99),
                   "latencies_ms": latency, "vm_before": vm_before, "vm_after": vm_counters()}
    finally:
        stop.set()
        if gpu_thread is not None:
            gpu_thread.join()
        if mapping is not None:
            mapping.close()
        os.close(fd)
    if gpu_errors:
        raise RuntimeError("GPU traffic failed") from gpu_errors[0]
    return {**metrics, "gpu_iterations": sum(gpu_iterations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--depths", nargs="+", type=int, choices=(1, 2, 4, 8), default=(1, 2, 4, 8))
    parser.add_argument("--blocks-kib", nargs="+", type=int, default=(2688, 5504))
    parser.add_argument("--mib", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--disjoint", action="store_true", help="never reuse a window, including between trials")
    args = parser.parse_args()
    args.file = [path.expanduser() for path in args.file]
    if (any(not path.is_file() or path.is_symlink() for path in args.file) or not 1 <= args.mib <= 512
            or not 0 <= args.warmups <= 3 or not 1 <= args.trials <= 5
            or any(n < 64 or n > args.mib * 1024 or n % 64 for n in args.blocks_kib)):
        parser.error("use a regular file, 1–512 MiB per arm, and 64 KiB-aligned block sizes")
    files = [(path, path.stat().st_size) for path in args.file]
    if min(size for _, size in files) < args.mib * 2**20:
        parser.error("a source is shorter than the requested arm size")
    if len({(path.stat().st_dev, path.stat().st_ino) for path, _ in files}) != len(files):
        parser.error("source files must be distinct")
    result_path = output_path("flashnext", "nvme-depth", "nvme-depth.jsonl")
    run = MeasurementRun(result_path, runtime="flashnext", experiment="nvme-depth", metadata={
        "diagnostic": True, "settings": vars(args), "source_sha256": source_hash(Path(__file__)),
        "file_bytes": {str(path): size for path, size in files},
        "offset_policy": ("strictly disjoint" if args.disjoint else "disjoint until wrap"),
    })
    run.start()
    from models.flashnext.tests.bench.bench_layer_gate import machine_sample, safety_reason

    file_index, offset, sequence, samples = 0, 0, 0, {}
    conditions = [(mode, depth, kib) for mode in args.modes for depth in args.depths for kib in args.blocks_kib]
    try:
        for trial in range(args.warmups + args.trials):
            for mode, depth, kib in conditions if trial % 2 == 0 else reversed(conditions):
                block = kib * 1024
                total = args.mib * 2**20 // block * block
                path, size = files[file_index]
                if offset + total > size:
                    file_index += 1
                    if file_index >= len(files) and args.disjoint:
                        raise ValueError("disjoint benchmark windows exhausted")
                    file_index %= len(files)
                    path, _ = files[file_index]
                    offset = 0
                condition = f"{mode}-qd{depth}-{kib}kib"
                sequence += 1
                before = machine_sample()
                reason = safety_reason(before, memory_rules="swap-only", power_rules="battery-ok")
                if reason:
                    raise RuntimeError(reason)
                row = read_arm(path, mode, depth, block, total, offset, gpu=args.gpu)
                after = machine_sample()
                warmup = trial < args.warmups
                run.arm(arm_id=str(sequence), condition=condition, round_index=trial + 1,
                        command=[sys.executable, "-m", __spec__.name, *sys.argv[1:]],
                        metrics=row, warmup=warmup, offset=offset, logical_bytes=total, source_file=str(path),
                        machine_before=before, machine_after=after)
                reason = safety_reason(after, memory_rules="swap-only", power_rules="battery-ok")
                if reason:
                    raise RuntimeError(reason)
                if not warmup:
                    samples.setdefault(condition, []).append((row["read_gb_s"], row["physical_bytes"], total))
                offset += total
    except (Exception, KeyboardInterrupt) as error:
        run.failure(str(error), arm_id=str(sequence))
        run.finish("failed")
        raise
    # Two-sided 95% Student t critical values for n=2..5 trials.
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    summaries = {}
    for key, measured in samples.items():
        values = [rate for rate, _, _ in measured]
        average = statistics.mean(values)
        margin = t95[len(values)] * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None
        fractions = [physical / logical for _, physical, logical in measured if physical is not None]
        summaries[key] = {"median_gb_s": statistics.median(values), "mean_gb_s": average,
                          "mean_95pct_ci": [average - margin, average + margin] if margin is not None else None,
                          "median_physical_read_fraction": statistics.median(fractions) if fractions else None,
                          "trials": len(values)}
    run.finish("diagnostic", throughput=summaries)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
