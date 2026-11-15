"""Compare checkpoint and packed readers on recorded Flash-Next decode routes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from veystrix.results import output_path


MODES = ("checkpoint", "record", "coalesced", "buffered", "mmap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--pool-mib", type=int, default=512)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if min(args.tokens, args.rounds, args.pool_mib) < 1:
        parser.error("token, round and pool sizes must be positive")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be unique")

    from models.flashnext import expert_cache, expert_pool
    from models.flashnext.diskio import disk_bytes_read
    from models.flashnext.slab_pack import checkpoint_identity
    from models.flashnext.store import SafeTensorStore
    from models.nemotron35.safetensors_index import index_checkpoint
    from models.shared.expert_pack import ExpertPack
    from veystrix.bench.provenance import collect
    from veystrix.bench.telemetry import Trace

    checkpoint, pack = args.checkpoint.expanduser(), args.pack.expanduser()
    sidecar = ExpertPack(pack, checkpoint, index_checkpoint(checkpoint), "flashnext")
    try:
        layers = sorted(layer for layer in sidecar.layers if layer > 0)
    finally:
        sidecar.close()
    trace_data = json.loads(gzip.decompress(args.trace.read_bytes()))
    events = [(turn, token, layer, experts) for turn, phase, token, layer, experts in trace_data["events"]
              if phase == "decode" and 1 <= token <= args.tokens and layer in layers]
    if not events:
        parser.error("no recorded decode routes match the sidecar layers and token limit")
    distinct_tokens = len({(turn, token) for turn, token, _, _ in events})
    out = output_path("flashnext", "expert-pack-replay", "expert-pack-replay.json")
    evidence = {
        "schema_version": 1,
        "checkpoint_identity": checkpoint_identity(checkpoint),
        "provenance": collect(Path(__file__).resolve().parents[4], checkpoint),
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "pack": pack.name, "layers": layers, "trace_tokens": distinct_tokens,
        "pool_bytes": args.pool_mib * 2**20, "page_cache": "as found", "arms": [],
    }
    store = SafeTensorStore(checkpoint)
    try:
        for index in range(args.rounds):
            order = args.modes if index % 2 == 0 else list(reversed(args.modes))
            for mode in order:
                settings = {
                    "FLASHNEXT_EXPERT_PACK": "" if mode == "checkpoint" else str(pack),
                    "FLASHNEXT_EXPERT_PACK_READ": "coalesced" if mode == "buffered" else
                        "record" if mode == "checkpoint" else mode,
                    "FLASHNEXT_EXPERT_PACK_NATIVE": "0",
                    "FLASHNEXT_EXPERT_POOL_BUFFERED_FILL": str(int(mode == "buffered")),
                    "FLASHNEXT_EXPERT_POOL_POLICY": "lru",
                    "FLASHNEXT_EXPERT_POOL_LOCK": "0",
                }
                with patch.dict(os.environ, settings):
                    pool = expert_pool.ExpertPool(store, evidence["pool_bytes"])
                    try:
                        if mode != "checkpoint" and pool.pack is None:
                            raise RuntimeError(f"{mode} fell back before the comparison")
                        trace = Trace(capture_shapes=True)
                        read_start, cpu_start, start = disk_bytes_read(), time.process_time_ns(), time.perf_counter_ns()
                        with trace.activate():
                            for _, _, layer, experts in events:
                                _, misses = pool.assign(layer, experts)
                                prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp"
                                for future in pool.fill(prefix, misses, expert_cache._submit_read):
                                    future.result()
                        wall_ns = time.perf_counter_ns() - start
                        cpu_ns = time.process_time_ns() - cpu_start
                        read_end = disk_bytes_read()
                        if mode != "checkpoint" and pool.packed_reads != pool.misses:
                            raise RuntimeError(f"{mode} fell back during the comparison")
                        evidence["arms"].append({
                            "round": index + 1, "mode": mode, "wall_ns": wall_ns, "cpu_ns": cpu_ns,
                            "physical_bytes": read_end - read_start if 0 <= read_start <= read_end else None,
                            "pool_hits": pool.hits, "pool_misses": pool.misses,
                            "pack_extents": pool.pack_extents,
                            "average_pack_extent_bytes": pool.pack_extent_bytes / pool.pack_extents if pool.pack_extents else None,
                            "positioned_read_calls": trace.reads["positioned_read_calls"],
                            "positioned_read_bytes": trace.reads["positioned_read_bytes"],
                        })
                    finally:
                        pool.close()
                out.write_text(json.dumps(evidence, indent=2) + "\n")
        evidence["median_cpu_ns"] = {mode: statistics.median(row["cpu_ns"] for row in evidence["arms"]
                                     if row["mode"] == mode) for mode in args.modes}
        evidence["status"] = "completed"
        out.write_text(json.dumps(evidence, indent=2) + "\n")
        print(out)
    finally:
        store.close()


if __name__ == "__main__":
    main()
