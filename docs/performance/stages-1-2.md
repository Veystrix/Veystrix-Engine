# Flash-Next memory capacity and expert packs

On machines with up to 16 GiB, Flash-Next sizes the optional decode pool
before prefill and again before decode. The configured
`FLASHNEXT_EXPERT_POOL_GB` is a ceiling in decimal GB. The planner chooses
0, 0.5, 1 or 2 GiB below that ceiling and shrinks the pool when the next phase
needs the space or macOS reports elevated pressure. Growth after a shrink
requires a cooldown and spare capacity. The planner accounts separately for
resident weights, live cache state, new KV allowance, prefix RAM, prefill
scratch and the maximum configured verifier width. It budgets 8 MiB per
routed position, up to 640 MiB at width eight. This is a planning allowance
for the expert union and staging, and has not been measured as an allocator
peak. The Stage 0 arm includes both phase budgets.

The new planner can leave the pool off and allows inference to proceed
regardless of the reported available-memory floor. Existing separately
enabled runtime policies still apply. Use `VEYSTRIX_MEMORY_POLICY=legacy` to
restore the prior pool sizing or `VEYSTRIX_MEMORY_POLICY=elastic` to opt in
on a larger host. The existing `FLASHNEXT_MEMORY_MANAGER=1` explicitly
selects the earlier experimental planner.

## Build and select a sidecar

```sh
mkdir -p "$PACK_DIR"
.venv/bin/python -m models.shared.pack_experts flashnext "$QWEN_CHECKPOINT" \
  "$PACK_DIR" --content-addressed
```

The writer streams aligned Q4 expert records into a temporary file, hashes
the output as it writes, and publishes it atomically under its SHA-256
filename. The checkpoint remains read-only. Use `--layers 1 2` for a
smaller comparison. The reader validates the layout, source fingerprint and
file length. Missing layers and unavailable packs use the original shards.

Set `FLASHNEXT_EXPERT_PACK` to the output file with a decode pool enabled.
`FLASHNEXT_EXPERT_PACK_READ=record` reads each expert independently.
`coalesced` groups up to eight consecutive experts in a bounded vectored read
into their assigned slots. `mmap` copies those extents from a read-only mapping.
`FLASHNEXT_EXPERT_POOL_BUFFERED_FILL=1` lets positioned reads use the OS page
cache. An unset pack path selects the original checkpoint reader. The new
read modes are opt-in.

```sh
.venv/bin/python -m models.flashnext.tests.bench.bench_expert_pack_replay \
  --checkpoint "$QWEN_CHECKPOINT" --pack "$EXPERT_PACK" --trace "$ROUTE_TRACE" \
  --pool-mib 512 --tokens 16 --rounds 5
```

Replay holds pool capacity and recorded routes fixed, reverses mode order
each round, and measures process CPU, wall time, physical bytes, positioned
read counts and extent sizes. It preserves the observed page cache, so a
buffered or mapped result can benefit from earlier arms. Use full-model token
parity and a separately controlled storage comparison for promotion.


Pair the same checkpoint and sidecar through the production token digest
gate. The benchmark requires the sidecar arm to read packed experts.

```sh
FLASHNEXT_EXPERT_POOL_GB=0.6 .venv/bin/python   -m models.flashnext.tests.bench.bench_production --fresh-arms   --expert-pack "$EXPERT_PACK" --tokens 32 --arms 5 --min-arms 5 --drop 0
```
