# Second miracle hunt: can the single M4 Air SSD physically exceed ~3.36 GB/s with no new hardware?

Scope: 16 GB M4 MacBook Air, one internal NVMe, E0 measured **3.36 GB/s** sustained cold `F_NOCACHE` reads (8 threads, 4 MiB chunks, full 113 GB checkpoint). Question: is there any software-only mechanism that raises the *physical* cold-read ceiling (target ~6.7 GB/s = 2×), to help Qwen reach 5+ and Nemotron 15+ tok/s bit-exactly? Reads-only mechanisms are bit-exact by construction (no arithmetic changes); the only quality risk is if prefetch/prediction changes *which* experts are executed.

Bottom line up front: **3.36 GB/s is a hardware wall.** The M4 Air's NAND/interconnect caps cold uncached reads at ~2.9–3.4 GB/s; every public number above that is warm page-cache data, an overlap/apparent throughput metric, or a larger Pro/Max/M5 chip. The "6.06 GB/s double buffering" figure in our own report is *not* 6.06 GB/s of cold SSD bytes — it is I/O overlapped with compute. No software path breaks the wall; the remaining wins are latency hiding and byte reduction, both already catalogued in neighbouring notes.

---

## Q1. Is 3.36 GB/s a hard hardware wall on this M4 Air, or is there headroom?

### Takeaway
It is a hardware wall. Independent benchmarks put the M4 Air's sequential read at 2,833–3,031 MB/s (Blackmagic 5 GB read) and 2,904 MB/s (15-inch), so our 3.36 GB/s is already *at or above* the published device ceiling. The SSD bandwidth of Apple Silicon scales with NAND package count/interleaving and PCIe lane count, not capacity; the Air uses a 2-lane link where Pro/Max use 4 lanes, and the ~2× jump to ~6.5 GB/s arrives only with the M5 generation (hardware change). Double-buffering, queue depth, hints and layout cannot make the media deliver more cold bytes.

### Cited Findings
- Blackmagic Disk Speed Test (5 reads): M4 MacBook Air 10c/8c GPU (256 GB) read **2,883.5 MB/s**; 10c/10c GPU (512 GB+) read **3,031.2 MB/s**; AmorphousDiskMark agreed the 10c/10c model is faster. — [GIGAZINE](https://gigazine.net/gsc_news/en/20250317-apple-m4-macbook-air-benchmark)
- Notebookcheck comparison table: MacBook Air 15 M4 (Apple SSD AP1024Z) **2,904 MB/s** read, 3,023.9 MB/s write; M4 Pro 14 (AP0512Z) 3,028 MB/s. — [Notebookcheck](https://www.notebookcheck.org/Apple-Analisis-del-MacBook-Air-15-M5-Muy-potente-sin-ventilador-y-sin-competencia.1245645.0.html)
- The M5 Air jumps to 6,473.4 MB/s read "suggesting Apple has moved to a faster storage protocol, likely PCIe NVMe Gen 4, and may be utilizing a dual-NAND flash chip configuration" — a generational hardware change, not a software trick. — [Biggo Finance / Notebookcheck](https://finance.biggo.com/news/202603091052_M5_MacBook_Air_SSD_Speed_Review)
- "Apple Silicon SSD throughput tracks the number of NAND packages (interleaving), not capacity itself"; two 512 GB machines differed 40% due to chip count; a 256 GB M2 MacBook Pro with one chip read ~1,446 MB/s vs ~2,900 MB/s with two. — [llms-explorer](https://llms-explorer.com/downloads/concepts/apple-ssd-speed-dependence-on-nand-capacity-and.md)
- A (lower-quality) review describes the M4 Air SSD controller as a custom ASIC on PCIe 4.0 ×2 lanes vs ×4 on M4 Pro, with ~2,250 MB/s sequential read. Treat the exact number cautiously — it conflicts with the better-sourced ~2.9–3.0 GB/s above. — [Frame & Focal](https://frameandfocal.com/camera-reviews/apple-macbook-air-with-m4-review-thin-in-size-price-and-sometimes-performance)
- Third-party M4 Max reads 5–7 GB/s; Apple's first-party peak is 14.5 GB/s on M5 Max 8 TB; speed follows NAND package count, not capacity. — [llms-explorer](https://llms-explorer.com/tree/mac-local-llms-moe-streaming-and-offload)
- M3 Air 24 GB: 20 GiB file with `F_NOCACHE` = **2.85 GB/s**; an 8 GiB file read 8.8 GB/s purely from page cache — "which is why the file must exceed memory". Our 113 GB checkpoint comfortably exceeds 16 GB, so E0's 3.36 GB/s is a genuine device figure, not cache. — [Apple Silicon Quest / swap cliff](https://www.creativeworksofknowledge.com/en/cwk-quests/apple-silicon-quest/memory/swap-cliff)
- `F_NOCACHE` is essential for honest measurement: "BlackMagic and AmorphousDiskMark skip it and may be inflated by RAM cache on big-memory Macs"; DiskTruth was created specifically to measure with `F_NOCACHE` on every fd. — [llms-explorer](https://llms-explorer.com/tree/mac-local-llms-moe-streaming-and-offload)
- M4 Air base SSD ~3 GB/s; M4 Pro internal "more like 6–7 GB/s" — our machine is the base M4, so Pro-class figures do not apply. — [prior project note, os_io_tricks.md Q6](../MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)

### Inferences
- The device ceiling on this hardware is ~3.0–3.4 GB/s. E0's 3.36 GB/s is essentially the wall; a software-only 2× to 6.7 GB/s is not physically available. This matches the project's own "more I/O workers = flat, SSD saturated" result.
- The M5 Air's ~6.5 GB/s proves Apple can build a 2× faster SSD, but it is a *different chip generation* with a different NAND/protocol — buying hardware, which this hunt excludes.
- Any reported figure materially above ~3.4 GB/s on this machine should be presumed warm-cache or overlap until proven with `F_NOCACHE` on a file larger than RAM (we have that proof: E0).
- Bit-exactness: not applicable — this is a physical limit, not a mechanism.

### Gaps
- No teardown confirms the exact NAND package count of this specific 16 GB M4 Air unit, so the precise ceiling (±0.2 GB/s) cannot be pinned down from public data alone.
- No public measurement of sustained uncached reads *while the GPU is simultaneously saturating the memory controller* on a 16 GB M4 Air; the 3.36 GB/s was measured standalone. Whether GPU contention lowers the effective ceiling during real decode is untested (see Q2/Q3 for the shared-controller evidence).

---

## Q2. The "6.06 GB/s double buffering" claim — is it a real second miracle?

### Takeaway
No. The TinyGiant/llama.cpp #27149 figure of "6.06 GB/s double-buffered pipeline" is an *overlap/apparent throughput* metric (I/O for layer N+1 hidden behind compute for layer N), not 6.06 GB/s of cold bytes from the device. The same source's own baseline — "sequential layer read (286 MB chunks) 2.97 GB/s" — is the real physical device rate and matches the M-series Air ceiling. This is the same latency-hiding technique already listed as method #6 in our report, not new bandwidth.

### Cited Findings
- Original post (M1 MacBook Pro 16 GB): "Sequential layer read (286 MB chunks) **2.97 GB/s**; Double-buffered pipeline **6.06 GB/s**; Pipeline speedup 2.09×." The model file is a 17.3 GB GGUF with simulated compute ("matrix multiplication is not wired up") in the first PoC. — [llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- Later compiled C benchmark with real BLAS compute (Apple Accelerate, M1 16 GB): pipelined vs sequential tok/s — 50% hit 2.20→2.40, 75% hit 2.92→3.81, 88% hit 3.47→4.10, 100% hit 4.46→4.45. "Compute ceiling: 4.4 tok/s (CPU only). SSD: **3.0+ GB/s** from contiguous layout. At 88% hit, SSD reads are fully hidden behind compute." — [llama.cpp discussion #27149, 2026-08-19 update](https://github.com/ggml-org/llama.cpp/discussions/27149)
- Our own report lists "overlap / double-buffered async I/O" as method #6: Jetson 1.94→2.95 tok/s (+32%), "TinyGiant M1 16 GB: 2.97→6.06 GB/s (2.09×)", bit-exact. — [project report](../Faster%20Qwen%20Nemotron%20inference%20on%20M4.md)
- General correction from the bandwidth survey: figures like "17.5 GB/s (M3 Max)" and "36 GB/s (Anemll, 0.74 ms/27 MB)" "exceed Apple's ceiling and include page-cache hits"; the same setups' cold `F_NOCACHE` rates are 5.5 GB/s. — [llms-explorer](https://llms-explorer.com/tree/mac-local-llms-moe-streaming-and-offload)

### Inferences
- "Double buffering 2.09×" measures wall-clock reduction from overlapping two independent pipelines, not increased media bandwidth. The bytes still cross the SSD at ~2.97–3.36 GB/s. The 100%-hit row being no faster (4.46 vs 4.45) confirms the pipeline saturates compute, not SSD.
- This is already inside Veystrix's plan (overlap method #6) and is bit-exact if the second buffer is the *router-selected* next-layer experts. It is a real tok/s win via latency hiding — just not a bandwidth wall-breaker and not a "second miracle".
- Bit-exactness: automatic if the overlapped reads are the actual selected experts; a predicted-superset variant risks token divergence (see Q3 and the IO-08 result in the prior note).

### Gaps
- The #27149 author did not publish the raw pipeline benchmark methodology (chunk size, `F_NOCACHE` usage, whether the 6.06 GB/s numerator included cache hits), so the exact decomposition cannot be audited beyond the self-reported 2.97 GB/s baseline.
- No reproduction of the 2.09× pipeline speedup on an M4 Air with GPU compute, which is our actual workload.

---

## Q3. F_RDADVISE / F_RDADVISEV correctly structured — can it push sustained cold decode above 3.36 GB/s?

### Takeaway
It can overlap page-in with compute and occasionally show 3.61–3.78 GB/s in a probe, but on a memory-constrained 16 GB machine with GPU+SSD sharing the memory controller it is *unreliable and often harmful*: multiple independent 2026 implementations disabled it in production. It does not raise the physical wall; it is a latency-hiding hint. The earlier EPERM is not fundamental (EPERM is not a documented RDADVISE error); correct struct and regular-file fd are required.

### Cited Findings
- Man page: `struct radvisory { off_t ra_offset; int ra_count; }`; `F_RDADVISEV` uses `struct radvisoryv { unsigned int rav_flags; unsigned int rav_count; struct radvisory *rav_ranges; }`, `rav_count` 1..64. Documented errors are EBADF/EFBIG/EINVAL — **not EPERM**. — [Xcode fcntl(2) man page](https://keith.github.io/xcode-man-pages/fcntl.2.html)
- Bun (oven-sh/bun, 2026) uses `fcntl(F_RDADVISE)` on macOS arm64 for cold-start bytecode: issue returns in 0.5 ms, pages resident ~5 ms later; 40k-function app 156→125 ms. `MADV_WILLNEED` is synchronous on Darwin (11 ms/24 MB) so useless for overlap. Over-prefetching competed with the faults actually waited on and regressed first render. — [Bun commit](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca)
- PostgreSQL added an `F_RDADVISE` prefetch shim for macOS (Aug 2024). — [pgsql-hackers](https://www.postgresql.org/message-id/0827edec-1317-4917-a186-035eb1e3241d%40eisentraut.org)
- Our project probe: untimed ahead advice reached 3.61–3.78 GB/s; "simple hints are rejected; RDADVISE is evaluated separately" and never tested in real decode. — [Mference experiment log](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md); [prior note](os_io_tricks.md is at ../MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)
- Mference OPTIMIZATION_JOURNEY: "F_RDADVISE through fcntl was a particularly convincing false start. In one paired median, advice reduced I/O from 87.4 to 72.2 ms/token, throughput 5.176→5.449 tok/s. One 1,536-token probe fell from 5.687 to 4.028 tok/s. Later repeats did not reproduce the collapse, and some variants won again. No tested policy could reliably identify when the hint would help. Production keeps it off." — [Mference](https://raw.githubusercontent.com/NeelM0906/Mference/main/docs/OPTIMIZATION_JOURNEY.md)
- TurboFieldfare (Gemma 4 26B-A4B, M-series): "短期测试确实好看：解码 5.176 → 5.449 tok/s。但长上下文门禁中，无限制的 advise 直接把速度打到 2.334 tok/s" — production keeps RDADVISE **off**. — [GitCode blog](https://blog.gitcode.com/2b915a7ca79c5413c0cff11f04b42e9d.html)
- Anemll/Flash-MoE exploration (M5 Max): "Read hints fail: `MADV_RANDOM` harmful, `F_RDADVISE` −8%/−4%, `F_RDAHEAD` 0%." — [Anemll io-and-gpu-exploration](https://raw.githubusercontent.com/Anemll/flash-moe/m5-nax/docs/io-and-gpu-exploration.md)
- Flash-MoE's 58-experiment log: "使用 F_RDADVISE 预读 | SSD DMA 和 GPU 共用内存控制器——并发访问会导致 GPU 速度降低 73%" (F_RDADVISE prefetch: SSD DMA and GPU share the memory controller, concurrent access slows GPU by 73%). — [MCP Field skill summary of Flash-MoE](https://mcpfield.com/skills/flash-moe-inference)
- Contrasting positive result on a *large-memory* machine: tkhr-sait/ds4 on M4 Max 128 GiB found F_RDADVISE "the single biggest lever" — prefill 111.9→182–222 t/s (1.6–2.0×), CPU-gen decode 5.4→~10 tok/s; "On Darwin posix_madvise(WILLNEED) is only a vm hint and issues no I/O; F_RDADVISE issues a real read-ahead, so the SSD page-in overlaps compute." — [ds4 issue #172](https://github.com/antirez/ds4/issues/172)
- F_RDADVISE issues real read-ahead; `posix_madvise`/`MADV_WILLNEED` is a no-op on XNU for file pages. — [ds4 issue #172](https://github.com/antirez/ds4/issues/172); [Anemll](https://raw.githubusercontent.com/Anemll/flash-moe/m5-nax/docs/io-and-gpu-exploration.md)
- A prior project experiment (IO-08) found speculative reads caused emitted token IDs to diverge and prefill wall time to rise 82.5→123.6 s — failed speed and parity. — [prior note](../MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)

### Inferences
- The fix for EPERM is mechanical: pass a pointer to `struct radvisory`, on a freshly opened regular file, not an mmap-only/anonymous fd. EPERM is not a documented RDADVISE error, so the earlier failure was almost certainly struct/fd-type/sandbox — falsifiable in minutes.
- RDADVISE does **not** make the SSD deliver more cold bytes. Its value is a ~5 ms lead time per issued range so page-in overlaps compute; with `F_RDADVISEV` up to 64 ranges it fits top-k experts. On a 16 GB machine the overlap competes with the GPU for the shared memory controller (Flash-MoE's −73% GPU) and with the reads actually waited on (Bun's over-prefetch regression) — which explains the three independent production disables. The M4 Max/128 GB success has memory headroom Veystrix lacks.
- The 3.61–3.78 GB/s probe is the only cold-ish number above E0's 3.36 GB/s, but it sits within plausible measurement variance and was never validated in real decode; it should not be treated as a new wall.
- Bit-exactness: automatic for advice-only prefetch (pages copied into the unified buffer cache; the subsequent `pread` returns identical bytes). Disqualified only if prefetched/predicted experts are *substituted* for router-selected ones — IO-08 proved that diverges tokens.
- Rank: **#1 for further work** (highest expected value, lowest cost) but with honest expectation = latency hiding, not bandwidth.

### Gaps
- No end-to-end M4 Air 16 GB decode result with correctly-structured RDADVISE — the decisive untested experiment.
- No measurement of `F_RDADVISEV` batching >1 range on this hardware.
- No quantification of the GPU-contention penalty (−73% on M3 Max Flash-MoE) on an M4 Air at Veystrix's GPU+SSD overlap ratio.
- No rule for how much lead time is needed at 15+ tok/s, where the 67 ms/token budget is below the raw SSD floor and only multi-token overlap can work.

---

## Q4. MTLIO / Metal fast resource loading — can cold `loadBytes` beat pread?

### Takeaway
MTLIO's >3 GB/s numbers are warm-cache only (13.1–13.3 GB/s in our probe; 5–7% of misses warm). No cold `loadBytes` measurement exists, and the closest evidence (GPU faulting cold file-backed buffers) is *slower* than a parallel `F_NOCACHE pread` pool. `MTLIOPriority.high` is a real latency knob but unmeasured for cold reads. It cannot break the physical wall.

### Cited Findings
- WWDC22 "Load resources faster with Metal 3": asynchronous `MTLIOCommandQueue`/`MTLIOFileHandle`, IO command buffers execute concurrently and complete out of order; `MTLIOPriority.high` gives lower latency and prioritisation; Apple's demo compares single-threaded `pread` vs fast resource loading on M1 Pro. — [Apple WWDC22](https://developer.apple.com/videos/play/wwdc2022/10104)
- Our project probe (IO-09): warm reads 13.1–13.3 GB/s but only 5.4–7.5% of misses warm; "the fast state was too rare to control token time" — rejected. — [Mference log](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- GPU-demand-paging of cold file-backed no-copy buffers ran at ~2.0 GB/s, "1.3× slower than an explicit pread pool (7.5 GB/s)"; `newBufferWithBytesNoCopy` on warm file-backed mmap gave 252–365 GB/s. — [M1 Max capability report (low-confidence, garbled mirror)](http://128.254.224.85/IYEoBaLV-mMf-rUplm-SvQ) — corroborated qualitatively by [llms-explorer](https://llms-explorer.com/tree/mac-local-llms-moe-streaming-and-offload)
- "Warm decode is serial: SSD DMA and GPU do not overlap"; app-level Metal LRU caches lost to the OS page cache (3.14→1.99 tok/s). — [llms-explorer](https://llms-explorer.com/tree/mac-local-llms-moe-streaming-and-offload)
- Apple docs: file handles must be created with the same `MTLDevice` as the IO queue/buffer; no entitlement requirement found. — [Apple resource loading docs](https://developer.apple.com/documentation/metal/resource-loading)

### Inferences
- MTLIO wins only when bytes are already in the unified buffer cache. In a cold-miss streaming decode it cannot beat `pread` and adds complexity. Its one untested angle — cold `loadBytes` into CPU memory as a better-coalesced staging read — has no supporting measurement.
- `MTLIOPriority.high` could reduce queueing delay on cold reads but has no published cold-latency number; a 20-line probe would settle it.
- Bit-exactness: automatic (raw byte copy into MTLBuffer/CPU memory).
- Rank: **#2** — worth a cold `loadBytes` + high-priority probe, but expect no wall-breaking gain.

### Gaps
- No cold-miss `loadBytes` vs parallel `pread` measurement on this machine.
- No published `MTLIOPriority.high` cold-read latency measurement on Apple Silicon.
- No evidence whether MLX kernels can consume an externally-created `MTLBuffer` zero-copy to remove one memcpy per token (that would cut CPU overhead, not SSD bytes).

---

## Q5. kqueue / dispatch / io_uring equivalents / O_DIRECT / MADV_WILLNEED / POSIX_FADV / mmap / huge pages

### Takeaway
macOS has no io_uring and no O_DIRECT; kqueue's `EVFILT_READ` is unusable for regular files (always ready, EPERM without a thread pool), and GCD `dispatch_io` is a thread-pool wrapper that adds overhead. `MADV_WILLNEED` is synchronous on Darwin; `posix_fadvise` does not exist; `mmap` cold is ~3.5–5× slower than `pread`; huge pages are not exposed for file-backed data. None changes the wall. The one new nuance: the fastest "async" path on macOS is simply a pthread pool of `F_NOCACHE preads`.

### Cited Findings
- a10: "For the Apple family (macOS, iOS, etc.) this uses kqueue, which is a poll based API." — [a10](https://github.com/Thomasdezeeuw/a10)
- `ultar/dataloader.zig`: "kqueue (macOS) dispatches regular-file I/O to a thread pool; without one every read returns EPERM. io_uring (Linux) handles file I/O in-kernel." — [ultar](https://github.com/meshy-dev/ultar/blob/main/dataloader.zig)
- Apple kqueue man: `EVFILT_READ` is a readiness API; regular files are always ready. — [Apple man page](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/kqueue.2.html)
- M1 Max measurement reports: "The fastest async path is a plain pthread pool doing F_NOCACHE preads — measured 7.6–7.7 GB/s at 4–8 threads, which saturates the device; dispatch_io adds nothing over that but loses control." Low-confidence source (garbled mirror), consistent with the Apple-Silicon-Quest and llms-explorer material. — [mirror](http://128.254.224.85/dADXjzEnUxTaxthHKndqzw)
- `MADV_WILLNEED` synchronous on Darwin (Bun), `posix_fadvise` absent (fio shim maps to `F_RDAHEAD`/`F_RDADVISE`). — [Bun](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca); [fio](https://github.com/axboe/fio/commit/407491a84e257a60a09209c624014757b816ee8e)
- Project: cold mmap 9.88 ms vs pread 2.79 ms (~3.5×); whole-pool mmap, mlock, compression, speculative reads rejected. — [Mference log](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- Flash-MoE: "mmap() 专家文件 | 每页缺页开销：比 pread() 慢 5 倍"; "dispatch_io | dispatch_data 管理开销：−70%". — [MCP Field skill summary](https://mcpfield.com/skills/flash-moe-inference)
- Darwin `madvise`/`mmap` man pages list no huge-page advice flag for file-backed data; Apple Silicon's 16 KiB pages already reduce TLB pressure. — [prior note](../MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)

### Inferences
- This entire family changes *which thread blocks*, not how fast the device reads. Our 8-thread `pread` pool is already the known-best macOS async path.
- `F_NOCACHE` on a regular-file fd is the right primitive and is already in E0. Adding `O_DIRECT` emulation or GCD layers cannot lower the 3.36 GB/s floor.
- Bit-exactness: automatic for all read-only paths.
- Rank: **#5** — no predicted benefit; do not re-litigate.

### Gaps
- No file-backed huge-page control exposed to userspace on macOS (gap in prior note too).
- POSiX aio limits (`kern.aiomax` ~90, few kernel threads) are from low-confidence mirrors; not independently verified.

---

## Q6. NVMe queue depth / APFS tuning / striping partitions of the one SSD / raw device / file-cache control

### Takeaway
Queue depth is already saturated at 8 threads. Striping *partitions of the same physical SSD* cannot raise media bandwidth; APFS has no user-facing read-path knobs and documented global-lock contention under parallel metadata I/O (not our few-large-file pattern). `F_NOCACHE` already bypasses the unified buffer cache, which is the only file-cache control that matters. Random reads at QD≥4 cost nothing vs sequential, so on-disk layout is free — but that does not add bandwidth.

### Cited Findings
- "Four parallel pread threads gave 9.2× over sequential reads, which the authors call superlinear due to NVMe command queuing"; but on the M4 Air "more I/O workers = flat — SSD saturated". — [llms-explorer](https://llms-explorer.com/downloads/concepts/moe-expert-ssd-streaming-and-page-cache-residenc.md); [project context (internal_engine_map.md)](../Apple%20Neural%20Engine%20integration/internal_engine_map.md)
- Random reads at QD≥3–4 equal sequential throughput on Apple NVMe ("chunked random expert fetch costs nothing vs streaming — layout freedom"). — [low-confidence M1 Max report](http://128.254.224.85/WP_DZbApgGV6LHwuXWomqA)
- `F_NOCACHE` bypasses the UBC for 4 KiB-aligned transfers; unaligned head/tail silently go through cache. No O_DIRECT alignment constraint on the buffer. — [low-confidence M1 Max report](http://128.254.224.85/IYEoBaLV-mMf-rUplm-SvQ)
- APFS global-lock analysis: parallel read-only `readdir()` hits a global kernel lock; Mojave improved but stayed far behind ext4. This is metadata, not bulk `pread`. — [Gregory Szorc](https://gregoryszorc.com/blog/2018/10/29/global-kernel-locks-in-apfs/)
- APFS vs HFS+ bulk throughput "within a few percentage points". — [OWC](https://eshop.macsales.com/blog/42901-the-speed-of-apfs-just-how-fast-is-it/)
- On M5 Max, internal AP2048Z read ceiling ~13.0–13.7 GB/s at 4–8 threads; a second TB5 SSD adds ~6.4 GB/s with no loss to either — but that requires hardware and a TB link, out of scope. — [Chida82/sf-ds4-1flash](https://github.com/Chida82/sf-ds4-1flash/blob/main/docs/ssd.md)
- Prior note conclusion: striping internal+TB4 external ≈1.5 GB/s external → ≤1.5× aggregate, not the 2× needed, and requires new hardware. — [prior note](../MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)

### Inferences
- With E0 already saturated at 8 threads, queue-depth tuning is exhausted. Partition striping on one device only adds scheduling overhead.
- APFS raw-device access is not exposed to normal processes and would not change media bandwidth.
- `F_NOCACHE` (already used) is the decisive cache control; `purge`/`vm_purgeable` cannot give a per-process file-cache policy that beats the wall.
- Bit-exactness: automatic (filesystem returns identical bytes).
- Rank: **#5** — no software gain available.

### Gaps
- No benchmark of internal+TB4 external as one APFS RAID0 on an M4 Air for large sequential reads (and it is out of scope: hardware).
- No user-facing knob found for NVMe queue depth or APFS read-ahead on macOS.

---

## Q7. RAM disk / memory compressor / purgeable memory within 16 GB

### Takeaway
Infeasible and irrelevant to the wall. A useful hot-expert RAM disk (multi-GB) would consume most of the 16 GB and starve the GPU; page pinning (`mlock`) recovered ~0 ms in our project because the problem is read traffic, not eviction. These controls change *which pages lose*, not bytes/s.

### Cited Findings
- Project IO-06: "Locked resident buffers... mlock recovered essentially 0 ms. Skipping expert streaming recovered about 12–14 ms. The mechanism was memory-system contention, not resident-page eviction." — [Mference log](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- macOS compresses memory under pressure; "Cooperating apps drop NSCache / purgeable VM. The file cache stays" — purgeable APIs do not control the file cache. — [mac-ram-cleaner](https://github.com/chumafox/mac-ram-cleaner)
- Apple XNU `memorystatus` documents pressure levels/notifications but no per-process compressed-memory policy. — [Apple open-source XNU doc](https://github.com/apple-open-source/macos/blob/master/xnu/doc/vm/memorystatus_notify.md)
- Quantized weights / entropy-packed data are incompressible, so the VM compressor wastes CPU or swaps to the same SSD (7.7× penalty in one report). — [low-confidence M1 Max report](http://128.254.224.85/dADXjzEnUxTaxthHKndqzw)

### Inferences
- RAM disk cannot hold a useful expert subset beside the model, KV cache and OS reserve on 16 GB; even if it could, the bytes must still be read from SSD once per token.
- Bit-exactness: automatic; only residency changes.
- Rank: **#6** — reject.

### Gaps
- No per-process `vm_compressor` knob found; no published macOS memory-compressor experiment for ML inference.

---

## Q8. Increasing useful bytes/sec: layout, alignment, read size, read amplification

### Takeaway
This is where real, bit-exact, software-only gains remain — by reducing *wasted* bytes and kernel overhead rather than raising the device ceiling. The best-supported new lever is 2 MiB-aligned destination buffers (16.8 GB/s vs 4.7 GB/s warm on M3 Max, ~5% in a mostly-cold pipeline). Expert-contiguous re-layout cut page faults 36× and produced 1.55× end-to-end in one study; our "expert packing" was rejected at the server level, so treat layout as workload-specific. None breaks 3.36 GB/s; all improve useful throughput.

### Cited Findings
- Flash-MoE `fs_usage`: 45,414 `pread` calls produced 260,845 disk read operations (5.7 per pread); a 3.9 MB pread fragments into mostly 512 KB + 8–16 KB pieces because UBC pages map to non-contiguous physical pages; ~46 µs kernel overhead per pread (~11 ms/token at 240 preads). — [llms-explorer](https://llms-explorer.com/downloads/concepts/moe-expert-ssd-streaming-and-page-cache-residenc.md)
- 2 MiB-aligned `posix_memalign` buffers read at **16.8 GB/s** (234 µs) vs 4.7 GB/s (836 µs) for 16 KiB-aligned Metal buffers on warm data; the 3.6× gap shrank to ~5% in the full pipeline because most reads were cold. "posix_memalign plus newBufferWithBytesNoCopy is free to adopt." — [llms-explorer](https://llms-explorer.com/downloads/concepts/moe-expert-ssd-streaming-and-page-cache-residenc.md)
- TinyGiant: interleaved GGUF touches 6,912 pages per expert (111× amplification); contiguous re-layout = 192 pages (36× reduction); "SSD: 3.0+ GB/s from contiguous layout". — [llama.cpp #27149](https://github.com/ggml-org/llama.cpp/discussions/27149)
- "Layout: mbolt reorder+interleave **1.55×** end-to-end under explicit reads; Flash-MoE clustering 0%." — [llms-explorer](https://llms-explorer.com/tree/mac-local-llms-moe-streaming-and-offload)
- Lossless expert compression (ZipMoE/SSD-LLaMA) cuts SSD traffic ~30% with bit-identical reconstructed weights, targeting ~3.5 GB/s UMA devices — but alone cannot reach Nemotron 15 tok/s. — [prior note](../MoE%20decode%20IO%20bypass%20research/expert_reuse.md)
- Our project already rejected expert packing/coalescing and gap-merged pread (TTFT only). — [project context](../Apple%20Neural%20Engine%20integration/internal_engine_map.md)

### Inferences
- E0's 8×4 MiB `F_NOCACHE` reads already avoid UBC fragmentation and use large chunks, so the low-hanging fruit is partly taken. The untested lever is **destination-buffer alignment** (2 MiB `posix_memalign`) — cheap, bit-exact, and the only new "useful bytes/sec" idea with a large measured effect.
- Layout effects are model/engine-specific (1.55× vs 0% in two studies; our packing rejected), so they need a falsifiable A/B rather than assumption.
- Bit-exactness: automatic — all are read/layout/alignment changes.
- Rank: **#3** (alignment first, layout second) — improves effective throughput, not the physical wall.

### Gaps
- No measurement of 2 MiB-aligned buffers on the M4 Air with 4 MiB `F_NOCACHE` chunks specifically.
- No test of whether reducing the 5.7× pread fragmentation further (beyond E0's 4 MiB chunks) helps in the actual decode path.

---

## Q9. Overall ranked verdict and falsifiable tests

### Takeaway
No software-only mechanism raises the M4 Air's physical cold-read ceiling above ~3.36–3.4 GB/s. The "second miracle" (2× to 6.7 GB/s) does not exist without new hardware. Ranked by value for the *tok/s* goal: RDADVISE (latency hiding, unreliable on 16 GB), MTLIO cold/high-priority probe, buffer alignment, overlap/double-buffering (already planned), byte reduction via compression, and everything else negligible. All are bit-exact if reads only.

### Cited Findings / Ranked table
| Rank | Mechanism | Realistic GB/s on THIS hardware | Wall-breaker? | Bit-exact? | Falsifiable test | Source |
|---|---|---|---|---|---|---|
| — | Physical device ceiling (`F_NOCACHE`, 8 threads) | **3.0–3.4** (E0 = 3.36) | n/a — this IS the wall | n/a | already measured on 113 GB file | [GIGAZINE](https://gigazine.net/gsc_news/en/20250317-apple-m4-macbook-air-benchmark); [swap cliff](https://www.creativeworksofknowledge.com/en/cwk-quests/apple-silicon-quest/memory/swap-cliff) |
| 1 | `F_RDADVISE`/`F_RDADVISEV` (advice-only) | ~3.6 probe; overlapped, not higher device rate | No | Yes (advice only) | correct struct + regular fd; issue <1 ms, residency <10 ms; then paired 200-token decode with token-ID parity | [Bun](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca); [Mference](https://raw.githubusercontent.com/NeelM0906/Mference/main/docs/OPTIMIZATION_JOURNEY.md) |
| 2 | MTLIO cold `loadBytes` / `MTLIOPriority.high` | warm 13.1; cold ≤ pread (~3) | No | Yes | 200-token cold decode, both paths, + high-priority arm, token parity | [WWDC22](https://developer.apple.com/videos/play/wwdc2022/10104); [Mference](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md) |
| 3 | 2 MiB-aligned `posix_memalign` buffers | warm 16.8 vs 4.7 on M3 Max; ~5% cold | No | Yes | A/B 4 MiB `F_NOCACHE` reads with 4 KiB vs 2 MiB-aligned destination; measure GB/s + tok/s | [llms-explorer](https://llms-explorer.com/downloads/concepts/moe-expert-ssd-streaming-and-page-cache-residenc.md) |
| 4 | Overlap / double buffering (router-selected N+1) | apparent 6.06; device ~3 | No | Yes | already in plan; gate on token parity | [llama.cpp #27149](https://github.com/ggml-org/llama.cpp/discussions/27149) |
| 5 | Lossless expert compression (~30% fewer bytes) | ~30% less traffic | No (helps Qwen tier) | Yes | decompressed SHA-256 == original; measure bytes/token | [prior note](../MoE%20decode%20IO%20bypass%20research/expert_reuse.md) |
| 6 | Layout / expert-contiguous re-layout | 1.55× (one study); 0% (another) | No | Yes | A/B reorder on one layer, 200 tokens, check digests | [llms-explorer](https://llms-explorer.com/tree/mac-local-llms-moe-streaming-and-offload); [llama.cpp #27149](https://github.com/ggml-org/llama.cpp/discussions/27149) |
| 7 | kqueue/dispatch/O_DIRECT/MADV_WILLNEED/posix_fadvise/mmap/huge pages | ≤ pread | No | Yes | already rejected; do not re-run | [prior note](../MoE%20decode%20IO%20bypass%20research/os_io_tricks.md) |
| 8 | RAID0 partitions / APFS tuning / RAM disk / purgeable / mlock | ≤ pread or requires hardware | No | Yes | already rejected | [prior note](../MoE%20decode%20IO%20bypass%20research/os_io_tricks.md) |

### Concrete falsifiable tests, in order
1. **RDADVISE sanity (minutes):** `fcntl(fd, F_RDADVISE, &(struct radvisory){.ra_offset=off,.ra_count=1<<20})` on a freshly opened regular file; expect return 0, errno untouched. Falsifies "EPERM is fundamental".
2. **RDADVISE lead-time probe:** one 400 MB expert; measure issue time, residency time (1-byte `pread` at last page), bytes/s; target <1 ms / <10 ms / >3.5 GB/s.
3. **Decisive RDADVISE decode test:** advice-only for router-selected layer L+1 experts via `F_RDADVISEV` ≤64 ranges; gate on tok/s ≥ baseline, bit-identical token IDs, no prefill regression. This is the gap the project left open.
4. **MTLIO cold `loadBytes` + `MTLIOPriority.high`** vs parallel `pread`, 200 tokens, token parity.
5. **Alignment A/B:** 4 KiB vs 2 MiB-aligned destination buffers for 4 MiB `F_NOCACHE` reads; measure GB/s and tok/s.
6. **Over-prefetch contention sweep:** vary RDADVISE breadth 1×/2×/4×; any arm changing token IDs is rejected regardless of speed.

### Inferences
- For the actual goal: Qwen 5+ tok/s needs ~672 MB/token at 3.36 GB/s; more *effective* bandwidth helps Qwen reach the 5 tier via overlap (rank 1/4) and byte reduction (rank 5), but the physical ceiling stays. Nemotron 15+ tok/s (67 ms/token) is below the raw SSD floor and cannot be reached by any of these on 16 GB — consistent with the main report's honest conclusion.
- "Second miracle" verdict: **not found.** The only 2× numbers are overlap/compute-hiding or cache-served; no software-only path makes the single M4 Air SSD deliver >~3.4 GB/s of cold bytes.

### Gaps
- No end-to-end M4 Air 16 GB test of any Q3/Q4 mechanism at the 5/15 tok/s targets.
- No source confirms whether GPU+SSD simultaneous load lowers the effective cold ceiling below 3.36 GB/s (the opposite direction from a miracle).
- Low-confidence garbled mirrors were used only where corroborated by the llms-explorer aggregator or project notes; treat their exact GB/s as indicative, not authoritative.
