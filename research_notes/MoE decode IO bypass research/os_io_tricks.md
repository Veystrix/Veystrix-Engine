# macOS / Apple Silicon OS & SSD I/O-Path and Memory-Subsystem Tricks for Hiding Expert-Weight Read Latency (M4, 16 GB, single ~3 GB/s NVMe)

Scope: novel I/O paths and memory-subsystem controls on macOS/Apple Silicon that could beat or hide the ~3 GB/s effective SSD wall (Qwen 408 MB/token, Nemotron 436 MB/token → ~130–145 ms/token floor). NO quality loss. Background mechanisms already rejected on this machine (F_NOCACHE, F_RDAHEAD=0, wired-limit sysctls, more I/O workers, O_DIRECT pager, gap-merged pread, expert packing) are not re-litigated except where new evidence changes the verdict.

Important source note: a project-adjacent experiment log (`NeelM0906/Mference`, "Model installation and expert I/O") mirrors this exact experiment series and is used below as near-primary evidence for measurements on the same class of hardware. It is a third-party repo, not an independent academic benchmark, so treat its numbers as project-internal measurements. — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)

---

## Q1. Is there an io_uring equivalent on macOS that can issue file reads with lower latency/overhead than `pread` (kqueue, GCD/dispatch, NECP, MTLIO)?

### Takeaway
macOS has no io_uring; kqueue's `EVFILT_READ` on regular files degenerates into a blocking thread-pool read (and returns EPERM without a thread pool), and GCD `dispatch_io` is a kqueue/thread-pool wrapper. The only genuinely new async file-I/O path is Metal 3's `MTLIOCommandQueue` fast resource loading, which Apple explicitly demonstrates outperforming single-threaded `pread` and which reaches 13.1–13.3 GB/s on warm data in this project's own probe — but warm data was only 5.4–7.5% of observed misses, so it did not change token time.

### Cited Findings
- `a10` (cross-platform async I/O library): "For the BSD family ... and for the Apple family (macOS, iOS, etc.) this uses kqueue, which is a poll based API." — [Source](https://github.com/Thomasdezeeuw/a10)
- `ultar/dataloader.zig`: "kqueue (macOS) dispatches regular-file I/O to a thread pool; without one every read returns EPERM. io_uring (Linux) handles file I/O in-kernel, so we skip the pool there to avoid idle worker." — [Source](https://github.com/meshy-dev/ultar/blob/main/dataloader.zig)
- Apple kqueue man page: `EVFILT_READ` "Takes a file descriptor as the identifier, and returns whenever there is data available to read" — a readiness API, not a completion API; regular files are always "ready", so it cannot express an async file read by itself. — [Source](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/kqueue.2.html)
- Stack Overflow (IOCP on macOS): the asynchronous file APIs are `dispatch_read` / `dispatch_write`; "Like IOCP the asynchronous I/O functions in GCD signals at the completion of an I/O task, not when the file descriptor is ready like the raw kqueue API." — [Source](https://stackoverflow.com/questions/6288001/io-completion-ports-for-mac-os-x)
- WWDC22 "Load resources faster with Metal 3": fast resource loading is an **asynchronous** API; "the thread which issues the loads does not need to wait for the loads to finish. The load operations execute concurrently to better utilize the throughput of faster storage." IO command buffers and command queues execute concurrently and **complete out of order**. Apple's own demo compares single-threaded `pread` vs fast resource loading on an M1 Pro and shows reduced latency for sparse-tile streaming. API: `MTLIOFileHandle` + `MTLIOCommandQueue` + `load(buffer:...sourceHandle:sourceHandleOffset:)` / `loadBytes`. — [Source](https://developer.apple.com/videos/play/wwdc2022/10104)
- Same session: queues can be created with `MTLIOPriority.high`; "The storage system will ensure that high-priority IO requests have a lower latency and are prioritized over other requests." Buffers can be cancelled mid-execution (`tryCancel`). — [Source](https://developer.apple.com/videos/play/wwdc2022/10104)
- Apple docs: `MTLIOCommandQueue` is "A command queue that schedules input/output commands for reading files in the file system, and writing to GPU resources and memory." — [Source](https://developer.apple.com/documentation/metal/mtliocommandqueue)
- Apple docs: file handles must be created with the same `MTLDevice` that created the IO queue/buffer. — [Source](https://developer.apple.com/documentation/metal/resource-loading)
- Project log, IO-09 (MTLIO): "Warm reads reached 13.1-13.3 GB/s, but only 5.4-7.5% of observed misses were fully warm." Final disposition: **Rejected for the runtime** — "The fast state was too rare to control token time." — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- Project log, IO-04/IO-05: a dedicated expert-I/O executor (best 8.59 ms vs existing 8.42 ms) and a custom four-worker pool (mixed 5.964–6.132 vs 6.093–6.022 tok/s) both failed beat-the-existing-concurrency gates; worker-count changes did not move profile I/O (~71.3 ms/token). — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)

### Inferences
- There is no kernel-bypass async file-read primitive exposed to ordinary macOS processes. kqueue/dispatch cannot reduce the bytes-from-SSD time; at best they change which thread blocks. This is consistent with the already-rejected "more I/O workers" result.
- MTLIO is the one path with demonstrated >3 GB/s effective throughput (13.1–13.3 GB/s warm), i.e. it can beat the wall **when data is already cached**. Its failure mode is state distribution, not the API: in a streaming MoE decode the weights are cold misses 92.5–94.6% of the time. Revisiting MTLIO only makes sense if the working set can first be made warm (see Q4/Q6), or if `loadBytes` into CPU memory is used as a better-coalesced staging read for cold misses — which the existing probe did **not** test.
- `MTLIOPriority.high` is a genuine latency-control knob (storage-layer prioritization) that has not been tried in the project. It is theoretically able to reduce queueing delay even for cold reads, but there is no published measurement of cold-read latency improvement on Apple Silicon.
- Bit-exactness: automatic. All these paths only move bytes into a CPU buffer or `MTLBuffer`; no arithmetic is changed. `loadBytes` to application-allocated memory and `load(buffer:)` to an MTLBuffer are raw copies. Risk is only if changing I/O scheduling changes which experts are selected (see Q2).

### Gaps
- No measured cold-miss numbers for `MTLIOCommandQueue` / `loadBytes` vs parallel `pread` on this machine. The project probe measured only warm MTLIO reads.
- No public measurement of `MTLIOPriority.high` reducing cold-read latency. Apple's WWDC demo shows priority effects qualitatively, not numbers.
- Whether Metal fast resource loading requires special entitlements or only works for game-like asset files: Apple's docs describe ordinary file handles; no entitlement requirement was found in sources. Needs a 20-line falsifiable test.
- No evidence for NECP or any network-stack-based path affecting local file I/O; no source found. Treat as non-viable.

---

## Q2. Can `F_RDADVISE` / `F_RDADVISEV` advisory prefetch (revisiting the earlier EPERM) or `MADV_WILLNEED` hide decode latency by overlapping SSD reads with compute?

### Takeaway
This is the strongest untested lead. On Darwin, `F_RDADVISE` is a true non-blocking async prefetch: it issues for 24 MB in ~0.5 ms and pages are resident ~5 ms later, and the project's own adv#ance probe reached 3.61–3.78 GB/s (above the ~3 GB/s wall). `MADV_WILLNEED` by contrast is **synchronous on Darwin** (~11 ms for 24 MB) and therefore cannot overlap I/O. The earlier EPERM is suspicious because EPERM is not among the documented errors for `F_RDADVISE`.

### Cited Findings
- Xcode `fcntl(2)` man page: `F_RDADVISE` = "Issue an advisory read async with no copy to user." `F_RDADVISEV` = "Issue multiple advisory reads async with no copy to user." Structures: `struct radvisory { off_t ra_offset; int ra_count; }` and `struct radvisoryv { unsigned int rav_flags; unsigned int rav_count; struct radvisory *rav_ranges; }`; `rav_count` must be 1..64. — [Source](https://keith.github.io/xcode-man-pages/fcntl.2.html)
- Same man page, documented errors for `F_RDADVISE`/`F_RDADVISEV`: **EBADF** ("fildes does not correspond to a valid regular file"), **EFBIG** (ra_offset exceeds file size), **EINVAL** (negative ra_offset/ra_count). **EPERM is documented only for `F_PUNCHHOLE`, `F_SETSIZE`, and `F_SETFL`** — not for RDADVISE. — [Source](https://keith.github.io/xcode-man-pages/fcntl.2.html)
- Bun commit (oven-sh/bun, "prefetch the startup bytecode on a cold start"): on macOS arm64 (16 KiB pages), "touching a 24 MB bytecode run sequentially costs ~1,500 I/Os (36 ms), while one read request for the same bytes completes in ~5 ms." They use `fcntl(F_RDADVISE)` on the executable for the backing file range (found via `getsegbyname("__BUN")` + slide). Explicitly: "**`MADV_WILLNEED` is synchronous on Darwin (11 ms for 24 MB), so it is no use for overlapping I/O.**" And: "**`F_RDADVISE` for 24 MB returns in 0.5 ms; the pages are resident ~5 ms later.**" Warm starts unaffected. — [Source](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca)
- Same Bun commit, measured cold-start gains with F_RDADVISE: 40,000-function app 156 ms → **125 ms** (off-variant 155 ms, warm 31–32 ms); 1,800-chunk `--help` 388 ms → **344 ms**; first UI render 636 ms → **596 ms**. Over-prefetching (62 MB) was rejected: `--help` improved to 264 ms but first render regressed (389 vs 367 ms) "because the large read competes with the faults startup is actually waiting on." — [Source](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca)
- PostgreSQL patch: "macOS doesn't have posix_fadvise(), but fcntl() with the F_RDADVISE command does the same thing." — [Source](https://postgresql.org/message-id/attachment/164300/v3-0001-Add-prefetching-support-on-macOS.patch)
- fio's macOS `posix_fadvise()` shim maps readahead control to `fcntl(fd, F_RDAHEAD, ...)`; confirms Darwin lacks POSIX_FADV flags as such. — [Source](https://github.com/axboe/fio/commit/407491a84e257a60a09209c624014757b816ee8e)
- Project log, IO-03 (Darwin file hints): "The first two [F_RDAHEAD=0, F_NOCACHE] were neutral. **Untimed ahead advice reached 3.61-3.78 GB/s** and justified a real-decode candidate." Disposition: "simple hints are rejected; **RDADVISE is evaluated separately**." — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- Project log, IO-08 (speculative expert reads): probes showed post-advice residency and "up to 10.63 GB/s"; but in the first end-to-end pair decode fell 4.937 → 4.742 tok/s, prefill wall time rose 82.50 → 123.64 s, **and emitted IDs diverged** — failed both speed and token-parity gates. Lesson recorded: "The probe measured page residency, not lead time, contention, or pipeline behavior." — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- Polars commit: "madvise_willneed performed the best on both MacOS on Apple Silicon and Ubuntu on x86-64, using PDS-H query 3 SF=10 after clearing file cache as a benchmark" (but see Bun's synchronous-on-Darwin finding — for a columnar scan the blocking fault avoidance can still win even if the call blocks). — [Source](https://github.com/pola-rs/polars/commit/e950592c1b220c0e60ce722ac116bc3ca4c70e48)

### Inferences
- **Why the prior F_RDADVISE EPERM is likely fixable:** EPERM is not a documented RDADVISE error, and a production runtime (Bun, 2026) uses it successfully on macOS arm64. Probable causes of the earlier failure: passing an `int` instead of a pointer to `struct radvisory`; using an fd that is not a *regular file* (EBADF, often misreported by wrappers as "operation not permitted"); calling it on an mmap-only/anonymous mapping; sandbox/entitlement restriction on the process; or confusing it with the kqueue regular-file thread-pool EPERM (Q1). Falsifiable in minutes: call `fcntl(fd, F_RDADVISE, &(struct radvisory){.ra_offset=off, .ra_count=count})` on a freshly opened regular file with a 1 MB range and print errno.
- Mechanism to beat the wall: RDADVISE is asynchronous and moves the read into the kernel/storage queue **without a user-space copy or a blocking thread**, giving ~5 ms lead time per issued range and allowing many ranges to be in flight. `F_RDADVISEV` can issue up to 64 ranges in one syscall — a natural fit for top-k experts across layers. The 3.61–3.78 GB/s project probe suggests the SSD can actually deliver above the naive 3 GB/s floor when the kernel issues the read as a large clean advisory rather than many pread calls; this is the only measured >wall number for **cold-ish** data in the project's notes.
- **Why it can hide latency:** issue RDADVISE for the next layer's likely expert ranges *while the current layer computes*, so the ~100–130 ms transfer overlaps ~compute time. Bit-exactness is automatic for prefetch-only: advice copies pages into the unified-buffer cache; it does not change the bytes subsequently read by `pread`/MTLIO.
- **Critical constraint from IO-08:** naive speculative reads caused token divergence. For NO quality loss, prefetch must be *advice-only* (never substitute an advised/predicted expert for the router-selected one) and must not introduce a contention path that changes scheduling/numerics. Any "prefetch a superset" variant must be tested for bit-identical token IDs, not just speed. Bun's over-prefetch regression shows that too much advice competes with the reads actually being waited on — the same mechanism likely explains IO-08's slowdown.
- `MADV_WILLNEED` should NOT be used on the decode critical path on Darwin: it blocks the calling thread (~11 ms/24 MB → ~190 ms for 436 MB), which is worse than the read it is meant to hide. It can still be used on a background thread as a blocking warm, but then it is equivalent to a dummy `pread` + discard with extra steps.
- Rank: **#1** for further work. Highest expected value, lowest implementation cost, strongest supporting evidence (documented API + production usage + project probe above wall).

### Gaps
- No end-to-end real-decode result for `F_RDADVISE` (the project explicitly deferred it). This is the single most important missing experiment.
- Exact cause of the earlier EPERM unknown without a reproduction; cannot assert it was a struct error vs sandbox vs fd type.
- No published numbers for `F_RDADVISEV` batching >1 range, or for RDADVISE on a 400+ MB expert blob (Bun measured 24 MB; project probe's blob size not stated in the fetched note).
- No measurement of how long RDADVISE-advised pages stay resident under a 16 GB memory guard / streaming pressure; "resident ~5 ms later" is an unloaded-system figure.
- No quantitative rule for how much lead time is needed to hide ~130 ms at 5+ tok/s (200 ms/token) vs 15+ tok/s (67 ms/token) — at 15+ tok/s the per-token budget is already below the SSD floor, so only overlap across *many* tokens in flight can work.

---

## Q3. Do `mmap` (shared vs private), huge pages, `MADV_DONTNEED`, or `POSIX_FADV_*` hints beat explicit reads for expert weights?

### Takeaway
No. On this hardware cold `mmap` was ~3.5× slower than `pread` (9.88 ms vs 2.79 ms), whole-pool mmap was rejected, and macOS lacks `posix_fadvise()` entirely. `MADV_WILLNEED` is synchronous on Darwin and `MADV_DONTNEED`/huge pages have no citable benefit for this workload; the only mmap-adjacent win is advisory prefetch of mapped ranges (Q2).

### Cited Findings
- Project log, IO-01 (mmap vs pread): "A cold expert read took 9.88 ms with mmap and 2.79 ms with pread, about a 3.5x difference. Warm mmap won its local comparison, but the expert working set exceeds usable cache on the target machine. The simulator reached 3.97 tok/s with parallel pread and 0.50 tok/s with mmap." Disposition: bounded parallel pread in production. — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- Project log header disposition: "Whole-pool mmap, mlock, compression, speculative reads, and MTLIO — Rejected." — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- Darwin `madvise(2)` man page lists `MADV_WILLNEED` ("expects to access this address range soon") and `MADV_DONTNEED`; no huge-page advice flag is documented. — [Source](https://leancrew.com/all-this/man/man2/madvise.html)
- macOS `mmap(2)` man page documents `MAP_JIT` only in the context of hardened-runtime W+X code with `com.apple.security.cs.allow-jit`; no `MAP_HUGETLB`-style flag for data. — [Source](https://leancrew.com/all-this/man/man2/mmap.html)
- PostgreSQL patch and fio shim confirm macOS has no `posix_fadvise()`; the surrogate is `fcntl(F_RDADVISE)` / `F_RDAHEAD`. — [Source](https://postgresql.org/message-id/attachment/164300/v3-0001-Add-prefetching-support-on-macOS.patch); [Source](https://github.com/axboe/fio/commit/407491a84e257a60a09209c624014757b816ee8e)
- Linux-side evidence for `MADV_WILLNEED` as a file-cache prefetch (force_page_cache_readahead) and for `MADV_SEQUENTIAL`/`MADV_WILLNEED` reducing a large sequential read from ~29 s to ~17 s / ~14 s in a diskdump benchmark — useful as mechanism background, but Darwin's WILLNEED is synchronous (Q2) so the overlap benefit does not transfer. — [Source](https://stackoverflow.com/questions/31215250/is-mmap-madvise-really-a-form-of-async-i-o); [Source](https://lists.crash-utility.osci.io/archives/list/devel@lists.crash-utility.osci.io/message/7QZV4RZCX2XVXUUJF3ER2SKKJZK3MMJD)
- `llama-cpp-expert-sniper` (third-party experiment): router calls `madvise(MADV_WILLNEED)` on each active expert's memory range to prefetch pages before compute. No benchmark numbers were present in the search snippet; treat as an idea, not evidence. — [Source](https://huggingface.co/waltgrace/llama-cpp-expert-sniper)

### Inferences
- `mmap` removes one copy but adds minor-fault overhead and loses control over read sizing/queueing; with a working set larger than usable cache, faults are the dominant cost. Already decisively rejected — do not repeat.
- Shared vs private mapping would not change the cold-fault cost; read-only private mappings would COW only on write (none here), so bit-exactness is moot. No source suggests a throughput difference for read-only expert blobs.
- Huge pages on macOS for anonymous/data mappings are not exposed as a general userspace API in the cited man pages; Apple Silicon's 16 KiB page size (Bun note) actually reduces TLB pressure versus 4 KiB. No citable mechanism to force 2 MB file-backed pages on macOS — gap.
- `MADV_DONTNEED` could theoretically free page-cache pages promptly to avoid the memory guard evicting the resident model, but the project showed the contention was not resident-page eviction (IO-06 recovered ~0 ms with mlock), so the premise is weak.
- Bit-exactness: any read-only mapping is bit-exact. Risk only if mapping lifetime/eviction races with GPU consumption.

### Gaps
- No test of `MADV_WILLNEED` issued from a background thread against mapped expert ranges (blocking but parallel to compute) on this machine. Low priority given Q2.
- No source found on file-backed huge pages or `vm_page` clustering controls exposed to userspace on Apple Silicon.

---

## Q4. Can page-cache pinning (`mlock`, `vmpage`), memory-compressor control, purgeable memory, or `memorystatus` pressure APIs protect the hot working set and reduce I/O stalls?

### Takeaway
No. The project's `mlock` experiment recovered essentially 0 ms; skipping expert streaming recovered 12–14 ms, proving the mechanism was memory-system **contention**, not resident-page eviction. Combined with the already-rejected wired-limit sysctls/MLX wired limit, memory-policy knobs are not a path to beating the SSD wall here.

### Cited Findings
- Project log, IO-06 (mlock and resident-model pages): "Locked resident buffers and a diagnostic run that skipped streaming. Evidence: mlock recovered essentially 0 ms. Skipping expert streaming recovered about 12-14 ms. What changed the conclusion: The mechanism was memory-system contention, not resident-page eviction." — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)
- Apple open-source XNU `memorystatus_notify.md`: documents pressure levels (`Normal`, `Warning`/`Urgent`, `Critical`) and `DISPATCH_MEMORYPRESSURE_*` notifications; cooperating apps drop `NSCache` / purgeable VM under pressure. — [Source](https://github.com/apple-open-source/macos/blob/master/xnu/doc/vm/memorystatus_notify.md)
- `mac-ram-cleaner` utility description: sending `memory_pressure -S -l warn -s 1` makes "Cooperating apps drop NSCache / purgeable VM. The file cache stays." — i.e. purgeable-memory APIs do not give direct control over the file cache. — [Source](https://github.com/chumafox/mac-ram-cleaner)
- `purge` (the classic "flush file cache" tool) is described as obsolete/churning: "Don't run `purge` from a launchd timer every minute. It's safe but it churns the disk cache; you'll lose the performance benefit of the cache existing." — [Source](https://github.com/daviddwlee84/dotfiles/blob/main/docs/tools/macos-swap.md)
- macOS compresses memory and swaps under pressure; user-facing guidance on `purge`/compressed memory is generic and offers no per-process control of compressed-page policy. — [Source](https://apple.stackexchange.com/questions/340333/tell-macos-to-only-use-swap-and-compressed-memory-when-really-needed-macos-10-1?rq=1)

### Inferences
- The OS's file cache and the unified-memory GPU allocation compete for the same 16 GB. Pinning more pages cannot create capacity; it only changes *which* pages lose. Since `mlock` gained ~0 ms and the gain came from removing streaming entirely, the bottleneck is the read traffic itself, not cache eviction of the 1.58 GB resident model.
- Any "protect the hot set" trick (purgeable markings, `memorystatus` observer that drops other caches, controlling the memory compressor) would at best reduce contention marginally and cannot change the ~130 ms of bytes that must cross the SSD bus. Rank: low.
- Bit-exactness: these controls do not alter byte values; only residency. Risk is indirect (if eviction forces re-reads during a kernel, that changes timing not values).

### Gaps
- No published experiment on macOS controlling the memory compressor (`vm_compressor`) for ML inference; no reliable source found for a per-process knob.
- No measurement of `vm_purgeable` / `NSPurgeableData` applied to expert blobs; unlikely to help for traffic that must be re-read each token.

---

## Q5. Is there GPU-direct storage / SSD→GPU DMA on Apple Silicon (MTLStorageModeManaged/shared, GPUDirect equivalents) that bypasses CPU staging?

### Takeaway
No public Apple API exposes direct SSD→GPU DMA. The closest supported primitive is Metal 3 fast resource loading (Q1), which writes directly into `MTLBuffer`/textures via the IO command queue but still goes through Apple's storage stack. Unified memory means CPU and GPU share DRAM pages, so "shared storage" removes a *copy*, not the SSD transfer. No measured evidence shows this beating the cold-read wall.

### Cited Findings
- WWDC22: fast resource loading writes directly into Metal textures/buffers via `load(texture:...)`/`load(buffer:...)` with `MTLIOFileHandle`, or to CPU memory via `loadBytes`; the API is explicitly built on "Apple silicon unified memory architecture and fast SSD storage." — [Source](https://developer.apple.com/videos/play/wwdc2022/10104)
- Research note on PyTorch MPS: "When PyTorch allocates an MPS tensor in shared Metal storage (Apple, 2026a), the GPU's MTLBuffer and the CPU's pointer are the same DRAM pages" — confirms zero-copy CPU/GPU sharing, not SSD/GPU direct DMA. — [Source](https://arxiv.org/html/2610.39029v1)
- GPUDirect Storage literature: GDS/GPUDirect RDMA "enable direct DMA but retain CPU orchestration"; research like GNoF proposes moving orchestration onto the GPU for NVMe-over-Fabrics. This is NVIDIA/CUDA ecosystem; no Apple equivalent was found. — [Source](https://arxiv.org/html/2611.21288v1); [Source](https://www.weka.io/learn/what-is-gpudirect-storage)
- `ssd-llm` for Apple Silicon uses "Metal Compute" kernels and keeps embeddings/early attention pinned in memory while streaming layers from SSD, but does not claim direct SSD→GPU DMA. — [Source](https://github.com/quantumnic/ssd-llm)
- Project log IO-09: MTLIO's 13.1–13.3 GB/s warm reads were "too rare to control token time"; no cold-DMA benefit was found. — [Source](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md)

### Inferences
- On Apple Silicon the "DMA" problem is different from discrete GPUs: there is no separate VRAM, so the only avoidable cost is the CPU bounce buffer. `MTLIOCommandQueue.load(buffer:)` already avoids that copy by having the storage system write the shared `MTLBuffer` directly. If MLX kernels can consume an externally-created `MTLBuffer` (zero-copy), loading experts straight into reuse pools could reduce one memcpy per token — but the bytes still traverse the SSD at ~3 GB/s cold, so this cannot break the floor; it can only improve the CPU-side overhead.
- Bit-exactness: automatic (raw bytes into a buffer).
- Rank: medium as an *overhead* optimization layered on top of Q1/Q2, not a wall-breaker. No falsifiable cold-read benefit is predicted.

### Gaps
- No source confirming any private/public Apple API for NVMe→GPU DMA or a Metal "GPUDirect Storage" equivalent. Explicitly: no reliable sources found.
- No measurement of end-to-end token/s from `load(buffer:)` directly into an MLX-consumable `MTLBuffer` vs `loadBytes` + `MLX` copy.

---

## Q6. Can hardware topology (RAM disk, NVMe striping/partitions, Thunderbolt second SSD, APFS tuning) raise the ~3 GB/s wall or add a second I/O lane?

### Takeaway
Only adding a real second device can raise aggregate bandwidth, and on a base M4 (16 GB, ~3 GB/s internal NVMe, Thunderbolt 4) the practical external RAID0 bandwidth is ~1.5 GB/s — not enough to double the wall, and it does not reduce the per-token bytes. RAM disk cannot hold the model (16 GB total) and APFS has no user-facing read-path tuning knobs; APFS has documented global-lock contention under parallel metadata I/O.

### Cited Findings
- Base M4 mini internal SSD ~3 GB/s; M4 Pro mini internal storage "more like 6-7 GB/sec" and TB5 (120 Gb/s) — the user's machine is the base M4, so neither applies. — [Source](https://reddit.com/r/macmini/comments/1hdl36i/heres_my_base_m4_mini_in_16_gb_425_tb); [Source](https://macstadium.com/blog/m4-mac-mini-highlights)
- AppleInsider review of a 40 Gbps Thunderbolt 4 dock/RAID on M4 mini: with dual displays, "RAID 0 array speeds slowed to about 1,500 megabytes per second read and write. This is the nature of a 40-gigabit connection" — i.e. external TB4 RAID0 ≈ 1.5 GB/s, not 3 GB/s. — [Source](https://appleinsider.com/articles/25/10/06/acasis-40gbps-mac-mini-m4-dock-review-all-the-ports-you-could-want-for-your-m4-mac-mini)
- Two-bay Thunderbolt SSD RAID0 measured ~800 MB/s in one user report; TB4 external bandwidth and APFS lock overhead are cited as limiters. — [Source](https://reddit.com/r/Thunderbolt/comments/1kchdrq/looking_for_a_decent_raid_enclosure_for_a_mac)
- APFS global-lock analysis: profiling showed `readdir()` → `apfs_vnop_readdir()` → btree → `lck_mtx_lock_grab_mutex()`; author concludes "APFS obtains a global kernel lock during read-only operations such as readdir(). In other words, APFS slows down when attempting to perform parallel read-only I/O." Mojave improved but remained far behind Linux/ext4. This is metadata/readdir evidence, **not** a measurement of bulk `pread` throughput. — [Source](https://gregoryszorc.com/blog/2018/10/29/global-kernel-locks-in-apfs/)
- macOS 10.13 readdir benchmark: 12 parallel walks 172 s (sys 23m42s) vs single process 523 s; 10.14 improved to 98 s; Linux/ext4 12 parallel walks 25 s. — [Source](https://gregoryszorc.com/blog/2018/10/29/global-kernel-locks-in-apfs/)
- APFS vs HFS+ bulk-throughput comparisons ("within a few percentage points") suggest little to gain from filesystem choice for sequential reads. — [Source](https://eshop.macsales.com/blog/42901-the-speed-of-apfs-just-how-fast-is-it/)
- RAM disk discussion is generic; with 16 GB unified memory shared by GPU and file cache, a 400+ MB/token streaming model leaves no headroom for a multi-GB RAM disk. — [Source](https://github.com/chumafox/mac-ram-cleaner)

### Inferences
- NVMe striping across *partitions of a single physical SSD* cannot increase media bandwidth and may hurt (extra I/O scheduling, no parallelism benefit). Already consistent with "SSD saturated" finding.
- A Thunderbolt second SSD could in principle double *aggregate* bandwidth if experts were striped across internal+external and both read in parallel, but at ~1.5 GB/s external the aggregate would be ~4.5 GB/s vs 3 GB/s — a ~1.5× ceiling, not the 2× needed, and per-token critical path is set by the slower lane unless every token splits its ~400 MB across both. Requires new hardware (out of scope for a software trick) and careful APFS/RAID0 setup.
- RAM disk: to hold a useful hot expert subset (say 8–16 GB of weights) would consume most/all of the 16 GB and starve the GPU; infeasible without capacity relief.
- APFS: no user-facing knob to tune read-ahead/allocation for expert blobs; the documented global-lock issue affects metadata-heavy parallelism (many small files), not the few-large-file streaming pattern here. Low priority.
- Bit-exactness: striping/RAID0 reorders blocks but the filesystem returns identical bytes; bit-exact. Risk only from hardware errors.
- Rank: low, because it requires hardware and offers at most ~1.5× aggregate gain while leaving the per-token critical path largely unchanged.

### Gaps
- No benchmark found for striping *internal + TB4 external* as a single APFS RAID0 on an M4 for large sequential reads; the ~1.5 GB/s figure is for two external drives on one TB4 bus, which is the more relevant constraint.
- No source on whether Apple's internal NVMe exposes multiple queues usable by two independent reader processes beyond what the already-saturated worker test covered.

---

## Q7. Ranked candidate list and falsifiable tests (synthesis for the report writer)

### Takeaway
Rank by (expected latency benefit × feasibility) given the evidence: (1) `F_RDADVISE`/`F_RDADVISEV` async advice revisited with correct struct and end-to-end decode test; (2) MTLIO with cold-miss/`loadBytes` and `MTLIOPriority.high` variants; (3) background-thread `MADV_WILLNEED` (low); (4) topology/RAID0 (hardware, ~1.5× ceiling); (5) all memory-purge/pin/huge-page/kqueue/dispatch variants (no predicted benefit). Speculative *use* of predicted experts is explicitly disqualified by the token-divergence result.

### Cited Findings
- Rank-1 evidence: [Bun F_RDADVISE async timing and cold-start gains](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca); [project RDADVISE probe 3.61–3.78 GB/s and deferral to real decode](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md); [man page struct and error list](https://keith.github.io/xcode-man-pages/fcntl.2.html).
- Rank-2 evidence: [WWDC22 MTLIO API + priority + pread comparison](https://developer.apple.com/videos/play/wwdc2022/10104); [project IO-09 warm 13.1–13.3 GB/s, 5.4–7.5% warm](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md).
- Disqualifying evidence for speculation: [IO-08 token divergence + slowdown](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md).
- Disqualifying evidence for memory pinning: [IO-06 mlock ~0 ms](https://github.com/NeelM0906/Mference/blob/main/docs/experiments/summaries/01-model-install-and-expert-io.md).

### Inferences
Concrete falsifiable tests, in priority order:
1. **RDADVISE syscall sanity (minutes):** open expert file as a regular file; `fcntl(fd, F_RDADVISE, &radvisory{offset,count=1MB})`; expect return 0 and errno untouched; then time a subsequent `pread` of the same range with a cache-clear between. Falsifies the "EPERM is fundamental" hypothesis immediately.
2. **RDADVISE lead-time probe:** for one 400 MB expert, issue `F_RDADVISE`, measure (a) issue time, (b) time until a 1-byte `pread` at the last page completes (residency), (c) bytes/s. Compare against the 3.61–3.78 GB/s project probe. Falsifiable target: issue <1 ms, residency <10 ms, sustained >3.5 GB/s.
3. **End-to-end RDADVISE decode (the decisive test):** issue advice for the *router-selected* experts of layer L+1 as soon as their IDs are known (advice-only, no change to which weights are read), using `F_RDADVISEV` batches up to 64. Gate on (a) tok/s ≥ baseline, (b) bit-identical output token IDs, (c) no increase in prefill wall time. This directly addresses the gap the project left open in IO-03.
4. **MTLIO cold `loadBytes` vs parallel `pread`:** 200-token decode, both paths, measure cold-miss ms/token and output parity; add a `MTLIOPriority.high` arm. Falsifies "MTLIO only helps warm data" only if a cold arm wins.
5. **Over-prefetch contention sweep:** vary RDADVISE prefetch breadth (1×, 2×, 4× top-k) and confirm the Bun/IO-08 over-prefetch regression boundary; any arm that changes token IDs is rejected regardless of speed.
6. **Topology (only if hardware available):** stripe internal + TB4 external as APFS RAID0; measure large sequential read GB/s and tok/s; expect ≤1.5× aggregate and no per-token critical-path win.

### Gaps
- No test has yet combined RDADVISE with MTLIO (kernel prefetch into page cache, then MTLIO `loadBytes` from cache) — this could theoretically combine the 3.6 GB/s cold-ish prefetch with the 13 GB/s warm load path and is the most interesting untried combination.
- No end-to-end measurement exists for any Q1–Q6 mechanism at the 15+ tok/s target, where the per-token budget (67 ms) is below the raw SSD floor; only multi-token overlap could possibly satisfy it.
