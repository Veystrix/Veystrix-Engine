# General methods to "split I/O time" / hide memory-storage I/O latency in inference — applicability to MoE SSD-offloaded decode on a 16 GB Apple M4 (no new hardware, no quality loss)

Scope note. This note is written for the report writer and is self-contained. "Split I/O time" is interpreted broadly: (a) dividing one token's read across time (pipelines/prefetch), (b) across resources (multiple queues, devices, tiers, DMA/copy engines), (c) across tokens/requests (batching, disaggregation), and (d) across deadlines (scheduling/time-slicing). The hard constraints are the target machine (16 GB base M4, one internal NVMe measured at 3.36 GB/s under load), batch-1 bit-exact decode, and NO new hardware. In-repo facts are cited with relative links from this file's location (`../../...` reaches the repo root); external facts with URLs. All "in-repo" measurements are on the exact target machine unless stated.

---

## KQ0. Framing — what the E0/E1 results imply for ANY "split I/O time" scheme

### Takeaway
E0 proved that GPU compute and SSD reads genuinely overlap on this M4 (3.36 GB/s physical read rate, only 10% decode-step inflation), and E1 proved that even a PERFECT prefetch oracle — which moves reads off the critical path with unchanged byte volume — yields only +24.6% (2.19→2.72 tok/s). Therefore every method in this note that merely *retimes, reorders, or overlaps the same bytes* is capped near 1.25× on this workload; only reducing bytes or adding bandwidth can go further, and both are outside "no new hardware / no quality loss" for this machine.

### Cited Findings
- E0 (concurrency proof): physical SSD read bandwidth under load **3.36 GB/s** with 8 reader threads × 4 MiB F_NOCACHE buffers over a 113 GB checkpoint; decode step 130.1→143.2 ms/token (**+10.08% inflation**, well under the 25% kill line); verdict **KEEP — overlap is real**. Note: a naive 64 MB-allocation attempt hit 24.79 GB/s but was a memcpy storm (discarded). — [in-repo: E0_CONCURRENCY_RESULT.md](../../docs/performance/E0_CONCURRENCY_RESULT.md)
- E1 (oracle pipeline, corrected): perfect one-token-lookahead prefetch of the REAL greedy route gives 2.19→**2.72 tok/s (+24.6%)**, −90 ms/token, but physical MB/token only 857.5 vs 873.1 (**−1.8%**) — "the win is overlap, not fewer bytes." Verdict **KILL vs the 4.5–5.5 tok/s gate**. — [in-repo: E1_ORACLE_RESULT.md](../../docs/performance/E1_ORACLE_RESULT.md)
- E1's own ceiling arithmetic: ~873 MB/token ÷ 3.36 GB/s ≈ **260 ms of drive time** vs ~200 ms of compute → **~3.8 tok/s absolute ceiling at this byte volume, and that requires a PERFECT predictor**. Offline zero-training predictors reach only ~0.36 recall (E3). — [in-repo: E1_ORACLE_RESULT.md](../../docs/performance/E1_ORACLE_RESULT.md)
- Independent framing: the decode budget is ~289/385 ms SSD wait (75%), 70 ms GPU, 51 ms router sync; Amdahl caps even *magically* removing all GPU+router work at 385/289 = **1.33×**. — [in-repo: reports/Apple Neural Engine integration.md](../../reports/Apple%20Neural%20Engine%20integration.md); another profile gives per-token 41% Python/`mx.eval` sync, 29% SSD, 27% GPU. — [in-repo: reports/Faster Qwen Nemotron inference on M4.md](../../reports/Faster%20Qwen%20Nemotron%20inference%20on%20M4.md)
- Rejected baselines that define "already tried": expert prefetch 2.39→2.38 tok/s; decode overlap 2.39→2.41 (within band); I/O workers 6/8/16/24/32 all 2.55–2.57 tok/s (SSD saturated); perfect predictor + overlap tied; early submit −8.2%. — [in-repo: DO_NOT_IMPLEMENT.md](../../docs/performance/DO_NOT_IMPLEMENT.md)

### Inferences
- The decisive question for any candidate is NOT "does it overlap I/O?" (E0 says overlap is already possible) but "does it change bytes/token or add bandwidth?" If neither, physics caps it at ~+25% (E1), usually much less once the rejected baseline is accounted for.
- "Splitting I/O time" on an oversubscribed single device is largely an accounting illusion: the bytes still traverse one 3.36 GB/s bus. The practical prize on this machine is removing the ~41% host/sync tax and shrinking bytes, not rescheduling SSD reads.
- Bit-exactness is automatic for pure read-scheduling/prefetch/stripe methods (only byte movement changes). It is NOT automatic for speculative *use* of predicted experts (changes which weights are read) or for fused GEMMs that change reduction order in prefill — both have failed local gates.

### Gaps
- No source gives a per-layer I/O-time vs compute-time breakdown for the exact Qwen/Nemotron decode, which is what would settle whether per-stage double buffering can ever have enough slack to hide a full stage on this machine.
- The E0/E1 numbers are for one checkpoint/model profile; generalization across the 200–870 MB/token range stated in the brief is by inference, not measurement.

---

## KQ1. Double buffering / ping-pong buffers / multi-stage pipelines (prefetch layer L+1 while computing L)

### Takeaway
This is the textbook meaning of "split I/O time," and it is well proven in batch/offload systems — but on this machine it is the closest match to the already-rejected layer-prefetch and decode-overlap+early-submit attempts, and E1's oracle establishes its absolute ceiling (+25%). The only defensible novel variant is a fine-grained, multi-command-buffer pipeline that removes the per-layer host sync and starts each expert's GEMM on its own bytes' arrival, without changing read-issue time.

### Cited Findings
- **Mechanism.** Two buffers A/B: compute layer L on GPU while asynchronously loading L+1's experts into the other buffer; swap at the layer boundary. Classic form documented in AirLLM issue #298 ("load layer N into buffer A; while executing prefetch layer N+1 into buffer B; swap"). — [AirLLM #298](https://github.com/lyogavin/airllm/issues/298)
- **When it works.** It hides I/O only when a stage's compute time ≥ the next stage's I/O time; if I/O > compute, bubbles remain. MoE-Lightning's CGOPipe chunks each layer's weights into pages and interleaves them with intermediate-result transfers to reach the throughput upper bound T = max(comm, T_cpu, T_gpu) (up to 10.3× batch throughput vs FlexGen/DeepSpeed, Mixtral 8×7B, single T4 16 GB — explicitly "we do not consider disk offloading"). — [arXiv 2411.11217](https://arxiv.org/html/2411.11217)
- **Quantified pipeline gain elsewhere.** Double-buffered next-layer reads reported at ~2.09× pipeline throughput for explicit-read MoE streaming; colibri on M4 Max submits resident-expert GPU work before missed-expert disk reads and gets ~1.4× end-to-end (CPU 0.30→Metal 0.42 tok/s, warm cache), token-exact decode but prefill GEMM can flip near-tie logits. — [llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149); [colibri metal.md](https://github.com/JustVugg/colibri/blob/v2.1.0/docs/metal.md)
- **Strongest novel variant (compute-on-arrival + one sync).** llama.cpp PR #31469 replaces 48 per-layer scheduler splits with ONE device sync; in-stream `KICK`/`WAIT` doorbell resolves top-k→slot on-device; `MOE_NOSPLIT` diagnostic measured ~8 ms/layer × 48 ≈ 340 ms/token of host-sync (2.32 vs 10.89 tok/s), and PR reports 1.47/1.63→16.29/22.03 tok/s, 100% byte-exact (576/576 identical greedy tokens). **Caution from same source:** later commits say the 20 tok/s figure used a boosted GPU, not reproducible cold (~2.2 tok/s), and after the sync fix the wall became kernel/node count and launch (7.6% SM, ~6930-node graph). — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469)
- **Why rejected here.** In-repo: layer-prefetch, decode overlap + early-submit (−8.2%), and perfect-predictor + overlap (tied) are all rejected; expert prefetch 2.39→2.38. — [in-repo: DO_NOT_IMPLEMENT.md](../../docs/performance/DO_NOT_IMPLEMENT.md); [in-repo: E1_ORACLE_RESULT.md](../../docs/performance/E1_ORACLE_RESULT.md)
- **Bit-exactness:** reads-only = fine; slot-remap is a permutation/gather (PR measured 0.000e+00 abs diff). Speculative *use* of predicted experts is disqualified (prior token divergence). — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469); [in-repo: os_io_tricks.md](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)

### Inferences
- **Per-method card.** Mechanism: 2-deep ping-pong read/compute pipeline with per-expert completion events, ideally with the single-sync slot-resolution redesign. Bit-exact: yes if bytes and GEMV math unchanged. Realistic gain: ≤ +25% (E1 oracle); realistically 0 to +10% because the plain form is already rejected and the SSD is saturated. M4 feasibility: plain form already implemented/rejected; single-sync + per-expert arrival dispatch is medium (needs native Metal/C++, MLX has no Python doorbell). Falsifiable test: hold read-issue time constant, vary only sync/dispatch granularity; count host syncs/token (expect 48→1 if the pathology exists), measure `io_await` overlap fraction and bit-exact digest over ≥200 tokens; drop if digest changes or no gain. — Sources as above.
- The early layers have zero overlap slack (PR #31469's "arrival-slack slot budgeting"), so even a correct pipeline is back-loaded; it cannot hide the first layers' reads.
- If Veystrix already does one host sync/step and already overlaps shared-expert compute (as the overlap_fusion note suggests may be partly true), this lever's headroom collapses toward zero — verify before building.

### Gaps
- No Apple-M4 measurement of a true multi-command-buffer, per-expert-event pipeline exists; colibri is M4 Max with 128 GB and warm data, PR #31469 is Linux/CUDA with a later-retracted headline.
- No public per-layer I/O-vs-compute time series for the exact target model, so "does a full stage fit in a compute window?" is unproven.

---

## KQ2. Asynchronous I/O primitives (io_uring, libaio, POSIX AIO, kqueue, IOCP, MTLIO, GPUDirect Storage)

### Takeaway
Async APIs buy two things: they keep the compute thread from blocking, and (with deep queues/O_DIRECT) they raise effective throughput by batching submissions. On macOS there is no io_uring and kqueue/GCD cannot shorten bytes-from-SSD time; the only genuinely new path is Metal 3 `MTLIOCommandQueue`, which is fast only on warm data (too rare here). Async submission cannot reduce bytes or add media bandwidth, so on this already-saturated SSD its gain is bounded by E1.

### Cited Findings
- **io_uring (Linux):** shared SQ/CQ rings; larger queue depth parallelizes requests; `IORING_OP_READ`, `io_uring_enter`, SQPOLL kernel thread avoid syscalls; "io_uring is Linux-specific." — [io_uring(7) man page](https://man7.org/linux/man-pages/man7/io_uring.7.html); [Oracle intro](https://blogs.oracle.com/linux/an-introduction-to-the-io-uring-asynchronous-io-framework); survey: [arXiv 2512.04859](https://arxiv.org/html/2512.04859v2)
- **What it can buy (Linux proof, non-transferable):** llama.cpp PR #31469's io_uring + O_DIRECT out-of-core pager + KICK/WAIT doorbell reports 1.47→16.29/22.03 tok/s, 208→12–35 MiB/tok, 100% byte-exact, 576/576 greedy tokens — but with the boosted-clock reproducibility caveat above. — [llama.cpp PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469)
- **libaio / POSIX AIO / kqueue / IOCP:** native libaio is O_DIRECT-only and narrow; POSIX AIO is a user-space thread pool; on BSD/macOS `a10` uses kqueue which is "poll based," and regular-file I/O is dispatched to a thread pool (returns EPERM without one); GCD `dispatch_read/write` are completion-based wrappers analogous to IOCP. — [a10](https://github.com/Thomasdezeeuw/a10); [ultar dataloader.zig](https://github.com/meshy-dev/ultar/blob/main/dataloader.zig); [Apple kqueue man page](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/kqueue.2.html); [Stack Overflow IOCP/macOS](https://stackoverflow.com/questions/6288001/io-completion-ports-for-mac-os-x)
- **MTLIO (the one new macOS path):** Metal 3 fast resource loading is asynchronous, command queues/buffers execute concurrently and complete out of order; `MTLIOFileHandle` + `MTLIOCommandQueue` + `load(buffer:...)` / `loadBytes`; `MTLIOPriority.high` gives storage-layer prioritization; Apple's demo beats single-threaded `pread`. — [WWDC22 session 10104](https://developer.apple.com/videos/play/wwdc2022/10104); [Apple MTLIOCommandQueue docs](https://developer.apple.com/documentation/metal/mtliocommandqueue); [Apple resource-loading docs](https://developer.apple.com/documentation/metal/resource-loading)
- **MTLIO's measured failure mode here:** warm reads 13.1–13.3 GB/s but only 5.4–7.5% of observed misses were warm → rejected as "too rare to control token time." No cold-miss `loadBytes` test was run. — [in-repo: os_io_tricks.md](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)
- **macOS-native async advice (most promising untested I/O lever):** `fcntl(F_RDADVISE)` is a true non-blocking async prefetch — Bun measured 24 MB issued in ~0.5 ms, resident ~5 ms later, and `MADV_WILLNEED` is synchronous on Darwin (~11 ms/24 MB); `F_RDADVISEV` batches up to 64 ranges. In-repo probe reached 3.61–3.78 GB/s with untimed ahead advice. — [Bun commit](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca); [Xcode fcntl(2)](https://keith.github.io/xcode-man-pages/fcntl.2.html); [in-repo: os_io_tricks.md](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)
- **GPUDirect Storage / DMA engines:** GDS and GPUDirect RDMA use a copy engine / DMA near the NIC or storage to move data directly into GPU memory "without burdening the CPU or GPU," skipping the CPU bounce buffer; GPUDirect Async lets the GPU trigger RDMA transfers to overlap I/O with compute. This is NVIDIA/PCIe ecosystem. — [NVIDIA GDS blog](https://developer.nvidia.com/blog/gpudirect-storage/); [CoPilotIO OSDI'26](https://www.usenix.org/system/files/osdi26-chen-guanyi.pdf); [GPUDirect Async](https://www.emergentmind.com/topics/gpudirect-async); [Lightweight async GPU-SSD integration, arXiv 2504.19365](https://arxiv.org/html/2504.19365v3)

### Inferences
- **Per-method card (async primitives as a class).** Mechanism: ring/queue-based submission decouples I/O from the compute thread; deep queues raise throughput. Bit-exact: yes (raw byte moves). Realistic gain on M4: ~0–25% ceiling; async alone cannot beat the byte wall. M4 feasibility: io_uring/libaio/IOCP/GDS **impossible (no API/hardware)**; MTLIO feasible but warm-state-limited; `F_RDADVISE/F_RDADVISEV` feasible and untested end-to-end. Falsifiable test: E2E advice-only RDADVISE decode — issue `F_RDADVISEV` for the router-selected L+1 ranges, gate on (a) tok/s ≥ baseline, (b) bit-identical token IDs, (c) no prefill regression; or MTLIO cold `loadBytes` vs parallel `pread` with a `MTLIOPriority.high` arm. — Sources as above.
- What async "buys" on this machine is mostly removing host blocking/sync — which overlaps with the compute-side tax in the other 41% profile, not with SSD media time. E0 already demonstrates 8-thread read concurrency saturating the drive, so substituting a fancier submission mechanism has little byte-level headroom.
- `F_RDADVISE` is advice-only (never substitutes predicted weights), so it is the one async variant that is cleanly bit-exact and not already killed by E1 — the main risk is over-prefetch contention (Bun's over-prefetch regression; in-repo IO-08 token divergence when predicted experts were actually used).

### Gaps
- No measured cold-read latency for `MTLIOCommandQueue` vs parallel `pread` on this machine; no published cold-latency number for `MTLIOPriority.high`.
- No end-to-end `F_RDADVISE` decode result exists (the project explicitly deferred it); cause of an earlier EPERM is unknown.
- No Apple API for kernel-bypass/GPUDirect-style storage DMA was found; treat as absent.

---

## KQ3. RAID / striping across multiple SSDs and across memory tiers (RAM, SSD, page cache)

### Takeaway
Splitting one logical read across devices raises aggregate bandwidth only if there really are multiple devices; on a single M4 NVMe, striping across partitions cannot help and striping to a Thunderbolt SSD requires new hardware and gives at most ~1.5× aggregate. Tiering (RAM/SSD/page cache) is software-only and bit-exact but capacity-starved on 16 GB; even perfect residency is bounded by E1's +25%.

### Cited Findings
- **Mechanism (devices):** RAID0 stripes blocks across N SSDs so one read is served in parallel; aggregate bandwidth ≈ sum of device bandwidths; reordering is transparent so bytes are bit-identical. — background: [NVIDIA GDS blog](https://developer.nvidia.com/blog/gpudirect-storage/) (DMA/copy-engine analogy)
- **The M4 bands:** internal base-M4 NVMe ~3 GB/s; a 40 Gbps TB4 RAID0 with dual displays measured **~1,500 MB/s** (the bus limit); a two-bay TB4 RAID0 measured ~800 MB/s in one report. Striping partitions of ONE SSD cannot raise media bandwidth. — [AppleInsider TB4 dock review](https://appleinsider.com/articles/25/10/06/acasis-40gbps-mac-mini-m4-dock-review-all-the-ports-you-could-want-for-your-m4-mac-mini); [Reddit TB RAID](https://reddit.com/r/Thunderbolt/comments/1kchdrq/looking_for_a_decent_raid_enclosure_for_a_mac); [in-repo: os_io_tricks.md Q6](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)
- **APFS caveat:** APFS has a documented global kernel lock on parallel metadata operations (`apfs_vnop_readdir` → btree → `lck_mtx_lock_grab_mutex`); 12 parallel readdir walks 172 s vs 98 s on 10.14 vs 25 s on Linux/ext4. This is metadata evidence, not bulk-`pread` throughput. — [Gregory Szorc](https://gregoryszorc.com/blog/2018/10/29/global-kernel-locks-in-apfs/)
- **Tiering mechanism:** FlexGen-style GPU/CPU/disk tiers and "pool-as-sole-cache + F_NOCACHE" reduce SSD bytes by serving hot experts from a faster tier; in-repo kept the exact-LRU slot cache (+31% on this model) as a structural byte-reduction lever, and E1 says full residency caps this path near 9 tok/s. — [FlexGen via CXL staging paper](https://dl.acm.org/doi/10.1145/3837053.3837356); [in-repo: E1_ORACLE_RESULT.md](../../docs/performance/E1_ORACLE_RESULT.md)
- **RAM disk infeasible:** 16 GB unified memory is shared by GPU and file cache; a useful multi-GB hot-expert RAM disk would starve the GPU. — [in-repo: os_io_tricks.md Q6](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)

### Inferences
- **Per-method card (multi-device striping).** Mechanism: RAID0 across internal+external SSD. Bit-exact: yes. Realistic gain: ≤ ~1.34× aggregate (3.36+1.5 ≈ 4.86 GB/s), and only per-token if every token splits its reads across both lanes. M4 feasibility: **IMPOSSIBLE — NEW HARDWARE** (no second SSD). Falsifiable test (only if hardware appears): stripe internal+TB4 external as one APFS RAID0, measure large sequential read GB/s and tok/s; expect ≤1.5× aggregate and no per-token critical-path win. — Sources as above.
- **Per-method card (memory-tier splitting).** Mechanism: hot experts in RAM/page cache, cold on SSD; software-only. Bit-exact: yes. Realistic gain: bounded by hit rate and E1 (+25% ceiling); 16 GB capacity prevents a useful hot set. M4 feasibility: feasible but capacity-starved. Falsifiable test: A/B a larger S3-FIFO pool vs native page cache on cold-start decode; measure SSD bytes/token and digest (larger pools already clamped at ~2.59 GiB in-repo). — Sources as above.
- "Splitting one read across RAM and SSD tiers" is really a caching problem, not an I/O-timing trick; it reduces bytes, which is the one thing that can beat E1's bound.

### Gaps
- No benchmark of internal+TB4 external as a single APFS RAID0 for large sequential reads on M4; the ~1.5 GB/s figure is two external drives on one bus.
- No source on whether the internal NVMe exposes multiple hardware queues usable by two independent readers beyond the already-saturated 8-thread test.

---

## KQ4. Concurrency / batching / pipeline parallelism / prefill-decode disaggregation (splitting I/O cost across tokens and users)

### Takeaway
These amortize fixed and I/O costs across many tokens in flight — the standard server answer — but the target workload is batch-1 local decode. Continuous batching is feasible in software and bit-exact for decode; pipeline/tensor parallelism and PD disaggregation need multiple devices/workers (impossible here) and in any case would only hide I/O if there are concurrent sequences whose compute can cover each other's reads.

### Cited Findings
- **Continuous batching & disaggregation mechanism:** continuous batching mixes prefill of new requests with decode of ongoing ones; chunked prefill trades TTFT for TPOT but cannot eliminate interference. DistServe disaggregates prefill and decode onto different instances so decode can accumulate a larger batch; decode is "heavily bandwidth-bound" so batching is key; uses queuing theory (M/D/1) for prefill instances. — [DistServe arXiv 2401.09670](https://arxiv.org/html/2401.09670v3)
- **PD disaggregation practice:** removes prefill-induced stalls of decode, improves tail latency; cross-worker KV transfer needs a high-speed interconnect (RDMA/InfiniBand/RoCE); "disaggregation isn't always a silver bullet — local prefill can be faster for short prompts / high prefix-cache hit." — [Modular inference handbook](https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation/); [SGLang PD+EP blog](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- **Batching hides part of expert-loading time:** SP-MoE notes "other works increase request batch size to accumulate computation time, which can hide a portion of expert loading time." — [SP-MoE arXiv 2510.10302v2](https://arxiv.org/html/2510.10302v2)
- **Batch-1 reality check:** ANE report states batch-1 decode never reaches alternative engines and the bottleneck is SSD wait; MoE offload energy paper shows increasing batch to 1024 amortizes expert loads but SSD energy stays 4.7–9.8× HBM. — [in-repo: ANE report](../../reports/Apple%20Neural%20Engine%20integration.md); [arXiv 2508.06978](https://arxiv.org/html/2508.06978v1)
- **Bit-exactness:** colibri's Metal decode is token-exact, but prefill GEMM's different accumulation order can flip near-tie logits (a `COLI_METAL_GEMM_MIN` switch keeps GEMMs on CPU for bit-exact prefill). — [colibri metal.md](https://github.com/JustVugg/colibri/blob/v2.1.0/docs/metal.md)
- **Amortization via hot-expert residency (not batching per se):** 1bit.gg pinning 75% of expert slots in cache reached 28–39 tok/s, 37–39 after "moving prefetch off decode path." — [1bit.gg](https://1bit.gg/moe-streaming.html)

### Inferences
- **Per-method card (continuous batching / multi-stream decode).** Mechanism: run several decode sequences concurrently so one sequence's CPU/GPU work covers another's SSD wait. Bit-exact: yes if per-token GEMV math is unchanged (decode is independent per token). Realistic gain: for batch-1 workload ≈ 0; for multi-request serving potentially large via amortization. M4 feasibility: feasible in software. Falsifiable test: run 2–4 concurrent decode streams over one pool, measure aggregate tok/s, bytes/token and per-stream digest; prediction — bytes/token unchanged, so no amortization benefit unless hot-expert overlap rises. — Sources as above.
- **Per-method card (pipeline/tensor parallelism, PD disaggregation).** Mechanism: split model or phases across workers; needs NVLink/RDMA-class interconnect and multiple instances. Bit-exact: yes in principle. Realistic gain: high for throughput, but not for a single-user local token. M4 feasibility: **IMPOSSIBLE — NEW HARDWARE** (no second Mac/GPU/interconnect). Falsifiable test: N/A on single machine. — Sources as above.
- Concurrency "splits I/O time" only in the sense of hiding one request's wait behind another's compute; with a single in-flight request there is nothing to hide behind, which is exactly why E0/E1 had to use synthetic readers/oracles.

### Gaps
- No measurement of multi-request continuous batching on the 16 GB M4 with this MoE; whether the SSD can serve concurrent streams without raising per-token latency is unknown.
- DistServe-class gains are datacenter numbers; no Apple-Silicon replication found.

---

## KQ5. Computational storage / near-data compute / processing-in-memory (PIM) / CXL memory

### Takeaway
These move computation to where the bytes live to avoid moving them — the only architectural route that truly beats a bus limit. They are uniformly datacenter/CXL research requiring programmable storage controllers or CXL-attached PNM devices; the M4 has neither, so all of this is **IMPOSSIBLE — NEW HARDWARE** and is included only for completeness of the "split I/O" taxonomy.

### Cited Findings
- **CXL-attached PNM for LLM:** "Scalable Processing-Near-Memory for 1M-Token LLM Inference" stores the full KV-cache in CXL memory and offloads page selection / attention to PNM accelerators to eliminate KV data transfers; related CXL-PNM (Park et al. 2024) and CENT (Gu et al. 2025) are GPU-free LLM systems. — [arXiv 2511.00321](https://arxiv.org/html/2511.00321v1)
- **CXL staging with on-device DMA:** "CXL Memory for LLM Inference Staging via an On-Device DMA Controller" keeps CXL only CPU-side and uses PCIe P2P for the GPU-facing path. — [ACM DL 10.1145/3837053.3837356](https://dl.acm.org/doi/10.1145/3837053.3837356)
- **CXL memory pooling** is positioned to augment GPU VRAM for KV cache and improve TTFT/token decode; KV-cache-as-storage-appliance and KVserver proposals combine CXL+PNM with CPU/SSD offload fallbacks. — [CXL Consortium blog](https://computeexpresslink.org/blog/overcoming-the-ai-memory-wall-how-cxl-memory-pooling-powers-the-next-leap-in-scalable-ai-computing-4267/); [SNIA session](https://www.snia.org/sniadeveloper/session/19665); [KVserver hotinfra26](https://www.cs.virginia.edu/~skadron/Papers/KVserver_hotinfra26.pdf)
- **In-storage/computational storage:** the broader field puts filters/aggregates in the SSD controller so fewer bytes cross the bus; no source found exposing such a programmable path on Apple Silicon. — general background; no Apple-specific source found (gap).

### Inferences
- **Per-method card (computational storage / PIM / CXL).** Mechanism: compute near data to eliminate transfers. Bit-exact: possible in principle but many PNM schemes use approximations; irrelevant to the gate here. Realistic gain on other hardware: potentially large (eliminates the dominant movement term). M4 feasibility: **IMPOSSIBLE — NEW HARDWARE** (no CXL, no programmable SSD controller, opaque NVMe). Falsifiable test: none possible on target. — Sources as above.
- The M4's unified memory already gives zero-copy CPU/GPU sharing, which is the one "near-data" property it has; it does not extend to the SSD.

### Gaps
- No Apple Silicon computational-storage or CXL API exists in any source found; explicitly no viable path.
- In-storage computing survey references could not be re-fetched in this session (search returned empty); the CXL papers above are the concrete anchor.

---

## KQ6. Prefetch lookahead, dedicated I/O cores, DMA engines, copy engines, RDMA

### Takeaway
Lookahead is already calibrated: a PERFECT oracle gives +25% (E1) and real predictors ~0.36 recall (E3), so it cannot reach the targets. Dedicated I/O cores/threads have been tried and failed to beat existing concurrency. DMA/copy engines exist only in the NVIDIA/RDMA ecosystem and have no M4-exposed equivalent; RDMA is a network technology for multi-node KV transfer.

### Cited Findings
- **Lookahead upper bound:** E1's perfect oracle = +24.6%; real zero-training predictors recall ~0.36; verdict DROP all prefetch-hint variants. — [in-repo: E1_ORACLE_RESULT.md](../../docs/performance/E1_ORACLE_RESULT.md)
- **Dedicated I/O workers/cores already saturated:** I/O workers 6/8/16/24/32 all 2.55–2.57 tok/s; a dedicated expert-I/O executor measured 8.59 ms vs existing 8.42 ms; a custom four-worker pool was mixed (5.964–6.132 vs 6.093–6.022 tok/s). — [in-repo: DO_NOT_IMPLEMENT.md](../../docs/performance/DO_NOT_IMPLEMENT.md); [in-repo: os_io_tricks.md](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)
- **io_uring SQPOLL** is the Linux "dedicated kernel I/O polling thread" version of the same idea (avoids syscalls), and **POSIX AIO / GCD** are user-level thread-pool versions. — [io_uring(7)](https://man7.org/linux/man-pages/man7/io_uring.7.html); [a10](https://github.com/Thomasdezeeuw/a10)
- **DMA/copy engines & GPUDirect Async:** copy engines move blocks asynchronously; GDS uses a DMA near storage, GPUDirect Async lets the GPU trigger RDMA, all to overlap I/O with compute — NVIDIA/PCIe only. — [NVIDIA GDS blog](https://developer.nvidia.com/blog/gpudirect-storage/); [CoPilotIO OSDI'26](https://www.usenix.org/system/files/osdi26-chen-guanyi.pdf); [GPUDirect Async](https://www.emergentmind.com/topics/gpudirect-async)
- **RDMA** appears in PD-disaggregation KV transfer (InfiniBand/RoCE), not local file I/O. — [vLLM/LLM-D/NIXL discussion](https://www.youtube.com/watch?v=BaD3CTYf6V0); [SGLang PD+EP blog](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)
- **On Apple, the closest primitive is Metal's IO command queue**, which writes straight into an `MTLBuffer` (unified-memory zero-copy) — it removes a CPU bounce copy but not the SSD transfer. — [WWDC22 10104](https://developer.apple.com/videos/play/wwdc2022/10104); [in-repo: os_io_tricks.md Q5](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)

### Inferences
- **Per-method card (lookahead prefetch).** Mechanism: predict next experts and issue reads early. Bit-exact: issue-only/advice-only = fine; speculative use = disqualified. Realistic gain: ≤ +25% (E1), realistically ~0 (E3 recall 0.36, rejected prefetch). M4 feasibility: feasible but futile. Falsifiable test: none needed — E1 already ran the decisive oracle; any new scheme must beat 2.72 tok/s to be interesting, which is impossible by definition. — Sources as above.
- **Per-method card (dedicated I/O core/thread).** Mechanism: pin I/O submission to a core so compute never blocks. Bit-exact: yes. Realistic gain: ~0 on a saturated SSD. M4 feasibility: feasible but already tried/negative. Falsifiable test: pinned-I/O-thread A/B with digest — already run, negative. — Sources as above.
- **Per-method card (DMA/copy engine, RDMA).** Mechanism: hardware moves bytes without CPU/GPU orchestration. Bit-exact: yes. Realistic gain: reduces copy overhead only. M4 feasibility: **IMPOSSIBLE — NEW HARDWARE / NO API**; unified memory already removes the VRAM copy, and no Apple GDS exists. Falsifiable test: N/A. — Sources as above.

### Gaps
- No Apple-exposed DMA/copy-engine or GPUDirect-style storage API was found in any source; no cold-read evidence that MTLIO beats `pread` (Q2 gap).
- The in-repo "dedicated executor" and "four-worker pool" measurements are third-party (Mference) project logs, not independent benchmarks.

---

## KQ7. Time division — can one SSD be time-sliced so the decoder never waits? (scheduling theory, deadlines, worst-case latency)

### Takeaway
No — when the device is oversubscribed, no scheduling discipline can make the decoder never wait. E1's own arithmetic gives a per-token drive time of ~260 ms against a 200 ms (5 tok/s) or 67 ms (15 tok/s) period, i.e. utilization ρ ≈ 1.3 or 3.9. With ρ > 1, EDF/rate-monotonic/proportional-share schedulers all accumulate unbounded backlog; the only remedies are fewer bytes or more bandwidth. macOS also exposes no user-facing storage scheduler, only a priority hint.

### Cited Findings
- **Workload utilization (from measured numbers):** ~873 MB/token ÷ 3.36 GB/s ≈ 260 ms drive time; compute ~200 ms; 5 tok/s target = 200 ms period → ρ ≈ 1.3; 15 tok/s = 67 ms period → ρ ≈ 3.9. E1 concludes the byte-volume floor dominates and 5/15 are unreachable by prediction. — [in-repo: E1_ORACLE_RESULT.md](../../docs/performance/E1_ORACLE_RESULT.md); [in-repo: E0_CONCURRENCY_RESULT.md](../../docs/performance/E0_CONCURRENCY_RESULT.md)
- **Queuing-theory framing is standard in serving systems:** DistServe models a prefill instance as M/D/1 and explicitly analyzes queuing delay vs execution time; it notes priority scheduling "adversely affects the latency of the other" phase. — [DistServe arXiv 2401.09670](https://arxiv.org/html/2401.09670v3)
- **Deadline/proportional-share I/O schedulers exist** (e.g., mClock: reservation + proportion + time-based priority for hypervisor I/O scheduling, Gulati et al., OSDI 2010 — canonical URL [USENIX OSDI'10](https://www.usenix.org/legacy/event/osdi10/tech/full_papers/Gulati.pdf), NOT re-fetched this session) and general real-time result is that schedulability requires total utilization ≤ 1. — [USENIX OSDI'10](https://www.usenix.org/legacy/event/osdi10/tech/full_papers/Gulati.pdf) (mark: unverified this session)
- **The only Apple-facing latency knob** is `MTLIOPriority.high` ("storage system will ensure high-priority IO requests have a lower latency and are prioritized"); no published cold-read latency number, and it still cannot change ρ. — [WWDC22 10104](https://developer.apple.com/videos/play/wwdc2022/10104)
- **Over-prefetch as self-inflicted oversubscription:** SP-MoE's Observation II — over-prefetching future layers without regard to I/O contention reduces accuracy, triggers more on-demand loads and delays computation; Bun over-prefetch and in-repo IO-08 show the same. — [SP-MoE arXiv 2510.10302v2](https://arxiv.org/html/2510.10302v2); [Bun commit](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca)

### Inferences
- **Per-method card (I/O time slicing / deadline scheduling).** Mechanism: reserve/prioritize SSD time slots for the decoder's reads. Bit-exact: yes (scheduling only). Realistic gain: **0 when ρ > 1**; bounded by E1 (+25%) even when ρ < 1. M4 feasibility: low — no user-facing storage scheduler; only `MTLIOPriority.high`. Falsifiable test: compute ρ from measured drive-time/period; if ρ ≈ 1.3, run a long decode and show deadline misses grow with run length under any priority arm; prediction — confirmed, so time-slicing cannot meet a 5 tok/s deadline. — Sources as above.
- "Splitting I/O time" by interleaving the decoder's reads with other traffic makes the decoder wait *more*, not less, unless the other traffic is itself useful work whose compute covers the decoder — which is just KQ1/KQ4 again.
- The honest answer to "can a single SSD's I/O be time-sliced so the decoder never waits" is a firm no under these byte volumes; it is a throughput-capacity problem, not a scheduling problem.

### Gaps
- mClock paper was not re-fetched (search returned empty); the "utilization ≤ 1" real-time condition is standard theory stated here as inference from cited measurements, not from a retrieved scheduling source this session.
- No Apple storage-scheduler internals source (XNU I/O queue priorities) was retrieved; only the Metal priority hint is documented.

---

## KQ8. 2024–2026 paper landscape on hiding/splitting storage I/O in LLM inference and MoE offload

### Takeaway
The recent literature uniformly confirms the same two findings as E0/E1: overlap/prefetch hides *latency* but cannot remove the *bytes*, and over-prefetch or coarse overlap creates stalls. The most relevant systems (SP-MoE, SSD-offload-energy, SolidAttention, async-offload, EStream/BigMoMo/EdgeMoE, OD-MoE, SSD-LLaMA, MoE-Lightning) are mostly non-Apple or larger-hardware, so their absolute numbers are not transferable; their mechanism lessons are.

### Cited Findings
- **SP-MoE (arXiv 2510.10302v2):** expert loading is 69.4% of SD-enabled MoE latency; "prior offloading systems achieve only coarse-grained overlap," prefetch is a continuous background service with pipelined runtime, batched I/O and accurate draft-stage prefetch; over-prefetching hurts (Observation II). TPOT baselines 1.40 s Mixtral / 0.48 s Phi-MoE / 0.32 s DeepSeek. — [arXiv 2510.10302v2](https://arxiv.org/html/2510.10302v2)
- **SSD offloading considered harmful in energy (arXiv 2508.06978):** prefetching hides latency (penalty as low as 1.25× DeepSeek / 1.32× Mixtral with NVLink5-class bandwidth) but SSD energy is ~4.9–12× HBM and 4.7–9.8× even at batch 1024 — "latency hiding cannot mitigate the energy penalty." — [arXiv 2508.06978](https://arxiv.org/html/2508.06978v1)
- **SolidAttention (FAST'26):** low-latency SSD-based serving on memory-constrained PCs with a speculative prefetcher and out-of-order overwrite — [USENIX FAST26](https://www.usenix.org/system/files/fast26-zheng.pdf)
- **Asynchronous model offload (SSDBM 2025):** selectively offload transformer blocks by interval k and asynchronously prefetch them — [SSDBM 2025 poster](https://ssdbm.org/2025/assets/poster/8884-Jie.pdf)
- **Mobile/flash MoE streaming (design templates):** EStream — one compiled expert graph with runtime binding, UFS-flash virtualization and UFS–NPU pipelining, 2.25–27.57× TTFT; BigMoMo — speculative multi-token windows to batch expert movement, mean 4.83× vs on-demand; EdgeMoE — experts on external storage with bitwidth adaptation + preloading. — [EStream 2609.06551](https://arxiv.org/html/2609.06551v1); [BigMoMo 2609.14643](https://arxiv.org/abs/2609.14643); [EdgeMoE IEEE TMC](https://ieeexplore.ieee.org/document/10906629)
- **OD-MoE (arXiv 2512.03927):** cacheless-SSD CPU–GPU parallel shadow model, Mixtral-8×7B 3.69 tok/s (75.5% of full-GPU), full-precision answers — a rare "full quality" SSD-offload result. — [arXiv 2512.03927](https://arxiv.org/html/2512.03927)
- **SSD-LLaMA lossless rANS + 3-tier cache:** 2.78× over no-cache, decode 2.10–15.58× — lossless by construction (digest-test, not reads-only). — [in-repo: Faster Qwen Nemotron report](../../reports/Faster%20Qwen%20Nemotron%20inference%20on%20M4.md)
- **MoE-Lightning CGOPipe (ASPLOS'25):** fine-grained per-layer GPU–CPU–I/O pipeline with paged weights; up to 10.3× batch throughput, reaches throughput bound with 2–3× less CPU memory (no disk offload). — [arXiv 2411.11217](https://arxiv.org/html/2411.11217)
- **Negative result on prefetching:** "Rethinking Expert Prefetching for Offloaded Diffusion MoE" finds that when recency + asynchronous demand loading already hide latency, temporal prefetching removes almost no demand misses and `none + async` is faster — a direct caveat to adding any predictor to this workload. — [OpenReview PDF](https://openreview.net/pdf?id=LCHQU8sZ32)
- **Shipping frameworks:** vllm-metal added MoE expert offloading (experts stay on SSD, waits for routing, misses read from SSD) and an RFC for incremental GPU cache + async pipeline; PR #29941 adds async H2D prefetch. — [vllm-metal MoE offload docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/moe_offload/); [vLLM RFC #38256](https://github.com/vllm-project/vllm/issues/38256)

### Inferences
- The literature's "fully pipelined, no-stall" claims are all measured on multi-GPU/NVMe/NVLink or mobile-NPU+UFS platforms where compute windows are much larger relative to I/O than the ~200 ms compute vs ~260 ms drive time here; they do not overturn E1's +25% bound.
- Recurring mechanism lessons that DO transfer: (1) keep prefetch off the critical path (1bit.gg 37–39 vs 28–38); (2) get I/O issue off the decode path (PR #31469 doorbell, one sync); (3) avoid over-prefetch (SP-MoE, Bun, IO-08); (4) prefer async demand loading + good residency over prediction (diffusion-MoE negative result).
- Bit-exactness map: OD-MoE full precision and PR #31469 byte-exact are clean; SP-MoE/<1.1% loss and cache-aware reranking are NOT bit-exact; rANS is lossless but must be digest-gated.

### Gaps
- No paper measured a 4-bit MoE decode on an Apple M4 with 16 GB doing SSD expert streaming under a bit-exact gate; closest are colibri (M4 Max, 128 GB, warm) and vllm-metal docs (no M4 numbers in source).
- Many 2026 items are preprints/single-team with unreplicated ratios to custom baselines; treat their speedups as mechanism evidence, not forecasts.

---

## KQ9. Ranked synthesis by plausibility × payoff, with explicit "impossible without new hardware" flags

### Takeaway
Under the gate (bit-exact, batch-1, one 3.36 GB/s SSD, no new hardware), almost every "split I/O time" idea is either already rejected, theoretically capped at +25%, or physically impossible. The highest plausibility×payoff items are those that attack the host/sync tax or open an untested I/O path; multi-device, near-data and scheduling tricks are dead ends here.

### Cited Findings — ranked cards (plausibility × payoff, best first)
1. **Fine-grained compute-on-arrival + ONE host sync (single-sync slot remap).** Mechanism: device-side KICK/WAIT slot resolution, per-expert GEMM on byte arrival. Bit-exact: yes (permutation). Gain: potentially large if Veystrix currently has per-layer syncs (~8 ms/layer pathology); ~40–55 ms hidden realistically, not 100%. M4: medium-high feasibility (native Metal/C++ likely needed). Test: sync count 48→1, digest equality, ms/token drop. — [PR #31469](https://github.com/ggml-org/llama.cpp/pull/31469); [in-repo: E1 result](../../docs/performance/E1_ORACLE_RESULT.md)
2. **`F_RDADVISE` / `F_RDADVISEV` async advice (untested end-to-end).** Mechanism: kernel async prefetch into page cache, advice-only. Bit-exact: yes. Gain: ≤ +25% ceiling; strongest untested I/O lever (3.6 GB/s+ cold-ish probe). M4: feasible. Test: E2E advice-only decode, gate on tok/s + identical token IDs; sweep prefetch breadth for contention. — [Bun commit](https://github.com/oven-sh/bun/commit/adc354d9983fe1866ceeb28e6a6ecd30ce443cca); [in-repo: os_io_tricks.md](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)
3. **MTLIO cold `loadBytes` + `MTLIOPriority.high`.** Mechanism: Metal async IO queue into shared MTLBuffer. Bit-exact: yes. Gain: uncertain; warm-only path failed (5.4–7.5% warm). M4: feasible. Test: cold `loadBytes` vs parallel `pread`, add priority arm, digest parity. — [WWDC22 10104](https://developer.apple.com/videos/play/wwdc2022/10104); [in-repo: os_io_tricks.md](../../research_notes/MoE%20decode%20IO%20bypass%20research/os_io_tricks.md)
4. **MLX command-buffer limit tuning** (`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`). Mechanism: fewer command buffers → less launch overhead. Bit-exact: hashes matched. Gain: +5–14% on compute/launch slice, but hides ~0 ms of I/O. M4: high feasibility, free. Test: env-var A/B + digest. — [MLX issue #4521](https://github.com/ml-explore/mlx/issues/4521); [vllm-metal config](https://docs.vllm.ai/projects/vllm-metal/en/latest/configuration/)
5. **Multi-stream / double-buffered layer pipeline (classic "split I/O time").** Mechanism: prefetch L+1 while computing L. Bit-exact: yes. Gain: ≤ +25% and plain form already rejected; realistically 0–10%. M4: medium (collides with rejected layer-prefetch). Test: hold read-issue time constant, vary sync/dispatch granularity; drop if no gain. — [in-repo: E1 result](../../docs/performance/E1_ORACLE_RESULT.md); [DO_NOT_IMPLEMENT.md](../../docs/performance/DO_NOT_IMPLEMENT.md); [2411.11217](https://arxiv.org/html/2411.11217)
6. **Continuous batching / multi-request overlap.** Mechanism: amortize across sequences. Bit-exact: yes for decode. Gain: ~0 for batch-1; high only for multi-request serving. M4: feasible. Test: 2–4 concurrent streams, bytes/token + digest. — [DistServe](https://arxiv.org/html/2401.09670v3)
7. **Software tiering / larger hot-expert cache.** Mechanism: serve hot experts from RAM/cache. Bit-exact: yes. Gain: bounded by hit rate and 16 GB capacity. M4: feasible but capacity-starved. Test: pool-size A/B, bytes/token + digest. — [in-repo: E1 result](../../docs/performance/E1_ORACLE_RESULT.md)

### Cited Findings — IMPOSSIBLE without new hardware (flagged, not candidates)
- **Second SSD / RAID0 striping / TB4 external:** **IMPOSSIBLE — NEW HARDWARE**; ≤1.5× aggregate, no per-token win. — [AppleInsider](https://appleinsider.com/articles/25/10/06/acasis-40gbps-mac-mini-m4-dock-review-all-the-ports-you-could-want-for-your-m4-mac-mini)
- **Pipeline/tensor parallelism, PD disaggregation across workers:** **IMPOSSIBLE — NEW HARDWARE** (no second GPU/Mac/interconnect). — [DistServe](https://arxiv.org/html/2401.09670v3)
- **GPUDirect Storage / DMA copy engines / GPUDirect Async / RDMA:** **IMPOSSIBLE — NO API/HARDWARE** on Apple Silicon; unified memory already removes the VRAM bounce copy. — [NVIDIA GDS](https://developer.nvidia.com/blog/gpudirect-storage/); [CoPilotIO](https://www.usenix.org/system/files/osdi26-chen-guanyi.pdf)
- **Computational storage / PIM / CXL PNM:** **IMPOSSIBLE — NEW HARDWARE**; no CXL or programmable storage controller on M4. — [arXiv 2511.00321](https://arxiv.org/html/2511.00321v1); [ACM DL](https://dl.acm.org/doi/10.1145/3837053.3837356)
- **I/O time-slicing / deadline scheduling to eliminate decoder wait:** **IMPOSSIBLE IN PRINCIPLE here** because ρ > 1 (260 ms work in 200/67 ms periods) and macOS exposes no storage scheduler. — [in-repo: E1 result](../../docs/performance/E1_ORACLE_RESULT.md)

### Inferences
- The single sentence for the report writer: on this machine, "splitting I/O time" cannot manufacture bandwidth or remove bytes, so it is bounded by E1's +25%; the actionable work is the compute/sync side (rank 1) and the one genuinely untested OS path (rank 2), while all multi-device/near-data/scheduling variants are hardware-blocked or theoretically void.
- Rankings 1–4 compose (different walls: sync, page cache, copy path, launch overhead); rank 5 overlaps rank 1 and should be tested only after it.

### Gaps
- No single source provides a bit-exact M4/16 GB measurement of any ranked item on the exact Qwen/Nemotron models; all are proposals or evidence from other hardware.
- Whether Veystrix already has one host sync/step (rank 1 headroom) is unverified in this note — check the current engine before implementation.
