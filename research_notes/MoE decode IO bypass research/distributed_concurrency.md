# Distributed, Multi-Mac, and Concurrency Techniques for MoE Decode (Veystrix)

Scope: pooling resources across 2+ Macs (and other devices) over Thunderbolt/RDMA, and concurrency techniques, to reach MoE decode targets that a single 16 GB M4 cannot (Qwen3.8-Flash-Next 125B MoE: 2.3 tok/s, need 5+; Nemotron 30B-A3B: ~7 tok/s, need 15+). Baseline steady-state OPT requires 10 GB (Nemotron) / 12 GB (Qwen) of *unique expert RAM resident* — impossible in one 16 GB Mac. "No quality loss" is a hard constraint. All measured numbers below are from the cited sources; inferences are explicitly labeled.

> Epistemic note: this timeline's sources (WWDC26 / macOS 26.2 / M5 Ultra) are treated as real per the search results. Apple's MLX research post is dated 2026-10-15 and is the most authoritative primary source. One secondary source (InsiderLLM, updated 2026-07-17) claimed the M5 Ultra was "delayed to roughly October 2026" and current Studios "capped at 96GB" — this conflicts with Apple's own Oct 15 post describing M5 Ultra Mac Studios with up to 512GB. Defer to the Apple primary source and flag the conflict as a gap. None of the high-end Apple benchmarks are reproducible on Veystrix's 16 GB M4; they establish *mechanism and scaling*, not an attainable config.

## Q1. Can pooling 2+ Macs over Thunderbolt/RDMA hold all experts resident and raise MoE decode where one 16 GB M4 cannot?

### Takeaway
Yes in principle: Apple's own measurements show tensor-parallel decode scaling ~1.7× on 2 Macs and ~2.6× on 4 for an MoE, and the real benefit for Veystrix is *capacity* — doubling resident RAM eliminates SSD expert-swapping, which is the actual bottleneck. But a second Mac **hurts** if the model already fits on one Mac (an EXO two-node test was 18–23% *slower*), so pooling only pays when it keeps experts resident that would otherwise be read from SSD.

### Cited Findings
- Apple's MLX research post (primary source, 2026-10-15) reports decode speed (b=1, 8,192-token context, 128 generated tokens, `mlx_lm.benchmark`): **Muse Spark Open 816B-A42B MoE in NVFP4** — 2× M5 Ultra pipeline **15.74 tok/s**, 2× M5 Ultra tensor **24.99 tok/s**, 4× M5 Ultra tensor **36.73 tok/s** (single-machine value not given because the model does not fit). [Apple MLX Research](https://machinelearning.apple.com/research/mlx-distributed-inference)
- Same source, dense controls: Qwen 3.8 27B decode — 1× M3 Ultra **12.46**, 1× M5 Ultra **19.69**, 2× M5 pipeline **19.44**, 2× M5 tensor **33.27**, 4× M5 tensor **50.89 tok/s**. Mistral Medium 3.5 128B — 4.95 / 7.70 / 7.70 / 13.69 / 23.37 tok/s. Tensor parallelism thus gives ~1.7× (2 nodes) and ~2.6× (4 nodes) decode scaling. [Apple MLX Research](https://machinelearning.apple.com/research/mlx-distributed-inference)
- Pipeline parallelism leaves decode essentially unchanged (Qwen 3.8: 19.44 vs 19.69 single), but prefill at 65,536 context is up to **~1.8× faster** than a single node because MLX chunks the context (e.g. `--prefill-step-size 8192`) and pipelines consecutive chunks across machines. [Apple MLX Research](https://machinelearning.apple.com/research/mlx-distributed-inference)
- MoE tensor-parallel decode scales as well as dense: Apple states "Decode is unaffected [by prefill communication]; Muse Spark Open sustains a significant decode speedup under tensor parallelization even at large contexts, comparable to dense models like Qwen 3.8 and Mistral Medium 3.5." The exception is prefill only, where separate normalization of expert outputs adds communication (2-machine TP lags 2-machine PP at 65K prefill). [Apple MLX Research](https://machinelearning.apple.com/research/mlx-distributed-inference)
- A 5-node M3 Ultra 512GB cluster running Kimi-K2-Thinking (~1T, Q4, batch=1) found near-identical throughput across schemes: PP5 14.45, PP4 14.49, TP4 14.82, TP2 13.15 TPS. **TP4 beat PP4 by only 2.3%**, while PP loaded 4.5× faster (22.3s vs 99.9s) and used 21% less peak memory (145.9 vs 185.7 GB). "RDMA makes the all-reduce communication in TP nearly free." [MLX Discussion #2990](https://github.com/ml-explore/mlx/discussions/2990)
- Same source: **RDMA works in ring mode too**, so there is no TCP/IP overhead for pipeline parallelism if TB5 is configured with `mlx.distributed_config`; a ring of TB5 nodes with less RAM may outperform a fully-meshed TP cluster without cable spaghetti. TP requires the model dimension to divide evenly by node count (Kimi 12288÷5 failed); PP has no such constraint. [MLX Discussion #2990](https://github.com/ml-explore/mlx/discussions/2990)
- **Counterpoint (critical for Veystrix):** EXO benchmark of Qwen3.5-35B-A3B-4bit, M4 Max 128GB alone vs M4 Max + M4 Pro 48GB two-node tensor sharding over TB5 RDMA (aggregate tok/s): c1 108.0→85.6 (−21%), c2 167.4→137.8 (−18%), c3 207.8→169.4 (−18%), c4 177.6→136.8 (−23%). The model fit in one node, so adding a second node only added communication and reduced per-node bandwidth. [EXO benchmark gist](https://gist.github.com/nexus-studio-dev/b46fdc95269e55666da858dc9ebb387e)
- Wiring matters: Apple's setup used two Thunderbolt cables per link to double bandwidth; on 4 M5 Ultras at 65,536 context, two cables cut Muse Spark Open TTFT from ~50.5s to 40.6s. [Apple MLX Research](https://machinelearning.apple.com/research/mlx-distributed-inference)
- RDMA over Thunderbolt is available starting with **macOS 26.2** on Apple-silicon Macs with Thunderbolt 5; a System Settings toggle ("Enable RDMA over Thunderbolt") plus reboot is required. [Apple Tech Note TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt); [WWDC26 session 233 digest](https://github.com/ivan-magda/wwdc26-notes/blob/main/sessions/233-distributed-inference-training-mlx/digest.md)
- Bit-exactness: TP/PP keep the identical weight tensors and the same arithmetic; the only numerical difference is the order of floating-point reductions in all-reduce (an inference, not a sourced claim). Quantization is a separate, lossy choice: Apple's MoE figure used NVFP4, but that is orthogonal to the parallelism scheme. Veystrix must keep its current precision/expert format unchanged to satisfy "no quality loss."

### Inferences
- For Veystrix the win from a second Mac is **capacity**, not raw FLOPs: 2×16 GB ≈ 32 GB of unified memory lets the ~10–12 GB of unique experts plus shared weights/KV stay resident, removing the SSD read that causes 2.3/7 tok/s. This directly targets the documented steady-state OPT blocker.
- Expect ~1.6–1.8× decode scaling per doubling of nodes *only after* the model fits resident; if Veystrix distributes a configuration that already fits in 16 GB, expect an 18–23% regression per the EXO data.
- Pipeline parallelism is the cheapest first cluster experiment (minimal code: `mlx_lm ... --pipeline`), with decode unchanged and a prefill win at long context; tensor parallelism is the throughput lever but needs a low-latency mesh and divisible dimensions.
- The "2× M4 16GB got 25 tok/s on Qwen3.6-35B" figure supplied in the parent context could not be verified; the closest sourced data are MLX ~25–35 tok/s for Qwen3.5-35B-A3B on a single M1 Pro 16GB and ~60–70 tok/s on M4 Max. [InsiderLLM](https://insiderllm.com/guides/qwen35-mac-mlx-vs-ollama/) Treat the 25 tok/s 2×M4 claim as unverified.

### Gaps
- No published TP/PP decode number exists for the exact Veystrix models (Qwen3.8-Flash-Next 125B, Nemotron 30B-A3B) or for 16 GB M4 class nodes; all Apple numbers use M5 Ultra / 512 GB.
- The parent's "2× M4 16GB → 25 tok/s on Qwen3.6-35B" figure was not found in any source.
- No data on how TP/PP all-reduce ordering affects exact output equivalence at Veystrix's precision (likely negligible, but unmeasured).

## Q2. Which multi-Mac transport/framework stack is practical, and what does each deliver?

### Takeaway
Apple MLX + JACCL + RDMA-over-TB5 is the primary, best-documented, highest-performance path (1.7–2.6× decode scaling). EXO is the most production-hardened for heterogeneous/consumer setups (auto-discovery, P2P all-reduce, handles TB5 setup quirks) but is not faster on a single-fitting model. llama.cpp RPC is the mature, bit-exact fallback that works over ordinary networking but is bandwidth/latency-limited.

### Cited Findings
- Stack layers: RDMA over TB5 (raw point-to-point, macOS 26.2) → **JACCL** (Apple open-source collective library: `all_sum`, `all_max`, `all_min`, etc., with mesh or ring topologies) → MLX (uses JACCL, orchestrates via `mlx.launch`) → MLX LM (CLI/Python). [WWDC26 session 233 digest](https://github.com/ivan-magda/wwdc26-notes/blob/main/sessions/233-distributed-inference-training-mlx/digest.md); [JACCL README](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/README.md)
- JACCL topology: full mesh = lowest latency; ring = fewer cables, scales further, 2–3 cables per neighbor raise bandwidth; JACCL auto-picks mesh vs ring per message size. JACCL supports only fully connected topologies for the mesh backend. [WWDC26 digest](https://github.com/ivan-magda/wwdc26-notes/blob/main/sessions/233-distributed-inference-training-mlx/digest.md); [MLX distributed docs](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- One-command workflow: `mlx.distributed_config --hosts ... --backend jaccl` generates the hostfile and `--auto-setup` disables Thunderbolt Bridge and configures links; `mlx.launch --hostfile ... --backend jaccl --env MLX_METAL_FAST_SYNCH=1 -- <cmd>` runs model-parallel inference; MLX LM shards automatically, tensor parallelism is the default, `--pipeline` switches. [WWDC26 digest](https://github.com/ivan-magda/wwdc26-notes/blob/main/sessions/233-distributed-inference-training-mlx/digest.md)
- Python-level user control relevant to Veystrix: `mx.distributed.init(strict=True, backend="jaccl")`, `sharded_load(model, pipeline_group, tensor_group)`, `nn.layers.distributed.shard_linear(layer, strategy="all-to-sharded", group=group)`, `mx.distributed.all_sum`. This is the hook for a custom MoE placement (see Q3). [WWDC26 digest](https://github.com/ivan-magda/wwdc26-notes/blob/main/sessions/233-distributed-inference-training-mlx/digest.md)
- WWDC26 headline results: ~3× token rate for Qwen 3.6 27B across four Macs vs one; Kimi 2.6 (1T params, ~1 TB at 8-bit) runs across four M3 Ultras; fine-tuning Qwen 3.5 9B goes ~180 → ~600 tok/s (>3×). [WWDC26 digest](https://github.com/ivan-magda/wwdc26-notes/blob/main/sessions/233-distributed-inference-training-mlx/digest.md)
- EXO: macOS app, day-0 RDMA-over-TB5 support advertised as "99% reduction in latency between devices"; supports MLX/tinygrad engines, UDP/Tailscale discovery, ring memory-weighted partitioning, OpenAI-compatible endpoint. Requires Apple silicon + macOS Sequoia 15.1+ (26.2+ for RDMA). [exo README](https://github.com/exo-explore/exo/blob/main/README.md); [Exo Labs FAQ](https://exolabs.net/faq/what-is-rdma-over-thunderbolt-5)
- EXO's practical advantage: a cluster builder using raw `mlx.distributed` reported a "nightmare" of TB5 setup hacks (bridge0 auto-bringup, STP shutting down TB5 interfaces, enX renaming, ARP cache corruption) and credited EXO with solving these and adding an inter-node P2P all-reduce. [MLX Discussion #2990](https://github.com/ml-explore/mlx/discussions/2990)
- llama.cpp RPC: built with `-DGGML_RPC=ON`, each worker runs `ggml-rpc-server`, then `llama-server --rpc <host:port> ...`; it splits tensors across backends (CPU/Metal/CUDA), works over standard networking, and is a work in progress with limitations. [llama.cpp RPC README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md); [Reddit announcement](https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/)
- vLLM-metal offers Ray-based executor and two-stage pipeline parallelism over an MLX ring for multi-Mac serving, but some pieces are "not implemented." [vLLM-metal distributed docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/distributed/)

### Inferences
- Bit-exactness: all three stacks run the same weights and kernels; TP/PP differ only in reduction order. llama.cpp RPC is closest to trivially bit-exact (same GGUF, same Metal kernels, tensor splits). No source claims quality loss from distribution itself.
- Maturity/risk trade: JACCL/TB5 RDMA stack is bleeding-edge (the discussion author attributes repeated kernel panics to `com.apple.driver.AppleThunderboltRDMA` v0.0.1). [MLX Discussion #2990](https://github.com/ml-explore/mlx/discussions/2990) EXO is the safer integration path if Veystrix ships a commercial product and cannot tolerate node instability.

### Gaps
- No source quantifies JACCL vs EXO vs llama.cpp RPC head-to-head on the same MoE and hardware.
- Stability/reliability of macOS 26.2 RDMA in a commercial always-on product is undocumented; only a single user's kernel-panic report exists.
- Cost of TB5 cables/second Mac not covered in these sources beyond general hardware pricing (see Q6).

## Q3. MoE-specific partitioning: expert parallelism, "dense on one Mac / experts on another," and expert offloading — do they beat generic TP/PP?

### Takeaway
Generic MLX TP/PP already parallelizes expert layers as part of each layer (Apple's MoE scaling ≈ dense). For Veystrix's SSD-expert problem, the more targeted levers are **expert parallelism (EP)** and **expert offloading with concurrent-request scheduling** — but the strongest published gains come from *scheduling around expert fetches*, not from new hardware.

### Cited Findings
- vLLM supports Expert Parallelism (EP), placing experts of an MoE on separate accelerators to increase locality/efficiency; TensorRT-LLM supports TP (default), EP, and hybrid TP+EP. [vLLM EP docs](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/); [TensorRT-LLM EP docs](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html)
- NVIDIA: large EP configurations (EP32) achieved up to **1.8× higher per-GPU output token throughput** than EP8 at 100 tokens/s/user. [NVIDIA Technical Blog](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems)
- OSDI '26 paper (arXiv 2606.10493): distributed stream-loading prefill with **SmallEP expert parallelism** reaches 1,800 tokens/s prefill and 45K prompts in 30s on two RTX 5090s; fine-grained CPU parallelism attains **28 tok/s on INT4 DeepSeek-V3 and 21.5 tok/s on intact FP8 V3**, i.e. original precision, no distillation/quantization loss. [OSDI '26 paper](https://arxiv.org/html/2606.10493)
- vLLM MoE playbook: DP+EP scales better for high-throughput batch processing; "Expert Parallelism helps or hurts" depending on model and batch regime. [AMD/vLLM MoE guide](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- Bevy (arXiv 2611.04794, 2026-11-03): expert offloading to local storage; **prefetching can be harmful when an expert cache is in GPU memory** because prediction precision drops as the cache covers more needed experts. Bevy instead uses runahead inference of concurrent requests — a request whose experts are resident runs ahead to reveal which experts it needs next, letting the system fetch exactly those while computing ready requests, **without predicting expert selections** (therefore lossless). Self-balancing scheduler batches ready requests per layer; up to **3.46× decode throughput of llama.cpp** and **1.55× of the fastest prefetching baseline** across two local platforms. [Bevy paper](https://arxiv.org/abs/2611.04794); [Bevy HTML §4.2–4.3](https://arxiv.org/html/2611.04794)
- XShare: in-batch expert sharing reduces expert activation up to 30% under standard batching, cuts peak GPU load up to 3× in expert-parallel deployments, and achieves up to 14% throughput gains in speculative decoding. [XShare paper](https://arxiv.org/html/2602.07265v1)
- PROBE: real-time predictive prefetching reduces prefill latency up to 1.32× and improves decoding throughput up to 1.26× over state-of-the-art baselines without contending with all-to-all collectives. [PROBE paper](https://arxiv.org/pdf/2602.00509v2)
- SpecMoEOff: speculative decoding + expert offloading yields up to **2.5× decode throughput** improvement. [SpecMoEOff paper](https://arxiv.org/abs/2508.21706)

### Inferences
- "Dense/shared weights on one Mac, experts on another" is a natural fit for EP, but no source benchmarks that exact split on Apple silicon. MLX's `shard_linear` / `sharded_load` primitives are the implementation hook; the research-model evidence (OSDI '26, vLLM/NVIDIA EP) supports the concept, not a specific Mac deployment.
- Bevy's result is the most relevant to Veystrix's root cause (SSD expert misses): it attacks the *stall while fetching missing experts*, which is exactly the single-Mac failure mode, and it does so without quality loss. Its 3.46× is vs llama.cpp, so the absolute gain vs Veystrix's current backend must be measured.
- Prefetching (PROBE, SpecMoEOff) is lossless in the sense that it only changes *when* weights are fetched; speculative methods add a draft model whose acceptance must be exact to preserve quality — unverified here.

### Gaps
- No benchmark of EP / expert-offloading on 16 GB M4 Macs; all EP numbers are datacenter GPUs or dual-socket CPUs + RTX 5090.
- No source confirms bit-exactness of Bevy/XShare/PROBE at Veystrix's precision; they are described as systems/optims not quantization changes, but exact output equivalence is not stated.
- No direct measurement of "dense-on-Mac-A / experts-on-Mac-B" placement.

## Q4. Concurrency: continuous batching, request coalescing, and prefill–decode disaggregation — gains, costs, and whether they preserve quality

### Takeaway
Concurrency can amortize expert reads, but the naive assumption that "batching always helps" is false for local MoE: continuous batching multiplies active experts and can worsen per-request latency. The winning designs are *selective* — runahead scheduling (Bevy), intra-node prefill/decode disaggregation with zero-copy shared weights, and in-batch expert sharing.

### Cited Findings
- OSDI '26 paper: "In the cloud—large batches saturate expert-activation paths and raise compute utilization—the gains diminish locally. With small batch sizes and sparse expert activation, batching multiplies the number of active experts per step, amplifying memory-bandwidth pressure; consequently, each request experiences worse TPOT compared to non-batched local execution." [OSDI '26 paper](https://arxiv.org/html/2606.10493)
- Same paper: intra-node prefill–decode disaggregation with zero-copy shared weights (single resident weight copy, experts ring buffer) plus dual-batch attention–MoE overlap sustains concurrency with **<15% latency increase and 50% throughput gains**. Disaggregated engine worst-case total latency 1.67× vs baseline, compared with 2.53× for the competing approach; TPOT increases at most 1.54× vs 2.45× for the competitor. [OSDI '26 paper](https://arxiv.org/html/2606.10493)
- Same paper: attention ≈350 µs and MoE ≈450 µs per layer; alternating CPU-MoE / GPU-attention leaves idle time, motivating two CUDA streams interleaved to hide it (61 layers ≈50 ms/token). [OSDI '26 paper](https://arxiv.org/html/2606.10493)
- Chunked prefill reduces mutual blocking between long prefill and decode but destabilizes per-token latency and can raise TTFT; the paper's policy uses chunked prefill for short (<2K) requests and routes long prefills to stream-loading prefill (SLP/DSLP), with decode dynamically batched up to a target size (e.g. Y=6). [OSDI '26 paper](https://arxiv.org/html/2606.10493)
- llm-d P/D disaggregation: prefill is compute-bound (FLOPs), decode is memory-bandwidth-bound; separating them gives throughput via specialization and prevents long prefills blocking decode. Decode pulls KV blocks over RDMA (IB/RoCE/EFA) via NIXL. [llm-d PD docs](https://llm-d.ai/docs/0.9/well-lit-paths/foundations/pd-disaggregation)
- xDeepServe: disaggregated prefill/decode for large MoE with expert parallelism uses **TP=4 for prefill attention and TP=1 for decode**, requiring different DP groupings; length-aware routing avoids stragglers that degrade TTFT. [xDeepServe paper](https://arxiv.org/html/2508.02520v4)
- LMSYS GB200 NVL72 with PD + large-scale EP: 3.8× prefill and 4.8× decode throughput. [LMSYS blog](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/)
- EXO's own concurrency data (two Macs) show aggregate throughput rising c1→c3 then falling at c4 (see Q1), consistent with batching's diminishing returns at local scale. [EXO benchmark gist](https://gist.github.com/nexus-studio-dev/b46fdc95269e55666da858dc9ebb387e)

### Inferences
- Bit-exactness: batching, request coalescing, and PD disaggregation do not change model weights or arithmetic (KV-cache transfer is a data movement), so they are quality-neutral. Bevy explicitly avoids prediction. Treat as lossless pending Veystrix's own output-diff test.
- For a single 16 GB M4, the most promising concurrency lever is **runahead/interleaved scheduling of a few (2–4) concurrent requests** to cover expert-fetch stalls — analogous to Bevy — rather than naive large-batch continuous batching. The Apple/EXO data show that on capable hardware a modest batch (c2–c3) maximizes aggregate throughput.

### Gaps
- No tok/s figures for continuous batching or PD disaggregation on Apple silicon / MLX; all figures are from CPU+GPU or datacenter systems.
- No source quantifies the quality impact of batching on Veystrix's exact models (expected zero).
- No falsified test yet of Bevy-style runahead in llama.cpp/MLX on a 16 GB Mac.

## Q5. Can iPhone / iPad / Vision Pro act as extra memory, and is it worth it?

### Takeaway
Technically yes for iPhone/iPad via EXO (heterogeneous cluster, memory-weighted ring sharding), but it is the lowest-value lever: devices are network-bound on Wi-Fi, the slowest device sets the pace, and measured throughput is ~3–5 tok/s for a 70B-class model — far below Veystrix's targets. No source supports Vision Pro as a useful node.

### Cited Findings
- EXO demo ran Llama-3-70B using an iPhone 15 Pro Max, iPad Pro M4, Galaxy S24 Ultra, M2/M3 MacBooks, and two RTX 4090s; devices auto-discover via mDNS/UDP/Tailscale and use peer-to-peer connections with a default ring memory-weighted partitioning scheme. [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/new-software-lets-you-run-a-private-ai-cluster-at-home-with-networked-smartphones-tablets-and-computers-exo-software-runs-llama-and-other-models); [EXO README](https://github.com/exo-explore/exo/)
- iOS support is immature: "the code needs some work to be ready for mainstream use," the iOS version was pulled and access required emailing the lead developer; a TestFlight request for an iOS beta heterogeneous cluster was still open as of April 2026. [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/new-software-lets-you-run-a-private-ai-cluster-at-home-with-networked-smartphones-tablets-and-computers-exo-software-runs-llama-and-other-models); [EXO issue #1943](https://github.com/exo-explore/exo/issues/1943)
- A hands-on EXO cluster recipe (Mac + iPhone + Pixel + Linux laptop, Llama-3.3-70B 4-bit) reports ~**3–5 tok/s steady-state**, notes "tok/s is dominated by the slowest device. Adding a fifth device often *slows* the cluster if it's an underpowered phone with a lossy wifi," and warns Wi-Fi 5 can show 30+ ms tail latency; Wi-Fi 6 or a hardwired link recommended. [Distributed Edge / Mosaic course](https://jvoltci.github.io/mosaic/edge-ai/distributed/)
- EXO's own tagline lists phones, laptops, desktops, and watches as devices, but no source in these results names the Vision Pro specifically as a tested node. [exo GitHub org mirrors](https://github.com/MacOS/exo)

### Inferences
- A phone/iPad could contribute a few GB of RAM to hold a slice of experts in a pipeline split, which could in principle keep a few more experts off SSD; but the network hop latency (~12–18 ms in the cited trace) and 3–5 tok/s ceiling make it unsuitable for Veystrix's 5+/15+ decode targets. Use only as a last-resort capacity filler, never in the critical decode path.

### Gaps
- No Vision Pro-specific EXO/MLX support or benchmark found.
- No measurement of an iPhone/iPad contributing specifically to expert-resident MoE decode on Thunderbolt (iOS devices lack TB5), only Wi-Fi/LAN.
- No quality/bit-exactness data for heterogeneous device clusters.

## Q6. Ranked by gain/cost, with falsifiable tests on available hardware

### Takeaway
Rank: (1) two-Mac MLX/JACCL tensor-parallel memory pooling; (2) Bevy-style runahead/concurrency on the existing Mac; (3) two-Mac pipeline parallel as the cheap capacity-only option; (4) expert parallelism/EP hybrid; (5) EXO for integration maturity; (6) iPhone/iPad fillers; (7) llama.cpp RPC as compatibility fallback. The dominant cost is a second Mac (≈$799 for a Mac mini M4 16GB class device; far more for large-memory Studios).

### Cited Findings
- Hardware cost anchors: "Mac mini M4 16GB (was $799 new)" — the cheapest second node; the same chip delivers ~24 tok/s on a 7B in the llama.cpp thread with 120 GB/s bandwidth. [localaimaster buying guide](https://localaimaster.com/blog/apple-silicon-ai-buying-guide)
- macOS allocates only ~70% of unified memory to the GPU and OS uses 3–5 GB, so a 16 GB Mac has roughly 10–11 GB practically usable — right at Nemotron's 10 GB / Qwen's 12 GB unique-expert requirement, hence "tight." [localaimaster buying guide](https://localaimaster.com/blog/apple-silicon-ai-buying-guide); [VRAM First M4 Ultra page](https://vramfirst.com/what-can-i-run/mac-studio-m4-ultra-192gb)
- Large-memory Macs are expensive: M4 Ultra 192GB Mac Studio ≈ $6,999–$8,999. [VRAM First](https://vramfirst.com/what-can-i-run/mac-studio-m4-ultra-192gb)
- Gain anchors for ranking: TP 2 nodes ~1.7×, TP 4 nodes ~2.6× decode (Apple); pipeline decode ~1× but capacity + prefill gains (Apple, #2990); Bevy up to 3.46× vs llama.cpp (Bevy); EXO two-node −18 to −23% if model fits (EXO gist); phone cluster 3–5 tok/s (Mosaic). [Apple MLX Research](https://machinelearning.apple.com/research/mlx-distributed-inference); [MLX #2990](https://github.com/ml-explore/mlx/discussions/2990); [Bevy](https://arxiv.org/abs/2611.04794); [EXO gist](https://gist.github.com/nexus-studio-dev/b46fdc95269e55666da858dc9ebb387e); [Mosaic](https://jvoltci.github.io/mosaic/edge-ai/distributed/)

### Inferences — ranked recommendation table

| Rank | Technique | Expected gain/cost | Bit-exact | Evidence strength | Hardware cost |
|---|---|---|---|---|---|
| 1 | 2-Mac MLX/JACCL TP memory pooling (RDMA/TB5) | ~1.7× decode per 2 nodes **if** it removes SSD swaps; capacity unblock is the main win | Yes (reduction order only) | Strong (Apple primary) | 2nd Mac ~$799 + TB5 cables; macOS 26.2 |
| 2 | Bevy-style runahead / interleaved concurrent requests | up to 3.46× vs llama.cpp; targets exact expert-fetch stall | Yes (no prediction) | Strong paper, untested on Mac | Software only |
| 3 | 2-Mac pipeline parallel (PP) | ~1× decode, but all experts resident + prefill win at long context; 4.5× faster load, 21% less memory | Yes | Strong (Apple, #2990) | 2nd Mac + cables |
| 4 | Expert parallelism / dense-on-one-experts-on-another | EP32 1.8×/GPU (datacenter); concept applicable, Mac numbers absent | Likely yes | Weak on Apple silicon | 2nd Mac, custom MLX sharding |
| 5 | EXO (heterogeneous, mature setup) | Integration/RDMA latency (-99% claim) but −18–23% if model fits | Yes | Medium (vendor + user benchmarks) | 2nd Mac; iOS beta immature |
| 6 | iPhone/iPad memory filler | ~3–5 tok/s cluster; too slow for target | Likely yes | Medium (course/demo) | "Free" if owned; Wi-Fi latency |
| 7 | llama.cpp RPC fallback | Compatible, mature, bandwidth/latency-limited; no published MoE gain | Yes | Medium | Any second machine, LAN |

### Inferences — falsifiable tests on available hardware
All can be executed without modifying model code (use `mlx_lm.benchmark` / `mlx_lm.server` and config flags), satisfying the no-code-change constraint.

1. **Single-Mac baseline (no new hardware):** `mlx_lm.benchmark --model <Veystrix model> -p 8192 -g 128 -b 1`, then `-b 2,4,8`. Predictions: reproduces 2.3/7 tok/s at b=1; records whether b=2–4 raises aggregate throughput or worsens per-token latency (OSDI prediction: local MoE batching may hurt TPOT). This validates target gap and the batching hypothesis.
2. **RDMA prerequisite check (no new hardware):** verify macOS ≥26.2 and TB5 ports, and that "Enable RDMA over Thunderbolt" appears in System Settings. If absent, the entire RDMA/TP/PP lever is unavailable on current hardware — a clean falsification. [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
3. **Two-Mac capacity test (requires second Mac):** load Qwen3.8-Flash-Next 125B with `mlx.launch --backend jaccl ... sharded_load` on 2×16 GB. Falsifiable endpoint: Activity Monitor shows no SSD swapping, all experts resident, and tok/s rises toward/above 5 (Qwen) / 15 (Nemotron). If tok/s instead falls 18–23% (EXO result), the model was already fitting and pooling does not help.
4. **PP vs TP A/B (requires second Mac):** run the same model with `--pipeline` vs default TP. Predictions: PP decode ≈ single-node, TP decode ~1.7×; PP prefill better at 32K+ context; PP hits the ~1,500-token Metal timeout at long context while TP handles 4K+. [MLX #2990](https://github.com/ml-explore/mlx/discussions/2990)
5. **Concurrency/runahead test (no new hardware for baseline; comparative):** serve 2–4 concurrent requests via `mlx_lm.server`; measure aggregate vs per-request TPOT. Falsifiable prediction: a Bevy-style "lowest-layer-first + opportunistic layerwise grouping" scheduler should close most of the gap to the 15+/5+ targets without new hardware if expert-fetch stalls dominate; if it does not, the bottleneck is pure bandwidth and only more resident RAM (rank 1/3) helps.
6. **iPhone filler (optional):** EXO cluster with an iPhone on Wi-Fi; predict ~3–5 tok/s and that removing the phone barely changes decode — confirming it should not be in the critical path. [Mosaic](https://jvoltci.github.io/mosaic/edge-ai/distributed/)

### Gaps
- No falsifiable test has been run; all are proposed.
- The exact second-Mac cost and TB5 cable cost for Veystrix's region are not sourced (US list prices used as anchors).
- No source confirms that 2×16 GB can keep *both* Nemotron's 10 GB and Qwen's 12 GB of unique experts resident alongside shared weights/KV within the ~70% GPU-addressable memory limit; this is the key feasibility question for rank 1.
