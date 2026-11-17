# Direct search leads (coordinator, 2026-11-17) — highest-value novel results

## 1. TRAINED PREROUTER — the strongest new lead (Edge0)
Edge0 "The Other Half of the Memory Wall" (arXiv:2609.18063v2) and menonlab blog:
- A per-layer prerouter predicts the NEXT layer's routing one token ahead; the prediction
  IS consumed as the routing, so the staged set equals the routed set, nothing dropped.
- Expert reads overlap compute. **+80-84% decode** on a machine where checkpoint doesn't fit;
  Prerouter swap-in gave **+59% decode** (Mac mini M4 Pro).
- 35B MoE at ~20 tok/s inside 3 GiB peak active memory on a 24 GB machine; bit-exact routing
  pinned to vendored models, executor paths verified element-wise vs dequantized reference.
- An unmerged recovery LoRA pays back int4 + routing-replacement quality loss.
- Relevance: our "next-layer router prefetch" was rejected, but it was a HEURISTIC (mix 4
  hyper-connection streams). A TRAINED prerouter with committed-as-routing is a different,
  much stronger mechanism. Requires training a small head — quality risk unless training
  makes the student match the teacher exactly (Edge0 uses a LoRA to recover). Commercial-safe
  to be first. THIS IS THE LEAD TO INVESTIGATE.
- URLs: https://arxiv.org/html/2609.18063v2 , https://themenonlab.blog/blog/edge0-streaming-moe-ssd-expert-offload

## 2. PagedMoE bitplane direct-consume (arXiv:2611.03173, 2026-11-02)
- Cross-precision expert cache: progressively refinable precision state; "Direct" consumes
  small weight tiles from cached bitplanes in fused GEMV WITHOUT materializing full BF16
  expert: **3.94-8.79× decode vs Materialize**. 24 GiB bitplane cache → 5.71/8.89 tok/s.
- Relevance: a *lossless-ish* (bitplane) representation consumed directly might cut bytes
  AND memory. But quality: cross-precision likely LOSSY (needs a quality audit). Note it.

## 3. MoEShift / Strata dynamic expert cache — beats our numbers on similar HW
- MoEShift v0.1.0-alpha (Apple Silicon only): Qwen3.8-Flash-Next (~62 GiB quant) at
  **~10 tok/s on 24 GB M4 Pro**, <10 GB peak. Strata dynamic expert cache (hottest experts
  across ALL layers): **>60 tok/s steady** on RTX with DDR5. (openclawdc.com article 2026-11-09;
  carteakey.dev Strata 74.8 tok/s). Their "dynamic expert cache across all layers" is exactly
  the shared-global-cache idea — worth checking why our pool=0 policy differs.
- URL: https://openclawdc.com/blog/can-a-24gb-mac-run-qwen-3-8-flash-next/

## 4. oMLX / Palm-Infra cross-layer prefetch + shared cache
- Palm-Infra: shared cache + cross-layer prefetch 10.89→12.70 tok/s, hit 74.5→89.3%
  (16 GiB cache). Cache sweep 1 GiB 12.38 / 10 GiB 16.19 / 16 GiB 16.53 tok/s. Qwen3.8
  NVFP4 best 21.8 decode. These are 64 GB numbers but the shared-cache+prefetch mechanism
  is relevant. URL: https://github.com/TencentYoutuResearch/Palm-Infra/blob/main/docs/ssd-offload.md

## 5. 64 GB / 128 GB reference points (for the memory-limit argument)
- 95.5 GiB Qwen3.8 on M5 Pro 64 GB: 27.6 tok/s with draft head, 18-18.6 without; 36 GiB
  stream cache; at 38 GiB collapses to 3.7. On 16 GB: resident portion alone fills memory.
- 128 GB M5 Max: oMLX with MTP+n-gram on SSD ~60 tok/s full 262K context; llama.cpp 59/46.
  Mainline llama.cpp moved decode 34→46.3 tok/s (+36%) from PR #27742 alone.
- URLs: note.com/hacklog_stealth/n/nd5833cb2bfe2 , heretik.io/qwen38-flash-next-mtp-macbook

## 6. mitln gpt-oss-120b engine (M1 Max): 7.84 vs TUFF 5.78 tok/s (+36%), bigger lead under
   memory pressure (1.72x at 4 GiB competing). Their engine keeps less file-cache/compressed
   memory — relevant to our single-tier / memory-pressure findings.
   URL: https://mitln.github.io/moe-inference-engine

## Verdict so far
The single most promising NOVEL direction is #1 (trained prerouter that commits predictions
as routing) — it directly attacks the I/O wait via accurate next-layer prefetch, with published
+59-84% decode on the exact hardware class, and is commercial-safe to be first. Quality must
be handled (Edge0 uses a recovery LoRA). Investigate deeply in the final report.
