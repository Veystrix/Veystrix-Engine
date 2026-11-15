# Lookup drafting (from HyperQwen) on the 16 GB Air, 15 November 2026

**Result: no gain on either model. Every new flag stays off.**

The owner asked for Nemotron's GPU use to go up, for Qwen to be faster, and for ideas from `syv-ai/HyperQwen` (vLLM patches for NVIDIA cards) to be used. Most of that repository is CUDA- or VRAM-specific (int8 tensor-core GEMMs, 16-bit recurrent state, requantised embeddings, split-KV attention kernels, Marlin fixes). The part that transfers is *drafting from the context*: when the reply copies text that is in the prompt, propose the copy and verify it in one pass. It is implemented here, tested, and measured on the real checkpoints. It does not help.

## What was built (all default off, all lossless)

| Flag | What it does |
| --- | --- |
| `VEYSTRIX_LOOKUP_V2=1` | HyperQwen's drafter: suffix keys up to 12 tokens, the whole prompt as history (was the last 2,048 tokens), a match may overlap its own suffix, the block doubles while a copy runs (Qwen up to 7 drafts, Nemotron up to 6). `models/shared/prompt_lookup.py` |
| `VEYSTRIX_LOOKUP_SAMPLING=1` | Lossless speculative sampling against point-mass drafts (`Sampler.verify_point_mass`, `veystrix/sampling.py`), so sampled replies (the Rakazo bots, T=1.0) can use lookup. Before this, both lookup paths ran only for greedy requests. |
| `FLASHNEXT_LOOKUP_ROLLBACK=1` | Qwen: keep the verified prefix from the verifier's intermediate GDN states instead of restoring the snapshot and replaying. Matches singleton decoding to about 1e-6, not bit for bit. |
| (always on when lookup runs) | `lookup_stats` flight-recorder event: rounds, drafts proposed and accepted, tokens per round. |

The enabling switches that already existed are unchanged: `FLASHNEXT_PROMPT_LOOKUP=1`, `VEYSTRIX_NEMOTRON_PROMPT_LOOKUP=1`. The Qwen path still requires BF16 KV (`quantized_kv.fingerprint_parts()` empty). The owner's saved Qwen setting is `tiered4`, so Qwen lookup never runs in the app; making it run would need rollback support in `MixedPrecisionQSAKVCache` (its `state` setter refuses a restore, so a trim-based rollback that avoids the 256-token page-aging step is needed), and the measurement below says it is not worth building.

## Tests

`veystrix/tests/test_lookup_v2.py` (13): drafter, gate, and a statistical check that the token emitted by sampled verification follows the target distribution for likely, unlikely, in-support and out-of-support drafts, with top-k, top-p and presence penalty. `Sampler.__call__` was refactored onto `Sampler.distribution`; draw-for-draw equal to the old sampler on 36 seeded draws. `test_speculative_tiny.py` (+3): partial-accept rounds in v2 and rollback modes keep greedy tokens and an identical cache; top-k 1 sampled lookup equals plain sampling; sampled lookup leaves the cache holding exactly the emitted tokens. `test_speed_features.py` (+1): the same for Nemotron. Run by a Haiku agent: 164 tests over nine modules, all pass.

## Real-checkpoint measurements

All runs: router started from this tree under `scripts/flashnext/real_run_guard.py`, one model at a time, greedy and sampled (T=1.0, top-p 0.95, top-k 20) copy-the-module prompt and a prose prompt, pool 0, BF16 KV for Qwen. **Deviation, at the owner's instruction: the 3 GiB available-memory floor was disabled for these runs** (swap, thermal and battery stops stayed; the compressor stop was raised to 9,000 MiB because Qwen made macOS compress about 5 GiB with the owner's apps open). Evidence: `results/nemotron35/20261115-20*-lookup-ab-nemotron-*`, `results/flashnext/20261115-19*-lookup-ab-qwen-*`.

**Nemotron** (floor 6.2 GiB, swap flat, thermal 0; decode tok/s from the usage log):

| Request | lookup off | old drafter | v2 + sampling | v2 rounds (accepted/proposed, tokens per round) |
| --- | ---: | ---: | ---: | --- |
| copy, greedy | 7.30 | 7.20 | **6.15** | 10 rounds, 33/36, 4.3 |
| prose, greedy | 7.92 | 8.12 | 7.06 | 2 rounds, 1/3 |
| copy, sampled | 6.99 | 7.10 | **6.19** | 12 rounds, 57/60, 5.75 |
| prose, sampled | 7.98 | 8.10 | 7.32 | 1 round, 0/2 |

Even with 92–95% of drafts accepted and 4–6 tokens per round, v2 is 11–16% slower on copying prompts. The round costs about 850 ms for about 4.6 tokens, against 137 ms for one plain token: **each drafted token routes to different experts, so extra rows add SSD reads instead of sharing them.** HyperQwen's speed-up depends on the weights being resident in VRAM, where extra rows are nearly free. The earlier measurement that a second verified Qwen row costs 0.73 of a round says the same.

**Qwen** (BF16 KV): plain decode 1.9–2.5 tok/s. One v2 run fired (12 rounds, 44/52 accepted, 4.7 tokens per round) and decoded at 1.47 tok/s, but that run was under heavy memory compression and was stopped by the compressor rule after the first request, so it is indicative only. The v2 + rollback run rarely fired (the chat model answered with tool-style text instead of copying). Outputs were not stable between greedy runs of the same prompt, so no digest comparison is claimed.

**Qwen prefill, route-union** (the unmeasured item A of the 15 November handover): with the old `prefill-16gb` profile a cold 937-token prefill took 126 s and 112 s; with `FLASHNEXT_PREFILL_ROUTE_UNION=1` it took 43 s, token digest identical (`f99e951f…`), floor 4.3 GiB. But that baseline is not the app's profile. Through the router (the app's path, 2,048-token balanced chunks) the same prompt prefills in 30.5 s with tiered4 KV, and 29.9 s with the union flags set: **no gain over what the app does today.** The 3x is a property of the slow comparison profile.

## The Nemotron GPU question

Unchanged from the 15 November handover: a decoded token needs about 41 ms of GPU work and 78–92 ms waiting for expert reads from the SSD, so the GPU sits near 30%. Lookup drafting does not change that, as above. A Mac with enough RAM to hold the experts (32 GB) is the lever that works; nothing in software on 16 GB does.

## Real traffic observations (the owner's Rakazo bots on Qwen, from the usage log)

Prompts of 11–15K tokens (26 tools plus bot instructions). The first request prefilled cold in 154 s (72 tok/s); the following ones reused 98–99% of the prefix from the SSD cache, so the cache works. Per request: 8–45 s of prefill for the short uncached tail and 1.6–2.9 tok/s decode.

## Findings about the CI runner (separate from the above)

The Flash-Next step of CI had never passed on a hosted macOS runner. Causes, in order found: `socket.getfqdn` took 35 s in `HTTPServer.server_bind` (fixed: `aacfbc7`); the runner is a 7 GB virtual machine whose GPU's largest Metal buffer is 3.76 GB, so the 4.6 GB pool tests cannot be built and free memory sits below Veystrix's own 3 GiB floor (snapshots refused, restores COLD); one keep-warm test raced a 10 ms timer against a 1 ms loop. `models/flashnext/hardware_probe.py` (`requires_full_mac`) skips the affected 15 classes on the runner, with the reason, when `VEYSTRIX_CI_RUNNER=1` (set by the workflow). Local runs are unchanged.
