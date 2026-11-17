# Direct search leads 2 — second miracle, 2026-11-17

## 4. Global Agent 16 GB Mac benchmarks — IMPORTANT reference point
globalagent.dev/moe-mac-inference-results (M5 Pro Mac, RAM sandboxed, 2026-10-09):
- Qwen3.8-Flash-Next 229B 6-bit, 16 GB: exact 3.53 tok/s (top-1 agree 1.00), speed 17.1 (top-1 0.30).
- Coder-Next 80B-A3B 6-bit, 16 GB: exact 6.44 tok/s, speed 41.2 (top-1 0.91).
- 32 GB: Flash-Next exact 6.18, 48 GB 7.88. 235B 4-bit 16 GB exact 1.13.
- "speed" = speed slider 20 (lossy routing/experts), exact keeps every routed expert.
Implication: a COMMERCIAL engine already gets 3.53 exact on the same model class at 16 GB, vs our 2.3. Our target is 5 — between their 16 GB (3.53) and 32 GB (6.18). The 6-bit vs our Q4, and a different streaming engine, likely explain part. But it shows 5 exact at 16 GB is NOT unprecedented in the industry; worth asking Opus to find the delta. Speed mode's 17 tok/s confirms the lossy path is where 15+ lives — forbidden.

## 5. Edge0 detail (read the Origins AI comparison)
Edge0 on 16 GB MacBook M2 (18.4 GiB checkpoint doesn't fit): blocked time on expert loads 244.0 -> 101.9 ms/step at K=4, +82% decode, by making the prerouter prediction BE the routing. This is the committed-routing path that fails our no-quality-loss rule. As a HINT (our E1), perfect future knowledge gives only +25% — so committing the routing is where Edge0's extra gain comes from, and we cannot use it.

## 6. DeepInfer-Mac (750B on 24 GB, arXiv 2610.36564)
Mixed low-bit, 338 GiB on SSD, 7.6 GiB resident, routed experts+Engram loaded on demand; "each token waits for a succession of storage reads." 78 tests pass but quality loss from low-bit not established. Local judgement: another example of hitting the storage wall, not a solution.
