# Non-greedy sampling parity: C++ engine vs `veystrix.sampling.Sampler`

**Status (measured, 2026-11-18):** bit-exact for all 11 parity cases, including
the exact THINKING and INSTRUCT presets, presence penalty, `top_k=0`, `min_p>0`,
and seeded categorical draws. No fix to `sampling.cpp` was required — the
existing implementation already matches Python.

## What was compared

Python reference: `/Users/vincent/Documents/Veystrix/veystrix/sampling.py`
(`Sampler.distribution`, `Sampler._mask`, `Sampler.__call__` non-greedy branch).
C++ port: `cpp/engine/src/sampling.hpp` / `sampling.cpp`
(`sampling_distribution`, `sampling_mask`, `sample_non_greedy`, `SamplingParams`).

Op-by-op the two implementations agree (all claims below are *measured* via
bit-exact vector comparison unless noted):

| Python op | C++ op | Verdict |
|---|---|---|
| `row = logits.reshape(-1)` | `mx::reshape(logits, {-1})` | bit-exact |
| `hits.at[seen].add(penalty)` on sorted int32 `seen` | equality-mask `where(is_seen, pen, 0)` over unique ids | bit-exact (seen ids are a set) |
| `row.astype(float32) / temperature` | `mx::astype(row, float32) / temperature` | bit-exact |
| `argpartition(row, -k)[-k:]`; min floor; `-inf` mask; `sort(kept)` | same sequence with slice `[n-k:n]` | bit-exact |
| `softmax`; `min_p * max(p)`; `where(p<floor,-inf)`; re-softmax | same, with `-inf` as float32 scalar | bit-exact |
| `argsort(-p)`; `cumsum`; `keep = carried-ordered < top_p` | same | bit-exact |
| `allowed = zeros(bool); allowed.at[order].add(keep)` | `take(keep, argsort(order))` (inverse permutation) | bit-exact |
| `mx.random.categorical(masked)`; `take(kept, drawn)`; `reshape(1).astype(uint32)` | `mx::random::categorical(masked[, key])`; same remap/shape | bit-exact |

Key detail on `allowed`: Python's scatter `allowed[order[i]] = keep[i]` equals
`take(keep, argsort(order))` because `order` is a permutation of `0..n-1`, so
`argsort(order)` is its inverse. Verified by the passing top-p cases.

PRNG: both sides use MLX 0.32.2 (measured: `mx.__version__ == "0.32.2"` in the
`.venv` and `MLX_VERSION` used by CMake). `mx.random.seed(seed)` (Python) and
`mlx::core::random::seed(seed)` (C++) seed the same process-wide default key, and
`categorical` on the default key is bit-identical (all 11 seeded tokens match).
The optional explicit `key` overload on `sample_non_greedy` is not exercised by
the parity test (the production path uses the default key).

## Vector generation

Generator: `cpp/engine/tools/gen_sampling_vectors.py` (uses
`/Users/vincent/Documents/Veystrix/.venv/bin/python`). All expected values come
from the real production `Sampler` ops, not a reimplementation.

Fresh vectors were written to a temp directory (the fixed
`cpp/engine/tests/sampling_vectors` dir was deliberately left untouched):
`/private/var/folders/37/fn6fxqyd2rgfmvc4954dggbm0000gn/T/opencode/sampling_vectors_fresh`.

One case was added to the generator to cover the previously missing exact
INSTRUCT preset: `case10_instruct_topk` (temperature 0.7, top_p 0.80, top_k 20,
min_p 0.0, presence_penalty 1.5, seen ids `[2,7,13,31,55,80]`, seed 21).

Cases and coverage:

- `case5_thinking_full` — THINKING preset (1.0 / 0.95 / 20 / 0 / 0)
- `case10_instruct_topk` — INSTRUCT preset (0.7 / 0.80 / 20 / 0 / 1.5) + seen
- `case6_presence_full`, `case7_presence_topk`, `case10_instruct_topk` — presence penalty / seen tokens
- `case0`, `case1`, `case3`, `case4`, `case6`, `case9` — `top_k=0` full-vocab path
- `case3_minp_only`, `case7_presence_topk` — `min_p > 0`
- all 11 — fixed-seed categorical draw (seeds 11–21)

## Exact test output (measured)

Build: `cmake --build cpp/engine/build --target test_sampling -j4` succeeded.

Run: `./test_sampling <fresh-vector-dir>`:

```
  [case0_temp_only] OK (distribution + seeded draw bit-identical)
  [case1_temp_scaled] OK (distribution + seeded draw bit-identical)
  [case2_topk_only] OK (distribution + seeded draw bit-identical)
  [case3_minp_only] OK (distribution + seeded draw bit-identical)
  [case4_topp_only] OK (distribution + seeded draw bit-identical)
  [case5_thinking_full] OK (distribution + seeded draw bit-identical)
  [case6_presence_full] OK (distribution + seeded draw bit-identical)
  [case7_presence_topk] OK (distribution + seeded draw bit-identical)
  [case8_topk_one] OK (distribution + seeded draw bit-identical)
  [case9_dominant_nucleus] OK (distribution + seeded draw bit-identical)
  [case10_instruct_topk] OK (distribution + seeded draw bit-identical)
test_sampling OK (11 cases bit-identical to Python)
EXIT=0
```

The comparison is byte-for-byte (`mx::astype(..., uint32)` views), not tolerance
based; it covers `masked`, the top-k survivor ids `kept`, and the seeded final
`token`.

## Files changed

- `cpp/engine/tools/gen_sampling_vectors.py` — added `case10_instruct_topk`.
- This document.
- `cpp/engine/src/sampling.cpp`, `cpp/engine/src/sampling.hpp`,
  `cpp/engine/tests/test_sampling.cpp` — **unchanged** (no fix needed).

## Caveats / remaining work (measured)

1. **Not yet wired into production.** The native generation tool
   (`cpp/engine/tools/veystrix_native_generate.cpp:256`) still calls
   `sample_greedy`; `sample_non_greedy` is exposed in `sampling.hpp` and verified
   by `test_sampling`, but the decode loop does not call it yet. Wiring it in
   (and threading `SamplingParams` / seen-token history) is outside the scope of
   this task's allowed files and is left to the parent.
2. **Explicit-key overload untested.** `sample_non_greedy(..., key)` is provided
   but not covered by the parity vectors; production uses the default key.
3. Seeded draws are verified for seeds 11–21 on small synthetic vocab rows
   (32–128). This establishes PRNG parity for the tested seeds/shapes, not an
   exhaustive proof over all seeds/vocab sizes (estimate: the shared MLX PRNG
   makes this general, but it was not enumerated).
