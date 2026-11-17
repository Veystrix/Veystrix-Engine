# Final-head divergence: next diagnostic step

## Symptom

The real C++ generator (`cpp/engine/tools/real_checkpoint_generate.cpp`) produces
step-0 logits whose argmax matches Python (token **271**), but whose values differ
systematically:

- linear fit `c++ ≈ 0.77 * python + const`
- correlation `0.90`
- max abs diff `3.8`

All GDN / prefill / quantized-matmul primitives are already bit-exact against
Python in synthetic tests, and PLE was just fixed (one-centered RMS norm). The
remaining step-0 gap must therefore be in the **final head composition**:

```
hidden_states [B,S,hc*H]
   -> hyper_connection_mixer (Qwen4ExpGatedResidual, use_combine=False)
   -> mixed_input [B,S,H]
   -> lm_head (quantized matmul)
   -> logits [B,S,vocab]
```

We need to decide whether the `0.77` scale is introduced **before** `lm_head`
(in the mixer) or **by** the `lm_head` projection itself.

## Plan

1. Capture the Python intermediates with the OWNER-RUN-only diagnostic
   `cpp/engine/tools/dump_python_head.py`. It mirrors `capture_real_reference.py`
   (same `FlashNextBackend`, greedy, deterministic 16-token prompt from
   `np.random.default_rng(20261117)`, all production patches active) and dumps,
   for the **last prompt token** (index -1) as fp32 `.npy`:

   - `prefill_mixer_in.npy`  — `[B, hc*H]`, tensor fed into the final mixer
   - `prefill_mixer_out.npy` — `[B, H]`, mixer output / `lm_head` input
   - `prefill_logits.npy`    — `[B, vocab]`, final logits row
   - plus `decode_mixer_*.npy` / `decode_logits.npy` for the first decode step
   - `hc_norm_weight.npy` and `meta.json` (shapes, dtypes, sample values, and
     whether the one-centered RMSNorm patch is active)

   Verify the hook first with the tiny synthetic smoke test (no checkpoint):

   ```
   /Users/vincent/Documents/Veystrix/.venv/bin/python \
     cpp/engine/tools/dump_python_head.py --self-test --out /tmp/head_synth
   ```

2. Add a matching C++ dump of `final_mixed` (the mixer output) and `final_logits`
   for the same prompt, using the existing inline capture block. The C++ tool
   already writes `debug_capture::get().add("final_mixed", mixed)` and
   `("final_logits", logits)` at
   `cpp/engine/tools/real_checkpoint_generate.cpp:770-794`.

3. Compare, in order:

   | Compare | If they match | If they differ |
   |---|---|---|
   | `prefill_mixer_out.npy` vs C++ `final_mixed` | mixer is correct; scale is in `lm_head` | scale is in the mixer (RMSNorm gain, `hc_count` divide, sigmoid/silu, mean reduction) |
   | `prefill_logits.npy` vs C++ `final_logits` | head composition is fully matched | if `final_mixed` matches but logits do not, the `lm_head` quantized matmul is the culprit |

   A ratio of `c++ / python ≈ 0.77` already present on `final_mixed` localises the
   bug to the mixer; if it only appears on `final_logits`, it is the projection.

## File:line anchors

- Python final mixer call: `mlx_vlm/models/qwen4_exp/language.py:823-829`
  (`return self.hyper_connection_mixer(hidden_states)` in `Qwen4ExpModel.__call__`).
- Python gated residual: `mlx_vlm/models/qwen4_exp/language.py:442-454`
  (`Qwen4ExpGatedResidual.__call__`: grouped RMSNorm -> down/silu/up sigmoid ->
  reshape to `[hc_count, H]` -> `mean` over `hc_count`).
- Python one-centered RMSNorm patch: `models/flashnext/patch_rmsnorm.py:97-121,150-162`.
- C++ math: `cpp/engine/src/engine.cpp:165-186` (grouped norm + mixer math).
- C++ final head: `cpp/engine/src/engine.cpp:185-206` (`final_head`, incl.
  `lm_head`).
- C++ tool inline capture: `cpp/engine/tools/real_checkpoint_generate.cpp:770-794`
  (`final_mixed` before the quantized `lm_head`, `final_logits` after it).

> Note: line numbers above are from the installed `mlx_vlm` in the Veystrix venv
> at the time of writing; the diagnostic identifies the mixer by instance
> (`use_combine=False` / no `block_inject_weight`) rather than by line number, so
> it is robust to small version drift.
