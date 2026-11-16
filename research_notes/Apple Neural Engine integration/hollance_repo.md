# Cross-check: hollance/neural-engine (2.6k stars, MIT)

Author is an ML consultant, not Apple; everything is experimental and the repo
itself warns it is incomplete/possibly wrong. Community-maintained, last
substantive content largely iOS-era (A12). Source:
https://github.com/hollance/neural-engine — docs/ read 2026-11-17.

## Confirms what we measured
- You **cannot force** Core ML onto the ANE; `computeUnits = .all` is a hint,
  not a guarantee. A single unsupported layer makes Core ML fall back to GPU
  or CPU (`running-on-ane.md`).
- Unsupported ANE layers include **gather**, custom layers, RNN/LSTM/GRU,
  broadcastable and "ND" layers, some pooling/upsampling (`unsupported-layers.md`).
  This is why Qwen's routed experts and our gather-heavy paths cannot run
  end-to-end on ANE, and matches our private-API/dequant findings.
- ANE uses **float16 for everything**; activations >1e2 or <1e-4 lose precision
  (`16-bit.md`). Matches our found max abs diff ~0.004 and the CoreML-LLM
  bit-exactness failure in the handoff report.
- Shared memory still has a **format-conversion cost** when switching
  processors ("data needs to be converted into a suitable format; on GPU it
  must be put into a texture") (`ane-vs-gpu.md`). This is the same root cause
  as our ~21 ms marshalling, stated independently.
- No public API to program ANE; Core ML uses private frameworks; Metal cannot
  touch ANE (`programming-ane.md`, `ane-vs-gpu.md`).
- Placement verification methods we used are the documented ones: compare
  `.all` vs `.cpuAndGPU` vs `.cpuOnly`, `H11ANEServicesThread`, symbolic
  breakpoints on `-[_ANEModel program]` and Espresso engines, and `sudo
  powermetrics` (`is-model-using-ane.md`).

## New details useful to us
- Core ML can switch devices **more than once**: ANE→CPU→ANE→CPU does happen
  because ANE↔CPU switching is cheaper than ANE↔GPU (`unsupported-layers.md`).
  Relevant if a future split model has isolated unsupported ops.
- Anecdote: Core ML's "smart compute system" may run on GPU when GPU is idle
  and on NPU when GPU is **heavily loaded** (`other.md`). Our decode leaves the
  GPU mostly idle during SSD waits; prefill loads it. This may partly explain
  why placement differs by phase and why ANE side-load adds contention.
- Anecdote: FP16-weight mlmodels appear **slower** than FP32 on ANE (author
  unsure why) (`other.md`). We only compared fp16 vs INT8, not fp32 — a
  possible cheap follow-up.
- Mitigation when a model is mostly ANE-compatible: **model surgery** —
  replace `addBroadcastable` with old `add`, `ConcatND` with `Concat`, etc.,
  or split the model so the compatible prefix runs on ANE (`model-surgery.md`).

## What it does NOT solve
- No zero-copy interface, no buffer-sharing API, no solution to our two
  allocations. The repo predates Core AI/MLTensor.
- No LLM/MoE-specific guidance; no answer for 4-bit expert gather or KV state.
- Quantization note: "no evidence the ANE does 8-bit ops directly"; 8-bit
  support is for matmul layers and it is unknown whether ANE or CPU executes
  them (`16-bit.md`). Our compute plan showed ANE for INT8 mlprogram ops, but
  static plans are not proof of runtime placement — keep power-counter
  verification in any future claim.

## Bottom line
Good confirmation of the diagnosis (format conversion + placement hints +
gather/unsupported layers + FP16 numerics), and two cheap ideas (fp32-weight
comparison; watching GPU load before assuming ANE placement). No new mechanism
to remove the MLX↔Core ML boundary cost.
