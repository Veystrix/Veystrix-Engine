"""STAGED — DO NOT RUN without the owner's explicit permission (AGENT_INVARIANTS.md).

Design-in-code for expert-wise K=2 read sharing (the one bit-identical, 16 GB
candidate). Written 2026-11-17 from the offline route-union measurement
(ROUTE_UNION_FINDING.md: K=2 union ratio 0.796 -> ~59 ms/token potential save).

Mechanism (bit-identical by construction):
  For two consecutive decode tokens t, t+1 at a MoE layer:
    1. Compute the target router for BOTH tokens (both hidden states exist in one
       causal forward).
    2. Group tokens by expert. Distinct reads = |E_t U E_t+1|, not |E_t|+|E_t+1|.
    3. Read each distinct expert ONCE; run a 2-row grouped GEMM on its rows;
       accumulate into each token's residual in BASELINE EXPERT ORDER.
    4. Greedy accept: token t emitted only if its argmax matches the baseline.
  No rerouting, no gating change, no weight change, no pool change.

First guarded run (once authorized), under real_run_guard.py:
  - 2048-token prompt, 100 decoded tokens, greedy, fixed seed.
  - Arms: baseline vs K=2, alternating, 3 pairs each.
  - Collect: token SHA-256 (must be identical), physical SSD bytes/token
    (expect ~20% fewer on the shared layers), TPOT, guard telemetry.
  - Gate: byte reduction > 12% AND token digest identical AND guard clean.
  - Abort if available memory < 3 GiB or digest differs on any pair.

Status: NOT GRANTED. Design only — no runtime code patched.
"""
from __future__ import annotations

GATES = {
    "digest": "token_sha256 equal across all baseline and K=2 arms",
    "bytes": "physical bytes/token >= 12% lower in K=2 arms",
    "guard": "no 3 GiB floor crossing, no compressor growth stop",
}

# Where this would hook (read-only references, already audited):
HOOKS = {
    "router": "models/flashnext/routing.py (topk observer, set_route_observer)",
    "expert_read": "models/flashnext/expert_cache.py::_read_weights / _read_weights_with_pool",
    "grouped_gemm": "models/flashnext/metal_runtime.py::MetalMoEExecutor.execute (gather_qmm)",
    "bit_equal_model": "models/shared/dense_gemm.py::_bit_equal (verify reduction order)",
}


def main():
    raise SystemExit(
        "STAGED ONLY: K=2 read-sharing needs owner authorization for a guarded "
        "16 GB run. See experiments/ane/ROUTE_UNION_FINDING.md."
    )


if __name__ == "__main__":
    main()
