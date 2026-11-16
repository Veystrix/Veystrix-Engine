"""STAGED — DO NOT RUN without the owner's explicit permission (AGENT_INVARIANTS.md).

One baked-weight ANE matmul island (ANEForge e5rt) vs MLX, with bitwise checks.
This is a *plan made executable*, not a benchmark result. It was written from
source inspection of ANEForge (github.com/sbryngelson/ANEForge) on 2026-11-17.

What it would measure, once authorized:
  1. Compile a single fp16 1x1-conv/matmul [1,C,1,S] -> [1,O,1,S] program with
     baked weights, target h16s (M4), via `aneforge` Python API.
  2. Compare ANE output against MLX fp16 and fp32 references at S=32,64,256,512
     and W-lane batching of 1 and 4 tokens (the only demonstrated multi-token
     ANE path).
  3. Report per-shape: ANE ms, MLX ms, max abs err, bitwise BF16 mismatch count,
     and top-1 argmax agreement — the exact gates for a bit-identical claim.

Prerequisites (NOT installed/run here): isolated venv at experiments/ane/V already
exists; `pip install aneforge` + its local dylib build would be needed. No changes
to repo .venv. All runs must be under scripts/flashnext/real_run_guard.py.

Authorisation status: NOT GRANTED. This file is documentation-in-code only.
"""
from __future__ import annotations

SHAPES = [(2560, 4096), (2688, 4096)]        # (inner, out) — Qwen q/o, Nemotron q/o
SEQ = (32, 64, 256, 512)
W_LANES = (1, 4)

MIL_TEMPLATE = """
program(1.3) {{
  func main<ios18>( tensor<fp16, [1,{inner},1,{seq}]> x ) {{
    tensor<fp16, [1,{out},1,{seq}]> y =
      conv(x, weight=W, strides=[1,1], pad=[0,0,0,0], dilation=[1,1], groups=1);
  }} -> (y);
}}
"""

GATES = {
    "speed": "ane_ms <= 0.8 * mlx_ms  (must be >=1.25x faster to be worth an island)",
    "token_exact": "top1 argmax agreement == 100% on 100 fixed prompts",
    "bitwise": "BF16 bit pattern mismatch count == 0 (else label token-exact only)",
}


def main():
    raise SystemExit(
        "STAGED ONLY: no ANE hardware measurement is authorised. "
        "See experiments/ane/FINAL_ANSWER.md and AGENT_INVARIANTS.md."
    )


if __name__ == "__main__":
    main()
