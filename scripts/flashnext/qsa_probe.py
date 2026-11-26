"""Tiny qwen4: indexer budget 16 (block_topk 4) so the reference uses its sparse QSA mask above ~16 tokens."""
import sys
from veystrix import port_check as pc
pc.TINY_QWEN4_EXP = dict(pc.TINY_QWEN4_EXP, indexer_budget=int(sys.argv[1]))
for c in pc.run_tiny_qwen4(quick=True):
    print("PASS" if c.ok else "FAIL", c.name, str(c.detail)[:160])
