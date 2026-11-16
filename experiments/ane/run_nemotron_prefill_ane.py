"""Run one Nemotron prefill with the ANE hook on or off, under the guard.

  VEYSTRIX_NEMOTRON_ANE_PREFILL=1 PYTHONPATH=.:experiments/ane \
    V/bin/python experiments/ane/run_nemotron_prefill_ane.py --tag ane

Uses the qualification harness so the measurement is the project-owned one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The hook reads the env at import time, so import it before the backend.
import ane_nemotron_hook  # noqa: F401

from veystrix.bench import qualification

CKPT = str(Path.home() / "models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community")


def main():
    args = sys.argv[1:]
    tag = "ane" if "ane" in args else "base"
    args = [a for a in args if a not in ("ane", "base")]
    argv = ["--model", "nemotron35", "--checkpoint", CKPT,
            "--prompt-tokens", "2048", "--output-tokens", "128",
            "--mode", "cold-process"] + args
    print(f"tag={tag} enabled={os.environ.get('VEYSTRIX_NEMOTRON_ANE_PREFILL')} argv={argv}", flush=True)
    if os.environ.get("ANE_TRACE") == "1":
        import traceback
        try:
            return qualification.main(argv)
        except Exception:
            traceback.print_exc()
            raise
    return qualification.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
