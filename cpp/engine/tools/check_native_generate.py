#!/usr/bin/env python3
"""CTest wrapper: run the native-generation Python check with a given binary.

Usage: check_native_generate.py --out <vectordir> --binary <path> --steps N
Delegates to gen_native_generate_vectors.py --check with the same arguments.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--binary", required=True)
ap.add_argument("--steps", type=int, default=3)
args = ap.parse_args()

gen = Path(__file__).resolve().parent / "gen_native_generate_vectors.py"
cmd = [sys.executable, str(gen), "--check", "--out", args.out,
       "--binary", args.binary, "--steps", str(args.steps)]
sys.exit(subprocess.call(cmd))
