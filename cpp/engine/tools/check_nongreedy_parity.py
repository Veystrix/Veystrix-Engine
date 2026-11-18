#!/usr/bin/env python3
"""NON-GREEDY real-checkpoint parity: C++ sampler vs Python Sampler.

Claim under test
----------------
`veystrix.sampling.Sampler` (Python) and `veystrix::engine::sample_non_greedy`
(C++, cpp/engine/src/sampling.cpp) must produce the SAME token id for the same
logit row, the same Sampling preset and the same seed. Both implementations
call MLX's default-key categorical draw; Python seeds it with
`mx.random.seed(seed)` and C++ with `mx::random::seed(seed)`, which are the same
PRNG key sequence. Every other stage (presence penalty, temperature, top-k
argpartition, min-p, top-p nucleus) is already bit-exact for 11 synthetic cases.

Two modes
---------
* `sampler` (default, decisive): a guarded GREEDY C++ real-checkpoint run dumps
  raw per-step logits via REALGEN_DUMP_LOGITS. Those EXACT float32 rows are
  replayed through (a) Python Sampler and (b) C++ sample_non_greedy, both seeded
  with --seed. Identical inputs isolate the sampler from engine-logit drift.
* `e2e`: run the C++ real generator with the opt-in env
  VEYSTRIX_CPP_SAMPLING=<preset> + VEYSTRIX_CPP_SAMPLING_SEED=<seed>, and run the
  Python FlashNext backend with the same preset and mx.random.seed(seed).
  Compare the sampled token sequences AND logit digests step by step. The C++
  real generator and the Python backend are known to have DIFFERENT numerics
  (missing PLE / fused RoPE / chunked prefill; see
  research_notes/Cerebras inference and IO splitting/real_checkpoint_generate.md),
  so a mismatch here is an ENGINE gap, not a sampler gap.

HARD RULE: every real-checkpoint run is wrapped in
  scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 5120 \
    --compressor-growth-mib 8192 --thermal-level 2 --max-seconds <budget>
run from /Users/vincent/Documents/Veystrix. If the guard stops a run (exit 90)
this script aborts immediately and reports it.

Usage
-----
  .venv/bin/python cpp/engine/tools/check_nongreedy_parity.py \
      --mode sampler --preset thinking --seed 7 --tokens 8 \
      --outdir results/flashnext/nongreedy-parity-...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed environment of the owner's machine. Overridable via CLI for portability.
# ---------------------------------------------------------------------------
RUN_ROOT = Path("/Users/vincent/Documents/Veystrix")
CHECKPOINT = Path("/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
DEFAULT_REFERENCE = RUN_ROOT / "results/flashnext/20261117-222244-real-reference/reference.json"
GUARD = RUN_ROOT / "scripts/flashnext/real_run_guard.py"
VENV_PY = RUN_ROOT / ".venv/bin/python"
MLXROOT = RUN_ROOT / ".venv/lib/python3.12/site-packages/mlx"
WORKTREE = Path(__file__).resolve().parents[3]  # tools -> engine -> cpp -> <worktree>

STEP_RE = re.compile(r"^step=(\d+)\s+in=(-?\d+)\s+out=(-?\d+)\s+sha256=([0-9a-f]{64})$")

PRESETS = {
    # Mirrors veystrix.sampling.THINKING / INSTRUCT and SamplingParams in
    # cpp/engine/src/sampling.hpp. Values are float32 on both sides.
    "thinking": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
    "instruct": dict(temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sampler", "e2e", "all"), default="sampler")
    ap.add_argument("--preset", choices=("thinking", "instruct"), default="thinking")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--budget", type=int, default=900, help="--max-seconds for the guard")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--binary", default=None, help="prebuilt C++ real-gen binary; build if absent")
    ap.add_argument("--reuse-logits", default=None,
                    help="directory of existing REALGEN_DUMP_LOGITS .f32 rows; "
                         "skip the guarded greedy real run (logits are preset-independent)")
    return ap.parse_args()


def check_swap() -> None:
    import subprocess as sp
    out = sp.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    print(f"[preflight] {out}", flush=True)


# Invoke through xcrun: calling the absolute CLT clang++ path directly loses the
# SDK environment and breaks <algorithm> lookup. xcrun sets SDKROOT correctly.
CXX = ["xcrun", "clang++"]
CC = ["xcrun", "clang"]


def build_binary(bindir: Path) -> Path:
    """Compile the standalone real generator WITH sampling.cpp linked in.

    The historical build command in
    research_notes/.../real_checkpoint_generate.md omits sampling.cpp; adding
    the opt-in call requires it in the link line. This is the ONLY build change.
    """
    bindir.mkdir(parents=True, exist_ok=True)
    native_read_o = bindir / "native_read.o"
    eng = WORKTREE / "cpp/engine"
    # The historical note's short source list is stale: this tool now needs
    # prefix_cache/kv_quant/mtp/... . Build all engine sources except the two the
    # documented standalone command excludes (decoder.cpp, model_adapter.cpp).
    # sampling.cpp is in this set and is required by the new sample_non_greedy call.
    excluded = {"decoder.cpp", "model_adapter.cpp"}
    sources = sorted(p for p in (eng / "src").glob("*.cpp") if p.name not in excluded)
    cc = CC + ["-O2", "-c", str(RUN_ROOT / "models/flashnext/native_read.c"),
               "-o", str(native_read_o)]
    run(cc, cwd=RUN_ROOT, label="build native_read.o")
    cxx = CXX + ["-std=c++20", "-O2",
           f"-I{MLXROOT}/include", f"-I{eng}/src", f"-I{WORKTREE}/cpp/decode/src"]
    cxx += [str(s) for s in sources]
    cxx += [str(WORKTREE / "cpp/decode/src/io_pager.cpp"), str(native_read_o),
            str(eng / "tools/real_checkpoint_generate.cpp"),
            f"-L{MLXROOT}/lib", "-lmlx", f"-Wl,-rpath,{MLXROOT}/lib",
            "-o", str(bindir / "real_gen_sampling")]
    run(cxx, cwd=RUN_ROOT, label="build real_gen_sampling")
    return bindir / "real_gen_sampling"


def run(cmd: list[str], cwd: Path, label: str) -> subprocess.CompletedProcess:
    print(f"[$ {label}] " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-4000:], file=sys.stderr)
        print(p.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"{label} exited {p.returncode}")
    return p


def guarded(cmd: list[str], outdir: Path, name: str, budget: int,
            env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a real-checkpoint command under the owner's guard. Abort on guard stop."""
    log = outdir / f"{name}.guard.jsonl"
    wrapped = [
        str(VENV_PY), str(GUARD),
        "--floor-gib", "1.5",
        "--swap-growth-mib", "5120",
        "--compressor-growth-mib", "8192",
        "--thermal-level", "2",
        "--max-seconds", str(budget),
        "--log", str(log),
        "--", *cmd,
    ]
    e = os.environ.copy()
    if env:
        e.update({k: str(v) for k, v in env.items()})
    print(f"[guard:{name}] " + " ".join(wrapped), flush=True)
    p = subprocess.run(wrapped, cwd=str(RUN_ROOT), capture_output=True, text=True, env=e)
    (outdir / f"{name}.stdout").write_text(p.stdout)
    (outdir / f"{name}.stderr").write_text(p.stderr)
    if p.returncode == 90:
        raise RuntimeError(f"GUARD STOPPED {name} (exit 90). Aborting. See {log}")
    if p.returncode != 0:
        print(p.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"{name} exited {p.returncode}")
    return p


def parse_steps(stdout: str) -> list[dict]:
    steps = []
    for line in stdout.splitlines():
        m = STEP_RE.match(line.strip())
        if m:
            steps.append(dict(step=int(m.group(1)), token_in=int(m.group(2)),
                              token_out=int(m.group(3)), digest=m.group(4)))
    return steps


# ---------------------------------------------------------------------------
# Python reference: replay identical real logits through veystrix Sampler.
# ---------------------------------------------------------------------------
def python_replay(logit_dir: Path, vocab: int, n: int, preset: str, seed: int) -> dict:
    import mlx.core as mx
    sys.path.insert(0, str(WORKTREE))
    from veystrix.sampling import Sampler, Sampling, THINKING, INSTRUCT

    settings = Sampling(**(THINKING if preset == "thinking" else INSTRUCT))
    mx.random.seed(seed)                      # same key sequence as C++ random::seed
    sampler = Sampler(settings)               # tracks _seen like _note_sampled
    tokens = []
    for step in range(n):
        path = logit_dir / f"logit.step{step}.f32"
        raw = path.read_bytes()
        if len(raw) != vocab * 4:
            raise RuntimeError(f"{path}: {len(raw)} bytes, expected {vocab * 4}")
        row = mx.array(list(struct.unpack("<" + "f" * vocab, raw)), mx.float32).reshape(1, vocab)
        tok = sampler(row)
        mx.eval(tok)
        tokens.append(int(tok.item()))
    return dict(tokens=tokens, preset=preset, seed=seed,
                settings={k: getattr(settings, k) for k in PRESETS[preset]})


# ---------------------------------------------------------------------------
# C++ reference: compile a tiny stdin driver against sampling.cpp (no source
# file is added to the repo), replay the SAME logits through sample_non_greedy.
# ---------------------------------------------------------------------------
CPP_REPLAY_DRIVER = r"""
#include "sampling.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
namespace mx = mlx::core;
using namespace veystrix::engine;
int main(int argc, char** argv) {
  if (argc != 6) { std::fprintf(stderr, "usage: %s vocab seed preset dir prefix\n", argv[0]); return 2; }
  int vocab = std::atoi(argv[1]);
  uint64_t seed = std::strtoull(argv[2], nullptr, 10);
  std::string preset(argv[3]);
  std::string dir(argv[4]), prefix(argv[5]);
  SamplingParams p;
  if (preset == "thinking") p = SamplingParams{1.0f, 0.95f, 20, 0.0f, 0.0f};
  else if (preset == "instruct") p = SamplingParams{0.7f, 0.80f, 20, 0.0f, 1.5f};
  else { std::fprintf(stderr, "bad preset\n"); return 2; }
  mx::random::seed(seed);  // same sequence as Python mx.random.seed
  std::vector<int32_t> seen;
  for (int step = 0; ; ++step) {
    char path[2048];
    std::snprintf(path, sizeof(path), "%s/%s.step%d.f32", dir.c_str(), prefix.c_str(), step);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) break;
    std::vector<float> buf(vocab);
    size_t n = std::fread(buf.data(), sizeof(float), vocab, f);
    std::fclose(f);
    if (n != (size_t)vocab) { std::fprintf(stderr, "short read step %d\n", step); return 2; }
    mx::array logits(buf.data(), mx::Shape{1, vocab}, mx::float32);
    mx::array tok = sample_non_greedy(logits, p, seen);
    mx::eval(tok);
    int id = tok.data<uint32_t>()[0];
    std::printf("%d\n", id);
    if (p.presence_penalty != 0.0f) {
      bool found = false;
      for (int32_t s : seen) if (s == id) { found = true; break; }
      if (!found) seen.push_back((int32_t)id);
    }
  }
  return 0;
}
"""


def cpp_replay(binary_dir: Path, logit_dir: Path, vocab: int, n: int,
               preset: str, seed: int) -> dict:
    binary_dir.mkdir(parents=True, exist_ok=True)
    src = binary_dir / "replay_driver.cpp"
    src.write_text(CPP_REPLAY_DRIVER)
    drv = binary_dir / "cpp_replay"
    eng = WORKTREE / "cpp/engine"
    cxx = CXX + ["-std=c++20", "-O2",
           f"-I{MLXROOT}/include", f"-I{eng}/src",
           str(src), str(eng / "src/sampling.cpp"),
           f"-L{MLXROOT}/lib", "-lmlx",
           f"-Wl,-rpath,{MLXROOT}/lib",
           "-o", str(drv)]
    run(cxx, cwd=RUN_ROOT, label="build cpp_replay driver")
    p = run([str(drv), str(vocab), str(seed), preset, str(logit_dir), "logit"],
            cwd=RUN_ROOT, label="cpp_replay")
    toks = [int(x) for x in p.stdout.split()]
    return dict(tokens=toks[:n], preset=preset, seed=seed)


def compare(a: list[int], b: list[int]) -> dict:
    n = min(len(a), len(b))
    diffs = [i for i in range(n) if a[i] != b[i]]
    return dict(n=n, match=(len(a) == len(b) and not diffs),
                diffs=diffs, a=a, b=b)


# ---------------------------------------------------------------------------
# Python backend capture for the e2e arm (guarded, real checkpoint).
# ---------------------------------------------------------------------------
def python_backend_snippet(preset: str, seed: int, prompt_tokens: int,
                           ntokens: int, checkpoint: Path) -> str:
    return f'''
import os, json, hashlib
import mlx.core as mx
import numpy as np
os.environ["VEYSTRIX_SAMPLING_GREEDY"] = "0"
from veystrix.sampling import Sampler, Sampling, THINKING, INSTRUCT
from veystrix.backends.flashnext import FlashNextBackend

preset, seed = {preset!r}, {seed}
settings = Sampling(**(THINKING if preset == "thinking" else INSTRUCT))
orig = Sampler.__call__
digests = []
def cap(self, logits):
    row = np.asarray(logits.reshape(-1).astype(mx.float32), dtype="<f4")
    digests.append(hashlib.sha256(row.tobytes()).hexdigest())
    return orig(self, logits)
Sampler.__call__ = cap

backend = FlashNextBackend(model_path={str(checkpoint)!r})
backend.sampling = settings  # override the greedy startup default
# The C++ tool has no stop-token logic: it emits exactly --tokens ids. The chat
# backend stops on EOS. Disable stops so both harnesses run the same horizon;
# otherwise a sampled EOS would truncate Python and fake a token-sequence diff.
backend.stops = set()
rng = np.random.default_rng(20261117)
prompt = rng.integers(0, 200000, size={prompt_tokens}).tolist()
backend.append_tokens(prompt)
tokens = []
def emitted(t, _p): tokens.append(int(t))
mx.random.seed(seed)  # EXACTLY the moment before the sampler starts drawing
_text, stats = backend.generate({ntokens}, on_decode_token=emitted)
print("PY_SAMPLED_JSON=" + json.dumps({{
    "tokens": tokens, "digests": digests,
    "preset": preset, "seed": seed,
    "settings": {{k: getattr(settings, k) for k in ("temperature","top_p","top_k","min_p","presence_penalty")}},
}}))
'''


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    bindir = outdir / "bin"
    report: dict = dict(mode=args.mode, preset=args.preset, seed=args.seed,
                        tokens=args.tokens, prompt_tokens=args.prompt_tokens,
                        checkpoint=str(CHECKPOINT),
                        guard_flags=dict(floor_gib=1.5, swap_growth_mib=5120,
                                         compressor_growth_mib=8192,
                                         thermal_level=2, max_seconds=args.budget))

    check_swap()
    binary = Path(args.binary) if args.binary else build_binary(bindir)
    report["binary"] = str(binary)
    (outdir / "build_command.txt").write_text(
        "# standalone build with sampling.cpp linked in; see "
        "docs/engineering/nongreedy_real_parity.md\n" + str(binary) + "\n")

    reference = Path(args.reference)
    # Regenerate the flat config with THIS worktree's gen_real_config.py: the
    # 2026-11-17 reference's config predates PLE and lacks shard.*.ple.* lines,
    # which the current tool requires. Written only into our owned outdir.
    config = outdir / "real_config.txt"
    gen_cfg = WORKTREE / "cpp/engine/tools/gen_real_config.py"
    run([str(VENV_PY), str(gen_cfg), "--checkpoint", str(CHECKPOINT),
         "--reference", str(reference), "--out", str(config)],
        cwd=RUN_ROOT, label="gen_real_config")
    vocab = None
    for line in config.read_text().splitlines():
        if line.startswith("vocab="):
            vocab = int(line.split("=", 1)[1])
    if vocab is None:
        raise RuntimeError(f"no vocab= in {config}")
    report["vocab"] = vocab

    ok = True
    try:
        if args.mode in ("sampler", "all"):
            # Guarded greedy real run to dump real logits (or reuse an existing
            # dump: logits do not depend on the sampling preset).
            logit_dir = Path(args.reuse_logits).resolve() if args.reuse_logits else outdir / "logits"
            logit_dir.mkdir(parents=True, exist_ok=True)
            if args.reuse_logits:
                print(f"[reuse] logits from {logit_dir} (no real-checkpoint run)")
            else:
                env = dict(REALGEN_DUMP_LOGITS=str(logit_dir / "logit"))
                p = guarded([str(binary), str(CHECKPOINT), str(reference),
                             str(args.tokens), str(config)],
                            outdir, "greedy_logit_dump", args.budget, env=env)
                greedy = parse_steps(p.stdout)
                report["greedy_steps"] = greedy
            report["logit_dir"] = str(logit_dir)
            py = python_replay(logit_dir, vocab, args.tokens, args.preset, args.seed)
            cpp = cpp_replay(bindir, logit_dir, vocab, args.tokens,
                             args.preset, args.seed)
            cmp = compare(py["tokens"], cpp["tokens"])
            report["sampler"] = dict(python=py, cpp=cpp, comparison=cmp)
            print("\n=== SAMPLER PARITY (identical real logits) ===")
            print(f"python: {py['tokens']}")
            print(f"cpp:    {cpp['tokens']}")
            print(f"exact sampled token sequences: "
                  f"{'MATCH' if cmp['match'] else 'DIFF'} (diff steps {cmp['diffs']})")
            ok &= cmp["match"]

        if args.mode in ("e2e", "all"):
            # C++ real gen with the opt-in env.
            env = dict(VEYSTRIX_CPP_SAMPLING=args.preset,
                       VEYSTRIX_CPP_SAMPLING_SEED=str(args.seed))
            p = guarded([str(binary), str(CHECKPOINT), str(reference),
                         str(args.tokens), str(config)],
                        outdir, "cpp_sampled", args.budget, env=env)
            cxx_steps = parse_steps(p.stdout)
            report["cpp_sampled_steps"] = cxx_steps

            # Python backend with the same preset + seed (guarded).
            snippet = python_backend_snippet(args.preset, args.seed,
                                             args.prompt_tokens, args.tokens, CHECKPOINT)
            (outdir / "python_backend_capture.py").write_text(snippet)
            q = guarded([str(VENV_PY), str(outdir / "python_backend_capture.py")],
                        outdir, "python_sampled", args.budget)
            marker = "PY_SAMPLED_JSON="
            line = next((l for l in q.stdout.splitlines() if marker in l), None)
            if line is None:
                raise RuntimeError("python backend did not emit PY_SAMPLED_JSON")
            pycap = json.loads(line.split(marker, 1)[1])
            report["python_sampled"] = pycap

            cxx_tokens = [s["token_out"] for s in cxx_steps]
            cmp = compare(cxx_tokens, pycap["tokens"])
            dig_cmp = compare([s["digest"] for s in cxx_steps], pycap["digests"])
            report["e2e"] = dict(comparison=cmp, digest_comparison=dig_cmp)
            print("\n=== END-TO-END (independent engine logits) ===")
            print(f"cpp sampled:    {cxx_tokens}")
            print(f"python sampled: {pycap['tokens']}")
            print(f"token sequence: {'MATCH' if cmp['match'] else 'DIFF'}")
            print(f"logit digests:  {'MATCH' if dig_cmp['match'] else 'DIFF'}")
            print("(an e2e DIFF reflects the documented engine-logit gap, "
                  "not the sampler; see sampler mode above.)")
    finally:
        report["ok"] = bool(ok)
        (outdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {outdir / 'report.json'}")

    if args.mode == "e2e":
        # e2e is a demonstration, not the parity gate; do not fail on engine gap.
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
