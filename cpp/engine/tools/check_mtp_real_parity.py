#!/usr/bin/env python3
"""Bit-exact Python <-> C++ check of the REAL checkpoint's MTP draft head.

This replaces the synthetic-weight MTP smoke test as the parity evidence. The
same hidden stream and the same token id are fed to

  * Python: models.flashnext.mtp.forward(language, hidden, token,
                                         mtp.make_cache(language)) -- i.e. the
    checkpoint's own Qwen4ExpMTPModule, with the real shard-22 weights and the
    real quantized embed_tokens / lm_head; and
  * C++:    mtp_draft_forward_real() in cpp/engine/src/mtp.cpp, driven through
    real_checkpoint_generate.cpp's VEYSTRIX_CPP_MTP_REAL_PARITY entry point.

Only the 76 shard-22 MTP tensors, the embedding table and lm_head are loaded on
the C++ side -- the 48-layer backbone and its expert pools are not needed for
this head-level comparison. The hidden stream is genuine: Python runs the real
model on a short prompt and dumps the last layer's pre-mixer hidden state.

The comparison is raw byte equality of the f32 logits (and of the argmax token).
No tolerance and no "close enough" reporting.

The script runs in three phases so that the heavy Python model is fully torn
down before the C++ binary starts:

  --phase gen       load the model, dump hidden/token/logits + a flat config
  --phase compare   run the C++ binary on those inputs, compare bytes
  (no --phase)      run gen then compare as two child processes

Usage (from /Users/vincent/Documents/Veystrix, under real_run_guard.py):
  .venv/bin/python <worktree>/cpp/engine/tools/check_mtp_real_parity.py \
      --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --binary /tmp/real_gen_mtp --out /tmp/mtp_real_parity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
if __import__("importlib").util.find_spec("mlx_vlm") is None:
    _VENV = Path("/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages")
    if _VENV.is_dir():
        sys.path.insert(0, str(_VENV))

PROMPT_SEED = 20261117
PROMPT_TOKENS = 8
MTP_PREFIX = "language_model.mtp."


def _hidden_shape(config: dict) -> tuple[int, int]:
    tc = config["text_config"]
    return int(tc["hidden_size"]), int(tc["hc_count"])


def write_config(out: Path, checkpoint: Path, config: dict) -> None:
    """Flat config for DenseLoader: geometry + shard map for the MTP tensors."""
    tc = config["text_config"]
    rp = tc["rope_parameters"]
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    lines: list[str] = []

    def kv(k: str, v) -> None:
        lines.append(f"{k}={v}")

    kv("hidden_size", tc["hidden_size"])
    kv("hc_count", tc["hc_count"])
    kv("hc_lowrank", tc["hc_lowrank"])
    kv("n_heads", tc["num_attention_heads"])
    kv("kv_heads", tc["num_key_value_heads"])
    kv("head_dim", tc["head_dim"])
    kv("n_experts", tc["num_experts"])
    kv("top_k", tc["num_experts_per_tok"])
    kv("inter", tc["moe_intermediate_size"])
    kv("shared_inter", tc["shared_expert_intermediate_size"])
    kv("vocab", tc["vocab_size"])
    kv("eps", tc["rms_norm_eps"])
    quant = config.get("quantization", {})
    kv("group_size", quant.get("group_size", 32))
    kv("bits", quant.get("bits", 4))
    kv("rope_base", rp["rope_theta"])
    kv("partial_rotary_factor", rp["partial_rotary_factor"])
    kv("rope_max_pos", tc["max_position_embeddings"])
    kv("rope_mrope", ",".join(str(x) for x in rp["mrope_section"]))
    # The real checkpoint uses the patched one-centered norm gains and an
    # untied lm_head (config.json tie_word_embeddings=false).
    kv("one_centered_norms", 1)
    kv("tie_word_embeddings", 1 if config.get("tie_word_embeddings", False) else 0)

    wanted = [k for k in wmap if k.startswith(MTP_PREFIX)]
    wanted += [k for k in wmap if k.startswith("language_model.model.embed_tokens.")]
    wanted += [k for k in wmap if k.startswith("language_model.lm_head.")]
    for name in wanted:
        kv("shard." + name, wmap[name])
    (out / "config.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'config.txt'}: {len(wanted)} shard entries, "
          f"{len(lines) - len(wanted)} geometry lines", flush=True)


def digest_f32(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise SystemExit(f"{path}: byte count {len(raw)} is not a multiple of 4")
    return hashlib.sha256(raw).hexdigest(), len(raw) // 4


def phase_gen(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)

    from models.flashnext import mtp as mtp_mod
    from models.flashnext.loader import load_streaming

    model, config, _store = load_streaming(
        str(checkpoint), verbose=False, keep_vision=False, use_mtp=True)
    language = model.language_model
    H, HC = _hidden_shape(config)

    prompt = np.random.default_rng(args.prompt_seed).integers(
        0, 200000, size=args.prompt_tokens).tolist()
    ids = mx.array([prompt], dtype=mx.int32)
    # capture_layer_ids=[] exposes the pre-final-mixer hidden stream (HC*H wide)
    # at every layer boundary; hidden_sink[0] is the last layer's raw stream.
    result = language(ids, cache=None, return_hidden=True,
                      skip_logits=True, capture_layer_ids=[])
    hc_hidden = result.hidden_states[0][:, -1:, :]  # [1, 1, HC*H]
    mx.eval(hc_hidden)
    if hc_hidden.shape[-1] != HC * H:
        raise SystemExit(
            f"unexpected MTP hidden width {hc_hidden.shape[-1]}, want {HC*H}")
    token = int(prompt[-1])

    # The checkpoint's own MTP head, fresh cache (position_ids=None -> dense
    # one-token softmax, the path the primitive implements).
    cache = mtp_mod.make_cache(language)
    tok = mx.array([[token]], dtype=mx.int32)
    logits, _hc = mtp_mod.forward(language, hc_hidden, tok, cache,
                                  return_hidden=True)
    mx.eval(logits)

    (out / "hidden.f32").write_bytes(bytes(mx.astype(hc_hidden, mx.float32)))
    (out / "token.txt").write_text(f"{token}\n")
    (out / "py_logits.f32").write_bytes(bytes(mx.astype(logits, mx.float32)))
    py_argmax = int(mx.argmax(mx.reshape(logits, (-1,)), axis=0).item())
    meta = {
        "checkpoint": str(checkpoint),
        "prompt_tokens": args.prompt_tokens,
        "prompt_seed": args.prompt_seed,
        "prompt_sha256": hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
        ).hexdigest(),
        "token": token,
        "hidden_shape": list(hc_hidden.shape),
        "hidden_dtype": str(hc_hidden.dtype),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "python_argmax": py_argmax,
        "mlx_version": mx.__version__,
    }
    (out / "python_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_config(out, checkpoint, config)
    print(f"python golden: token={token} logits={logits.shape} "
          f"argmax={py_argmax} sha256={digest_f32(out / 'py_logits.f32')[0][:16]}",
          flush=True)
    return 0


def phase_compare(args) -> int:
    out = Path(args.out)
    env = dict(os.environ)
    env["VEYSTRIX_CPP_MTP_REAL_PARITY"] = str(out)
    cmd = [args.binary, args.checkpoint, str(out / "python_meta.json"), "1",
           str(out / "config.txt")]
    print("running C++: " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    if proc.returncode != 0:
        print(f"C++ binary failed: exit {proc.returncode}", file=sys.stderr)
        return proc.returncode

    py_path, cpp_path = out / "py_logits.f32", out / "cpp_logits.f32"
    py_sha, py_n = digest_f32(py_path)
    cpp_sha, cpp_n = digest_f32(cpp_path)
    py = np.fromfile(py_path, dtype="<f4")
    cpp = np.fromfile(cpp_path, dtype="<f4")
    meta = json.loads((out / "python_meta.json").read_text())

    print("=" * 72)
    print(f"python logits: {py_n} f32  sha256={py_sha}  argmax={int(py.argmax())}")
    print(f"C++    logits: {cpp_n} f32  sha256={cpp_sha}  argmax={int(cpp.argmax())}")
    ok = py_n == cpp_n and py_sha == cpp_sha
    if py_n == cpp_n:
        diffs = int(np.count_nonzero(
            py.view(np.uint32) != cpp.view(np.uint32)))
        print(f"byte-level differing f32 elements: {diffs} / {py_n}")
    else:
        print("element count MISMATCH")
    tok_ok = int(py.argmax()) == int(meta["python_argmax"])
    print(f"python argmax matches recorded golden: {tok_ok}")
    print(f"REAL MTP DRAFT PARITY: {'BIT-EXACT PASS' if ok else 'FAIL'} "
          f"(tokens {'OK' if tok_ok else 'DIFF'}, logits "
          f"{'OK' if ok else 'DIFF'})")
    report = {
        "python_sha256": py_sha,
        "cpp_sha256": cpp_sha,
        "python_argmax": int(py.argmax()),
        "cpp_argmax": int(cpp.argmax()),
        "elements": [py_n, cpp_n],
        "bit_exact": bool(ok),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    ap.add_argument("--prompt-seed", type=int, default=PROMPT_SEED)
    ap.add_argument("--phase", choices=["gen", "compare"], default=None)
    args = ap.parse_args()
    if args.phase == "gen":
        return phase_gen(args)
    if args.phase == "compare":
        return phase_compare(args)
    # Orchestrate: two child processes so Python's model memory is released
    # before the C++ loader allocates its packed MTP weights.
    common = ["--checkpoint", args.checkpoint, "--binary", args.binary,
              "--out", args.out, "--prompt-tokens", str(args.prompt_tokens),
              "--prompt-seed", str(args.prompt_seed)]
    r1 = subprocess.run([sys.executable, __file__, *common, "--phase", "gen"])
    if r1.returncode != 0:
        return r1.returncode
    return subprocess.run(
        [sys.executable, __file__, *common, "--phase", "compare"]
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
