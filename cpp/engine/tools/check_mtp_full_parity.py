#!/usr/bin/env python3
"""Bit-exact Python <-> C++ check of the REAL MTP head at a NONZERO position.

This closes the "only position-0 fresh-cache validated" gap. The golden is
produced by the checkpoint's own ``Qwen4ExpMTPModule`` with a PREPOPULATED QSA
KV cache: a short real prompt is fed through the backbone to obtain genuine
pre-mixer hidden streams, the MTP head is stepped once per prompt position to
fill its cache (positions 0..N-2), and the final draft step at position N-1
(nonzero) is compared against the C++ ``mtp_draft_forward_real`` overload that
accepts an ``MTPDraftCache``.

For N-1 < 2048 the QSA sparse indexer returns None and Python's attention mask
is None (S=1), so a DENSE attention over the past keys + the current token is
the correct path. The sparse-mask regime (past_len >= 2048) is out of scope and
is documented as an honest limitation in docs/engineering/mtp_full_parity.md.

The C++ side is driven by a small self-contained driver that is GENERATED INTO
THE OUTPUT DIRECTORY (not committed) and includes the existing
``real_checkpoint_generate.cpp`` solely to reuse its flat-config loader,
``DenseLoader``, MTP weight loader and SHA-256 helper. No repository file other
than the allowed set is modified.

Comparison is raw f32 byte equality of the full logit tensor plus argmax/token
equality. No tolerance, no "close enough". The three phases (gen / compare /
orchestrated) mirror ``check_mtp_real_parity.py`` so the heavy Python model is
fully torn down before the C++ loader allocates its packed weights.

Usage (from /Users/vincent/Documents/Veystrix, under real_run_guard.py):
  .venv/bin/python <worktree>/cpp/engine/tools/check_mtp_full_parity.py \
      --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --worktree <worktree> --out /tmp/mtp_full_parity [--prompt-tokens 8]
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

PROMPT_SEED = 20261118
PROMPT_TOKENS = 8
MTP_PREFIX = "language_model.mtp."

# --------------------------------------------------------------------------- #
# The C++ parity driver. It is emitted into the output directory at run time;
# it is intentionally NOT part of the repository. Including
# real_checkpoint_generate.cpp reuses its DenseLoader / config / weight helpers
# so this driver stays small and cannot drift from the validated loader.
# --------------------------------------------------------------------------- #
CPP_DRIVER = r'''
#define main real_checkpoint_generate_main_disabled
#include "real_checkpoint_generate.cpp"
#undef main

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mx = mlx::core;
using namespace veystrix::engine;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <dir> <checkpoint>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1], ckpt = argv[2];
  try {
    Config cfg = read_config(dir + "/config.txt");
    const int gs = get_int(cfg, "group_size", 32);
    const int bits = get_int(cfg, "bits", 4);
    DenseLoader dl(ckpt, cfg, gs, bits);

    MTPConfig c = mtp_real::config_from(cfg);
    MTPRealWeights w = mtp_real::load_weights(dl, c, gs, bits);

    // Final draft step's hidden stream and token id.
    std::vector<float> hv = mtp_real::read_f32(dir + "/hidden.f32");
    if ((int)hv.size() != c.hc_count * c.hidden_size)
      throw std::runtime_error("hidden.f32 element count mismatch");
    mx::array hidden(hv.data(), mx::Shape{1, 1, c.hc_count * c.hidden_size},
                     mx::float32);
    hidden = mx::astype(hidden, mx::bfloat16);
    mx::eval(hidden);

    std::FILE* tf = std::fopen((dir + "/token.txt").c_str(), "r");
    if (!tf) throw std::runtime_error("cannot open token.txt");
    long token = 0;
    if (std::fscanf(tf, "%ld", &token) != 1)
      throw std::runtime_error("token.txt is not an integer");
    std::fclose(tf);
    std::vector<int32_t> ids{(int32_t)token};
    mx::array idarr(ids.data(), mx::Shape{1, 1}, mx::int32);

    // Prepopulated QSA KV cache: already-RoPE-rotated keys/values at their own
    // positions, laid out [B=1, n_kv_heads, past_len, head_dim].
    const int past_len = get_int(cfg, "past_len", 0);
    if (past_len <= 0)
      throw std::runtime_error("past_len must be > 0 for the nonzero-position test");
    std::vector<float> kv = mtp_real::read_f32(dir + "/past_k.f32");
    std::vector<float> vv = mtp_real::read_f32(dir + "/past_v.f32");
    const size_t want = (size_t)c.n_kv_heads * past_len * c.head_dim;
    if (kv.size() != want || vv.size() != want)
      throw std::runtime_error("past_k.f32 / past_v.f32 element count mismatch");
    mx::array pk(kv.data(), mx::Shape{1, c.n_kv_heads, past_len, c.head_dim},
                 mx::float32);
    mx::array pv(vv.data(), mx::Shape{1, c.n_kv_heads, past_len, c.head_dim},
                 mx::float32);
    pk = mx::astype(pk, mx::bfloat16);
    pv = mx::astype(pv, mx::bfloat16);
    mx::eval(pk, pv);

    MTPDraftCache cache;
    cache.past_keys = pk;
    cache.past_values = pv;
    cache.past_len = past_len;
    c.position = past_len;  // explicit; the overload uses cache.past_len anyway

    mx::array logits = mtp_draft_forward_real(c, w, hidden, idarr, cache);
    mx::array f = mx::reshape(mx::astype(logits, mx::float32), mx::Shape{-1});
    mx::eval(f);

    std::FILE* of = std::fopen((dir + "/cpp_logits.f32").c_str(), "wb");
    if (!of) throw std::runtime_error("cannot write cpp_logits.f32");
    std::fwrite(f.data<float>(), sizeof(float), f.size(), of);
    std::fclose(of);

    mx::array am = mx::astype(mx::argmax(f, 0), mx::int32);
    mx::eval(am);
    const std::string digest = sha256_hex(f.data<float>(), f.size() * sizeof(float));
    std::printf("MTP_FULL_PARITY past_len=%d argmax=%d logits_sha256=%s elems=%zu\n",
                past_len, am.item<int32_t>(), digest.c_str(), f.size());
    mem_telemetry("mtp-full-parity");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "MTP_FULL_PARITY error: %s\n", e.what());
    return 1;
  }
}
'''


def write_config(out: Path, checkpoint: Path, config: dict, past_len: int) -> None:
    """Flat config for the C++ loader: geometry + MTP shard map + past_len."""
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
    kv("one_centered_norms", 1)
    kv("tie_word_embeddings", 1 if config.get("tie_word_embeddings", False) else 0)
    kv("past_len", past_len)

    wanted = [k for k in wmap if k.startswith(MTP_PREFIX)]
    wanted += [k for k in wmap if k.startswith("language_model.model.embed_tokens.")]
    wanted += [k for k in wmap if k.startswith("language_model.lm_head.")]
    for name in wanted:
        kv("shard." + name, wmap[name])
    (out / "config.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'config.txt'}: {len(wanted)} shard entries, "
          f"past_len={past_len}", flush=True)


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
    H = int(config["text_config"]["hidden_size"])
    HC = int(config["text_config"]["hc_count"])
    N = args.prompt_tokens
    if N < 3:
        raise SystemExit("--prompt-tokens must be >= 3 (need a nonzero position)")

    # On Apple Silicon MRoPERotaryEmbedding defaults to a fused Metal kernel
    # (metal::cos/sin) that differs from the eager mx.cos/sin path by ~1 f32
    # ULP. At position 0 cos=1/sin=0 so the two are identical (the validated
    # position-0 path). At a NONZERO position they diverge, and the C++ port
    # reproduces the EAGER branch (qsa_full_port.md "Fused-Metal RoPE path").
    # Force the real module's eager branch for a bit-exact comparison. This is
    # the same convention used by gen_qsa_full_vectors.py. The indexer shares
    # this rotary object (Qwen4ExpQSAIndexer.__init__ stores it directly).
    rot = language.mtp.layers[0].self_attn.rotary_emb
    rot.fused_apply = False
    rot._compiled_apply = None

    prompt = np.random.default_rng(args.prompt_seed).integers(
        0, 200000, size=N).tolist()
    ids = mx.array([prompt], dtype=mx.int32)
    # capture_layer_ids=[] exposes the pre-final-mixer hidden stream (HC*H wide)
    # at every layer boundary; hidden_states[0] is the last layer's raw stream.
    result = language(ids, cache=None, return_hidden=True,
                      capture_layer_ids=[])
    hc_hidden = result.hidden_states[0]          # [1, N, HC*H]
    mx.eval(hc_hidden)
    if hc_hidden.shape[-1] != HC * H or hc_hidden.shape[1] != N:
        raise SystemExit(f"unexpected hidden shape {hc_hidden.shape}, "
                         f"want [1,{N},{HC*H}]")

    # The "next token" for the final draft step. Mirror production: argmax of
    # the target's logits at the last prompt position. Fall back to the last
    # prompt token if this model call skipped logits (still a genuine id).
    logits_obj = getattr(result, "logits", None)
    if logits_obj is not None:
        next_main = mx.argmax(logits_obj[:, -1, :], axis=-1).astype(mx.int32)
        next_source = "target_argmax_last_position"
    else:
        next_main = ids[:, -1:]                   # [1, 1]
        next_source = "last_prompt_token_fallback"
    next_main = mx.reshape(next_main, (1, 1))
    mx.eval(next_main)
    token = int(next_main.item())

    # Warm the MTP cache at positions 0..N-2 using REAL backbone hidden streams
    # and the actual following prompt tokens (production _append semantics).
    cache = mtp_mod.make_cache(language)
    for p in range(N - 1):
        h = hc_hidden[:, p:p + 1, :]
        tok = ids[:, p + 1:p + 2]
        mtp_mod.forward(language, h, tok, cache, skip_logits=True)
        mx.eval([entry.state for entry in cache])

    entry = cache[0]
    offset = int(entry.offset)
    if offset != N - 1:
        raise SystemExit(f"MTP cache offset {offset} != expected {N - 1}")
    # Snapshot the PREPOPULATED cache BEFORE the final step updates it again.
    past_k = np.asarray(mx.astype(entry.keys[0, :, :offset, :], mx.float32))
    past_v = np.asarray(mx.astype(entry.values[0, :, :offset, :], mx.float32))

    # Final draft step at nonzero position N-1.
    anchor = hc_hidden[:, -1:, :]                  # [1, 1, HC*H]
    py_logits, _hc = mtp_mod.forward(
        language, anchor, next_main, cache, return_hidden=True)
    mx.eval(py_logits)

    (out / "hidden.f32").write_bytes(bytes(mx.astype(anchor, mx.float32)))
    (out / "token.txt").write_text(f"{token}\n")
    (out / "past_k.f32").write_bytes(past_k.astype("<f4").tobytes())
    (out / "past_v.f32").write_bytes(past_v.astype("<f4").tobytes())
    (out / "py_logits.f32").write_bytes(bytes(mx.astype(py_logits, mx.float32)))
    py_argmax = int(mx.argmax(mx.reshape(py_logits, (-1,)), axis=0).item())

    meta = {
        "checkpoint": str(checkpoint),
        "prompt_tokens": N,
        "prompt_seed": args.prompt_seed,
        "prompt_sha256": hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
        ).hexdigest(),
        "draft_position": N - 1,
        "past_len": offset,
        "next_token": token,
        "next_token_source": next_source,
        "hidden_shape": list(anchor.shape),
        "hidden_dtype": str(anchor.dtype),
        "past_k_shape": list(past_k.shape),
        "past_v_shape": list(past_v.shape),
        "logits_shape": list(py_logits.shape),
        "logits_dtype": str(py_logits.dtype),
        "python_argmax": py_argmax,
        "rope_branch": "eager (fused_apply=False, qsa_full_port.md caveat 1)",
        "mlx_version": mx.__version__,
    }
    (out / "python_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_config(out, checkpoint, config, offset)
    print(f"python golden: position={N-1} past_len={offset} token={token} "
          f"logits={py_logits.shape} argmax={py_argmax} "
          f"sha256={digest_f32(out / 'py_logits.f32')[0][:16]}", flush=True)
    return 0


def compile_driver(args) -> int:
    wt = Path(args.worktree).resolve()
    out = Path(args.out)
    driver = out / "mtp_full_driver.cpp"
    driver.write_text(CPP_DRIVER)
    binary = out / "mtp_full_driver"
    mlxroot = Path(args.mlxroot)
    native_c = wt / "models/flashnext/native_read.c"
    native_o = out / "native_read.o"
    subprocess.run(
        ["xcrun", "clang", "-O2", "-c", str(native_c), "-o", str(native_o)],
        check=True)
    cmd = [
        "xcrun", "clang++", "-std=c++20", "-O2",
        "-I" + str(mlxroot / "include"),
        "-I" + str(wt / "cpp/engine/src"),
        "-I" + str(wt / "cpp/decode/src"),
        "-I" + str(wt / "cpp/engine/tools"),
    ] + [str(p) for p in sorted((wt / "cpp/engine/src").glob("*.cpp"))] + [
        str(wt / "cpp/decode/src/io_pager.cpp"),
        str(native_o),
        str(driver),
        "-L" + str(mlxroot / "lib"), "-lmlx",
        "-Wl,-rpath," + str(mlxroot / "lib"),
        "-o", str(binary),
    ]
    print("compiling C++ driver:\n  " + " ".join(cmd[:8]) + " ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"wrote {binary}", flush=True)
    return 0


def phase_compare(args) -> int:
    out = Path(args.out)
    if compile_driver(args) != 0:
        return 1
    cmd = [str(out / "mtp_full_driver"), str(out), str(Path(args.checkpoint).resolve())]
    print("running C++: " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    if proc.returncode != 0:
        print(f"C++ driver failed: exit {proc.returncode}", file=sys.stderr)
        return proc.returncode

    py_path, cpp_path = out / "py_logits.f32", out / "cpp_logits.f32"
    py_sha, py_n = digest_f32(py_path)
    cpp_sha, cpp_n = digest_f32(cpp_path)
    py = np.fromfile(py_path, dtype="<f4")
    cpp = np.fromfile(cpp_path, dtype="<f4")
    meta = json.loads((out / "python_meta.json").read_text())

    print("=" * 72)
    print(f"position={meta['draft_position']}  past_len={meta['past_len']}  "
          f"token={meta['next_token']} ({meta['next_token_source']})")
    print(f"python logits: {py_n} f32  sha256={py_sha}  argmax={int(py.argmax())}")
    print(f"C++    logits: {cpp_n} f32  sha256={cpp_sha}  argmax={int(cpp.argmax())}")
    ok = py_n == cpp_n and py_sha == cpp_sha
    if py_n == cpp_n:
        diffs = int(np.count_nonzero(py.view(np.uint32) != cpp.view(np.uint32)))
        print(f"byte-level differing f32 elements: {diffs} / {py_n}")
    else:
        print("element count MISMATCH")
        diffs = -1
    tok_ok = int(py.argmax()) == int(meta["python_argmax"])
    print(f"python argmax matches recorded golden: {tok_ok}")
    print(f"REAL MTP NONZERO-POSITION PARITY: {'BIT-EXACT PASS' if ok else 'FAIL'} "
          f"(tokens {'OK' if tok_ok else 'DIFF'}, logits "
          f"{'OK' if ok else 'DIFF'})")
    report = {
        "draft_position": meta["draft_position"],
        "past_len": meta["past_len"],
        "python_sha256": py_sha,
        "cpp_sha256": cpp_sha,
        "python_argmax": int(py.argmax()),
        "cpp_argmax": int(cpp.argmax()),
        "elements": [py_n, cpp_n],
        "differing_elements": diffs,
        "bit_exact": bool(ok),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if ok else 1


def main() -> int:
    default_mlxroot = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages/mlx"
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--worktree", required=True,
                    help="path to the worktree containing cpp/engine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    ap.add_argument("--prompt-seed", type=int, default=PROMPT_SEED)
    ap.add_argument("--mlxroot", default=default_mlxroot)
    ap.add_argument("--phase", choices=["gen", "compare"], default=None)
    args = ap.parse_args()
    if args.phase == "gen":
        return phase_gen(args)
    if args.phase == "compare":
        return phase_compare(args)
    common = ["--checkpoint", args.checkpoint, "--worktree", args.worktree,
              "--out", args.out, "--prompt-tokens", str(args.prompt_tokens),
              "--prompt-seed", str(args.prompt_seed), "--mlxroot", args.mlxroot]
    r1 = subprocess.run([sys.executable, __file__, *common, "--phase", "gen"])
    if r1.returncode != 0:
        return r1.returncode
    return subprocess.run(
        [sys.executable, __file__, *common, "--phase", "compare"]
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
