#!/usr/bin/env python3
"""OWNER-RUN diagnostic: dump the hidden state after EVERY decoder layer.

This mirrors ``capture_real_reference.py`` (same backend, same greedy setup,
same deterministic 16-token prompt) but hooks ``Qwen4ExpDecoderLayer.__call__``
at class level and records the OUTPUT hidden state of every decoder layer for
the LAST prefill forward. Its purpose is to find the FIRST layer where the C++
engine and the Python/MLX path diverge.

Production prefills the whole prompt in ONE call (S=16), so every layer's
``__call__`` runs exactly once with ``hidden_states.shape[1] == 16``. Decode
steps run with S=1 and are skipped. (With opt-in chunked prefill there can be
several S>1 calls; the LAST one wins, matching "last prefill forward".)

For each layer l the LAST-token slice is written as raw float32 (little-endian,
NO header) to::

    --out/layer_L##.f32        # output[:, -1, :]  ->  [1, 10240]

Layer 0's INPUT (the tiled input embedding, before any decoder layer) is also
captured to ``--out/layer_L00_input.f32``. These are byte-comparable with the
C++ side's ``layer.L##.f32`` dumps. ``meta.json`` records shapes, sha256 and a
few sample values per layer.

DO NOT run against the real checkpoint unless you are the owner and have the
memory guard in place (see capture_real_reference.py's docstring). This script
never writes to any shared file; it only creates .f32 dumps + meta.json in
--out. Do NOT use this to run the real checkpoint from a subagent.

Run the synthetic hook smoke test first (no checkpoint, no guard needed)::

    /Users/vincent/Documents/Veystrix/.venv/bin/python \
        cpp/engine/tools/dump_python_layers.py --self-test --out /tmp/layers_synth

Real capture (OWNER-RUN)::

    scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048 \
      --compressor-growth-mib 4096 --thermal-level 99 --log <dir>/guard.jsonl -- \
      .venv/bin/python cpp/engine/tools/dump_python_layers.py \
        --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
        --out <dir>/layers --tokens 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# The Veystrix venv is NOT on sys.path when invoked from other environments.
REPO = "/Users/vincent/.local/share/opencode/worktree/dd9a0e/cpp-engine-qdense"
VENV_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
for _p in (REPO, VENV_SITE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer  # noqa: E402

PROMPT_SEED = 20261117
ANCHORS = {
    "python_decoder_layer": "mlx_vlm/models/qwen4_exp/language.py:744-772 "
                            "(Qwen4ExpDecoderLayer.__call__)",
    "python_tiled_embed": "mlx_vlm/models/qwen4_exp/language.py:804-807 "
                          "(Qwen4ExpModel.__call__: embed + mx.tile)",
    "cpp_layer_dump": "cpp/engine/tools/real_checkpoint_generate.cpp "
                      "(debug_capture: layer.L##.f32 / attn.L##.f32)",
}

# Module-level hook state. The class-level __call__ patch is process-wide, so
# the active capture and the original callable live in globals that install() /
# uninstall() manage. (Overriding __call__ on an instance does NOT work for
# `layer(...)` because Python looks up special methods on the type.)
_ORIG_LAYER_CALL = [None]
_CAPTURE = [None]


def _to_numpy(a) -> np.ndarray:
    """BF16/F16 -> fp32, then to a numpy fp32 array (numpy cannot hold BF16)."""
    mx.eval(a)
    if getattr(a, "dtype", None) in (mx.bfloat16, mx.float16):
        a = mx.astype(a, mx.float32)
    return np.asarray(a, dtype=np.float32)


def _last_token(a: np.ndarray) -> np.ndarray:
    """[B,S,D] -> [B,D] keeping only the last (prefill) token."""
    if a.ndim >= 3:
        return a[:, -1, ...]
    return a


def _describe_bytes(raw: bytes) -> dict:
    arr = np.frombuffer(raw, dtype="<f4")
    return {
        "n_floats": int(arr.size),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_bytes": len(raw),
        "first_values": arr[:6].tolist(),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
    }


class LayerCapture:
    """Records the last S>1 call's (input, output) for every decoder layer.

    MLX arrays are immutable, so holding references is safe; we eval + convert
    to numpy fp32 only at dump time. Records are keyed by layer index and
    overwritten on each S>1 (prefill) call so the LAST prefill forward wins.
    """

    def __init__(self, layers):
        # Identity map: layer object -> index in the model's layer list.
        self._index = {id(layer): i for i, layer in enumerate(layers)}
        self.n_layers = len(layers)
        self.outputs: dict[int, object] = {}
        self.inputs: dict[int, object] = {}
        self.call_counts: dict[int, int] = {i: 0 for i in range(self.n_layers)}
        self.prefill_calls: dict[int, int] = {i: 0 for i in range(self.n_layers)}
        self.info: dict[int, dict] = {}

    def record(self, layer, hidden_states, out, seq_len: int):
        idx = self._index.get(id(layer))
        if idx is None:
            return
        self.call_counts[idx] += 1
        if seq_len <= 1:
            return  # decode step: skip
        self.prefill_calls[idx] += 1
        # Keep the LAST prefill forward for this layer.
        self.outputs[idx] = out
        self.inputs[idx] = hidden_states
        self.info[idx] = {
            "seq_len": seq_len,
            "input_shape": list(hidden_states.shape),
            "input_dtype": str(hidden_states.dtype),
            "output_shape": list(out.shape),
            "output_dtype": str(out.dtype),
        }


def install(capture: LayerCapture):
    """Install the class-level hook on Qwen4ExpDecoderLayer.__call__."""
    _CAPTURE[0] = capture
    _ORIG_LAYER_CALL[0] = Qwen4ExpDecoderLayer.__call__

    def _capturing_layer_call(self, *args, **kwargs):
        out = _ORIG_LAYER_CALL[0](self, *args, **kwargs)
        cap = _CAPTURE[0]
        if cap is not None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is not None and getattr(hidden, "ndim", 0) >= 2:
                cap.record(self, hidden, out, int(hidden.shape[1]))
        return out

    Qwen4ExpDecoderLayer.__call__ = _capturing_layer_call


def uninstall():
    if _ORIG_LAYER_CALL[0] is not None:
        Qwen4ExpDecoderLayer.__call__ = _ORIG_LAYER_CALL[0]
        _ORIG_LAYER_CALL[0] = None
    _CAPTURE[0] = None


def _raw_f32(a) -> bytes:
    """Last-token slice -> contiguous, little-endian float32 bytes (no header)."""
    last = _last_token(_to_numpy(a))
    return np.ascontiguousarray(last, dtype="<f4").tobytes()


def dump(out_dir: Path, capture: LayerCapture, checkpoint: str,
         prompt: list[int], tokens: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict = {}

    def save(name: str, raw: bytes, shape: list):
        (out_dir / name).write_bytes(raw)
        d = _describe_bytes(raw)
        d["shape"] = shape
        d["file"] = name
        files[name] = d

    for l in range(capture.n_layers):
        if l not in capture.outputs:
            raise RuntimeError(
                f"layer {l} never captured an S>1 (prefill) call; "
                f"prefill_calls={capture.prefill_calls[l]}, "
                f"total_calls={capture.call_counts[l]}"
            )
        raw = _raw_f32(capture.outputs[l])
        shape = [int(v) for v in capture.info[l]["output_shape"]]
        save(f"layer_L{l:02d}.f32", raw, shape[:1] + shape[2:])

    # Layer 0 input is the tiled input embedding (before any decoder layer).
    if 0 in capture.inputs:
        raw0 = _raw_f32(capture.inputs[0])
        shape0 = [int(v) for v in capture.info[0]["input_shape"]]
        save("layer_L00_input.f32", raw0, shape0[:1] + shape0[2:])

    meta = {
        "purpose": "Python hidden state after every decoder layer (last prefill token)",
        "checkpoint": checkpoint,
        "prompt_seed": PROMPT_SEED,
        "prompt_tokens": prompt,
        "prompt_sha256": hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
        ).hexdigest(),
        "requested_tokens": tokens,
        "mlx_version": mx.__version__,
        "numpy_version": np.__version__,
        "anchors": ANCHORS,
        "n_layers": capture.n_layers,
        "format": "raw little-endian float32, no header (one [1,hidden] row per file)",
        "call_counts": capture.call_counts,
        "prefill_calls": capture.prefill_calls,
        "layer_info": {str(k): v for k, v in sorted(capture.info.items())},
        "files": files,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"layers captured: {capture.n_layers} "
          f"(prefill calls/layer: {sorted(set(capture.prefill_calls.values()))})")
    for name, d in files.items():
        print(f"  {name}: shape={d['shape']} sha256={d['sha256'][:16]}... "
              f"first={d['first_values'][:4]}")
    print(f"wrote {len(files)} .f32 dumps + meta.json to {out_dir}")
    return meta


def run_real(args):
    from veystrix.backends.flashnext import FlashNextBackend

    os.environ.setdefault("VEYSTRIX_SAMPLING_GREEDY", "1")
    backend = FlashNextBackend(model_path=args.checkpoint)
    if hasattr(backend, "sampling") and backend.sampling is not None:
        try:
            backend.sampling.greedy = True
        except Exception:
            pass

    # backend.language is LanguageModel; .model is Qwen4ExpModel with .layers.
    layers = backend.language.model.layers
    assert len(layers) > 0, "model has no decoder layers"
    assert all(isinstance(l, Qwen4ExpDecoderLayer) for l in layers), \
        "unexpected decoder layer type"

    capture = LayerCapture(layers)
    install(capture)
    try:
        rng = np.random.default_rng(PROMPT_SEED)
        prompt = rng.integers(0, 200000, size=args.prompt_tokens).tolist()
        backend.append_tokens(prompt)
        token_ids: list[int] = []

        def emitted(token, _piece):
            token_ids.append(int(token))

        backend.generate(args.tokens, on_decode_token=emitted)
    finally:
        uninstall()

    dump(Path(args.out).expanduser().resolve(), capture,
         args.checkpoint, prompt, args.tokens)


def run_self_test(args):
    """Tiny synthetic model: 2 real Qwen4ExpDecoderLayer instances, S=4.

    Confirms the class-level hook attaches to the real Qwen4ExpDecoderLayer,
    captures the S=4 prefill output (and skips an S=1 decode call), and dumps
    byte-round-trippable fp32 files + meta.json. No checkpoint / guard / sampler.
    """
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp.config import TextConfig

    cfg = TextConfig(
        model_type="qwen4_exp",
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        num_experts=2,
        num_experts_per_tok=1,
        shared_expert_intermediate_size=8,
        moe_intermediate_size=8,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=64,
        hc_count=2,
        hc_lowrank=4,
        full_attention_interval=10**9,  # all layers linear (no QSA indexer)
        ple_layer_ids=[],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        rope_parameters=None,
    )
    layers = [Qwen4ExpDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
    mx.eval([p for l in layers for p in l.parameters()])
    caches = [ArraysCache(size=2) for _ in layers]
    B, S = 1, 4
    hidden = cfg.hc_count * cfg.hidden_size  # 32

    capture = LayerCapture(layers)
    install(capture)
    try:
        # Prefill: S=4 through both layers, then a decode step S=1 (must be skipped).
        x = mx.random.normal((B, S, hidden))
        for layer, cache in zip(layers, caches):
            x = layer(x, mx.array([[0, 1, 2, 3]]), mask=None,
                      cache=cache, position_ids=None)
        mx.eval(x)
        prefill_out_last = _to_numpy(x)[:, -1, :]  # reference for layer 1
        xd = mx.random.normal((B, 1, hidden))
        for layer, cache in zip(layers, caches):
            xd = layer(xd, mx.array([[4]]), mask=None,
                       cache=cache, position_ids=None)
        mx.eval(xd)
    finally:
        uninstall()

    assert capture.n_layers == 2
    for l in range(2):
        assert capture.prefill_calls[l] == 1, capture.prefill_calls
        assert capture.call_counts[l] == 2, capture.call_counts  # prefill + decode
        assert l in capture.outputs, f"layer {l} output not captured"
    assert 0 in capture.inputs, "layer 0 input not captured"

    out_dir = Path(args.out).expanduser().resolve()
    meta = dump(out_dir, capture, "<synthetic-self-test>",
                prompt=[], tokens=0)
    assert meta["n_layers"] == 2
    assert (out_dir / "meta.json").is_file()

    # Round-trip the raw bytes and verify the LAST prefill token was captured.
    for name, exp_shape in (("layer_L00.f32", (B, hidden)),
                            ("layer_L01.f32", (B, hidden)),
                            ("layer_L00_input.f32", (B, hidden))):
        raw = (out_dir / name).read_bytes()
        arr = np.frombuffer(raw, dtype="<f4").reshape(exp_shape)
        assert arr.dtype == np.float32 and arr.shape == exp_shape, (name, arr.shape)
    got_last = np.frombuffer(
        (out_dir / "layer_L01.f32").read_bytes(), dtype="<f4"
    ).reshape(B, hidden)
    assert np.allclose(got_last, np.ascontiguousarray(prefill_out_last, dtype="<f4"),
                       rtol=1e-6, atol=1e-6), "captured layer-1 output != prefill output"
    print("SELF-TEST PASS: hook attached to Qwen4ExpDecoderLayer, captured "
          "the S=4 prefill output (S=1 decode skipped), dumped raw fp32 .f32 "
          "files + meta.json with byte round-trip verified.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", help="real checkpoint (OWNER-RUN only)")
    ap.add_argument("--out", required=True, help="directory for .f32 dumps + meta.json")
    ap.add_argument("--tokens", type=int, default=2,
                    help="generated tokens (so the last prefill forward is complete)")
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--self-test", action="store_true",
                    help="run the tiny synthetic hook smoke test (no checkpoint)")
    args = ap.parse_args()
    if args.self_test:
        run_self_test(args)
    else:
        if not args.checkpoint:
            ap.error("--checkpoint is required unless --self-test is given")
        run_real(args)


if __name__ == "__main__":
    main()
