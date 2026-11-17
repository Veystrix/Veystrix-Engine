#!/usr/bin/env python3
"""OWNER-RUN diagnostic: dump the final-head intermediates from the Python path.

This is a *diagnostic only*. It mirrors ``capture_real_reference.py`` (same
backend, same greedy settings, same deterministic prompt) but additionally
hooks the FINAL ``hyper_connection_mixer`` and the sampler so we can compare
the Python final-head composition against the C++ generator's ``final_mixed``
and ``final_logits`` tensors:

    mixer input   [B, S, hc*H]   ->  mixer out [B, S, H]  ->  lm_head  ->  logits

The C++ side is instrumented at ``cpp/engine/tools/real_checkpoint_generate.cpp:770-794``
(= ``debug_capture::get().add("final_mixed", mixed)`` and ``("final_logits", ...)``)
and the math lives in ``cpp/engine/src/engine.cpp:185-206`` (``final_head``).
The Python anchors are ``mlx_vlm/models/qwen4_exp/language.py:823-829``
(the final ``self.hyper_connection_mixer(hidden_states)`` call in
``Qwen4ExpModel.__call__``) and ``language.py:442-454``
(``Qwen4ExpGatedResidual.__call__``).

DO NOT run on the real checkpoint unless you are the owner and have the memory
guard in place (see capture_real_reference.py's docstring). This script never
writes to any shared file; it only creates .npy dumps + meta.json in --out.

Run the synthetic hook smoke test first (no checkpoint, no guard needed)::

    /Users/vincent/Documents/Veystrix/.venv/bin/python \
        cpp/engine/tools/dump_python_head.py --self-test --out /tmp/head_synth

Real capture (OWNER-RUN)::

    scripts/flashnext/real_run_guard.py --floor-gib 1.5 --swap-growth-mib 2048 \
      --compressor-growth-mib 4096 --thermal-level 99 --log <dir>/guard.jsonl -- \
      .venv/bin/python cpp/engine/tools/dump_python_head.py \
        --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
        --out <dir>/head --tokens 2
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# The Veystrix venv is NOT on sys.path when invoked from other environments.
MLX_VLM_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
if MLX_VLM_SITE not in sys.path:
    sys.path.insert(0, MLX_VLM_SITE)

import mlx.core as mx  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpGatedResidual,
    Qwen4ExpRMSNorm,
)

PROMPT_SEED = 20261117
ANCHORS = {
    "python_mixer_call": "mlx_vlm/models/qwen4_exp/language.py:823-829 "
                         "(Qwen4ExpModel.__call__ final hyper_connection_mixer)",
    "python_gated_residual": "mlx_vlm/models/qwen4_exp/language.py:442-454 "
                             "(Qwen4ExpGatedResidual.__call__)",
    "python_rms_norm": "models/flashnext/patch_rmsnorm.py:97-121,150-162 "
                       "(_rms_norm / configure, one-centered gain)",
    "cpp_final_head": "cpp/engine/src/engine.cpp:185-206 (final_head)",
    "cpp_tool_inline": "cpp/engine/tools/real_checkpoint_generate.cpp:770-794 "
                       "(debug_capture: final_mixed / final_logits)",
}

# Module-level hook state. The class-level __call__ patch is process-wide, so we
# keep the target instance and the active capture in globals that install() /
# uninstall() manage. (Overriding __call__ on the instance does NOT work for
# `mixer(x)` because Python looks up special methods on the type.)
_TARGET = [None]          # the final Qwen4ExpGatedResidual (use_combine=False)
_CAPTURE = [None]         # the active HeadCapture
_ORIG_GATED_CALL = [None]
_ORIG_SAMPLER_CALL = [None]


def _to_numpy(a) -> np.ndarray:
    """BF16/F16 -> fp32, then to a numpy fp32 array (numpy cannot hold BF16)."""
    mx.eval(a)
    if getattr(a, "dtype", None) in (mx.bfloat16, mx.float16):
        a = mx.astype(a, mx.float32)
    return np.asarray(a, dtype=np.float32)


def _last_token(a: np.ndarray) -> np.ndarray:
    """Take the last (prompt) token slice: [B,S,D] -> [B,D]; [B,D] -> [B,D]."""
    if a.ndim >= 3:
        return a[:, -1, ...]
    return a


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _describe(a: np.ndarray) -> dict:
    flat = np.ascontiguousarray(a).reshape(-1)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": _digest(a),
        "first_values": flat[:6].tolist(),
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "mean": float(flat.mean()) if flat.size else None,
    }


class HeadCapture:
    """Collects the final mixer's (input, output) per forward pass and the
    logits handed to the sampler, in call order. MLX arrays are immutable, so
    holding references is safe; we eval + convert only when dumping."""

    def __init__(self):
        self.mixer_calls: list[dict] = []   # [{input, output, shapes/dtypes}]
        self.sampler_logits: list = []

    def record_mixer(self, hyper_input, mixed_input):
        self.mixer_calls.append({
            "input": hyper_input,
            "output": mixed_input,
            "in_shape": list(hyper_input.shape),
            "in_dtype": str(hyper_input.dtype),
            "out_shape": list(mixed_input.shape),
            "out_dtype": str(mixed_input.dtype),
        })

    def record_logits(self, logits):
        self.sampler_logits.append({
            "array": logits,
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
        })


def install(target_mixer, capture: HeadCapture, *, sampler: bool = True):
    """Install the class-level hooks. `target_mixer` is the final mixer
    instance; only calls on that instance are captured."""
    _TARGET[0] = target_mixer
    _CAPTURE[0] = capture

    _ORIG_GATED_CALL[0] = Qwen4ExpGatedResidual.__call__

    def _capturing_gated_call(self, hyper_input):
        out = _ORIG_GATED_CALL[0](self, hyper_input)
        if self is _TARGET[0] and _CAPTURE[0] is not None:
            _CAPTURE[0].record_mixer(hyper_input, out)
        return out

    Qwen4ExpGatedResidual.__call__ = _capturing_gated_call

    if sampler:
        from veystrix.sampling import Sampler
        _ORIG_SAMPLER_CALL[0] = Sampler.__call__

        def _capturing_sampler_call(self, logits):
            if _CAPTURE[0] is not None:
                _CAPTURE[0].record_logits(logits)
            return _ORIG_SAMPLER_CALL[0](self, logits)

        Sampler.__call__ = _capturing_sampler_call


def uninstall():
    if _ORIG_GATED_CALL[0] is not None:
        Qwen4ExpGatedResidual.__call__ = _ORIG_GATED_CALL[0]
        _ORIG_GATED_CALL[0] = None
    if _ORIG_SAMPLER_CALL[0] is not None:
        from veystrix.sampling import Sampler
        Sampler.__call__ = _ORIG_SAMPLER_CALL[0]
        _ORIG_SAMPLER_CALL[0] = None
    _TARGET[0] = None
    _CAPTURE[0] = None


def _norm_info(mixer) -> dict:
    """Report the final mixer's hc_norm gain and the one-centered patch state.

    Production replaces Qwen4ExpRMSNorm.__call__ with patch_rmsnorm._rms_norm
    and tags each instance with `_flashnext_one_centered`. We record both the
    raw signals and a derived `one_centered_convention` so the C++ side knows
    which convention to compare against.
    """
    norm = mixer.hc_norm
    info: dict = {"class": type(norm).__name__}
    try:
        w = _to_numpy(norm.weight)
        np.save(_norm_info.out_dir / "hc_norm_weight.npy", w)
        info.update({
            "weight_shape": list(w.shape),
            "weight_dtype": str(w.dtype),
            "weight": _describe(w),
            "gain = weight (one-centered)": _describe(w),
            "gain = 1.0 + weight (zero-centered)": _describe((w + 1.0).astype(np.float32)),
        })
    except Exception as exc:
        info["weight_read_error"] = f"{type(exc).__name__}: {exc}"

    patch_applied = False
    class_call_src = ""
    try:
        from models.flashnext import patch_rmsnorm
        patch_applied = bool(patch_rmsnorm._applied)
        class_call_src = inspect.getsource(patch_rmsnorm._rms_norm)
    except Exception as exc:
        info["patch_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        class_call_src = class_call_src or inspect.getsource(type(norm).__call__)
    except Exception:
        pass

    instance_flag = getattr(norm, "_flashnext_one_centered", None)
    # _float_gain defaults to True when the instance flag is absent.
    one_centered = (True if instance_flag is None else bool(instance_flag))
    info.update({
        "patch_rmsnorm_applied": patch_applied,
        "call_replaced_by_patch": (
            getattr(Qwen4ExpRMSNorm.__call__, "__name__", "") == "_rms_norm"
        ),
        "instance__flashnext_one_centered": instance_flag,
        "one_centered_convention": one_centered,
        "call_source_has_plus_1": "1.0 +" in class_call_src or "1 +" in class_call_src,
        "call_source_snippet": class_call_src[:600],
    })
    return info


def dump(out_dir: Path, capture: HeadCapture, mixer,
         checkpoint: str, prompt: list[int], tokens: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _norm_info.out_dir = out_dir  # small channel for _norm_info to save a npy
    files: dict = {}

    def save(name: str, arr: np.ndarray):
        np.save(out_dir / name, arr)
        files[name] = _describe(arr)

    # PREFILL = first mixer call; FIRST DECODE = second mixer call.
    if not capture.mixer_calls:
        raise RuntimeError("no mixer calls captured - hook did not attach")
    prefill = capture.mixer_calls[0]
    save("prefill_mixer_in.npy", _last_token(_to_numpy(prefill["input"])))
    save("prefill_mixer_out.npy", _last_token(_to_numpy(prefill["output"])))
    if len(capture.mixer_calls) > 1:
        dec = capture.mixer_calls[1]
        save("decode_mixer_in.npy", _last_token(_to_numpy(dec["input"])))
        save("decode_mixer_out.npy", _last_token(_to_numpy(dec["output"])))

    # Logits: first sampler call is the prefill (seeded) logits row, second is
    # the first decode step - same assumption as capture_real_reference.py.
    if len(capture.sampler_logits) > 0:
        save("prefill_logits.npy", _last_token(_to_numpy(capture.sampler_logits[0]["array"])))
    if len(capture.sampler_logits) > 1:
        save("decode_logits.npy", _last_token(_to_numpy(capture.sampler_logits[1]["array"])))

    norm_info = _norm_info(mixer)
    meta = {
        "purpose": "Python final-head intermediates for C++ 0.77-scaling localisation",
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
        "n_mixer_calls": len(capture.mixer_calls),
        "n_sampler_logits": len(capture.sampler_logits),
        "raw_mixer_calls": [
            {"in_shape": c["in_shape"], "in_dtype": c["in_dtype"],
             "out_shape": c["out_shape"], "out_dtype": c["out_dtype"]}
            for c in capture.mixer_calls
        ],
        "raw_sampler_logits": [
            {"shape": s["shape"], "dtype": s["dtype"]}
            for s in capture.sampler_logits
        ],
        "final_mixer_norm": norm_info,
        "files": files,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Console report.
    print(f"mixer calls captured: {len(capture.mixer_calls)} "
          f"(prefill in={prefill['in_shape']} {prefill['in_dtype']} -> "
          f"out={prefill['out_shape']} {prefill['out_dtype']})")
    print(f"sampler logits captured: {len(capture.sampler_logits)}")
    print(f"hc_norm: weight={norm_info.get('weight_shape')} "
          f"one_centered={norm_info.get('one_centered_convention')} "
          f"patch_applied={norm_info.get('patch_rmsnorm_applied')}")
    for name, d in files.items():
        print(f"  {name}: shape={d['shape']} dtype={d['dtype']} "
              f"first={d['first_values'][:4]}")
    print(f"wrote {len(files)} .npy files + meta.json to {out_dir}")
    return meta


def run_real(args):
    from veystrix.backends.flashnext import FlashNextBackend
    from veystrix.sampling import Sampler

    os.environ.setdefault("VEYSTRIX_SAMPLING_GREEDY", "1")
    backend = FlashNextBackend(model_path=args.checkpoint)
    if hasattr(backend, "sampling") and backend.sampling is not None:
        try:
            backend.sampling.greedy = True
        except Exception:
            pass

    # backend.language is the Qwen4Exp LanguageModel; .model is Qwen4ExpModel.
    final_mixer = backend.language.model.hyper_connection_mixer
    assert final_mixer.__class__.__name__ == "Qwen4ExpGatedResidual", \
        f"unexpected final mixer type: {type(final_mixer)}"
    assert "block_inject_weight" not in final_mixer, \
        "final mixer is not use_combine=False - wrong instance?"

    capture = HeadCapture()
    install(final_mixer, capture, sampler=True)
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

    dump(Path(args.out).expanduser().resolve(), capture, final_mixer,
         args.checkpoint, prompt, args.tokens)


def run_self_test(args):
    """Tiny synthetic model: confirm the class-level hook attaches to a
    Qwen4ExpGatedResidual(use_combine=False), captures in/out and dumps files.
    No checkpoint, no real model, no guard, no Sampler patch."""
    cfg = SimpleNamespace(hc_count=2, hidden_size=8, hc_lowrank=16,
                          rms_norm_eps=1e-6)
    mixer = Qwen4ExpGatedResidual(cfg, use_combine=False)
    mx.eval([p for p in mixer.parameters()])

    capture = HeadCapture()
    install(mixer, capture, sampler=False)
    try:
        B, S = 1, 4
        x_np = (np.random.default_rng(0).standard_normal(
            (B, S, cfg.hc_count * cfg.hidden_size)) * 0.5).astype(np.float32)
        y = mixer(mx.array(x_np))
        mx.eval(y)
    finally:
        uninstall()

    assert len(capture.mixer_calls) == 1, \
        f"hook failed: captured {len(capture.mixer_calls)} mixer calls"
    got_in = capture.mixer_calls[0]["input"]
    assert got_in.shape == (B, S, cfg.hc_count * cfg.hidden_size), got_in.shape
    out_dir = Path(args.out).expanduser().resolve()
    meta = dump(out_dir, capture, mixer, "<synthetic-self-test>",
                prompt=[], tokens=0)
    # Verify the files really landed and round-trip with the expected shapes.
    for name, exp_shape in (("prefill_mixer_in.npy", (B, cfg.hc_count * cfg.hidden_size)),
                            ("prefill_mixer_out.npy", (B, cfg.hidden_size))):
        arr = np.load(out_dir / name)
        assert arr.shape == exp_shape and arr.dtype == np.float32, (name, arr.shape, arr.dtype)
    assert (out_dir / "meta.json").is_file()
    assert meta["n_mixer_calls"] == 1
    print("SELF-TEST PASS: hook attached, captured prefill mixer in/out, "
          "dumped fp32 .npy + meta.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", help="real checkpoint (OWNER-RUN only)")
    ap.add_argument("--out", required=True, help="directory for .npy dumps + meta.json")
    ap.add_argument("--tokens", type=int, default=2,
                    help="generated tokens (>=1 so first decode is captured)")
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
