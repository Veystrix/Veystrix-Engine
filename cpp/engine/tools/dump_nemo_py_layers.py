#!/usr/bin/env python3
"""Guarded diagnostic: dump the hidden state after EVERY Nemotron-H decoder layer.

Mirrors ``gen_nemotron_reference.py`` exactly (same pinned env, same streamed
backend, same chunk_tokens=1 prefill, same zero-seeded Mamba caches) but hooks
``mlx_lm.models.nemotron_h.NemotronHBlock.__call__`` at class level and records
the OUTPUT residual of every decoder layer for one prefill step.

The captured tensor is the same one the C++ tool digests: ``NemotronHBlock``
returns ``x + mixer(norm(x))`` (nemotron_h.py:455), which is exactly C++'s
``h = h + delta`` (nemotron_generate.cpp:674) — after the residual add, before
``norm_f`` / ``lm_head``.

For each layer L the digest is SHA-256 of the float32 little-endian bytes of
the hidden (flattened), the same convention as C++ ``digest_array``
(nemotron_generate.cpp:128-132). Lines are printed as::

    [PYNEMOLAYER step=S L=XX] <sha256>

so they can be diffed directly against the C++ ``[NEMOLAYER step L=XX]`` lines
from ``NEMOTRON_DUMP_LAYERS=1``.

With ``chunk_tokens=1`` the Python prefill runs one forward per prompt token;
the LAST prefill step (index len(prompt)-1) produces the logits that yield the
first generated token, i.e. C++/Python "step 0". By default this script dumps
that step. ``--all-steps`` dumps every prefill step.

Run under real_run_guard.py with the same flags as the reference capture.
No shared files are modified; output goes to --out (JSON) and stderr/stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Pin the effective path BEFORE importing any model code (identical to
# gen_nemotron_reference.py).
os.environ["VEYSTRIX_NEMOTRON_KV"] = "bf16"
os.environ["VEYSTRIX_NEMOTRON_GATHER"] = "1"
os.environ["VEYSTRIX_NEMOTRON_FAST_SSM"] = "0"
os.environ["VEYSTRIX_NEMOTRON_PARALLEL_MAMBA"] = "0"
os.environ["VEYSTRIX_NEMOTRON_NATIVE_MTP"] = "0"
os.environ["VEYSTRIX_NEMOTRON_PREFIX_CACHE"] = "0"
os.environ["VEYSTRIX_NEMOTRON_EXPERT_PACK"] = ""

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

REPO = "/Users/vincent/.local/share/opencode/worktree/dd9a0e/cpp-engine-qdense"
VENV_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
for _p in (REPO, VENV_SITE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ANCHORS = {
    "python_block": "mlx_lm/models/nemotron_h.py:443-455 "
                    "(NemotronHBlock.__call__ -> x + mixer(norm(x)))",
    "cpp_layer_dump": "cpp/engine/tools/nemotron_generate.cpp:674-677 "
                      "(h = h + delta; digest_array(h))",
    "cpp_digest": "cpp/engine/tools/nemotron_generate.cpp:128-132 "
                  "(reshape astype float32 -> sha256 over LE bytes)",
}

# Module-level hook state (class-level patch is process-wide).
_ORIG_BLOCK_CALL = [None]
_CAPTURE = [None]


class LayerCapture:
    """Records every layer's output residual, keyed by [step][layer].

    ``chunk_tokens=1`` means every forward calls each of the 52 blocks exactly
    once. A new step begins when layer index 0 is invoked again.
    """

    def __init__(self, layers):
        self._index = {id(layer): i for i, layer in enumerate(layers)}
        self.n_layers = len(layers)
        self.outputs: dict[int, dict[int, object]] = {}
        self._step = 0
        self._calls = 0

    def record(self, layer, out):
        idx = self._index.get(id(layer))
        if idx is None:
            return
        if idx == 0 and self._calls > 0:
            self._step += 1
        self._calls += 1
        self.outputs.setdefault(self._step, {})[idx] = out


def install(capture: LayerCapture):
    from mlx_lm.models.nemotron_h import NemotronHBlock

    _CAPTURE[0] = capture
    _ORIG_BLOCK_CALL[0] = NemotronHBlock.__call__

    def _capturing_block_call(self, *args, **kwargs):
        out = _ORIG_BLOCK_CALL[0](self, *args, **kwargs)
        cap = _CAPTURE[0]
        if cap is not None:
            cap.record(self, out)
        return out

    NemotronHBlock.__call__ = _capturing_block_call
    return NemotronHBlock


def uninstall(block_cls):
    if _ORIG_BLOCK_CALL[0] is not None and block_cls is not None:
        block_cls.__call__ = _ORIG_BLOCK_CALL[0]
        _ORIG_BLOCK_CALL[0] = None
    _CAPTURE[0] = None


def digest_array(a) -> str:
    """C++ digest_array: reshape(-1), astype float32, sha256 over LE bytes."""
    f = mx.reshape(mx.astype(a, mx.float32), (-1,))
    mx.eval(f)
    row = np.ascontiguousarray(np.asarray(f, dtype="<f4"))
    return hashlib.sha256(row.tobytes()).hexdigest()


class MoeCapture:
    """Records layer-L MoE intermediates for the matching prefill step."""

    def __init__(self, mixer, step_getter):
        self.mixer = mixer
        self.step_getter = step_getter
        self.data: dict[int, dict] = {}

    def rec(self, key, value):
        self.data.setdefault(self.step_getter(), {})[key] = value


def install_moe_hooks(mixer, cap: MoeCapture):
    """Class-level hooks (special methods resolve on the type), filtered to
    the one target mixer instance by identity."""
    from mlx_lm.models.nemotron_h import MoEGate, NemotronHMLP
    from models.nemotron35.streamed_experts import StreamedSwitchMLP
    from models.nemotron35.streamed_model import OverlappedSharedMoE

    up, down = mixer.shared_experts.up_proj, mixer.shared_experts.down_proj
    # Each class gets ONE wrapper; if up/down share a class, its wrapper
    # records both (by identity), avoiding a second wrapper shadowing the first.
    plans = [
        (OverlappedSharedMoE, [(mixer, "moe_in", lambda s, a, o: a[0])]),
        (MoEGate, [(mixer.gate, "inds", lambda s, a, o: o[0]),
                   (mixer.gate, "scores", lambda s, a, o: o[1]),
                   (mixer.gate, "gate_logits", lambda s, a, o: a[0] @ s.weight.T)]),
        (StreamedSwitchMLP, [(mixer.switch_mlp, "routed_y", lambda s, a, o: o)]),
        (NemotronHMLP, [(mixer.shared_experts, "shared", lambda s, a, o: o)]),
    ]
    qplan = []
    if type(up) is type(down):
        qplan = [(up, "su", lambda s, a, o: o), (down, "sd", lambda s, a, o: o)]
        plans.append((type(up), qplan))
    else:
        plans.append((type(up), [(up, "su", lambda s, a, o: o)]))
        plans.append((type(down), [(down, "sd", lambda s, a, o: o)]))

    classes, originals, seen = [], [], {}
    for cls, entries in plans:
        if cls in seen:
            seen[cls][1].extend(entries)
            continue
        originals.append(cls.__call__)
        classes.append(cls)
        seen[cls] = (cls, list(entries))

    for c, o in zip(classes, originals):
        entries = seen[c][1]

        def hooked(self, *a, _o=o, _entries=entries, **kw):
            out = _o(self, *a, **kw)
            for inst, key, extract in _entries:
                if self is inst:
                    cap.rec(key, extract(self, a, out))
            return out
        c.__call__ = hooked
    return classes, originals


def uninstall_moe_hooks(classes, originals):
    for c, o in zip(classes, originals):
        c.__call__ = o


class MambaCapture:
    def __init__(self, mixer, step_getter):
        self.mixer = mixer
        self.step_getter = step_getter
        self.data: dict[int, dict] = {}

    def rec(self, key, value):
        self.data.setdefault(self.step_getter(), {})[key] = value


def install_mamba_hooks(mixer, cap: MambaCapture):
    cls = type(mixer)
    orig_call, orig_conv, orig_ssm = cls.__call__, cls._conv, cls._ssm

    def call_hook(self, *a, **kw):
        out = orig_call(self, *a, **kw)
        if self is mixer:
            cap.rec("x", a[0])
            cap.rec("out", out)
        return out

    def conv_hook(self, conv_input, cache, mask):
        before = cache[0] if cache is not None else None
        out = orig_conv(self, conv_input, cache, mask)
        if self is mixer:
            cap.rec("conv_before", before)
            cap.rec("conv_after", cache[0] if cache is not None else None)
            cap.rec("conv_input", conv_input)
            cap.rec("conv_out", out)
        return out

    def ssm_hook(self, hidden_states, b, c, dt, cache, mask):
        before = cache[1] if cache is not None else None
        out = orig_ssm(self, hidden_states, b, c, dt, cache, mask)
        if self is mixer:
            cap.rec("ssm_before", before)
            cap.rec("ssm_after", cache[1] if cache is not None else None)
            cap.rec("ssm_y", out)
            cap.rec("ssm_x", hidden_states)
            cap.rec("ssm_b", b)
            cap.rec("ssm_c", c)
            cap.rec("ssm_dt", dt)
            cap.rec("ssm_state_in", before)
        return out

    cls.__call__, cls._conv, cls._ssm = call_hook, conv_hook, ssm_hook
    return (cls, (orig_call, orig_conv, orig_ssm))


def uninstall_mamba_hooks(h):
    cls, origs = h
    cls.__call__, cls._conv, cls._ssm = origs


def dump_mamba(cap, step):
    d = cap.data.get(step)
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        else:
            out[k] = digest_array(v)
    if "ssm_dt" in d:
        from mlx_lm.models.ssm import compute_dt
        dt = d["ssm_dt"]
        bias = cap.mixer.dt_bias
        dt_py = compute_dt(dt, bias, (0.0, float("inf")))
        dt_cpp = mx.clip(
            mx.logaddexp(mx.astype(dt, mx.float32) + bias,
                         mx.array(0.0, mx.float32)),
            mx.array(0.0, mx.float32), mx.array(float("inf"), mx.float32))
        out["dt_py"] = digest_array(dt_py)
        out["dt_cpp"] = digest_array(dt_cpp)
    return out


def dump_moe(cap, capture, step, n_layers):
    import mlx.nn as nn
    d = cap.data.get(step)
    if d is None:
        return None
    out = {}

    out["_dtypes"] = {k: str(v.dtype) for k, v in d.items()}

    def add(name, arr, ints=False):
        if ints:
            a = np.asarray(mx.reshape(mx.astype(arr, mx.int32), (-1,)), dtype="<i4")
            out[name] = {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                         "values": a.tolist()}
        else:
            out[name] = digest_array(arr)

    add("moe_in", d["moe_in"])
    add("gate_logits", d["gate_logits"])
    add("scores", d["scores"])
    add("inds", d["inds"], ints=True)
    ids = [int(x) for x in np.asarray(mx.reshape(d["inds"], (-1,))).tolist()]
    unique = sorted(set(ids))
    local = [unique.index(i) for i in ids]
    out["local"] = {"values": local}
    out["unique"] = {"values": unique}
    if "gate_logits" in d:
        from mlx_lm.models.nemotron_h import group_expert_select
        gl = d["gate_logits"]
        bias = cap.mixer.gate.e_score_correction_bias
        _, py_scores = group_expert_select(gl, bias, 6, 1, 1, 2.5, True)
        orig = mx.sigmoid(mx.astype(gl, mx.float32))
        sc = orig + bias
        inds_all = mx.argpartition(mx.negative(sc), 5, axis=-1)
        ids = inds_all[..., :6]
        top = mx.take_along_axis(orig, ids, axis=-1)
        denom = mx.sum(top, axis=-1, keepdims=True) + mx.array(1e-20, mx.float32)
        cpp_scores = (top / denom) * mx.array(2.5, mx.float32)
        out["scores_py_compiled"] = digest_array(py_scores)
        out["scores_cpp_unfused"] = digest_array(cpp_scores)
    y = d["routed_y"]
    scores = d["scores"]
    add("routed_y", y)
    # y shape: [1,1,k,H]; scores shape [1,1,k] -> [1,1,k,1]
    sw = mx.reshape(scores, list(y.shape[:-2]) + [y.shape[-2], 1])
    comb_f32 = (y * sw).sum(axis=-2)  # [1,1,H]
    add("combined_f32", comb_f32)
    add("combined_bf16", mx.astype(comb_f32, mx.bfloat16))
    add("su", d["su"])
    add("su_bf16", mx.astype(d["su"], mx.bfloat16))
    add("sa", nn.relu2(d["su"]))
    add("sa_weak0", mx.square(mx.maximum(d["su"], 0)))
    add("sa_f32zero", mx.square(mx.maximum(d["su"], mx.array(0.0, mx.float32))))
    add("sa_from_bf16", nn.relu2(mx.astype(d["su"], mx.bfloat16)))
    add("sd", d["sd"])
    add("shared", d["shared"])
    add("delta", mx.astype(comb_f32, mx.bfloat16) + d["shared"])
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="JSON file for the digest dump")
    ap.add_argument("--reference", default=None,
                    help="reference.json with prompt_ids (default: /tmp/nemo_ref.json)")
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="if no --reference, regenerate this many with seed 20261118")
    ap.add_argument("--tokens", type=int, default=1,
                    help="generated tokens after prefill (default 1, enough for step 0)")
    ap.add_argument("--step", type=int, default=None,
                    help="prefill step index to dump (default: last prefill step)")
    ap.add_argument("--all-steps", action="store_true",
                    help="dump every prefill step instead of just --step")
    ap.add_argument("--dump-moe-layer", type=int, default=None,
                    help="also dump MoE intermediates for this layer index")
    ap.add_argument("--dump-mamba-layer", type=int, default=None,
                    help="also dump mamba cache-state intermediates for this layer")
    return ap.parse_args()


def get_prompt(args):
    if args.reference:
        ref = json.loads(Path(args.reference).read_text())
        prompt = [int(t) for t in ref["prompt_ids"]]
        return prompt, "reference.json"
    n = args.prompt_tokens or 6
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    rng = np.random.default_rng(20261118)
    prompt = [int(t) for t in rng.integers(3, config["vocab_size"] - 1, size=n)]
    return prompt, "rng:20261118"


def main() -> int:
    args = parse_args()

    from models.nemotron35.backend_core import EngineCore
    from models.nemotron35.config import validate_config
    from models.nemotron35.streamed_model import load_streamed
    from mlx_lm.models.nemotron_h import NemotronHBlock

    prompt, prompt_src = get_prompt(args)
    prompt_sha = hashlib.sha256(
        b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
    ).hexdigest()
    print(f"[pydump] prompt ({prompt_src}, {len(prompt)} tok) sha={prompt_sha[:12]}",
          file=sys.stderr, flush=True)

    pool_bytes = 256 * 2**20
    print(f"[pydump] loading streamed model (pool={pool_bytes // (2**20)} MiB)",
          file=sys.stderr, flush=True)
    streamed = load_streamed(args.checkpoint, pool_bytes=pool_bytes,
                             config_check=validate_config, verify=True)
    core = EngineCore(streamed, chunk_tokens=1, int8_kv=False)

    # Prime Mamba caches with explicit zero state (same as gen_nemotron_reference.py).
    H = int(streamed.config["mamba_num_heads"])
    Dh = int(streamed.config["mamba_head_dim"])
    Ds = int(streamed.config["ssm_state_size"])
    G = int(streamed.config["n_groups"])
    K = int(streamed.config["conv_kernel"])
    conv_dim = H * Dh + 2 * G * Ds
    primed = 0
    for entry in core.cache:
        if hasattr(entry, "cache") and len(entry.cache) == 2 and entry.cache[1] is None:
            entry[0] = mx.zeros((1, K - 1, conv_dim), mx.bfloat16)
            entry[1] = mx.zeros((1, H, Dh, Ds), mx.float32)
            primed += 1
    print(f"[pydump] primed {primed} Mamba caches", file=sys.stderr, flush=True)

    layers = streamed.model.backbone.layers
    assert len(layers) == 52, f"expected 52 layers, got {len(layers)}"
    assert all(isinstance(l, NemotronHBlock) for l in layers), "unexpected layer type"

    capture = LayerCapture(layers)
    block_cls = install(capture)
    mb_cap = None
    mb_hooks = None
    if args.dump_mamba_layer is not None:
        li = args.dump_mamba_layer
        mb_cap = MambaCapture(layers[li].mixer, lambda: capture._step)
        mb_hooks = install_mamba_hooks(layers[li].mixer, mb_cap)
        print(f"[pydump] mamba hooks on layer {li}", file=sys.stderr, flush=True)
    moe_cap = None
    moe_hooks = None
    if args.dump_moe_layer is not None:
        li = args.dump_moe_layer
        mixer = layers[li].mixer
        moe_cap = MoeCapture(mixer, lambda: capture._step)
        moe_hooks = install_moe_hooks(mixer, moe_cap)
        print(f"[pydump] MoE hooks on layer {li} ({type(mixer).__name__})",
              file=sys.stderr, flush=True)
    try:
        print(f"[pydump] prefill {len(prompt)} tokens (chunk=1)", file=sys.stderr,
              flush=True)
        logits = core.prefill(prompt)
        # Also run one decode so the captured step set is complete (step 0's
        # logits are from the last prefill step; decode not needed for the dump,
        # but confirms the path stays alive).
        for i in range(args.tokens):
            if i == 0:
                d = digest_array(logits)
                tok = int(mx.argmax(mx.reshape(logits, (-1,))).item())
                print(f"[pydump] step0 token={tok} logit_sha={d[:12]}",
                      file=sys.stderr, flush=True)
            if i + 1 < args.tokens:
                logits = core.decode(int(mx.argmax(mx.reshape(logits, (-1,))).item()))
    finally:
        uninstall(block_cls)
        if mb_hooks is not None:
            uninstall_mamba_hooks(mb_hooks)
        if moe_hooks is not None:
            uninstall_moe_hooks(*moe_hooks)

    steps_seen = sorted(capture.outputs)
    print(f"[pydump] captured steps: {steps_seen} "
          f"({len(capture.outputs[steps_seen[0]])} layers/step)",
          file=sys.stderr, flush=True)

    last_step = len(prompt) - 1
    if args.step is not None:
        selected = [args.step]
    elif args.all_steps:
        selected = steps_seen
    else:
        selected = [last_step]
    for s in selected:
        if s not in capture.outputs:
            raise RuntimeError(f"step {s} not captured; have {steps_seen}")

    result = {
        "checkpoint": args.checkpoint,
        "prompt_ids": prompt,
        "prompt_sha256": prompt_sha,
        "prompt_source": prompt_src,
        "n_layers": capture.n_layers,
        "steps_captured": steps_seen,
        "selected_steps": selected,
        "anchors": ANCHORS,
        "mlx_version": mx.__version__,
        "numpy_version": np.__version__,
        "digests": {},
    }

    for s in selected:
        per_layer = capture.outputs[s]
        digests = {}
        for l in range(capture.n_layers):
            if l not in per_layer:
                raise RuntimeError(f"step {s} layer {l} missing")
            d = digest_array(per_layer[l])
            digests[l] = d
            print(f"[PYNEMOLAYER step={s} L={l:02d}] {d}", flush=True)
        result["digests"][str(s)] = digests

    if mb_cap is not None:
        result["mamba"] = {}
        for s in selected:
            md = dump_mamba(mb_cap, s)
            result["mamba"][str(s)] = md
            if md is not None:
                for k, v in md.items():
                    print(f"[PYMB step={s} {k}] {v}", flush=True)

    if moe_cap is not None:
        result["moe"] = {}
        for s in selected:
            md = dump_moe(moe_cap, capture, s, capture.n_layers)
            result["moe"][str(s)] = md
            if md is not None:
                for k, v in md.items():
                    print(f"[PYMOE step={s} {k}] {v}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"[pydump] wrote {args.out}", file=sys.stderr, flush=True)
    streamed.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
