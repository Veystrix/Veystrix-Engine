#!/usr/bin/env python3
"""OWNER-RUN diagnostic: dump layer-0 MoE intermediates from the REAL model.

Mirrors dump_python_intermediates.py but hooks the MoE block of layer 0 (a
linear-attention layer, top_k=10, 512 experts, routed experts streamed via
mx.gather_qmm). Captures the exact S=16 prefill inputs and every MoE
intermediate, plus the stacked expert weights and the shared-expert weights,
so probe_moe_intermediates.cpp can replay the C++ MoE on IDENTICAL inputs.

Captured (raw little-endian f32 except *.weight which is uint32):
  m2d          [1,16,2560]   input to the MoE block (MLP hyper-connection mixed)
  gate_logits  [1,16,512]    self.gate(m2d)  (dense Linear)
  gate_w       [512,2560]    router weight (bf16)
  inds         [1,16,10]     router indices actually passed to switch_mlp
  scores       [1,16,10]     router scores (f32)
  local        [16,10]       first-occurrence expert -> stacked-row mapping
  wanted       [n_exp]       expert ids in stacked-row order
  routed_y     [1,16,10,2560]  switch_mlp(m2d, inds) output (before score combine)
  combined     [1,16,2560]   (routed_y * scores).sum(-2)
  sgl, sg      [1,16,1]      shared gate logits / sigmoid
  se_g, se_u   [1,16,640]    shared gate/up projections
  se_swiglu    [1,16,640]    silu(se_g)*se_u
  se_s         [1,16,2560]   shared down projection
  sg_s         [1,16,2560]   sg * se_s
  moe2d        [1,16,2560]   combined + sg_s (final MoE branch)
  w_gate/w_up/w_down .weight/.scales/.biases  stacked expert rows in `wanted` order
  w_shared_{gate,up,down,sgate} .weight/.scales/.biases  shared-expert weights
  sorted_flag  text: sorted_indices passed by Python's gather_qmm

The hooks wrap the classes the backend already patched at init; patching only
the original functions would be bypassed (same lesson as linear_state.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = "/Users/vincent/.local/share/opencode/worktree/dd9a0e/cpp-engine-qdense"
VENV_SITE = "/Users/vincent/Documents/Veystrix/.venv/lib/python3.12/site-packages"
for _p in (REPO, VENV_SITE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_vlm.models.activations import swiglu  # noqa: E402
from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    Qwen4ExpGatedResidual, Qwen4ExpDecoderLayer, Qwen4ExpGatedDeltaNet,
)
from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP  # noqa: E402
from models.flashnext.expert_cache import (  # noqa: E402
    StreamingSwitchGLU, StreamingSwitchLinear,
)
from models.flashnext import compiled as moe_compiled  # noqa: E402

PROMPT_SEED = 20261117
_STATE = [None]
_ORIG = {}


def _to_np(a) -> np.ndarray:
    mx.eval(a)
    if getattr(a, "dtype", None) in (mx.bfloat16, mx.float16):
        a = mx.astype(a, mx.float32)
    return np.asarray(a)


def _dump(out_dir: Path, manifest: dict, name: str, a, dtype_tag=None):
    dtype = getattr(a, "dtype", None)
    shape = [int(v) for v in a.shape]
    if dtype == mx.uint32 or dtype_tag == "uint32":
        raw = np.asarray(a, dtype=np.uint32).tobytes()
        tag = "uint32"
    else:
        raw = _to_np(a).astype("<f4").tobytes()
        tag = str(dtype)
    (out_dir / (name + ".bin")).write_bytes(raw)
    manifest["files"][name] = {
        "file": name + ".bin", "shape": shape, "dtype": tag,
        "n": int(np.prod(shape)), "sha256": hashlib.sha256(raw).hexdigest(),
        "first": np.frombuffer(raw, dtype="<u4" if tag == "uint32" else "<f4")[:4].tolist(),
    }


def install(target_layer_idx: int):
    _ORIG["moe"] = Qwen3_5MoeSparseMoeBlock.__call__
    _ORIG["glu"] = StreamingSwitchGLU.__call__
    _ORIG["lin"] = StreamingSwitchLinear.__call__
    _ORIG["mlp"] = Qwen3_5MLP.__call__
    _ORIG["renorm"] = moe_compiled.renorm
    _ORIG["gr"] = Qwen4ExpGatedResidual.__call__
    _ORIG["layer"] = Qwen4ExpDecoderLayer.__call__
    _ORIG["gdn"] = Qwen4ExpGatedDeltaNet.__call__

    def gr_call(self, hyper_input):
        st = _STATE[0]
        layer = st["target_layer"]
        is_t = (self is layer.attn_hyper_connection or
                self is layer.mlp_hyper_connection)
        if is_t and int(hyper_input.shape[1]) > 1 and st["gr_calls"] < 2:
            idx = st["gr_calls"]
            st["gr_calls"] += 1
            out = _ORIG["gr"](self, hyper_input)
            hc, H = self.hc_count, self.hidden_size
            normed = self.hc_norm(hyper_input)
            down = self.input_mix_weight_down(normed)
            lowpre = down / self.hc_count          # pre-SiLU, the nn.silu input
            lo = nn.silu(lowpre)
            st["cap"][f"gr{idx}_down"] = down
            st["cap"][f"gr{idx}_lowpre"] = lowpre
            mix = mx.sigmoid(self.input_mix_weight_up(lo))
            mixg = mix.reshape(*mix.shape[:-1], hc, H)
            normg = normed.reshape(*normed.shape[:-1], hc, H)
            st["cap"][f"gr{idx}_input"] = hyper_input
            st["cap"][f"gr{idx}_normed"] = normed
            st["cap"][f"gr{idx}_low"] = lo
            st["cap"][f"gr{idx}_mix"] = mix
            st["cap"][f"gr{idx}_prod"] = mixg * normg
            st["cap"][f"gr{idx}_mixed"] = out[0]
            st["cap"][f"gr{idx}_inj"] = out[2]
            return out
        return _ORIG["gr"](self, hyper_input)

    def layer_call(self, hidden_states, input_ids, mask=None, cache=None,
                   position_ids=None):
        st = _STATE[0]
        first = (self is st["target_layer"] and int(hidden_states.shape[1]) > 1
                 and not st["layer_captured"])
        if first:
            st["layer_captured"] = True
            st["cap"]["layer_input"] = hidden_states
        out = _ORIG["layer"](self, hidden_states, input_ids, mask=mask,
                             cache=cache, position_ids=position_ids)
        if first:
            st["cap"]["layer_output"] = out
        return out

    def gdn_call(self, inputs, mask=None, cache=None):
        st = _STATE[0]
        out = _ORIG["gdn"](self, inputs, mask=mask, cache=cache)
        target_la = getattr(st["target_layer"], "linear_attn", None)
        if (self is target_la
                and int(inputs.shape[1]) > 1 and "gdn_out" not in st["cap"]):
            st["cap"]["gdn_out"] = out
        return out

    Qwen4ExpGatedResidual.__call__ = gr_call
    Qwen4ExpDecoderLayer.__call__ = layer_call
    Qwen4ExpGatedDeltaNet.__call__ = gdn_call

    def renorm_hook(scores, topk_mass, blend):
        out, norm = _ORIG["renorm"](scores, topk_mass, blend)
        st = _STATE[0]
        if st["moe_calls"] == 1 and "scores" not in st["cap"]:
            st["cap"]["scores"] = out
        return out, norm

    moe_compiled.renorm = renorm_hook

    def moe_call(self, x):
        st = _STATE[0]
        pre = None
        if self is st["target_moe"] and x.shape[1] > 1 and st["moe_calls"] == 0:
            st["moe_calls"] += 1
            gate_logits = self.gate(x)
            sgl = self.shared_expert_gate(x)
            sg = mx.sigmoid(sgl)
            se = self.shared_expert
            se_g = se.gate_proj(x)
            se_u = se.up_proj(x)
            se_swiglu = swiglu(se_g, se_u)
            se_s = se.down_proj(se_swiglu)
            pre = dict(m2d=x, gate_logits=gate_logits, gate_w=self.gate.weight,
                       sgl=sgl, sg=sg, se_g=se_g, se_u=se_u,
                       se_swiglu=se_swiglu, se_s=se_s, sg_s=sg * se_s)
        out = _ORIG["moe"](self, x)
        if pre is not None:
            pre["moe2d"] = out
            st["cap"].update(pre)
        return out

    def glu_call(self, x, *args, **kwargs):
        st = _STATE[0]
        is_t = self is st["target_glu"] and x.shape[1] > 1 and st["glu_calls"] == 0
        if is_t:
            st["glu_calls"] += 1
            st["cap"]["glu_x"] = x
            st["cap"]["glu_indices"] = args[0] if args else kwargs.get("indices")
        out = _ORIG["glu"](self, x, *args, **kwargs)
        if is_t:
            st["cap"]["routed_y"] = out
        return out

    def lin_call(self, x, indices, *args, **kwargs):
        st = _STATE[0]
        prefix = self.cache.prefix
        is_t = (f".layers.{st['layer']}.mlp.switch_mlp." in prefix
                and st["lin_calls"] < 3 and st["glu_calls"] >= 1)
        if is_t:
            st["lin_calls"] += 1
            proj = prefix.rsplit(".", 1)[-1]  # gate_proj / up_proj / down_proj
            rec = st["cap"].setdefault("lin", {}).setdefault(proj, {})
            rec.update({
                "weights": kwargs.get("weights"),
                "sorted_indices": kwargs.get("sorted_indices"),
                "indices": indices,
                "x": x,
            })
        out = _ORIG["lin"](self, x, indices, *args, **kwargs)
        if is_t:
            st["cap"]["lin"][proj]["out"] = out
        return out

    Qwen3_5MoeSparseMoeBlock.__call__ = moe_call
    StreamingSwitchGLU.__call__ = glu_call
    StreamingSwitchLinear.__call__ = lin_call


def uninstall():
    Qwen3_5MoeSparseMoeBlock.__call__ = _ORIG["moe"]
    StreamingSwitchGLU.__call__ = _ORIG["glu"]
    StreamingSwitchLinear.__call__ = _ORIG["lin"]
    moe_compiled.renorm = _ORIG["renorm"]
    Qwen4ExpGatedResidual.__call__ = _ORIG["gr"]
    Qwen4ExpDecoderLayer.__call__ = _ORIG["layer"]
    Qwen4ExpGatedDeltaNet.__call__ = _ORIG["gdn"]


def run_real(args):
    from veystrix.backends.flashnext import FlashNextBackend

    os.environ.setdefault("VEYSTRIX_SAMPLING_GREEDY", "1")
    backend = FlashNextBackend(model_path=args.checkpoint)
    if getattr(backend, "sampling", None) is not None:
        try:
            backend.sampling.greedy = True
        except Exception:
            pass

    layers = backend.language.model.layers
    assert len(layers) > args.layer
    moe = layers[args.layer].mlp
    glu = moe.switch_mlp
    _STATE[0] = {"target_moe": moe, "target_glu": glu, "layer": args.layer,
                 "target_layer": layers[args.layer], "gr_calls": 0,
                 "layer_captured": False,
                 "moe_calls": 0, "glu_calls": 0, "lin_calls": 0, "cap": {}}

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    install(args.layer)
    try:
        rng = np.random.default_rng(PROMPT_SEED)
        prompt = rng.integers(0, 200000, size=args.prompt_tokens).tolist()
        backend.append_tokens(prompt)
        backend.generate(args.tokens, on_decode_token=lambda *_: None)
    finally:
        uninstall()

    cap = _STATE[0]["cap"]
    if "moe2d" not in cap or "routed_y" not in cap:
        raise RuntimeError(f"L{args.layer} MoE not captured: keys={sorted(cap.keys())}")
    if "scores" not in cap:
        # Fallback when compiled.installed() is False (then _moe_call computes
        # scores inline and never calls compiled.renorm).
        k = moe.top_k
        gates = mx.softmax(cap["gate_logits"], axis=-1, precise=True)
        ik = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        sc = mx.take_along_axis(gates, ik, axis=-1)
        cap["scores"] = sc / sc.sum(axis=-1, keepdims=True)

    # Derive the local (expert -> stacked-row) mapping from the router inds.
    inds = cap["glu_indices"]
    flat = np.asarray(_to_np(mx.astype(inds, mx.int32)), dtype=np.int32).reshape(-1)
    wanted = list(dict.fromkeys(flat.tolist()))
    order = {e: i for i, e in enumerate(wanted)}
    local = np.array([order[int(e)] for e in flat], dtype=np.int32).reshape(inds.shape)

    manifest = {
        "purpose": f"L{args.layer} MoE intermediates, S>1 prefill, REAL checkpoint",
        "checkpoint": args.checkpoint, "layer": args.layer,
        "prompt_seed": PROMPT_SEED, "prompt_tokens": prompt,
        "mlx_version": mx.__version__,
        "top_k": int(moe.top_k), "n_experts": int(moe.num_experts),
        "group_size": int(glu.gate_proj.group_size), "bits": int(glu.gate_proj.bits),
        "files": {},
    }

    # Straight captures.
    for name in ("m2d", "gate_logits", "gate_w",
                 "inds", "scores", "routed_y"):
        _dump(out_dir, manifest, name, cap[name] if name != "inds" else inds)
    _dump(out_dir, manifest, "local", mx.array(local, dtype=mx.int32))
    _dump(out_dir, manifest, "wanted", mx.array(wanted, dtype=mx.int32))
    for name in ("combined", "sgl", "sg", "se_g", "se_u", "se_swiglu",
                 "se_s", "sg_s", "moe2d"):
        if name == "combined":
            continue  # computed below from routed_y * scores
        _dump(out_dir, manifest, name, cap[name])
    # Python's exact combine: (routed_y * scores[...,None]).sum(-2)
    combined = (cap["routed_y"] * mx.expand_dims(cap["scores"], -1)).sum(axis=-2)
    _dump(out_dir, manifest, "combined", combined)

    # Stacked expert weights (in `wanted` order) captured from gather_qmm.
    lin = cap.get("lin", {})
    for proj in ("gate_proj", "up_proj", "down_proj"):
        e = lin.get(proj)
        if e is None or e["weights"] is None:
            raise RuntimeError(f"missing weights for {proj}: {sorted(lin.keys())}")
        w, s, b = e["weights"]
        for tag, arr in (("weight", w), ("scales", s), ("biases", b)):
            _dump(out_dir, manifest, f"w_{proj}_{tag}", arr)
        _dump(out_dir, manifest, f"{proj}_out", e["out"])
        _dump(out_dir, manifest, f"{proj}_indices", e["indices"])
        manifest[f"sorted_{proj}"] = e["sorted_indices"]

    # ---- Reference replay: identical x/local/scores/weights, NO gather_sort
    # and sorted_indices=False (the exact inputs the C++ probe uses). This
    # isolates gather_qmm compute from Python's sorted/permuted prefill path.
    Tn = int(cap["m2d"].shape[1])
    x2 = mx.reshape(cap["m2d"], (Tn, -1))
    rlocal = mx.array(local, dtype=mx.uint32)
    rs = mx.reshape(cap["scores"], (Tn, -1))
    x4 = mx.expand_dims(x2, (-2, -3))
    lw = {p: lin[p]["weights"] for p in ("gate_proj", "up_proj", "down_proj")}
    gs = glu.gate_proj.group_size
    bits = glu.gate_proj.bits
    mode = glu.gate_proj.mode

    def gq(x, proj):
        w, s, b = lw[proj]
        return mx.gather_qmm(x, w, s, b, rhs_indices=rlocal, transpose=True,
                             group_size=gs, bits=bits, mode=mode,
                             sorted_indices=False)

    ref_g = gq(x4, "gate_proj")
    ref_u = gq(x4, "up_proj")
    # SwiGLU.__call__(x_up, x_gate) -> swiglu(gate=x_gate, x=x_up) = silu(g)*u
    ref_a = swiglu(ref_g, ref_u)
    ref_d = gq(ref_a, "down_proj")
    ref_d2 = mx.squeeze(ref_d, -2)
    ref_weighted = ref_d2 * rs[..., None]
    ref_comb = ref_weighted.sum(axis=-2)
    for nm, arr in (("ref_g", ref_g), ("ref_u", ref_u), ("ref_a", ref_a),
                    ("ref_d", ref_d2), ("ref_weighted", ref_weighted),
                    ("ref_comb", ref_comb)):
        _dump(out_dir, manifest, nm, arr)

    # Shared expert weights.
    se = moe.shared_expert
    for tag, mod in (("gate", se.gate_proj), ("up", se.up_proj),
                     ("down", se.down_proj), ("sgate", moe.shared_expert_gate)):
        for kind in ("weight", "scales", "biases"):
            _dump(out_dir, manifest, f"w_shared_{tag}_{kind}", getattr(mod, kind))

    # ---- Hyper-connection path (upstream of the MoE) -----------------------
    layer = layers[args.layer]
    for nm in ("layer_input", "layer_output", "gdn_out",
               "gr0_input", "gr0_normed", "gr0_down", "gr0_lowpre", "gr0_low",
               "gr0_mix", "gr0_prod", "gr0_mixed", "gr0_inj",
               "gr1_down", "gr1_lowpre",
               "gr1_input", "gr1_normed", "gr1_low", "gr1_mix", "gr1_prod",
               "gr1_mixed", "gr1_inj"):
        if nm in cap:
            _dump(out_dir, manifest, nm, cap[nm])
    for tag, hc in (("attn", layer.attn_hyper_connection),
                    ("mlp", layer.mlp_hyper_connection)):
        _dump(out_dir, manifest, f"hc_{tag}_norm_w", hc.hc_norm.weight)
        for part in ("input_mix_weight_down", "input_mix_weight_up",
                     "block_inject_weight"):
            sub = getattr(hc, part)
            for kind in ("weight", "scales", "biases"):
                v = getattr(sub, kind, None)
                if v is not None:
                    _dump(out_dir, manifest, f"hc_{tag}_{part}_{kind}", v)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cfg_lines = [f"{k}={v}" for k, v in manifest.items()
                 if k in ("top_k", "n_experts", "group_size", "bits")]
    cfg_lines += [f"sorted_{p}={manifest.get('sorted_'+p)}"
                  for p in ("gate_proj", "up_proj", "down_proj")]
    (out_dir / "config.txt").write_text("\n".join(cfg_lines) + "\n")

    print(f"captured L{args.layer} MoE: {len(manifest['files'])} files -> {out_dir}")
    print(f"  routed_y={manifest['files']['routed_y']['shape']} "
          f"n_experts_used={len(wanted)} "
          f"sorted_indices={manifest.get('sorted_gate_proj')}")
    for name in ("m2d", "gate_logits", "inds", "scores", "routed_y",
                 "combined", "moe2d", "sg_s"):
        d = manifest["files"][name]
        print(f"  {name:12s} shape={d['shape']} sha={d['sha256'][:12]} first={d['first'][:3]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    args = ap.parse_args()
    run_real(args)


if __name__ == "__main__":
    main()
