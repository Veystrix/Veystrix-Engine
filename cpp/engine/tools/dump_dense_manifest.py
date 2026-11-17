#!/usr/bin/env python3
"""Dump a per-module dense-parameter manifest for the Flash-Next checkpoint.

This mirrors ``models/flashnext/loader.py`` model construction for the purpose
of classifying dense parameters: lazy ``mx.load`` of the shards, the exact
``class_predicate`` from loader.py:209-235, and the same ``nn.quantize`` call.
It deliberately STOPS before ``model.load_weights`` / ``mx.eval`` / ``model.eval``:
there is no forward pass and no checkpoint weight is read beyond its header
(shape / dtype). Module weights stay at their random initialization; only the
module classes and their quantization metadata are inspected.

The output is ``{module_path: {kind, group_size, bits, has_bias, weight_dtype}}``
for every NON-streamed module that owns a ``weight`` parameter. Path keys are
module paths (the owners of the ``.weight`` parameter), e.g.
``language_model.model.layers.0.mlp.gate_proj``. ``kind`` is derived from the
module class AFTER quantization:

  * ``nn.QuantizedLinear``     -> quantized_linear (with its group_size / bits)
  * ``nn.QuantizedEmbedding``  -> quantized_embedding
  * ``nn.Linear``              -> linear (a projection the predicate did not quantize)
  * anything else              -> raw

Swapped/streamed modules (``.switch_mlp.`` / ``.ngram_embedding.``) are excluded,
as are those whose path is a streamed marker. MTP is off (production default:
``FLASHNEXT_NATIVE_MTP`` defaults to ``off``); rmsnorm convention detection is
skipped because it reads tensor bytes, which this tool must not do.

Run real checkpoint (do NOT need a forward pass, and do NOT run this lightly):
  .venv/bin/python cpp/engine/tools/dump_dense_manifest.py \
      --checkpoint /path/to/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --out /path/to/dense_manifest.json

Run synthetic unit-test fixture (tiny, fast, no checkpoint):
  .venv/bin/python cpp/engine/tools/dump_dense_manifest.py \
      --synthetic-fixture --out /tmp/synthetic_manifest.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Mirrors models/flashnext/loader.py:51.
STREAMED = (".switch_mlp.", ".ngram_embedding.")


def _is_streamed(key: str) -> bool:
    """Mirrors models/flashnext/loader.py:58-59."""
    return any(marker in key for marker in STREAMED)


def make_class_predicate(weights: dict, mode: str):
    """Exact copy of the closure from loader.py:209-235.

    ``weights`` maps parameter path -> lazily-loaded mx.array (headers only).
    Only ``.shape`` is read -- no tensor values, no eval.
    """
    def class_predicate(module_path, module):
        if _is_streamed(f"{module_path}."):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if hasattr(module, "weight") and module.weight.size % 64 != 0:
            return False
        weight_key = f"{module_path}.weight"
        scales_key = f"{module_path}.scales"
        if weight_key not in weights or scales_key not in weights:
            return False

        # Infer every module from its logical and packed tensor shapes.
        logical_in = int(module.weight.shape[-1])
        packed_in = int(weights[weight_key].shape[-1])
        groups = int(weights[scales_key].shape[-1])
        bits = packed_in * 32 // logical_in
        group_size = logical_in // groups
        return {"group_size": group_size, "bits": bits, "mode": mode}

    return class_predicate


def classify_module(module: nn.Module) -> dict:
    """Derive the manifest entry for one (weight-owning) module."""
    if isinstance(module, nn.QuantizedLinear):
        return {
            "kind": "quantized_linear",
            "group_size": int(module.group_size),
            "bits": int(module.bits),
            "has_bias": "bias" in module,
            "weight_dtype": str(module["weight"].dtype),
        }
    if isinstance(module, nn.QuantizedEmbedding):
        return {
            "kind": "quantized_embedding",
            "group_size": int(module.group_size),
            "bits": int(module.bits),
            "has_bias": False,
            "weight_dtype": str(module["weight"].dtype),
        }
    if isinstance(module, nn.Linear):
        return {
            "kind": "linear",
            "group_size": None,
            "bits": None,
            "has_bias": "bias" in module,
            "weight_dtype": str(module["weight"].dtype),
        }
    return {
        "kind": "raw",
        "group_size": None,
        "bits": None,
        "has_bias": False,
        "weight_dtype": str(module.weight.dtype) if hasattr(module, "weight") else None,
    }


def collect_manifest(model: nn.Module) -> dict:
    """Collect one entry per non-streamed module that owns a ``weight``."""
    manifest = {}
    for path, module in model.named_modules():
        if not path or _is_streamed(f"{path}."):
            continue
        if not hasattr(module, "weight"):
            continue
        manifest[path] = classify_module(module)
    return manifest


def build_from_checkpoint(checkpoint: str, keep_vision: bool = False) -> tuple[nn.Module, dict]:
    """Build the model like loader.load_streaming, stopping before weight load."""
    from mlx_vlm.utils import (
        apply_generation_config_defaults,
        get_model_and_args,
        load_config,
        update_module_configs,
    )

    path = Path(os.path.expanduser(checkpoint))
    config = load_config(path)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_class, _ = get_model_and_args(config=config, model_path=path)
    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, model_class, config,
        ["text", "vision", "perceiver", "projector", "audio"])
    model_config = apply_generation_config_defaults(model_config, config)
    model = model_class.Model(model_config)
    if not keep_vision:
        model.vision_tower = None
    # MTP attach and rmsnorm fingerprint are intentionally skipped: MTP is off
    # by production default, and the fingerprint reads tensor bytes.

    # Lazy load: shapes are known from headers, nothing is evaluated.
    weights = {}
    for shard in sorted(glob.glob(str(path / "*.safetensors"))):
        weights.update(mx.load(shard))

    quantization = config.get("quantization") or {}
    mode = quantization.get("mode", "affine")
    nn.quantize(
        model,
        group_size=quantization.get("group_size"),
        bits=quantization.get("bits"),
        mode=mode,
        class_predicate=make_class_predicate(weights, mode),
    )
    return model, manifest_note(mode)


def manifest_note(mode: str) -> dict:
    return {"mode": mode, "mtp": "off (production default)",
            "stop_before": "load_weights / eval"}


def build_synthetic() -> tuple[nn.Module, dict]:
    """Tiny deterministic model exercising every manifest kind.

    Modules and expected kinds:
      norm                  -> raw
      dense                 -> quantized_linear (gs=32, bits=4, bias)
      head                  -> linear (no scales in the synthetic weight map)
      emb                   -> quantized_embedding
      block.switch_mlp      -> excluded (streamed marker)
    """
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(64)
            self.dense = nn.Linear(64, 128, bias=True)
            self.head = nn.Linear(64, 128, bias=False)
            self.emb = nn.Embedding(256, 64)
            self.block = nn.Module()
            self.block.switch_mlp = nn.Linear(64, 128, bias=True)

    model = Tiny()

    # Build a synthetic weight map with packed weights + scales for the modules
    # we want nn.quantize to convert. ``head`` deliberately has no scales entry
    # so it stays a plain linear.
    weights = {}
    for name, module in model.named_modules():
        if not name or not hasattr(module, "to_quantized"):
            continue
        if name == "head" or _is_streamed(f"{name}."):
            # head has no scales entry -> stays a plain Linear; the streamed
            # module is left unquantized and excluded from the manifest.
            continue
        qw, qs, qb = mx.quantize(module.weight, group_size=32, bits=4, mode="affine")
        weights[f"{name}.weight"] = qw
        weights[f"{name}.scales"] = qs
        weights[f"{name}.biases"] = qb

    nn.quantize(
        model, group_size=32, bits=4, mode="affine",
        class_predicate=make_class_predicate(weights, "affine"),
    )
    return model, {"mode": "affine", "mtp": "n/a (synthetic)",
                   "stop_before": "load_weights / eval"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None,
                    help="Flash-Next checkpoint dir (real mode)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--synthetic-fixture", action="store_true",
                    help="build a tiny fake model instead of reading a checkpoint")
    ap.add_argument("--keep-vision", action="store_true",
                    help="keep the vision tower (default: drop, matching the backend)")
    args = ap.parse_args()

    if args.synthetic_fixture:
        model, note = build_synthetic()
    elif args.checkpoint:
        model, note = build_from_checkpoint(args.checkpoint, args.keep_vision)
    else:
        ap.error("provide --checkpoint or --synthetic-fixture")

    manifest = collect_manifest(model)
    out = {"note": note, "manifest": manifest}
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path}: {len(manifest)} non-streamed parameter paths")
    for path, entry in manifest.items():
        print(f"  {path}: {json.dumps(entry)}")


if __name__ == "__main__":
    main()
