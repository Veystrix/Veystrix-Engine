#!/usr/bin/env python3
"""Generate flat, bit-exact config vectors for test_model_adapter.

Reads the REAL checkpoint config.json files with Python's ``json`` module and
writes one flat ``key=value`` text file per model. The C++ test then feeds the
scalars back through from_qwen_config() / from_nemotron_config() and checks
every field of the resulting ModelSpec, including the Python-side ``expected_*``
ground truth for derived fields (per-layer kind mapping, MoE mask, prefix
format).

No MLX / numpy needed — pure stdlib.

Flat format (see also cpp/engine/src/model_adapter.hpp):
  * UTF-8, one ``key=value`` per line.
  * '#' comments and blank lines ignored.
  * Vector values are comma-separated, no spaces.
  * Floats written in repr form (e.g. 1e-06) and parsed with std::stof.
  * ``expected_*`` keys are the Python-side ground truth for derived values.

Usage:
  python cpp/engine/tools/gen_model_adapter_vectors.py OUTPUT_DIR

  # with the verify command in the handoff:
  python cpp/engine/tools/gen_model_adapter_vectors.py /tmp/model_adapter_flat
  /tmp/test_model_adapter /tmp/model_adapter_flat

Default checkpoint paths can be overridden with --qwen-config / --nemotron-config.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

QWEN_DEFAULT = "/Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP/config.json"
NEMOTRON_DEFAULT = os.path.expanduser(
    "~/Models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community/config.json"
)

# Python-side mirror of the C++ layer-name -> LayerKind mapping. Kept here as
# the ground truth the C++ builder is checked against.
QWEN_KIND = {
    "linear_attention": "LINEAR_ATTN_GDN",
    "full_attention": "FULL_ATTN_QSA",
}
NEMOTRON_KIND = {
    "mamba": "MAMBA_SSM",
    "attention": "FULL_ATTN_QSA",   # plain dense attention; use_qsa_indexer=0
    "moe": "MOE",
}
QWEN_PREFIX = "language_model.model.layers.{L}.mlp.switch_mlp.{P}.{T}"
NEMOTRON_PREFIX = "backbone.layers.{L}.mixer.switch_mlp.{P}.{T}"


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fmt(x) -> str:
    """Stable scalar string for the flat file (no surrounding quotes)."""
    if isinstance(x, bool):
        return "1" if x else "0"
    return str(x)


def _vec(items) -> str:
    return ",".join(str(x) for x in items)


def qwen_lines(cfg: dict) -> list[str]:
    t = cfg["text_config"]
    layer_types = t["layer_types"]
    kinds = [QWEN_KIND[k] for k in layer_types]
    has_moe = [1] * len(layer_types)
    q = t

    kv = {
        "model_family": "QWEN4_EXP",
        "hidden_size": q["hidden_size"],
        "n_layers": q["num_hidden_layers"],
        "layer_types": _vec(layer_types),
        "expected_layer_kinds": _vec(kinds),
        "expected_has_moe": _vec(has_moe),
        "n_experts": q["num_experts"],
        "top_k": q["num_experts_per_tok"],
        "group_size": cfg["quantization"]["group_size"],
        "moe_intermediate": q["moe_intermediate_size"],
        "expert_activation": "SWIGLU" if q.get("hidden_act", "silu") == "silu" else "UNKNOWN",
        "expert_prefix": QWEN_PREFIX,
        "norm_eps": q["rms_norm_eps"],
        "hc_count": q["hc_count"],
        "hc_lowrank": q["hc_lowrank"],
        "output_gate_type": q["output_gate_type"],
        "use_qsa_indexer": 1,
        "qsa_n_heads": q["indexer_n_heads"],
        "qsa_kv_heads": q["indexer_kv_heads"],
        "qsa_head_dim": q["indexer_head_dim"],
        "qsa_compress_ratio": q["indexer_compress_ratio"],
        "qsa_token_budget": q["indexer_budget"],
        "qsa_eps": q["rms_norm_eps"],
        "gdn_conv_kernel": q["linear_conv_kernel_dim"],
        "gdn_n_key_heads": q["linear_num_key_heads"],
        "gdn_n_value_heads": q["linear_num_value_heads"],
        "gdn_key_head_dim": q["linear_key_head_dim"],
        "gdn_value_head_dim": q["linear_value_head_dim"],
    }
    return ["# Qwen3.8-Flash-Next model-adapter flat config (generated)"] + [
        f"{k}={_fmt(v)}" for k, v in kv.items()
    ]


def nemotron_lines(cfg: dict) -> list[str]:
    layer_types = cfg["layers_block_type"]
    kinds = [NEMOTRON_KIND[k] for k in layer_types]
    has_moe = [1 if k == "moe" else 0 for k in layer_types]

    kv = {
        "model_family": "NEMOTRON_H",
        "hidden_size": cfg["hidden_size"],
        "n_layers": cfg["num_hidden_layers"],
        "layer_types": _vec(layer_types),
        "expected_layer_kinds": _vec(kinds),
        "expected_has_moe": _vec(has_moe),
        "n_experts": cfg["n_routed_experts"],
        "top_k": cfg["num_experts_per_tok"],
        "group_size": cfg["quantization"]["group_size"],
        "moe_intermediate": cfg["moe_intermediate_size"],
        "expert_activation": "RELU2" if cfg.get("mlp_hidden_act") == "relu2" else "UNKNOWN",
        "expert_prefix": NEMOTRON_PREFIX,
        "norm_eps": cfg["layer_norm_epsilon"],
        "hc_count": 0,
        "hc_lowrank": 0,
        "output_gate_type": "",
        "use_qsa_indexer": 0,
        "qsa_n_heads": 0,
        "qsa_kv_heads": 0,
        "qsa_head_dim": 0,
        "qsa_compress_ratio": 0,
        "qsa_token_budget": 0,
        "qsa_eps": 0,
        "gdn_conv_kernel": 0,
        "gdn_n_key_heads": 0,
        "gdn_n_value_heads": 0,
        "gdn_key_head_dim": 0,
        "gdn_value_head_dim": 0,
    }
    return ["# Nemotron 3.5 Lightning model-adapter flat config (generated)"] + [
        f"{k}={_fmt(v)}" for k, v in kv.items()
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_dir", help="directory to write the flat config files into")
    ap.add_argument("--qwen-config", default=QWEN_DEFAULT)
    ap.add_argument("--nemotron-config", default=NEMOTRON_DEFAULT)
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    qwen = qwen_lines(_read(args.qwen_config))
    nemotron = nemotron_lines(_read(args.nemotron_config))

    (out / "qwen_model_adapter.txt").write_text("\n".join(qwen) + "\n", encoding="utf-8")
    (out / "nemotron_model_adapter.txt").write_text("\n".join(nemotron) + "\n", encoding="utf-8")

    print(f"wrote {out / 'qwen_model_adapter.txt'}")
    print(f"wrote {out / 'nemotron_model_adapter.txt'}")


if __name__ == "__main__":
    main()
