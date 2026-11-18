#!/usr/bin/env python3
"""Emit a flat key=value config for `nemotron_generate` from a real
NVIDIA-Nemotron-3.5-Lightning checkpoint plus its captured reference.json.

This helper is intentionally tiny: it ONLY reads config.json, the safetensors
index and reference.json. It never loads tensor data. Its output is a flat
config whose `shard.<tensor-name>` lines tell the C++ loader which shard each
dense (or routed-expert) tensor lives in, so the C++ side never has to guess
names or scan all four shards.

The reference stores the raw prompt ids, so no RNG regeneration is needed.

Usage:
  .venv/bin/python cpp/engine/tools/gen_nemotron_config.py \
      --checkpoint /Users/vincent/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit-mlx-community \
      --reference results/nemotron35/<ts>-cpp-ref/reference.json \
      --out /tmp/nemotron_config.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def needed_tensors(cfg: dict, n_layers: int) -> list[str]:
    """Every tensor the C++ decode tool reads, by exact checkpoint name."""
    types = cfg["layers_block_type"]
    out: list[str] = []
    for li in range(n_layers):
        base = f"backbone.layers.{li}"
        out.append(f"{base}.norm.weight")
        kind = types[li]
        if kind == "mamba":
            for suffix in (
                "mixer.norm.weight",
                "mixer.in_proj.weight", "mixer.in_proj.scales", "mixer.in_proj.biases",
                "mixer.conv1d.weight", "mixer.conv1d.bias",
                "mixer.A_log", "mixer.dt_bias", "mixer.D",
                "mixer.out_proj.weight", "mixer.out_proj.scales", "mixer.out_proj.biases",
            ):
                out.append(f"{base}.{suffix}")
        elif kind == "attention":
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                for part in ("weight", "scales", "biases"):
                    out.append(f"{base}.mixer.{proj}.{part}")
        elif kind == "moe":
            out += [
                f"{base}.mixer.gate.weight",
                f"{base}.mixer.gate.e_score_correction_bias",
                f"{base}.mixer.shared_experts.up_proj.weight",
                f"{base}.mixer.shared_experts.up_proj.scales",
                f"{base}.mixer.shared_experts.up_proj.biases",
                f"{base}.mixer.shared_experts.down_proj.weight",
                f"{base}.mixer.shared_experts.down_proj.scales",
                f"{base}.mixer.shared_experts.down_proj.biases",
            ]
            # Routed experts are streamed row-by-row by the C++ tool, but the
            # shard map is still needed to locate them.
            for proj in ("fc1", "fc2"):
                for part in ("weight", "scales", "biases"):
                    out.append(f"{base}.mixer.switch_mlp.{proj}.{part}")
        else:
            raise SystemExit(f"unknown layer kind {kind!r} at layer {li}")
    out += [
        "backbone.embeddings.weight", "backbone.embeddings.scales",
        "backbone.embeddings.biases", "backbone.norm_f.weight",
        "lm_head.weight", "lm_head.scales", "lm_head.biases",
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text())
    ref = json.loads(Path(args.reference).read_text())
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]

    prompt = [int(t) for t in ref["prompt_ids"]]
    prompt_sha = hashlib.sha256(
        b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
    ).hexdigest()
    if prompt_sha != ref["prompt_sha256"]:
        raise SystemExit(
            f"prompt SHA mismatch: regenerated {prompt_sha} != reference "
            f"{ref['prompt_sha256']}"
        )

    n_layers = int(cfg["num_hidden_layers"])
    quant = cfg.get("quantization", cfg.get("quantization_config", {}))
    lines: list[str] = []

    def kv(key: str, value) -> None:
        lines.append(f"{key}={value}")

    kv("hidden_size", cfg["hidden_size"])
    kv("n_layers", n_layers)
    kv("layer_types", ",".join(cfg["layers_block_type"]))
    kv("n_experts", cfg["n_routed_experts"])
    kv("top_k", cfg["num_experts_per_tok"])
    kv("group_size", quant.get("group_size", 64))
    kv("bits", quant.get("bits", 4))
    kv("moe_intermediate", cfg["moe_intermediate_size"])
    kv("shared_intermediate", cfg["moe_shared_expert_intermediate_size"])
    kv("norm_eps", cfg["norm_epsilon"] if "norm_epsilon" in cfg else cfg["layer_norm_epsilon"])
    kv("routed_scaling_factor", cfg.get("routed_scaling_factor", 2.5))
    kv("vocab", cfg["vocab_size"])
    mamba_heads = int(cfg["mamba_num_heads"])
    mamba_head_dim = int(cfg["mamba_head_dim"])
    kv("mamba_num_heads", mamba_heads)
    kv("mamba_head_dim", mamba_head_dim)
    kv("mamba_n_groups", cfg["n_groups"])
    kv("mamba_ssm_state_size", cfg["ssm_state_size"])
    kv("mamba_conv_kernel", cfg["conv_kernel"])
    kv("mamba_intermediate_size", mamba_heads * mamba_head_dim)
    kv("mamba_conv_dim", mamba_heads * mamba_head_dim + 2 * int(cfg["n_groups"]) * int(cfg["ssm_state_size"]))
    # mlx-lm NemotronH ModelArgs has no `time_step_limit` in config.json, so
    # its __post_init__ uses (0.0, inf); the time_step_min/max keys in the
    # checkpoint are NOT read by this revision of mlx-lm. Mirror that exactly.
    kv("mamba_time_step_min", 0.0)
    kv("mamba_time_step_max", "inf")
    kv("n_heads", cfg["num_attention_heads"])
    kv("kv_heads", cfg["num_key_value_heads"])
    kv("head_dim", cfg["head_dim"])
    kv("attn_scale", float(cfg["head_dim"]) ** -0.5)

    kv("prompt_tokens", len(prompt))
    kv("prompt_ids", ",".join(str(t) for t in prompt))
    kv("prompt_sha256", prompt_sha)

    for name in needed_tensors(cfg, n_layers):
        shard = wmap.get(name)
        if shard is None:
            raise SystemExit(f"tensor missing from checkpoint index: {name}")
        kv("shard." + name, shard)

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(
        f"wrote {args.out}: {n_layers} layers, "
        f"{sum(k.startswith('shard.') for k in lines)} tensor->shard entries, "
        f"prompt sha={prompt_sha[:12]}"
    )


if __name__ == "__main__":
    main()
