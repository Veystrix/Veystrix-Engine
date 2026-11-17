#!/usr/bin/env python3
"""Emit a flat key=value config for `real_checkpoint_generate` from a REAL
Qwen3.8-Flash-Next MLX-4bit checkpoint plus its captured reference.json.

This helper is intentionally tiny: it ONLY reads config.json, the safetensors
index, and reference.json. It never loads tensor data. Its output is a flat
config whose `shard.<tensor-name>` lines tell the C++ loader which shard each
NON-expert dense tensor lives in, so the C++ side never has to guess names or
scan all 22 shards.

The captured reference does not store the raw prompt ids (only their SHA-256),
so we regenerate them with the exact same numpy RNG call that
`capture_real_reference.py` used and verify the digest before proceeding.

Usage:
  .venv/bin/python cpp/engine/tools/gen_real_config.py \
      --checkpoint /Users/vincent/models/Qwen3.8-Flash-Next-MLX-4bit-MTP \
      --reference results/flashnext/20261117-222244-real-reference/reference.json \
      --out /tmp/real_config.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# Dense (non-expert, non-ngram, non-vision) tensor prefixes we need. Experts are
# streamed by StreamingMoE and are deliberately NOT in this map.
def needed_dense_tensors(cfg: dict, n_layers: int) -> list[str]:
    tc = cfg["text_config"]
    hc = ["hc_norm.weight",
          "input_mix_weight_down", "input_mix_weight_up",
          "block_inject_weight"]
    gdn = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
           "conv1d.weight", "A_log", "dt_bias", "norm.weight", "out_proj"]
    fa = ["q_proj", "k_proj", "v_proj", "o_proj",
          "q_norm.weight", "k_norm.weight",
          "indexer.index_qk_proj", "indexer.q_layernorm.weight",
          "indexer.k_layernorm.weight"]
    mlp = ["gate.weight",
           "shared_expert.gate_proj", "shared_expert.up_proj",
           "shared_expert.down_proj", "shared_expert_gate.weight"]
    out: list[str] = []
    types = tc["layer_types"]
    for li in range(n_layers):
        base = f"language_model.model.layers.{li}"
        out += [f"{base}.attn_hyper_connection.{s}" for s in hc]
        if types[li] == "linear_attention":
            out += [f"{base}.linear_attn.{s}" for s in gdn]
        else:
            out += [f"{base}.self_attn.{s}" for s in fa]
        out += [f"{base}.mlp_hyper_connection.{s}" for s in hc]
        out += [f"{base}.mlp.{s}" for s in mlp]
    # PLE at layers whose 1-based index is in ple_layer_ids (layer index 1 here).
    ple_layers = tc.get("ple_layer_ids", [])
    for pl in ple_layers:
        li = int(pl) - 1
        pb = f"language_model.model.layers.{li}.ple"
        out += [f"{pb}.key_proj", f"{pb}.value_proj",
                f"{pb}.norm_key.weight", f"{pb}.norm_query.weight",
                f"{pb}.norm_conv.weight", f"{pb}.conv1d.weight",
                f"{pb}.ple_embedding.layer_multipliers",
                f"{pb}.ple_embedding.ngram_heads_offsets",
                f"{pb}.ple_embedding.ngram_heads_vocab_sizes"]
        # 128 n-gram embedding shards, each a Q4 triplet.
        for s in range(int(tc.get("split_ngram_parts", 128))):
            out += [f"{pb}.ple_embedding.ngram_embedding.shard_{s}.weight",
                    f"{pb}.ple_embedding.ngram_embedding.shard_{s}.scales",
                    f"{pb}.ple_embedding.ngram_embedding.shard_{s}.biases"]
    out += ["language_model.model.embed_tokens",
            "language_model.model.hyper_connection_mixer.hc_norm.weight",
            "language_model.model.hyper_connection_mixer.input_mix_weight_down",
            "language_model.model.hyper_connection_mixer.input_mix_weight_up",
            "language_model.lm_head"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text())
    tc = cfg["text_config"]
    ref = json.loads(Path(args.reference).read_text())
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]

    # Recover the deterministic prompt and verify it against the reference.
    prompt = np.random.default_rng(20261117).integers(
        0, 200000, size=ref["prompt_tokens"]).tolist()
    prompt_sha = hashlib.sha256(
        b"".join(int(t).to_bytes(4, "little", signed=True) for t in prompt)
    ).hexdigest()
    if prompt_sha != ref["prompt_sha256"]:
        raise SystemExit(
            f"prompt SHA mismatch: regenerated {prompt_sha} != reference "
            f"{ref['prompt_sha256']}"
        )

    n_layers = int(tc["num_hidden_layers"])
    lines: list[str] = []

    def kv(key: str, value) -> None:
        lines.append(f"{key}={value}")

    kv("hidden_size", tc["hidden_size"])
    kv("n_layers", n_layers)
    kv("hc_count", tc["hc_count"])
    kv("hc_lowrank", tc["hc_lowrank"])
    kv("vocab", tc["vocab_size"])
    kv("eps", tc["rms_norm_eps"])
    quant = cfg.get("quantization", {})
    kv("group_size", quant.get("group_size", 32))
    kv("bits", quant.get("bits", 4))
    kv("top_k", tc["num_experts_per_tok"])
    kv("n_experts", tc["num_experts"])
    kv("expert_prefix",
       "language_model.model.layers.{L}.mlp.switch_mlp")
    kv("num_slots", max(16, tc["num_experts_per_tok"] + 4))

    # GDN geometry.
    kv("key_dim", tc["linear_num_key_heads"] * tc["linear_key_head_dim"])
    kv("value_dim", tc["linear_num_value_heads"] * tc["linear_value_head_dim"])
    kv("num_k_heads", tc["linear_num_key_heads"])
    kv("num_v_heads", tc["linear_num_value_heads"])
    kv("head_k_dim", tc["linear_key_head_dim"])
    kv("head_v_dim", tc["linear_value_head_dim"])
    kv("conv_kernel", tc["linear_conv_kernel_dim"])

    # Full-attention / QSA geometry.
    kv("n_heads", tc["num_attention_heads"])
    kv("kv_heads", tc["num_key_value_heads"])
    kv("head_dim", tc["head_dim"])
    kv("n_idx_heads", tc["indexer_n_heads"])
    kv("n_idx_kv", tc["indexer_kv_heads"])
    kv("idx_head_dim", tc["indexer_head_dim"])
    kv("compress_ratio", tc["indexer_compress_ratio"])
    kv("token_budget", tc["indexer_budget"])
    rp = tc["rope_parameters"]
    kv("rope_dim", int(tc["head_dim"] * tc["partial_rotary_factor"]))
    kv("rope_base", rp["rope_theta"])
    kv("rope_scaling", 1.0)
    kv("rope_max_pos", tc["max_position_embeddings"])
    kv("rope_style", "interleaved")
    kv("rope_mrope", ",".join(str(x) for x in rp["mrope_section"]))
    kv("layer_types", ",".join(tc["layer_types"]))
    kv("ple_layer_ids", ",".join(str(x) for x in tc.get("ple_layer_ids", [])))

    kv("prompt_tokens", len(prompt))
    kv("prompt_ids", ",".join(str(t) for t in prompt))
    kv("prompt_sha256", prompt_sha)

    # Tensor -> shard map for every dense tensor the C++ side must load.
    for name in needed_dense_tensors(cfg, n_layers):
        # The loader probes raw `name` and, for quantized tensors,
        # `base + .weight/.scales/.biases` (base = name without a trailing
        # `.weight`). Emit every spelling so it can decide at runtime.
        probes = [name, name + ".weight", name + ".scales", name + ".biases"]
        base = name[:-7] if name.endswith(".weight") else name
        probes += [base, base + ".weight", base + ".scales", base + ".biases"]
        for probe in probes:
            shard = wmap.get(probe)
            if shard is not None:
                kv("shard." + probe, shard)

    # Expert tensor -> shard map. StreamingMoE handles single-shard layers;
    # the C++ side uses these entries for its cross-shard fallback (12 of the
    # 48 layers split their 9 expert tensors across two adjacent shards, which
    # the shared StreamingMoE intentionally refuses).
    for li in range(n_layers):
        prefix = f"language_model.model.layers.{li}.mlp.switch_mlp"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for part in ("weight", "scales", "biases"):
                name = f"{prefix}.{proj}.{part}"
                shard = wmap.get(name)
                if shard is not None:
                    kv("shard." + name, shard)

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(
        f"wrote {args.out}: {n_layers} layers, "
        f"{sum(k.startswith('shard.') and not k.endswith('.scales') for k in lines)} "
        f"dense tensors, prompt sha={prompt_sha[:12]}"
    )


if __name__ == "__main__":
    main()
