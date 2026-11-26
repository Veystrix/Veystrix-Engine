"""Recipe generator for the generic C++ runtime (universal engine).

A recipe is DATA: it tells ``cpp/engine/src/generic`` which registered blocks
make up a model, their parameters, the tensor-name templates in the checkpoint,
the quantization of every tensor group and the KV-cache policy. It is derived
from ``config.json`` plus the safetensors headers (metadata only, no weights,
no MLX) and a table keyed by ``model_type`` whose defaults mirror
``mlx_lm.models.<type>`` (the parity oracle).

``veystrix models inspect <ckpt> --emit-recipe <out.json>`` writes one; the C++
bridge (``veystrix/backends/cpp_engine.py``) generates it on demand into a
cache directory and passes it as profile key ``recipe``.

See docs/engineering/UNIVERSAL_CPP_ENGINE.md section 2.1 for the format.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

FORMAT = "veystrix-recipe/3"  # schema: docs/engineering/RECIPE_SCHEMA.md


class RecipeError(ValueError):
    """The checkpoint cannot be expressed by the registered blocks."""


def _read_header(path: Path) -> dict:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def checkpoint_tensors(root: Path) -> dict[str, dict]:
    """name -> {dtype, shape} for every tensor in every shard (headers only)."""
    out: dict[str, dict] = {}
    for shard in sorted(Path(root).glob("*.safetensors")):
        if shard.name.startswith(".") or shard.name == "mtp.safetensors":
            continue
        for name, meta in _read_header(shard).items():
            if name == "__metadata__":
                continue
            out[name] = {"dtype": meta["dtype"], "shape": list(meta["shape"])}
    if not out:
        raise RecipeError(f"no safetensors found in {root}")
    return out


def _text_config(config: dict) -> dict:
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(config.get(key), dict):
            merged = dict(config[key])
            merged.setdefault("model_type", config.get("model_type"))
            return merged
    return dict(config)


def _ignored(name: str) -> bool:
    return (name.startswith("vision_tower") or name.startswith("model.visual") or
            ".visual." in name or name.startswith("mtp.") or ".mtp." in name)


def _find_suffix(tensors: dict, suffix: str) -> str:
    hits = [n for n in tensors if n.endswith(suffix) and not _ignored(n)]
    if not hits:
        raise RecipeError(f"checkpoint has no tensor ending in {suffix!r}")
    return min(hits, key=len)


def _quant_block(config: dict, tensors: dict) -> dict | None:
    quant = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quant, dict):
        if any(n.endswith(".scales") for n in tensors):
            raise RecipeError("tensors are quantized but config.json has no quantization block")
        return None
    default = {k: quant[k] for k in ("group_size", "bits", "mode") if k in quant and not isinstance(quant[k], dict)}
    default.setdefault("mode", "affine")
    overrides: dict[str, dict] = {}
    for key, value in quant.items():
        if not isinstance(value, dict):
            continue
        path = _checkpoint_path(key, tensors)
        if path is None:
            continue
        overrides[path] = {k: value[k] for k in ("group_size", "bits", "mode") if k in value}
    return {"default": default, "overrides": overrides}


def _checkpoint_path(module_path: str, tensors: dict) -> str | None:
    """Map an mlx_lm module path (post-sanitize names) to the checkpoint's names."""
    candidates = [module_path]
    if module_path.startswith("language_model.model."):
        candidates.append("model.language_model." + module_path[len("language_model.model."):])
    if module_path.startswith("language_model.lm_head"):
        candidates.append(module_path[len("language_model."):])
    for cand in candidates:
        if cand + ".weight" in tensors or cand + ".scales" in tensors:
            return cand
    return None


def _quant_for(quant: dict | None, path: str) -> dict | None:
    if quant is None:
        return None
    q = dict(quant["default"])
    q.update(quant["overrides"].get(path, {}))
    return q


def _elem_bits(dtype: str) -> int:
    return {"U8": 8, "I8": 8, "F8_E4M3": 8, "U16": 16, "U32": 32, "I32": 32}.get(dtype.upper(), 32)


def verify_quantization(tensors: dict, quant: dict | None) -> list[str]:
    """Cross-check every quantized tensor's packed shape against the recipe's bits/group."""
    problems = []
    for name, meta in tensors.items():
        if not name.endswith(".scales") or _ignored(name):
            continue
        base = name[: -len(".scales")]
        weight = tensors.get(base + ".weight")
        if weight is None:
            problems.append(f"{base}: scales without weight")
            continue
        q = _quant_for(quant, base)
        if not q or "bits" not in q or "group_size" not in q:
            problems.append(f"{base}: quantized but no bits/group_size in config")
            continue
        w_bits = _elem_bits(str(weight.get("dtype", "U32")))
        in_w = weight["shape"][-1] * w_bits // int(q["bits"])
        in_s = meta["shape"][-1] * int(q["group_size"])
        if in_w != in_s:
            problems.append(f"{base}: bits={q['bits']} group={q['group_size']} do not match shapes "
                            f"{weight['shape']} ({weight.get('dtype')}) / {meta['shape']}")
    return problems


# --------------------------------------------------------------------------- #
# model_type table                                                             #
# --------------------------------------------------------------------------- #

def _qwen3_5(config: dict, tensors: dict) -> dict:
    """mlx_lm.models.qwen3_5 / qwen3_5_moe (TextModelArgs defaults mirrored)."""
    t = _text_config(config)
    hidden = int(t.get("hidden_size", 4096))
    heads = int(t.get("num_attention_heads", 32))
    head_dim = int(t.get("head_dim") or hidden // heads)
    rope_params = t.get("rope_parameters")
    if rope_params is None and "rope_parameters" not in t:
        rope_params = {"type": "default", "mrope_section": [11, 11, 10], "rope_theta": 100000,
                       "partial_rotary_factor": 0.25}
    rope_params = dict(rope_params or {})
    if "type" not in rope_params and "rope_type" in rope_params:
        rope_params["type"] = rope_params.pop("rope_type")
    partial = float(rope_params.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
    theta = float(rope_params.get("rope_theta", t.get("rope_theta", 100000.0)))
    rope_type = rope_params.get("type") or "default"
    if rope_type not in ("default", "mrope", "linear"):
        raise RecipeError(f"rope type {rope_type!r} is not ported to the generic runtime")
    rope_scale = 1.0 / float(rope_params["factor"]) if rope_type == "linear" else 1.0
    eps = float(t.get("rms_norm_eps", 1e-6))
    layers = int(t.get("num_hidden_layers", 32))
    interval = int(t.get("full_attention_interval", 4))
    experts = int(t.get("num_experts", 0))

    embed = _find_suffix(tensors, "embed_tokens.weight")[: -len(".weight")]
    prefix = embed[: -len("embed_tokens")]  # e.g. "language_model.model."
    tied = bool(t.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    head = None if tied else _find_suffix(tensors, "lm_head.weight")[: -len(".weight")]
    if any(".experts.gate_up_proj" in n or ".experts.0." in n for n in tensors if not _ignored(n)):
        raise RecipeError("experts are not stacked as switch_mlp.{gate,up,down}_proj; convert the "
                          "checkpoint with mlx_lm (it stacks them) before running the generic engine")
    # mlx_lm qwen3_5 sanitize: shift 1-D norm weights by +1 when the checkpoint
    # still carries MTP weights or unsanitized conv1d weights (HF layout).
    has_mtp = any("mtp." in n for n in tensors)
    raw_conv = any("conv1d.weight" in n and meta["shape"][-1] != 1 for n, meta in tensors.items())
    norm_shift = bool(has_mtp or raw_conv)

    L = prefix + "layers.{L}."
    blocks = {
        "gdn": {
            "type": "linear.gdn", "tensors": L + "linear_attn",
            "num_k_heads": int(t.get("linear_num_key_heads", 16)),
            "num_v_heads": int(t.get("linear_num_value_heads", 64)),
            "head_k_dim": int(t.get("linear_key_head_dim", 192)),
            "head_v_dim": int(t.get("linear_value_head_dim", 128)),
            "conv_kernel": int(t.get("linear_conv_kernel_dim", 4)), "eps": eps,
        },
        "attn": {
            "type": "attn.gated_gqa", "tensors": L + "self_attn",
            "num_heads": heads, "num_kv_heads": int(t.get("num_key_value_heads", 8)),
            "head_dim": head_dim, "qk_norm": True, "output_gate": True, "eps": eps,
            "norm_shift": norm_shift,
            "rope": {"type": rope_type, "dims": int(head_dim * partial), "base": theta,
                     "scale": rope_scale, "traditional": False},
        },
    }
    if experts > 0:
        blocks["moe"] = {
            "type": "moe.softmax_topk", "tensors": L + "mlp", "router": "gate",
            "num_experts": experts, "top_k": int(t.get("num_experts_per_tok", 0)),
            "norm_topk_prob": bool(t.get("norm_topk_prob", True)),
            "experts": ["switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj"],
            "activation": "swiglu", "streamed": True,
        }
        if int(t.get("shared_expert_intermediate_size", 0)) > 0:
            blocks["moe"]["shared_expert"] = "shared_expert"
            blocks["moe"]["shared_expert_gate"] = "shared_expert_gate"
        mlp = "moe"
    else:
        blocks["mlp"] = {"type": "mlp.swiglu", "tensors": L + "mlp"}
        mlp = "mlp"
    pattern = [{"mixer": "gdn" if (i + 1) % interval != 0 else "attn", "mlp": mlp} for i in range(interval)]
    if layers % interval:
        raise RecipeError(f"num_hidden_layers {layers} is not a multiple of full_attention_interval {interval}")
    return {
        "hidden_size": hidden,
        "vocab_size": int(t.get("vocab_size", 151936)),
        "rms_norm_eps": eps,
        "norm_shift": norm_shift,
        "max_position_embeddings": int(t.get("max_position_embeddings", 131072)),
        "tie_word_embeddings": tied,
        "embed": embed,
        "final_norm": prefix + "norm",
        "head": head,
        "layer_norms": {"input": L + "input_layernorm", "post": L + "post_attention_layernorm"},
        "blocks": blocks,
        "layers": {"count": layers, "pattern": pattern, "overrides": {}},
        "mtp": {"present": has_mtp, "used": False, "note": "MTP measured no gain on this hardware; kept off"},
    }


def _norm_gain(config: dict) -> str:
    """Qwen4ExpRMSNorm gain convention, as models/flashnext/loader.py:_norm_one_centered decides it:
    the shipped Flash-Next conversion stores ONE-centered gains (gain = w); a converter can declare the
    zero-centered convention (gain = 1 + w) in config.json."""
    for source in (config, config.get("text_config", {})):
        value = source.get("norm_convention") or source.get("rms_norm_convention")
        if isinstance(value, str):
            value = value.lower().replace("_", "-")
            if value in {"zero", "zero-centered"}:
                return "zero"
            if value in {"one", "one-centered"}:
                return "one"
    return "one"


def _qwen4_exp(config: dict, tensors: dict) -> dict:
    """mlx_vlm.models.qwen4_exp (Qwen3.8 Flash-Next / Qwen 4 preview): hyper-connection residual streams,
    Qwen4 gated delta net, gated attention with the QSA sparse indexer, softmax top-k MoE with a shared
    expert, and the n-gram PLE layer. Numerics follow the production Flash-Next runtime (one-centered
    RMSNorm gains, threshold-1.0 routing renormalised by the top-k mass) that the custom C++ runtime matches."""
    t = _text_config(config)
    hidden = int(t["hidden_size"])
    heads = int(t["num_attention_heads"])
    head_dim = int(t.get("head_dim") or hidden // heads)
    layers = int(t["num_hidden_layers"])
    eps = float(t.get("rms_norm_eps", 1e-6))
    hc = int(t.get("hc_count", 4))
    lowrank = int(t.get("hc_lowrank", 320))
    rope_params = dict(t.get("rope_parameters") or {})
    partial = float(rope_params.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
    theta = float(rope_params.get("rope_theta", t.get("rope_theta", 10000000.0)))
    section = list(rope_params.get("mrope_section") or [11, 11, 10])
    if not rope_params.get("mrope_interleaved", True):
        raise RecipeError("qwen4_exp: only interleaved MRoPE is ported")
    layer_types = t.get("layer_types")
    if not layer_types:
        interval = int(t.get("full_attention_interval", 4))
        layer_types = ["linear_attention" if (i + 1) % interval else "full_attention" for i in range(layers)]
    if len(layer_types) != layers:
        raise RecipeError(f"qwen4_exp: {len(layer_types)} layer_types for {layers} layers")
    gain = _norm_gain(config)
    eos = t.get("eos_token_id", config.get("eos_token_id"))
    eos = int(eos[0] if isinstance(eos, list) else eos)

    embed = _find_suffix(tensors, "embed_tokens.weight")[: -len(".weight")]
    prefix = embed[: -len("embed_tokens")]
    tied = bool(t.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    head = None if tied else _find_suffix(tensors, "lm_head.weight")[: -len(".weight")]
    L = prefix + "layers.{L}."
    experts = int(t["num_experts"])
    blocks = {
        "gdn": {
            "type": "linear.gdn_qwen4", "tensors": L + "linear_attn",
            "num_k_heads": int(t["linear_num_key_heads"]), "num_v_heads": int(t["linear_num_value_heads"]),
            "head_k_dim": int(t["linear_key_head_dim"]), "head_v_dim": int(t["linear_value_head_dim"]),
            "conv_kernel": int(t.get("linear_conv_kernel_dim", 4)), "eps": eps,
            "output_gate": str(t.get("output_gate_type") or t.get("hidden_act") or "sigmoid"),
        },
        "attn": {
            "type": "attn.qsa_gated", "tensors": L + "self_attn",
            "num_heads": heads, "num_kv_heads": int(t["num_key_value_heads"]), "head_dim": head_dim,
            "eps": eps, "gain": gain,
            "rope": {"type": "mrope_interleaved", "dims": int(head_dim * partial), "base": theta,
                     "mrope_section": section,
                     "max_position_embeddings": int(t.get("max_position_embeddings", 262144))},
            "indexer": {"n_heads": int(t.get("indexer_n_heads", 4)), "kv_heads": int(t.get("indexer_kv_heads", 1)),
                        "head_dim": int(t.get("indexer_head_dim", 128)), "budget": int(t.get("indexer_budget", 2048)),
                        "compress_ratio": int(t.get("indexer_compress_ratio", 4)),
                        # QSA sparse attention in the C++ runtime since 21 Nov 2026 (pooled indexer state in
                        # snapshots): part of the recipe digest, so dense-era snapshots are never restored.
                        "state": "qsa-sparse-pooled-v1"},
        },
        "moe": {
            "type": "moe.softmax_topk", "tensors": L + "mlp", "router": "gate",
            "num_experts": experts, "top_k": int(t["num_experts_per_tok"]), "norm_topk_prob": True,
            "experts": ["switch_mlp.gate_proj", "switch_mlp.up_proj", "switch_mlp.down_proj"],
            "activation": "swiglu", "streamed": True,
            "shared_expert": "shared_expert", "shared_expert_gate": "shared_expert_gate",
        },
    }
    pre = {}
    ple_ids = [int(i) for i in (t.get("ple_layer_ids") or [])]
    for n, one_based in enumerate(ple_ids):
        layer = one_based - 1  # Qwen4ExpDecoderLayer: ple_layer_ids.index(layer_idx + 1)
        if not 0 <= layer < layers:
            raise RecipeError(f"qwen4_exp: ple layer id {one_based} out of range")
        blocks[f"ple{n}"] = {
            "type": "pre.ple_ngram", "tensors": L + "ple", "ple_index": n,
            "ngram_size": int(t.get("ngram_size", 3)), "heads_per_ngram": int(t.get("heads_per_ngram", 8)),
            "embed_dim": int(t.get("ple_embed_dim") or hidden), "conv_kernel": int(t.get("ple_conv_kernel_size", 4)),
            "split_parts": int(t.get("split_ngram_parts", 128)), "eos_token_id": eos, "eps": eps, "gain": gain,
        }
        pre[str(layer)] = f"ple{n}"
    pattern = [{"mixer": "gdn" if kind == "linear_attention" else "attn", "mlp": "moe"} for kind in layer_types]
    return {
        "hidden_size": hidden,
        "vocab_size": int(t["vocab_size"]),
        "rms_norm_eps": eps,
        "norm_shift": False,
        "max_position_embeddings": int(t.get("max_position_embeddings", 262144)),
        "tie_word_embeddings": tied,
        "embed": embed,
        "final_norm": None,
        "head": head,
        "layer_norms": None,
        "residual": {"type": "hyper", "hc_count": hc, "lowrank": lowrank, "gain": gain,
                     "mixer": L + "attn_hyper_connection", "mlp": L + "mlp_hyper_connection",
                     "final": prefix + "hyper_connection_mixer"},
        "blocks": blocks,
        "layers": {"count": layers, "pattern": pattern, "overrides": {}, "pre": pre},
        "mtp": {"present": any(".mtp." in n or n.startswith("mtp.") for n in tensors), "used": False,
                "note": "MTP measured no gain on this hardware; kept off"},
        # Measured 21 Nov (ENGINE_AB_2026-11-20.md, qs1-qs3, identical text): per-expert launch as rows land,
        # GPU keep-warm and page-cache warm-up of the next layer's top-4 guesses: 3.64 tok/s vs 3.16 with the
        # code defaults (custom runtime 3.86). tensor_rows (qo2, n=8 each, identical text): decode expert rows
        # tensor-major as the custom runtime reads them, 3.88 vs 3.69 tok/s (custom 3.92); prefill unchanged.
        # gate_tail 3 (qo3, qo6): the last 3 experts' compute behind a GPU gate so the next layer is encoded while
        # their rows land; qo6 (n=8): 4.09 vs 4.05 at tail 2, 3.98 at tail 1. prefill_pipe 0 (qo3): read only routed experts in prefill, 43-47
        # instead of 74 GB per prompt, TTFT 21.3-24.5 vs 24.8-25.7 s at the same decode speed.
        "speed": dict(QWEN4_EXP_SPEED),
    }


def _inkling2_loop(config: dict, tensors: dict) -> dict:
    """Thinking Machines Inkling 2 Mini (inkling2_loop): 24 physical layers with a 3-pass
    looped core (layers 6..17 -> 48 visits), Block Attention Residuals (6-layer blocks),
    5:1 sliding-window (1,024-tok) / full relative-bias sink attention, FP32 rank-64 cross-pass
    message gates, and fused w13/w2 sigmoid-routed MoE (128 experts, top-8, NVFP4/affine 4-bit)."""
    t = _text_config(config)
    hidden = int(t["hidden_size"])
    heads = int(t["num_attention_heads"])
    kv_heads = int(t["num_key_value_heads"])
    head_dim = int(t.get("head_dim", 128))
    layers = int(t["num_hidden_layers"])
    eps = float(t.get("rms_norm_eps", 1e-6))
    loop_cfg = dict(t.get("loop_config") or {})
    core_start = int(loop_cfg.get("core_start", 6))
    core_end = int(loop_cfg.get("core_end", 18))
    num_passes = int(loop_cfg.get("num_passes", 3))
    gate_rank = int(loop_cfg.get("gate_rank", 64))
    local_ids = set(int(i) for i in (t.get("local_layer_ids") or []))
    sw_size = int(t.get("sliding_window_size", 1024))

    embed = _find_suffix(tensors, "embed.weight")[: -len(".weight")]
    prefix = embed[: -len("embed")]  # e.g. "model."
    tied = bool(t.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))
    head = None if tied else _find_suffix(tensors, "lm_head.weight")[: -len(".weight")]
    L = prefix + "layers.{L}."

    attn_common = {
        "tensors": L,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "d_rel": int(t.get("d_rel", 16)),
        "rel_extent": int(t.get("rel_extent", 1024)),
        "sconv_kernel_size": int(t.get("sconv_kernel_size", 4)),
        "log_scaling_n_floor": int(t.get("log_scaling_n_floor", 8192)),
        "log_scaling_alpha": float(t.get("log_scaling_alpha", 0.1)),
        "use_value_norm_no_gamma": bool(t.get("use_value_norm_no_gamma", True)),
        "qk_l2_clip_threshold": float(t.get("qk_l2_clip_threshold", 22.627416997969522)),
        "rel_l2_clip_threshold": float(t.get("rel_l2_clip_threshold", 8.0)),
        "eps": eps,
    }
    blocks = {
        "attn_local": {"type": "attn.inkling2_local", "sliding_window": sw_size, **attn_common},
        "attn_full": {"type": "attn.inkling2_full", "sliding_window": 0, **attn_common},
        "moe": {
            "type": "moe.inkling2_sigmoid",
            "tensors": L + "mlp",
            "num_experts": int(t.get("n_routed_experts", 128)),
            "top_k": int(t.get("num_experts_per_tok", 8)),
            "intermediate_size": int(t.get("intermediate_size", 1152)),
            "route_scale": float(t.get("route_scale", 8.0)),
            "use_gate_bias": bool(t.get("use_gate_bias", True)),
            "gate_softcap": float(t.get("mlp_gate_softcap", 4.0)),
            "up_softcap": float(t.get("mlp_up_softcap", 25.0)),
            "experts": ["experts.w13", "experts.w2"],
            "fused_gate_up": True,
            "streamed": True,
        },
    }
    pattern = [{"mixer": "attn_local" if i in local_ids else "attn_full", "mlp": "moe"} for i in range(layers)]
    mtp_cfg = config.get("mtp_config")
    return {
        "hidden_size": hidden,
        "vocab_size": int(t["vocab_size"]),
        "unpadded_vocab_size": int(t.get("unpadded_vocab_size", t["vocab_size"])),
        "final_logit_softcapping": float(t.get("final_logit_softcapping", 30.0)),
        "rms_norm_eps": eps,
        "norm_shift": False,
        "max_position_embeddings": int(t.get("model_max_length", 1048576)),
        "tie_word_embeddings": tied,
        "embed": embed,
        "embed_norm": prefix + "embed_norm",
        "final_norm": prefix + "norm",
        "head": head,
        "layer_norms": None,
        "residual": {
            "type": "block_attn_res",
            "block_size": int(t.get("n_layers_per_attn_res_block", 6)),
            "normalize_values": bool(t.get("attn_res_normalize_values", True)),
            "final_query": prefix + "final_input.query",
        },
        "loop": {
            "core_start": core_start,
            "core_end": core_end,
            "passes": num_passes,
            "gate_rank": gate_rank,
            "share_kv": False,
        },
        "blocks": blocks,
        "layers": {"count": layers, "pattern": pattern, "overrides": {}},
        "mtp": {
            "present": bool(mtp_cfg),
            "used": False,
            "draft_type": "dspark",
            "hidden_size": int(mtp_cfg.get("hidden_size", hidden)) if isinstance(mtp_cfg, dict) else hidden,
            "num_hidden_layers": int(mtp_cfg.get("num_hidden_layers", 3)) if isinstance(mtp_cfg, dict) else 3,
            "num_attention_heads": int(mtp_cfg.get("num_attention_heads", heads)) if isinstance(mtp_cfg, dict) else heads,
            "num_key_value_heads": int(mtp_cfg.get("num_key_value_heads", kv_heads)) if isinstance(mtp_cfg, dict) else kv_heads,
            "head_dim": int(mtp_cfg.get("head_dim", head_dim)) if isinstance(mtp_cfg, dict) else head_dim,
            "intermediate_size": int(mtp_cfg.get("intermediate_size", 9216)) if isinstance(mtp_cfg, dict) else 9216,
            "rms_norm_eps": float(mtp_cfg.get("rms_norm_eps", eps)) if isinstance(mtp_cfg, dict) else eps,
            "rope_theta": float(mtp_cfg.get("rope_theta", 1000000.0)) if isinstance(mtp_cfg, dict) else 1000000.0,
            "local_layer_ids": [int(i) for i in mtp_cfg.get("local_layer_ids", [0, 2])] if isinstance(mtp_cfg, dict) else [0, 2],
            "sliding_window_size": int(mtp_cfg.get("sliding_window_size", 1024)) if isinstance(mtp_cfg, dict) else 1024,
            "block_size": int(mtp_cfg.get("block_size", 8)) if isinstance(mtp_cfg, dict) else 8,
            "markov_rank": int(mtp_cfg.get("markov_rank", 256)) if isinstance(mtp_cfg, dict) else 256,
            "markov_head_type": str(mtp_cfg.get("markov_head_type", "rnn_hidden")) if isinstance(mtp_cfg, dict) else "rnn_hidden",
            "target_layer_ids": [int(i) for i in mtp_cfg.get("target_layer_ids", [5, 23, 41, 48])] if isinstance(mtp_cfg, dict) else [],
            "note": "Optional DSpark 3-layer draft (mtp.safetensors); enable via VEYSTRIX_INKLING_DSPARK=1 or profile",
        },
        "speed": dict(INKLING2_LOOP_SPEED),
    }


MODEL_TYPES = {
    "qwen3_5_moe": _qwen3_5,
    "qwen3_5": _qwen3_5,
    "qwen4_exp": _qwen4_exp,
    "inkling2_loop": _inkling2_loop,
    "inkling2_loop_text": _inkling2_loop,
}

# KV policy written into recipes (the bridge/memory plan may override per run).
# qwen4_exp (Flash-Next) speed knobs, measured in ENGINE_AB_2026-11-20.md (see the comment in _qwen4_exp).
# Changing them needs a new real parity record (veystrix/tests/test_parity_defaults.py).
QWEN4_EXP_SPEED = {"pipe": 1, "keepwarm": 1, "spec_top": 4, "tensor_rows": 1, "gate_tail": 3, "prefill_pipe": 0}
INKLING2_LOOP_SPEED = {"pipe": 0, "gate": 1, "keepwarm": 1, "spec_fetch": 4, "tensor_rows": 1}

DEFAULT_KV = {"mode": "bf16", "policy": "auto",
              "note": "bf16 is exact; the memory plan ages old pages into tiered4+ "
                      "(INT8 recent window + INT8 protected prefix + INT4 middle) only when "
                      "the bf16 KV would cost the page cache the experts need"}


def emit_recipe(checkpoint: str | Path) -> dict:
    root = Path(checkpoint).expanduser()
    try:
        config = json.loads((root / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise RecipeError(f"cannot read {root / 'config.json'}: {exc}") from exc
    model_type = str(config.get("model_type") or _text_config(config).get("model_type") or "")
    builder = MODEL_TYPES.get(model_type)
    if builder is None:
        raise RecipeError(f"model_type {model_type!r} has no recipe builder "
                          f"(known: {', '.join(sorted(MODEL_TYPES))})")
    tensors = checkpoint_tensors(root)
    body = builder(config, tensors)
    quant = _quant_block(config, tensors)
    problems = verify_quantization(tensors, quant)
    if problems:
        raise RecipeError("quantization does not match the checkpoint: " + "; ".join(problems[:5]))
    recipe = {"format": FORMAT, "model_type": model_type, "quant": quant, **body,
              "cache": {"kv": dict(DEFAULT_KV)}}
    recipe["source"] = {"checkpoint": str(root), "config_sha256": hashlib.sha256(
        (root / "config.json").read_bytes()).hexdigest()}
    return recipe


def recipe_cache_path(checkpoint: str | Path, cache_root: str | Path | None = None) -> Path:
    """Deterministic cache location for a checkpoint's generated recipe."""
    root = Path(checkpoint).expanduser().resolve()
    cfg = (root / "config.json").read_bytes()
    # The builder source is part of the key: a builder fix must not be shadowed by a recipe cached before it.
    builder = Path(__file__).read_bytes()
    key = hashlib.sha256(str(root).encode() + b"\0" + cfg + b"\0" + builder).hexdigest()[:16]
    base = Path(cache_root or Path("~/.cache/veystrix/recipes")).expanduser()
    return base / f"{root.name}-{key}.json"


def ensure_recipe(checkpoint: str | Path, cache_root: str | Path | None = None) -> Path:
    """Generate (once) and return the cached recipe path for a checkpoint."""
    path = recipe_cache_path(checkpoint, cache_root)
    if not path.is_file():
        recipe = emit_recipe(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(recipe, indent=1, sort_keys=False))
        tmp.replace(path)
    return path
