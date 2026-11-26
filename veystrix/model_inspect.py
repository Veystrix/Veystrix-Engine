"""Inspect a checkpoint before porting it: architecture, engine coverage, memory fit.

``veystrix models inspect <dir>`` answers the three questions that decide whether a new
model is cheap to add: (1) which architecture primitives does it need, (2) which of those
does each engine already have, (3) how would it run on a 16..128 GB Mac. Metadata only:
config.json and safetensors headers are read, no tensor payload and no MLX.

The coverage tables below are the single place to extend when a primitive is ported;
``docs/engineering/ADDING_A_MODEL.md`` describes the workflow.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from veystrix.memory_budget_runtime import plan
from veystrix.memory_tiers import GIB
from veystrix.model_memory import ModelMemory, probe_checkpoint

# Architectures with a validated, bit-exact C++ port and a Python runtime.
KNOWN_FAMILIES = {
    "qwen4_exp": {"family": "flashnext", "python": True, "cpp": True, "runtime": "generic"},
    "qwen4_exp_text": {"family": "flashnext", "python": True, "cpp": True, "runtime": "generic"},
    "nemotron_h": {"family": "nemotron35-lightning", "python": True, "cpp": True},
    # Generic recipe-driven C++ runtime (cpp/engine/src/generic); no Veystrix Python engine.
    "qwen3_5_moe": {"family": "ornith", "python": False, "cpp": True, "runtime": "generic"},
    "qwen3_5_moe_text": {"family": "ornith", "python": False, "cpp": True, "runtime": "generic"},
    "inkling2_loop": {"family": "inkling2-mini", "python": False, "cpp": True, "runtime": "generic"},
    "inkling2_loop_text": {"family": "inkling2-mini", "python": False, "cpp": True, "runtime": "generic"},
}

# Primitive name -> (python, cpp, note). "cpp" False means the primitive must be ported.
PRIMITIVES = {
    "moe-swiglu-affine-q4": (True, True, "Qwen-style routed experts, streamed from SSD"),
    "moe-relu2-affine-q4": (True, True, "Nemotron-style routed experts"),
    "gated-delta-net": (True, True, "linear attention (GDN)"),
    "qsa-sparse-attention": (True, True, "indexer-selected sparse attention (>2048 tokens)"),
    "mamba2-ssm": (True, True, "Mamba-2 state-space mixer, chunked prefill"),
    "dense-gqa-attention": (True, True, "full attention with GQA and rotary"),
    "kv-bf16": (True, True, "dense bf16 KV"),
    "kv-int8": (True, True, "INT8 block KV"),
    "kv-int4-tiered": (True, True, "INT4/tiered KV"),
    "mtp-draft": (True, True, "native multi-token-prediction head (measured no gain; off by default)"),
    "affine-q4-g32": (True, True, "affine 4-bit, group 32"),
    "affine-q4-g64": (True, True, "affine 4-bit, group 64"),
    "affine-q8": (True, True, "affine 8-bit weights (router / shared-expert gate): generic runtime QLinear, port-checked"),
    "gated-gqa-attention": (True, True, "gated full attention (q_proj -> queries+gate, QK norm, partial rotary): generic attn.gated_gqa"),
    "kv-tiered4plus": (True, True, "INT8 protected prefix + INT4 history + INT8 recent window: generic KV store"),
    "mxfp4-nvfp4": (True, False, "microscaling FP4 formats: not ported"),
    "sliding-window-attention": (True, False, "local attention window: not ported in C++"),
    "mla-attention": (True, False, "multi-head latent attention (DeepSeek style): not ported"),
    "sigmoid-router-bias": (True, False, "DeepSeek-style sigmoid router with correction bias: not ported"),
    "dense-swiglu-mlp": (True, True, "non-MoE gated MLP layer: generic mlp.swiglu"),
}


@dataclass
class Inspection:
    path: str
    model_type: str
    known_family: str | None
    runtime: str = ""                      # "custom" | "generic" | "" (no validated adapter)
    known_python: bool = True
    needs: list[str] = field(default_factory=list)
    missing_cpp: list[str] = field(default_factory=list)
    missing_python: list[str] = field(default_factory=list)
    quantization: dict = field(default_factory=dict)
    kv_bytes_per_token: dict = field(default_factory=dict)
    memory: dict | None = None
    fit: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Generic recipe runtime: does a recipe builder exist, which block types does the recipe use, and which
    # of them does this build's library lack ({"recipe": "ok"|error, "blocks": [...], "missing_blocks": [...]}).
    generic: dict = field(default_factory=dict)

    @property
    def cpp_ready(self) -> bool:
        return self.known_family is not None or not self.missing_cpp

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["cpp_ready"] = self.cpp_ready
        return data


def _text_config(config: dict) -> dict:
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(config.get(key), dict):
            return config[key]
    return config


def _needs(config: dict, text: dict) -> tuple[list[str], list[str]]:
    needs: list[str] = []
    notes: list[str] = []
    experts = text.get("num_experts") or text.get("n_routed_experts") or text.get("num_local_experts")
    act = str(text.get("mlp_hidden_act") or text.get("hidden_act") or "").lower()
    if experts:
        needs.append("moe-relu2-affine-q4" if "relu2" in act or "relu_squared" in act
                     else "moe-swiglu-affine-q4")
        if text.get("topk_method") or text.get("scoring_func") == "sigmoid" or text.get("e_score_correction_bias"):
            needs.append("sigmoid-router-bias")
    else:
        needs.append("dense-swiglu-mlp")
    kinds = {str(k).lower() for k in (text.get("layer_types") or [])}
    kinds |= {str(k).lower() for k in (text.get("layers_block_type") or [])}
    if any("linear" in k for k in kinds):
        needs.append("gated-delta-net")
    if any("mamba" in k for k in kinds) or text.get("ssm_state_size"):
        needs.append("mamba2-ssm")
    if any(k in ("full_attention", "attention") for k in kinds) or text.get("num_attention_heads"):
        needs.append("dense-gqa-attention")
    if text.get("indexer_budget") or text.get("indexer_n_heads"):
        needs.append("qsa-sparse-attention")
    if text.get("kv_lora_rank") or text.get("q_lora_rank"):
        needs.append("mla-attention")
    if text.get("sliding_window") or any("sliding" in k for k in kinds):
        needs.append("sliding-window-attention")
    if text.get("mtp") or text.get("num_nextn_predict_layers") or text.get("mtp_num_hidden_layers"):
        needs.append("mtp-draft")
        notes.append("MTP head present; speculative decoding measured no gain on this hardware (keep off).")
    quant = config.get("quantization") or config.get("quantization_config") or {}
    bits, group = quant.get("bits"), quant.get("group_size")
    mode = str(quant.get("mode") or ("affine" if bits else "")).lower()
    if mode in ("mxfp4", "nvfp4", "mxfp8"):
        needs.append("mxfp4-nvfp4")
    elif bits == 4 and group == 32:
        needs.append("affine-q4-g32")
    elif bits == 4 and group == 64:
        needs.append("affine-q4-g64")
    elif bits == 8:
        needs.append("affine-q8")
    if any(isinstance(v, dict) and v.get("bits") == 8 for v in quant.values()) and "affine-q8" not in needs:
        needs.append("affine-q8")       # per-module overrides (qwen3_5_moe: router + shared-expert gate)
    if str(config.get("model_type") or "").startswith("qwen3_5") or str(text.get("model_type") or "").startswith("qwen3_5"):
        if text.get("attn_output_gate", True):
            needs.append("gated-gqa-attention")
        if "gated-delta-net" not in needs:
            needs.append("gated-delta-net")
    elif bits:
        notes.append(f"unusual quantization bits={bits} group={group}: needs a projection gate")
    elif not quant:
        notes.append("no quantization block: weights are full precision; memory fit uses file sizes")
    needs += ["kv-bf16"]
    return needs, notes


def _kv_per_token(text: dict) -> dict:
    layers = int(text.get("num_hidden_layers") or 0)
    kinds = [str(k).lower() for k in (text.get("layer_types") or text.get("layers_block_type") or [])]
    interval = text.get("full_attention_interval")
    if kinds:
        attn_layers = sum(1 for k in kinds if k in ("full_attention", "attention"))
    elif isinstance(interval, int) and interval > 0 and str(text.get("model_type", "")).startswith("qwen3_5"):
        attn_layers = layers // interval           # qwen3_5: every interval-th layer is full attention
    else:
        attn_layers = layers
    kv_heads = int(text.get("num_key_value_heads") or text.get("num_attention_heads") or 0)
    head_dim = int(text.get("head_dim") or 0)
    if not head_dim and text.get("hidden_size") and text.get("num_attention_heads"):
        head_dim = int(text["hidden_size"]) // int(text["num_attention_heads"])
    per_layer = 2 * kv_heads * head_dim                   # K and V elements per token
    return {"attention_layers": attn_layers,
            "bf16": per_layer * attn_layers * 2, "int8": per_layer * attn_layers,
            "int4": per_layer * attn_layers // 2}


def fit_table(memory: ModelMemory | None, kv_bytes: int, sizes=(16, 24, 32, 48, 64, 96, 128)) -> list[dict]:
    """How the dynamic plan sizes the pool on each Mac size, assuming ~85% of RAM is free."""
    rows = []
    for gib in sizes:
        total = gib * GIB
        extra = ({"dense_bytes": memory.resident_dense_bytes, "routed_expert_bytes": memory.routed_expert_bytes,
                  "record_bytes": memory.record_bytes} if memory else {})
        budget = plan(total=total, available=int(total * 0.85), swap_used=0, pressure=1,
                      kv_bytes_per_token=kv_bytes, context_tokens=32768, **extra)
        rows.append({"ram_gib": gib, "mode": "resident" if budget.resident else "streamed",
                     "pool_gib": round(budget.pool_bytes / GIB, 1), "max_sessions": budget.max_sessions})
    return rows


def inspect(path: str | Path) -> Inspection:
    root = Path(path).expanduser()
    try:
        config = json.loads((root / "config.json").read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"veystrix: cannot read {root / 'config.json'}: {exc}") from exc
    text = _text_config(config)
    model_type = str(config.get("model_type") or text.get("model_type") or "unknown")
    known = KNOWN_FAMILIES.get(model_type) or KNOWN_FAMILIES.get(str(text.get("model_type")))
    needs, notes = _needs(config, text)
    missing_cpp = [n for n in needs if not PRIMITIVES.get(n, (True, False, ""))[1]]
    missing_py = [n for n in needs if not PRIMITIVES.get(n, (False, False, ""))[0]]
    memory = probe_checkpoint(root)
    kv = _kv_per_token(text)
    result = Inspection(
        path=str(root), model_type=model_type, known_family=known["family"] if known else None,
        runtime=(known.get("runtime", "custom") if known else ""),
        known_python=bool(known.get("python", True)) if known else False,
        needs=needs, missing_cpp=[] if known else missing_cpp, missing_python=[] if known else missing_py,
        quantization=config.get("quantization") or config.get("quantization_config") or {},
        kv_bytes_per_token=kv, notes=notes)
    if memory:
        result.memory = {"total_gib": round(memory.total_bytes / GIB, 2),
                         "routed_expert_gib": round(memory.routed_expert_bytes / GIB, 2),
                         "resident_dense_gib": round(memory.resident_dense_bytes / GIB, 2),
                         "streamed_table_gib": round(memory.streamed_table_bytes / GIB, 2),
                         "moe_layers": memory.moe_layers, "experts_per_layer": memory.experts_per_layer,
                         "expert_record_bytes": memory.record_bytes}
        result.fit = fit_table(memory, kv["int8"])
        if memory.total_bytes > 400 * GIB:
            result.notes.append("checkpoint is larger than any single Mac's RAM: it can only run by "
                                "streaming experts from SSD; needs a fast internal disk.")
    else:
        result.notes.append("no safetensors found: memory fit unavailable")
    result.generic = generic_report(root)
    if known is None and not missing_cpp:
        result.notes.append("every needed primitive exists, but the model has no validated adapter yet: "
                            "add a ModelSpec and run the parity gates (see ADDING_A_MODEL.md).")
    return result


RESIDUAL_TYPES = {"hyper", "block_attn_res"}  # generic_model.cpp; absent residual = plain pre-norm


def _library_block_types(lib: str | None = None) -> list[str] | None:
    """Block types the generic runtime in libveystrix_c implements (None if no library / older library)."""
    import ctypes
    import os
    path = lib or os.environ.get("VEYSTRIX_CPP_LIB") or str(
        Path(__file__).resolve().parents[1] / "cpp" / "engine" / "build" / "libveystrix_c.dylib")
    if not Path(path).is_file():
        return None
    try:
        dll = ctypes.CDLL(path)
        fn = dll.vx_generic_block_types
    except (OSError, AttributeError):
        return None
    fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_int], ctypes.c_int
    buf = ctypes.create_string_buffer(8192)
    n = fn(buf, len(buf))
    if n < 0:
        return None
    return [t for t in buf.value.decode().split(",") if t]


def generic_report(root: Path, lib: str | None = None) -> dict:
    """What the generic runtime needs for this checkpoint: recipe builder, block types, missing blocks."""
    from veystrix.recipe import RecipeError, emit_recipe
    try:
        recipe = emit_recipe(root)
    except RecipeError as exc:
        return {"recipe": f"no recipe: {exc}",
                "next": "add a builder to veystrix/recipe.py (docs/engineering/ADDING_A_MODEL.md)"}
    except Exception as exc:  # unreadable header etc.
        return {"recipe": f"error: {exc}"}
    blocks = sorted({b["type"] for b in recipe.get("blocks", {}).values()})
    report = {"recipe": "ok", "format": recipe.get("format"), "blocks": blocks}
    residual = (recipe.get("residual") or {}).get("type")
    if residual and residual not in RESIDUAL_TYPES:
        report["missing_residual"] = residual
    have = _library_block_types(lib)
    if have is None:
        report["missing_blocks"] = None
        report["library"] = "no libveystrix_c with vx_generic_block_types (build cpp/engine first)"
    else:
        report["missing_blocks"] = [b for b in blocks if b not in have]
    return report


def render(result: Inspection) -> str:
    lines = [f"model: {result.path}", f"  type: {result.model_type}   known family: {result.known_family or 'no'}"]
    quant = result.quantization
    lines.append(f"  quantization: {quant if quant else 'none declared'}")
    lines.append("  needs: " + ", ".join(result.needs))
    if result.known_family and result.runtime == "generic":
        lines.append("  engines: cpp ok (generic recipe runtime, port-checked); python: "
                     + ("ok" if result.known_python else "no Veystrix Python engine"))
        lines.append("  recipe: veystrix models inspect --emit-recipe PATH")
    elif result.known_family:
        lines.append("  engines: python ok, cpp ok (validated, bit-exact gates)")
    else:
        lines.append("  python: " + ("ready" if not result.missing_python else "missing " + ", ".join(result.missing_python)))
        lines.append("  cpp:    " + ("primitives available" if not result.missing_cpp
                                     else "needs ports: " + ", ".join(result.missing_cpp)))
    g = result.generic
    if g:
        if g.get("recipe") != "ok":
            lines.append(f"  generic runtime: {g.get('recipe')}" + (f"; next: {g['next']}" if g.get("next") else ""))
        else:
            missing = g.get("missing_blocks")
            state = ("all blocks in this build" if missing == [] else
                     f"MISSING blocks: {', '.join(missing)} (add one file per block family, see ADDING_A_MODEL.md)"
                     if missing else g.get("library", "library not checked"))
            lines.append(f"  generic runtime: recipe {g.get('format')} ok; blocks {', '.join(g['blocks'])}; {state}")
            if g.get("missing_residual"):
                lines.append(f"  generic runtime: residual type '{g['missing_residual']}' is not implemented")
    kv = result.kv_bytes_per_token
    lines.append(f"  KV per token ({kv['attention_layers']} attention layers): bf16 {kv['bf16']:,} B, "
                 f"int8 {kv['int8']:,} B, int4 {kv['int4']:,} B")
    if result.memory:
        m = result.memory
        lines.append(f"  weights {m['total_gib']} GiB = experts {m['routed_expert_gib']} + resident dense "
                     f"{m['resident_dense_gib']} + streamed tables {m['streamed_table_gib']}")
        lines.append("  fit by installed RAM (85% free assumed):")
        for row in result.fit:
            lines.append(f"    {row['ram_gib']:>4} GB  {row['mode']:<8} pool {row['pool_gib']:>6} GiB  "
                         f"sessions<={row['max_sessions']}")
    for note in result.notes:
        lines.append("  note: " + note)
    return "\n".join(lines)
