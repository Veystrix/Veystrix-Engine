"""Per-session recurrent-state + KV-size calculator (CPU-only, no MLX, no checkpoint).

This is a *calculator*, not a measurement: every number it prints is derived
arithmetic on config fields that the repo / deployed engine actually read.
It feeds the ``session_state_mb`` input of the multi-session Phase 0 simulator
(Task 1).  Nothing here loads a real checkpoint or runs a benchmark.

Units: ``MB`` below is SI megabytes (1 MB = 1_000_000 bytes), matching the
repo's ``GB = 1e9`` convention in ``models/flashnext/prefill_budget.py:17``.
``MIB`` (2**20 bytes) is exported for readers who prefer binary units.

--------------------------------------------------------------------------
DERIVED FORMULAS AND THEIR SOURCES
--------------------------------------------------------------------------
Qwen (Flash-Next / qwen4_exp), per session
  * linear/GDN layers: count of ``layer_types == "linear_attention"``
    (``models/flashnext/preflight.py:60`` -> 36 of 48; the engine maps the
    checkpoint's ``"full_attention"`` to ``"qwen_sparse_attention"`` in
    ``mlx_vlm/models/qwen4_exp/config.py:83-96``, so both spellings count).
  * conv state / linear layer = (linear_conv_kernel_dim - 1) * conv_dim * 2 B
    with conv_dim = 2*linear_num_key_heads*linear_key_head_dim
                    + linear_num_value_heads*linear_value_head_dim
    (``mlx_vlm/models/qwen3_5/language.py:1051-1065``; conv state shape
    ``(B, conv_kernel_size-1, conv_dim)`` at ``language.py:1129``;
    bf16 because the conv state takes ``inputs.dtype`` at
    ``models/flashnext/qwen4_verifier.py:127-138``).
  * recurrent state / linear layer =
        linear_num_value_heads * linear_value_head_dim
        * linear_key_head_dim * 4 B   (f32)
    (shape [B, Hv, Dv, Dk] f32 in ``mlx_vlm/models/qwen3_5/gated_delta.py:239,
    352,831-836``; 48*128*128*4 = 3_145_728 B in
    ``models/flashnext/prefill_budget.py:45``).
  * KV / full layer = 2 * ctx_len * num_key_value_heads * head_dim * 2 B
    (K+V, bf16; ``models/flashnext/prefill_budget.py:46`` -> 2 kv heads,
    head_dim 256, 2 B; full layers = 12 at ``preflight.py:60``).

Nemotron (nemotron_h), per session
  * layer counts from ``layers_block_type`` (23 mamba, 23 moe, 6 attention of
    52; ``models/nemotron35/config.py:8-16,43-49``).  MoE layers have no
    per-session recurrent state here (expert pool is a separate allocator).
  * SSM state / mamba layer =
        mamba_num_heads * mamba_head_dim * ssm_state_size * 4 B (f32)
    (shape [B, heads, head_dim, state_width] f32 in
    ``models/nemotron35/mamba_scan.py:34-37`` and
    ``models/nemotron35/memory.py:27-30``; 64*64*128*4 = 2_097_152 B).
    NOTE: the Phase 0 task brief suggested d_state*d_inner*n_groups; the
    engine allocates d_state*d_inner with NO n_groups factor (n_groups only
    widens the conv tail).  We follow the engine and flag the difference.
  * conv state / mamba layer = (conv_kernel - 1)
        * (mamba_num_heads*mamba_head_dim + 2*n_groups*ssm_state_size) * 2 B
    (``models/nemotron35/memory.py:27-30`` for the shape;
    ``mlx_lm/models/nemotron_h.py:144-147`` shows the tail follows
    ``conv_input.dtype``, i.e. bf16 in production -> 2 B ASSUMPTION.
    ``memory.py`` instead prices it at ``kv_element_bytes`` = 1 B under
    INT8 KV; that variant is reported separately by ``int8_conv=True``).
  * KV / attention layer = 2 * ctx_len * num_key_value_heads * head_dim * 1 B
    (INT8, 1 B per value per the task brief; 6 attention layers, 2 kv heads,
    head_dim 128 from ``models/nemotron35/config.py:8-11``).
    ``models/nemotron35/memory.py:32-36`` adds BF16 scale+bias per 64 values
    (effective 128 + 4*(128//64) = 136 B per head per token, rounded up to
    4096-token blocks) when ``int8_kv=True``; opt into that with
    ``int8_scale_bias=True``.  It gives 3_422_552_064 B at 1,048,576 tokens,
    matching ``models/nemotron35/tests/unit/test_int8_attention.py:118``.

--------------------------------------------------------------------------
OWNER INPUTS NEEDED (not guessable from this repo; see ``owner_inputs()``)
--------------------------------------------------------------------------
The values in ``PRODUCTION_CONFIGS`` below are ESTIMATES assembled from repo
constants (they are not read from the checkpoint's config.json, which we must
not open).  The checkpoint owner should confirm them, especially:
  1. Qwen: linear_num_key_heads=16 and linear_key_head_dim=128 individually
     (their product 2*16*128=4096 is fixed by the weight corpus, but the split
     comes from the checkpoint; see bench_kernel_lab.py:100-104).
  2. Qwen: linear_num_value_heads=48, linear_value_head_dim=128,
     linear_conv_kernel_dim=4 (repo-derived, confirm from checkpoint).
  3. Qwen: full-attention head_dim=256 and num_key_value_heads=2 (from
     prefill_budget.py:46); num_attention_heads is not stored in this repo.
  4. Qwen: layer_types of the deployed checkpoint (36 linear / 12 full) and
     the production KV dtype (bf16 assumed here, per the task brief).
  5. Nemotron: deployed config.json equals models/nemotron35/config.py
     EXPECTED (mamba_num_heads 64, mamba_head_dim 64, ssm_state_size 128,
     n_groups 8, conv_kernel 4, num_key_value_heads 2, head_dim 128,
     23 mamba / 6 attention layers).
  6. Nemotron: Mamba conv dtype (bf16 vs f32) and whether the simulator
     should count INT8 scale/bias and 4096-token block rounding.
  7. Whether the simulator's "session state" includes Qwen's QSA indexer
     keys (128*2+12 = 268 B/token/full layer, prefill_budget.py:47).  The
     task formula is K/V only, so this is opt-in via ``include_indexer=True``.
"""
from __future__ import annotations

import argparse
from typing import Any

MB = 1_000_000           # SI megabyte (1e6 bytes) -- see module docstring
MIB = 1 << 20            # binary megabyte, for convenience only

_QWEN_ALIASES = {"qwen", "qwen4_exp", "qwen4", "flashnext", "flash-next", "qwen3.8"}
_NEMOTRON_ALIASES = {"nemotron", "nemotron35", "nemotron_h", "nemotron3.5", "nemotron-3.5"}

_FULL_ATTN = {"full_attention", "qwen_sparse_attention"}
_LINEAR_ATTN = {"linear_attention"}
_NEMO_ATTN = {"attention", "full_attention"}
_NEMO_MAMBA = {"mamba"}

QWEN_FIELDS = (
    "linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
    "linear_value_head_dim", "linear_conv_kernel_dim", "num_key_value_heads",
    "head_dim",
)
NEMOTRON_FIELDS = (
    "layers_block_type", "mamba_num_heads", "mamba_head_dim", "ssm_state_size",
    "n_groups", "conv_kernel", "num_key_value_heads", "head_dim",
)

# ESTIMATE: assembled from repo constants, not read from a checkpoint.
PRODUCTION_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen": {
        # models/flashnext/preflight.py:52-60, prefill_budget.py:21-29,
        # tests/bench/bench_kernel_lab.py:100-104
        "num_hidden_layers": 48,
        "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "indexer_head_dim": 128,
    },
    "nemotron": {
        # models/nemotron35/config.py:8-16 (public BF16 config) and LAYER_ORDER
        "layers_block_type": None,  # filled in below from config.LAYER_ORDER
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "conv_kernel": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
    },
}

# Copy the reference layer order so this module stays stdlib-only/importable
# even where the model packages are not installed.  Source:
# models/nemotron35/config.py:16 (52 layers: 23 mamba, 23 moe, 6 attention).
_NEMOTRON_LAYER_ORDER = (
    'mamba', 'moe', 'mamba', 'moe', 'mamba', 'attention', 'moe', 'mamba',
    'moe', 'mamba', 'moe', 'mamba', 'attention', 'moe', 'mamba', 'moe',
    'mamba', 'moe', 'mamba', 'attention', 'moe', 'mamba', 'moe', 'mamba',
    'moe', 'mamba', 'attention', 'moe', 'mamba', 'moe', 'mamba', 'moe',
    'mamba', 'attention', 'moe', 'mamba', 'moe', 'mamba', 'moe', 'mamba',
    'moe', 'mamba', 'attention', 'moe', 'mamba', 'moe', 'mamba', 'moe',
    'mamba', 'moe', 'mamba', 'moe',
)
assert len(_NEMOTRON_LAYER_ORDER) == 52
assert _NEMOTRON_LAYER_ORDER.count("mamba") == 23
assert _NEMOTRON_LAYER_ORDER.count("attention") == 6
PRODUCTION_CONFIGS["nemotron"]["layers_block_type"] = list(_NEMOTRON_LAYER_ORDER)


def _normalise_model(model: str) -> str:
    key = str(model).strip().lower().replace("_", "-")
    if key in _QWEN_ALIASES or key in {a.replace("_", "-") for a in _QWEN_ALIASES}:
        return "qwen"
    if key in _NEMOTRON_ALIASES or key in {a.replace("_", "-") for a in _NEMOTRON_ALIASES}:
        return "nemotron"
    raise ValueError(
        f"unknown model {model!r}; expected one of "
        f"{sorted(_QWEN_ALIASES | _NEMOTRON_ALIASES)}"
    )


def _flatten(config: dict[str, Any]) -> dict[str, Any]:
    """Accept either a flat config.json or a VL wrapper with ``text_config``."""
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    merged = dict(config)
    text = config.get("text_config")
    if isinstance(text, dict):
        merged.update(text)
    return merged


def _get(config: dict[str, Any], name: str, model: str) -> Any:
    try:
        return config[name]
    except KeyError as exc:
        fields = QWEN_FIELDS if model == "qwen" else NEMOTRON_FIELDS
        raise KeyError(
            f"missing config field {name!r} for {model}; "
            f"owner inputs needed: {', '.join(fields)}"
        ) from exc


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config field {name!r} must be a positive int, got {value!r}")
    return value


def _layer_counts(config: dict[str, Any], model: str) -> tuple[int, int]:
    """Return (state_layers, kv_layers) = (GDN/mamba, full/attention)."""
    # The two models name this field differently; prefer the model's own name
    # so a config carrying both (e.g. a tiny test fixture) works for each.
    if model == "qwen":
        layers = config.get("layer_types") or config.get("layers_block_type")
    else:
        layers = config.get("layers_block_type") or config.get("layer_types")
    if isinstance(layers, (list, tuple)) and layers:
        if model == "qwen":
            full = sum(1 for t in layers if t in _FULL_ATTN)
            linear = sum(1 for t in layers if t in _LINEAR_ATTN)
            unknown = set(layers) - _FULL_ATTN - _LINEAR_ATTN
            if unknown:
                raise ValueError(f"unknown Qwen layer types: {sorted(unknown)}")
        else:
            full = sum(1 for t in layers if t in _NEMO_ATTN)
            linear = sum(1 for t in layers if t in _NEMO_MAMBA)
            unknown = set(layers) - _NEMO_ATTN - _NEMO_MAMBA - {"moe"}
            if unknown:
                raise ValueError(f"unknown Nemotron layer types: {sorted(unknown)}")
        stated = config.get("num_hidden_layers")
        if stated is not None and len(layers) != _int(stated, "num_hidden_layers"):
            raise ValueError("layer_types length disagrees with num_hidden_layers")
        return linear, full
    if model == "qwen":
        # Engine default: every `full_attention_interval`-th layer is full.
        n = _int(_get(config, "num_hidden_layers", model), "num_hidden_layers")
        interval = _int(config.get("full_attention_interval", 4), "full_attention_interval")
        full, remainder = divmod(n, interval)
        if remainder:
            full += 1
        return n - full, full
    raise KeyError(
        "Nemotron needs 'layers_block_type' (owner inputs needed: "
        + ", ".join(NEMOTRON_FIELDS) + ")"
    )


def state_breakdown(
    model: str,
    ctx_len: int,
    config: dict[str, Any],
    *,
    kv_bytes: int | None = None,
    conv_bytes: int | None = None,
    int8_scale_bias: bool = False,
    int8_conv: bool = False,
    include_indexer: bool = False,
) -> dict[str, int]:
    """Return byte-level pieces; every value is derived arithmetic, not measured."""
    model = _normalise_model(model)
    ctx_len = _int(ctx_len, "ctx_len")
    cfg = _flatten(config)
    state_layers, kv_layers = _layer_counts(cfg, model)
    out: dict[str, int] = {}

    if model == "qwen":
        hv = _int(_get(cfg, "linear_num_value_heads", model), "linear_num_value_heads")
        hk = _int(_get(cfg, "linear_num_key_heads", model), "linear_num_key_heads")
        dk = _int(_get(cfg, "linear_key_head_dim", model), "linear_key_head_dim")
        dv = _int(_get(cfg, "linear_value_head_dim", model), "linear_value_head_dim")
        kernel = _int(_get(cfg, "linear_conv_kernel_dim", model), "linear_conv_kernel_dim")
        hkv = _int(_get(cfg, "num_key_value_heads", model), "num_key_value_heads")
        dkv = _int(_get(cfg, "head_dim", model), "head_dim")
        kb = 2 if kv_bytes is None else _int(kv_bytes, "kv_bytes")   # bf16 default
        cb = 2 if conv_bytes is None else _int(conv_bytes, "conv_bytes")  # bf16

        conv_dim = 2 * hk * dk + hv * dv
        conv_per = (kernel - 1) * conv_dim * cb
        recurrent_per = hv * dv * dk * 4                            # f32, engine-confirmed
        out["conv_dim"] = conv_dim
        out["gdn_conv_bytes"] = state_layers * conv_per
        out["gdn_recurrent_bytes"] = state_layers * recurrent_per
        out["gdn_bytes"] = out["gdn_conv_bytes"] + out["gdn_recurrent_bytes"]
        out["kv_bytes"] = kv_layers * 2 * ctx_len * hkv * dkv * kb
        out["indexer_bytes"] = 0
        if include_indexer:
            idx_dim = _int(cfg.get("indexer_head_dim", 128), "indexer_head_dim")
            out["indexer_bytes"] = kv_layers * ctx_len * (idx_dim * 2 + 12)
        out["state_layers"] = state_layers
        out["kv_layers"] = kv_layers
    else:
        heads = _int(_get(cfg, "mamba_num_heads", model), "mamba_num_heads")
        dim = _int(_get(cfg, "mamba_head_dim", model), "mamba_head_dim")
        ssm = _int(_get(cfg, "ssm_state_size", model), "ssm_state_size")
        groups = _int(_get(cfg, "n_groups", model), "n_groups")
        kernel = _int(_get(cfg, "conv_kernel", model), "conv_kernel")
        hkv = _int(_get(cfg, "num_key_value_heads", model), "num_key_value_heads")
        dkv = _int(_get(cfg, "head_dim", model), "head_dim")
        kb = 1 if kv_bytes is None else _int(kv_bytes, "kv_bytes")   # INT8 default
        cb = 1 if int8_conv else (2 if conv_bytes is None else _int(conv_bytes, "conv_bytes"))

        conv_dim = heads * dim + 2 * groups * ssm
        conv_per = (kernel - 1) * conv_dim * cb
        ssm_per = heads * dim * ssm * 4                              # f32, engine-confirmed
        out["conv_dim"] = conv_dim
        out["mamba_ssm_bytes"] = state_layers * ssm_per
        out["mamba_conv_bytes"] = state_layers * conv_per
        out["mamba_bytes"] = out["mamba_ssm_bytes"] + out["mamba_conv_bytes"]
        if int8_scale_bias:
            rounded = ((ctx_len + 4095) // 4096) * 4096
            out["kv_bytes"] = (
                kv_layers * 2 * rounded * hkv * (dkv + 4 * (dkv // 64))
            )
        else:
            out["kv_bytes"] = kv_layers * 2 * ctx_len * hkv * dkv * kb
        out["indexer_bytes"] = 0
        out["state_layers"] = state_layers
        out["kv_layers"] = kv_layers

    out["fixed_bytes"] = out.get("gdn_bytes", out.get("mamba_bytes", 0))
    out["total_bytes"] = out["fixed_bytes"] + out["kv_bytes"] + out["indexer_bytes"]
    return out


def session_state_mb(model: str, ctx_len: int, config: dict[str, Any]) -> float:
    """Per-session recurrent-state + KV size in SI MB (1e6 bytes).

    ``model`` accepts ``"qwen"``/``"qwen4_exp"``/``"flashnext"`` and
    ``"nemotron"``/``"nemotron35"``/``"nemotron_h"``.  ``config`` is a flat
    config.json dict (or a VL wrapper with ``text_config``).  Defaults inside
    ``state_breakdown`` apply the task's production modes: bf16 Qwen KV and
    conv, INT8 (1-byte) Nemotron KV, bf16 Nemotron conv.
    """
    return state_breakdown(model, ctx_len, config)["total_bytes"] / MB


def owner_inputs() -> str:
    """The fields this calculator cannot verify without the checkpoint owner."""
    return "\n".join(
        [
            "Owner inputs needed (values below are repo-derived ESTIMATES, not measurements):",
            "  1. Qwen linear_num_key_heads / linear_key_head_dim split (16/128 assumed).",
            "  2. Qwen linear_num_value_heads / linear_value_head_dim / linear_conv_kernel_dim (48/128/4).",
            "  3. Qwen full-attention head_dim=256 and num_key_value_heads=2; num_attention_heads unknown.",
            "  4. Qwen deployed layer_types (36 linear / 12 full) and KV dtype (bf16 assumed).",
            "  5. Nemotron deployed config == models/nemotron35/config.py EXPECTED + LAYER_ORDER.",
            "  6. Nemotron Mamba conv dtype (bf16 assumed; memory.py prices it at INT8 1 B).",
            "  7. Whether the simulator counts Qwen QSA indexer keys (268 B/token/full layer).",
            "  8. Whether INT8 KV includes BF16 scale/bias + 4096-token rounding (memory.py does).",
        ]
    )


def table(
    ctx_lengths: tuple[int, ...] = (4096, 16_384, 65_536, 131_072),
    *,
    int8_scale_bias: bool = False,
    int8_conv: bool = False,
    include_indexer: bool = False,
) -> str:
    """Print the 4K/16K/64K/128K table for both models (all values ESTIMATE)."""
    lines = [
        "Per-session state (SI MB = 1e6 B). All values ESTIMATE from repo-derived configs.",
        "Qwen: bf16 KV + bf16 GDN conv + f32 recurrent. Nemotron: 1 B INT8 KV + bf16 conv + f32 SSM.",
        f"{'model':<10} {'ctx':>8} {'state MB':>11} {'fixed MB':>10} {'KV MB':>10} {'layers(s/k)':>12}",
    ]
    for model in ("qwen", "nemotron"):
        cfg = PRODUCTION_CONFIGS[model]
        for ctx in ctx_lengths:
            b = state_breakdown(
                model, ctx, cfg, int8_scale_bias=int8_scale_bias,
                int8_conv=int8_conv, include_indexer=include_indexer,
            )
            lines.append(
                f"{model:<10} {ctx:>8} {b['total_bytes'] / MB:>11.3f} "
                f"{b['fixed_bytes'] / MB:>10.3f} {b['kv_bytes'] / MB:>10.3f} "
                f"{b['state_layers']:>5}/{b['kv_layers']:<6}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ctx", type=int, nargs="+", default=[4096, 16_384, 65_536, 131_072])
    parser.add_argument("--int8-scale-bias", action="store_true",
                        help="Nemotron: include BF16 scale/bias + 4096-token rounding (memory.py int8_kv)")
    parser.add_argument("--int8-conv", action="store_true",
                        help="Nemotron: price Mamba conv at 1 B (memory.py) instead of bf16 2 B")
    parser.add_argument("--include-indexer", action="store_true",
                        help="Qwen: include QSA indexer keys (268 B/token/full layer)")
    parser.add_argument("--owner-inputs", action="store_true", help="print owner-input caveats")
    args = parser.parse_args(argv)
    print(table(tuple(args.ctx), int8_scale_bias=args.int8_scale_bias,
                int8_conv=args.int8_conv, include_indexer=args.include_indexer))
    if args.owner_inputs:
        print()
        print(owner_inputs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
