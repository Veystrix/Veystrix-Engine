"""Measured speed settings for the C++ engine.

The knobs below are the ones the A/B benchmark runs used to reach the recorded
decode rates: ~4 tok/s for Flash-Next (pipe + keep-warm + spec_top 4 +
tensor_rows + gate_tail 3 + prefill_pipe 0) and Inkling 2 Mini's measured
generic-runtime defaults (gate + keep-warm + spec_fetch 4 + tensor_rows +
256 MiB MLX free-buffer cache cap + rdahead 0). Values are exact switches, not guesses:
an env value set by the operator wins over the recipe, and the recipe's own
measured defaults are used when no env value is supplied.

The terminal tool exposes these as ``speed_mode`` / ``speed_knobs`` settings
and forwards them to the engine process at launch (``/speed`` in the TUI).
"""
from __future__ import annotations

# Knob name -> the env var the C++ generic runtime reads at process start.
KNOBS = {
    "rows": "VEYSTRIX_GENERIC_ROWS",
    "pipe": "VEYSTRIX_GENERIC_PIPE",
    "spec_fetch": "VEYSTRIX_GENERIC_SPEC_FETCH",
    "spec_top": "VEYSTRIX_GENERIC_SPEC_TOP",
    "prefill_pipe": "VEYSTRIX_GENERIC_PREFILL_PIPE",
    "prefill_advise": "VEYSTRIX_GENERIC_PREFILL_ADVISE",
    "read_lanes": "VEYSTRIX_GENERIC_READ_LANES",
    "keepwarm": "VEYSTRIX_CPP_KEEPWARM",
    "tensor_rows": "VEYSTRIX_GENERIC_ROW_ORDER",
    "gate": "VEYSTRIX_GENERIC_GATE",
    "gate_tail": "VEYSTRIX_CPP_GATE_TAIL",
    "mlx_cache_bytes": "VEYSTRIX_CPP_MLX_CACHE_BYTES",
    "rdahead": "VEYSTRIX_CPP_RDAHEAD",
}
ENV_TO_KNOB = {value: key for key, value in KNOBS.items()}

# Per-model measured maximum-speed settings. Only knobs evidenced by a real
# A/B run belong here; changing them requires a new parity/performance record.
MAX_SPEED = {
    "flashnext": {"pipe": 1, "keepwarm": 1, "spec_top": 4, "tensor_rows": 1,
                  "gate_tail": 3, "prefill_pipe": 0},
    "inkling2-mini": {"pipe": 0, "gate": 1, "keepwarm": 1, "spec_fetch": 4,
                      "tensor_rows": 1, "mlx_cache_bytes": 256 << 20, "rdahead": 0},
    "nemotron35-lightning": {},
}
MODES = ("max", "recipe", "custom")


def _format(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def selected(mode: str, knobs: dict | None, model: str) -> dict:
    """Resolve the effective knobs for a model, without looking at the environment."""
    if mode == "max":
        return dict(MAX_SPEED.get(model, {}))
    if mode == "custom":
        return {k: v for k, v in (knobs or {}).items() if k in KNOBS}
    return {}                                  # recipe mode: rely on the recipe's measured defaults


def speed_env(mode: str, knobs: dict | None, model: str, environ: dict | None = None) -> dict:
    """Env vars to hand to the engine process. An operator-set env value wins."""
    environ = environ or {}
    result = {}
    for knob, value in selected(mode, knobs, model).items():
        name = KNOBS[knob]
        if environ.get(name) in (None, ""):
            result[name] = _format(value)
    return result


def describe(mode: str, knobs: dict | None, model: str, environ: dict | None = None) -> list[str]:
    """Human-readable lines for the TUI's /speed command."""
    environ = environ or {}
    effective = selected(mode, knobs, model)
    lines = [f"  ◆ Veystrix · speed={mode} for {model}"]
    if not effective:
        lines.append("      recipe defaults (no env overrides)")
    for knob, value in sorted(effective.items()):
        name = KNOBS[knob]
        source = "env" if environ.get(name) not in (None, "") else mode
        lines.append(f"      {knob}={_format(value)} [{source}]")
    lines.append("      switches: /speed max | recipe | custom; /speed <knob> <value>")
    return lines
