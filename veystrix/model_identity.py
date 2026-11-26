"""Single source of truth for model-family identity.

The engine, the macOS app and the docs all need the same handful of facts about
each supported model family: which ``model_type`` values it accepts, how it is
displayed, its served-model alias, its checkpoint alias/folder and its context
ceiling. Keeping those facts here means a future model swap or day-1 addition
changes one module plus the data files it points at, rather than dozens of
scattered strings.

This module is metadata-only: it never imports MLX, never loads a checkpoint and
never starts a process.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelIdentity:
    family: str
    display_name: str
    model_types: frozenset[str]
    served_id: str
    download_repo: str
    context_limit: int
    checkpoint_alias: str = ""
    checkpoint_directory: str = ""
    # Distinct roles of the top-level and text-config model_type. Subsets of
    # model_types; kept explicit so consumers that need the distinction (the
    # preflight profile) do not have to guess from an unordered set.
    model_type: str = ""
    text_model_type: str = ""
    # Which engines can serve this family. "cpp" is the generic C ABI runtime.
    engines: tuple[str, ...] = ("cpp",)
    # C++ runtime: "generic" (recipe-driven universal runtime).
    cpp_runtime: str = "generic"
    # Recommended sampling defaults from the model card (overridable by preferences/requests).
    sampling: tuple[tuple[str, float], ...] = ()
    reasoning: bool = True                     # emits reasoning / thinking segments
    tool_format: str = "inkling"
    locked: bool = False                       # hidden and blocked in user-facing CLI/TUI/downloads


# Inkling 2 Mini (Apache 2.0, Thinking Machines Lab): 24-layer looped MoE (middle 12 layers x 3
# passes = 48 visits, 1M context, NVFP4/affine 4-bit + optional DSpark draft), served by the
# generic recipe-driven C++ runtime.
INKLING2_MINI = ModelIdentity(
    family="inkling2-mini",
    display_name="Inkling 2 Mini",
    model_types=frozenset({"inkling2_loop", "inkling2_loop_text"}),
    served_id="veystrix-inkling2-mini",
    download_repo="mlx-community/Inkling-2-Mini-oQ4e-mtp",
    context_limit=1_048_576,
    checkpoint_alias="inkling2-mini",
    checkpoint_directory="Inkling-2-Mini-oQ4e-mtp",
    model_type="inkling2_loop",
    text_model_type="inkling2_loop_text",
    engines=("cpp",),
    cpp_runtime="generic",
    sampling=(("temperature", 0.6), ("top_p", 0.95), ("top_k", 20)),
    tool_format="inkling",
)

# Register new model families here for day-1 support across CLI, TUI, checkpoints,
# recipes, and the generic C++ runtime.
IDENTITIES: dict[str, ModelIdentity] = {
    INKLING2_MINI.family: INKLING2_MINI,
}

# Short names the router and internal tools accept for --model.
MODEL_CHOICES: tuple[str, ...] = tuple(IDENTITIES)
# Unlocked models exposed to end users in the terminal tool, CLI, and downloads.
PUBLIC_MODEL_CHOICES: tuple[str, ...] = tuple(f for f, i in IDENTITIES.items() if not i.locked)


def is_locked(family: str | None) -> bool:
    return bool(for_family(family).locked)


UNKNOWN = ModelIdentity(
    family="unknown",
    display_name="Unknown model",
    model_types=frozenset(),
    served_id="",
    download_repo="",
    context_limit=32_768,
)


def for_family(family: str | None) -> ModelIdentity:
    """The identity for a family key, or a safe unknown identity."""
    return IDENTITIES.get(family or "", UNKNOWN)


def display_name(family: str | None) -> str:
    return for_family(family).display_name


def context_limit(family: str | None) -> int:
    return for_family(family).context_limit


def model_types(family: str | None) -> frozenset[str]:
    return for_family(family).model_types


def served_id(family: str | None) -> str:
    return for_family(family).served_id


def for_model_type(model_type: str | None) -> ModelIdentity | None:
    """The family whose checkpoints declare this model_type, if any."""
    for identity in IDENTITIES.values():
        if model_type and model_type in identity.model_types:
            return identity
    return None


def sampling_defaults(family: str | None) -> dict:
    return dict(for_family(family).sampling)
