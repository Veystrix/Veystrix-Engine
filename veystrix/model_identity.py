"""Single source of truth for model-family identity.

The engine, the macOS app and the docs all need the same handful of facts about
each supported model family: which ``model_type`` values it accepts, how it is
displayed, its served-model alias, its checkpoint alias/folder and its context
ceiling. Keeping those facts here means a future model swap changes one module
plus the data files it points at, rather than dozens of scattered strings.

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


FLASHNEXT = ModelIdentity(
    family="flashnext",
    display_name="Veystrix Flash-Next",
    model_types=frozenset({"qwen4_exp", "qwen4_exp_text"}),
    served_id="veystrix-flashnext",
    download_repo="Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP",
    context_limit=262_144,
    checkpoint_alias="vontra-mtp",
    checkpoint_directory="Qwen3.8-Flash-Next-MLX-4bit-MTP",
)

NEMOTRON = ModelIdentity(
    family="nemotron35-lightning",
    display_name="Veystrix Nemotron 3.5 Lightning",
    model_types=frozenset({"nemotron_h"}),
    served_id="veystrix-nemotron35-lightning",
    download_repo="mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit",
    context_limit=1_048_576,
)

IDENTITIES: dict[str, ModelIdentity] = {
    FLASHNEXT.family: FLASHNEXT,
    NEMOTRON.family: NEMOTRON,
}

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
