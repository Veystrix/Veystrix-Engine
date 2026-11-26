"""C++ backend construction for models served by the GENERIC recipe-driven runtime.

``session.install_cpp_selector()`` calls :func:`install_generic` after the custom-runtime
factory is installed. It wraps ``build_backend`` so that families whose identity says
``cpp_runtime == "generic"`` (Ornith today) are built here, and every other model still goes
to the custom factory unchanged.

What this adds over the custom factory:
* the recipe is generated from the checkpoint (headers only) and cached;
* the KV precision comes from the memory plan (``veystrix.kv_policy``), or from
  ``VEYSTRIX_KV_MODE`` (``veystrix start --kv``);
* the context ceiling is the identity's (Ornith 262,144) clamped by ``VEYSTRIX_CONTEXT_LIMIT``;
* the model card's sampling defaults (Ornith: temperature 0.6, top-p 0.95, top-k 20) apply
  unless the user changed sampling in preferences;
* ``runtime_info()`` feeds /health and /veystrix/status (runtime, KV mode and bytes/token,
  context in use vs ceiling, recipe digest).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from veystrix import model_identity
from veystrix.backends.cpp_engine import CppEngineBackend, _default_tokenizer

KV_ENV = "VEYSTRIX_KV_MODE"
MLX_CACHE_ENV = "VEYSTRIX_CPP_MLX_CACHE_BYTES"
GENERIC_MLX_CACHE_BYTES = 256 << 20


def generic_families() -> tuple[str, ...]:
    return tuple(f for f, i in model_identity.IDENTITIES.items() if i.cpp_runtime == "generic")


class GenericCppBackend(CppEngineBackend):
    """CppEngineBackend plus the generic runtime's identity, KV plan and status."""

    def __init__(self, *args, family: str = "", kv_plan=None, recipe_path: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.family = family
        self.kv_plan = kv_plan
        self.recipe_path = recipe_path

    def runtime_info(self) -> dict:
        info = {"runtime": "generic", "family": self.family, "recipe": self.recipe_path,
                "context_ceiling": self.context_limit}
        try:
            caps = self._binding.capabilities(self._engine)
        except Exception:
            caps = {}
        if isinstance(caps, dict):
            if caps.get("kv"):
                info["kv"] = caps["kv"]
            for key in ("model_type", "experts", "engine"):
                if key in caps:
                    info[key] = caps[key]
        if self.kv_plan is not None:
            info["kv_plan"] = self.kv_plan.as_dict()
            try:
                from veystrix.kv_policy import KVGeometry, bytes_per_token
                recipe = json.loads(Path(self.recipe_path).read_text())
                info["kv_bytes_per_token"] = round(bytes_per_token(KVGeometry.from_recipe(recipe),
                                                                   self.kv_plan.mode))
            except Exception:
                pass
        try:
            info["context_in_use"] = int(self._binding.tokens(self._session)) if self._session else 0
        except Exception:
            pass
        try:
            stats = self._binding.stats(self._engine)
            if isinstance(stats, dict):
                info["stats"] = {k: stats[k] for k in ("expert_bytes_read", "resident_bytes", "streamed_bytes",
                                                       "prefill_chunk", "load_seconds") if k in stats}
        except Exception:
            pass
        return info


def _stop_tokens(tokenizer, model_dir: str) -> set[int]:
    stops: set[int] = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        stops.update(int(t) for t in eos)
    elif eos is not None:
        stops.add(int(eos))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        unk = getattr(tokenizer, "unk_token_id", None)
        for marker in ("<|im_end|>", "<|endoftext|>", "<|content_model_end_sampling|>"):
            tid = convert(marker)
            if tid is not None and tid != unk:
                stops.add(int(tid))
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
        for source in (config, config.get("text_config") or {}):
            ids = source.get("eos_token_id")
            if isinstance(ids, (list, tuple)):
                stops.update(int(t) for t in ids)
            elif isinstance(ids, int):
                stops.add(ids)
    except (OSError, ValueError, AttributeError):
        pass
    return stops


def _context_ceiling(identity, model_dir: str) -> int:
    ceiling = identity.context_limit
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
        text = config.get("text_config") or config
        native = int(text.get("max_position_embeddings") or 0)
        if native > 0:
            ceiling = min(ceiling, native)
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    requested = os.environ.get("VEYSTRIX_CONTEXT_LIMIT", "")
    if requested.isdigit() and int(requested) > 0:
        ceiling = min(ceiling, int(requested))
    return ceiling


def apply_sampling_defaults(backend, family: str, prefs: dict | None) -> None:
    """Model-card sampling unless the user changed sampling away from the global defaults."""
    from veystrix.sampling import THINKING, Sampling
    defaults = model_identity.sampling_defaults(family)
    if not defaults:
        return
    prefs = prefs or {}
    user_changed = any(key in prefs and float(prefs[key]) != float(THINKING[key])
                       for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty") if key in THINKING)
    if user_changed:
        return
    current = getattr(backend, "sampling", None) or Sampling()
    backend.sampling = Sampling(
        temperature=float(defaults.get("temperature", current.temperature)),
        top_p=float(defaults.get("top_p", current.top_p)),
        top_k=int(defaults.get("top_k", current.top_k)),
        min_p=float(defaults.get("min_p", current.min_p)),
        presence_penalty=float(defaults.get("presence_penalty", current.presence_penalty)))


def _prefill_chunk(environ, family: str = "") -> int:
    """Prefill chunk in tokens (VEYSTRIX_GENERIC_PREFILL_CHUNK, default from RAM tier).
    The C++ runtime also scales chunks down dynamically under live memory pressure."""
    from veystrix.memory_tiers import current_tier
    default_chunk = current_tier(environ).prefill_chunk
    raw = environ.get("VEYSTRIX_GENERIC_PREFILL_CHUNK")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = default_chunk
    else:
        value = default_chunk
    return max(256, min(value, 16384))


def generic_profile(model_dir: str, family: str, *, environ=None):
    """(profile dict, KVPlan, recipe path) for a generic-runtime checkpoint. No weights are read."""
    from veystrix import kv_policy
    from veystrix.recipe import ensure_recipe
    environ = os.environ if environ is None else environ
    identity = model_identity.for_family(family)
    recipe_path = ensure_recipe(model_dir)
    recipe = json.loads(Path(recipe_path).read_text())
    context = _context_ceiling(identity, model_dir)
    requested = (environ.get(KV_ENV) or "auto").strip().lower()
    plan = kv_policy.plan_for_checkpoint(model_dir, recipe, context, requested=requested)
    profile = {"engine": "generic", "model": family, "recipe": str(recipe_path),
               "context_window": context, "prefill_chunk": _prefill_chunk(environ, family), **plan.profile()}
    if environ.get("VEYSTRIX_SHADOW_PATH") or environ.get("VEYSTRIX_CPP_DIGESTS") == "1":
        profile["digests"] = True
    return profile, plan, str(recipe_path)


def build_generic_backend(name, args, prefs, *, binding=None, tokenizer=None):
    family = str(name).strip().lower()
    model_dir = getattr(args, "model_path", None) or os.environ.get("VEYSTRIX_MODEL_DIR")
    if not model_dir:
        raise SystemExit(f"veystrix: a checkpoint path is required for {family!r} (--checkpoint PATH)")
    try:
        profile, plan, recipe_path = generic_profile(model_dir, family)
    except Exception as exc:
        raise SystemExit(f"veystrix: cannot prepare the generic runtime for {model_dir}: {exc}") from exc
    # MLX free-buffer cache at 256 MiB, as the custom factory ships it. Without this, the server's memory budget
    # (apply_runtime_memory_budget exports VEYSTRIX_CPP_RESIDENT) makes the C++ loader pick a 2 GiB cap: freed
    # prefill buffers then sit cold in MLX's cache, the kernel compresses them, and the expert page cache loses
    # ~1.7 GiB (Ornith through the server: hit rate 92% -> 86.5%, disk 43 -> 71 MB/token; 12.7 -> 14.1 tok/s
    # once capped, ENGINE_AB_2026-11-20.md "Ornith server gap"). Allocation policy only: outputs are unchanged.
    from veystrix.memory_tiers import current_tier
    tier_cache = current_tier().mlx_cache_mb << 20
    os.environ.setdefault(MLX_CACHE_ENV, str(tier_cache))
    if not profile.get("digests") and not os.environ.get("VEYSTRIX_CPP_SHADOW_LOG"):
        os.environ.setdefault("VEYSTRIX_CPP_NO_DIGEST", "1")
    # Overlap transformers/AutoTokenizer loading with C++ vx_engine_open() (which releases the GIL).
    tok_future = None
    tok_pool = None
    if tokenizer is None:
        from concurrent.futures import ThreadPoolExecutor
        tok_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="veystrix-tok")
        tok_future = tok_pool.submit(_default_tokenizer, model_dir)
    try:
        backend = GenericCppBackend(
            model_dir=model_dir, profile_json=json.dumps(profile, sort_keys=True), binding=binding,
            tokenizer=(tokenizer if tokenizer is not None else tok_future),
            session_dir=getattr(args, "session_dir", None) or f"~/.cache/veystrix/{family}/sessions",
            stop_tokens=set(), family=family, kv_plan=plan, recipe_path=recipe_path)
    finally:
        if tok_pool is not None:
            tok_pool.shutdown(wait=False)
    stops = _stop_tokens(backend.tokenizer, model_dir)
    if not stops:
        backend.close()
        raise SystemExit("veystrix: could not determine stop tokens from the checkpoint tokenizer; "
                         "refusing to generate unbounded.")
    backend.stop_tokens = stops
    if prefs:
        from veystrix.session import mirror_preferences
        mirror_preferences(backend, prefs, prefs.get("profile", "default"))
    apply_sampling_defaults(backend, family, prefs)
    return backend


def install_generic(module) -> None:
    """Route generic-runtime families to :func:`build_generic_backend` on ``module``."""
    if getattr(module, "_veystrix_generic_installed", False):
        return
    import functools
    previous = module.build_backend

    @functools.wraps(previous)          # keeps the custom factory's identity (selector tests, logs)
    def build_backend(name, args, prefs):
        if str(name).strip().lower() in generic_families():
            return build_generic_backend(name, args, prefs)
        return previous(name, args, prefs)

    module.build_backend = build_backend
    module._veystrix_generic_installed = True
