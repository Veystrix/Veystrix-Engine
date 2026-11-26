"""One-command engine/model swap and engine auto-start for the terminal tool.

``veystrix engine cpp`` / ``veystrix engine python`` (and ``/engine`` inside the UI)
change the selected engine, persist it, and restart a running engine so the change
takes effect. A swap that cannot run (no C++ build, no parity record) is refused
BEFORE anything is stopped or saved, so a failed swap never leaves the user without
an engine or with a config that points at one that cannot start.

Switching is always a restart, never a hot swap: both engines own the unified-memory
budget and must not be resident at the same time.
"""
from __future__ import annotations

import json
import os
import sys
import time

from pathlib import Path

from veystrix.tool import engine as engine_mod
from veystrix.tool.config import Config, LocalKeyStore, home_dir
from veystrix.tool.supervisor import StartOptions, guard_stop, health, logs, start, status, stop

from veystrix import model_identity

MODELS = model_identity.PUBLIC_MODEL_CHOICES

# A memory-guard stop is terminal: reloading an 8+ GiB model on every /start is
# what drives the machine into swap. Back off before allowing another attempt.
_GUARD_FILE = "guard-stop.json"
_GUARD_BACKOFF_SECONDS = 30.0


def _guard_gate(home: Path, *, record: bool = False) -> float:
    """Return the epoch time before which another start is refused."""
    path = home / "run" / _GUARD_FILE
    try:
        until = float(json.loads(path.read_text()).get("not_before", 0))
    except (OSError, ValueError, TypeError):
        until = 0.0
    if record:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"not_before": time.time() + _GUARD_BACKOFF_SECONDS}))
    return until


def checkpoint_refusal(model: str) -> str | None:
    """None when the model has an installed, structurally valid checkpoint (generic families only)."""
    identity = model_identity.IDENTITIES.get(model)
    if identity is None or identity.cpp_runtime != "generic":
        return None
    from veystrix import checkpoints
    try:
        checkpoints.resolve_for_family(model, None)
    except ValueError as exc:
        return str(exc)
    return None


def default_checkpoint(model: str) -> str:
    """The single installed checkpoint for ``model``: the router refuses to start without an
    explicit ``--checkpoint``. Raises SystemExit with the resolver's reason when there is none."""
    from veystrix import checkpoints
    if model not in model_identity.IDENTITIES:
        return ""
    try:
        return str(checkpoints.resolve_for_family(model, None))
    except ValueError as exc:
        raise SystemExit(f"veystrix: {exc}") from exc


def _installed_public_models() -> list[tuple[str, str]]:
    """Return (model_family, checkpoint_path) for every public model with a complete checkpoint."""
    found: list[tuple[str, str]] = []
    for name in MODELS:
        try:
            ckpt = default_checkpoint(name)
        except SystemExit:
            continue
        if ckpt:
            found.append((name, ckpt))
    return found


def resolve_model_and_checkpoint(
    model: str,
    *,
    checkpoint: str = "",
    explicit_model: bool = False,
    interactive: bool = False,
    echo=None,
    resolver_fn=None,
) -> tuple[str, str]:
    """Return (model_family, checkpoint_path), handling missing checkpoints cleanly."""
    from veystrix.tool.harnesses import family_of

    resolve = resolver_fn or default_checkpoint
    normalized = family_of(model) or model
    if model_identity.is_locked(normalized) or normalized not in MODELS:
        raise SystemExit(f"veystrix: unknown or unavailable model {model!r}; use: {', '.join(MODELS)}")
    model = normalized

    if checkpoint:
        ckpt_path = Path(checkpoint).expanduser()
        if not ckpt_path.is_dir():
            raise SystemExit(f"veystrix: checkpoint directory does not exist: {checkpoint}")
        return model, str(ckpt_path.resolve())

    try:
        return model, resolve(model)
    except SystemExit as missing_exc:
        if not explicit_model:
            installed = _installed_public_models()
            if len(installed) == 1:
                alt_model, alt_ckpt = installed[0]
                if echo:
                    echo(f"no checkpoint for {model}; using installed model {alt_model}")
                return alt_model, alt_ckpt
            if len(installed) > 1:
                names = ", ".join(m for m, _ in installed)
                raise SystemExit(
                    f"veystrix: no checkpoint found for {model}, but multiple models are installed ({names}); "
                    f"choose one with --model NAME or run 'veystrix models pull {model}'."
                ) from missing_exc
        if interactive:
            from veystrix.checkpoints import model_root
            from veystrix.model_pull import PullError, pull

            ident = model_identity.IDENTITIES.get(model)
            label = ident.display_name if ident else model
            root = model_root()
            print(f"veystrix: no complete checkpoint found for {label} in {root}.", file=sys.stderr)
            try:
                answer = input(f"Download {label} now? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in ("", "y", "yes"):
                try:
                    pull(model, root, echo=echo or (lambda m: print(f"veystrix: {m}", file=sys.stderr)))
                except (PullError, ImportError) as exc:
                    raise SystemExit(f"veystrix: download failed: {exc}") from exc
                return model, default_checkpoint(model)
        raise missing_exc


def availability(home: Path, model: str, environ=None) -> dict:
    """Which engines can run right now for ``model``, with the reason when one cannot."""
    identity = model_identity.IDENTITIES.get(model)
    if identity is not None and "python" not in identity.engines:
        result = {"python": {"available": False,
                             "reason": f"{identity.display_name} has no Python engine (C++ generic runtime only)"}}
    else:
        result = {"python": {"available": True, "reason": ""}}
    try:
        engine_mod.check_engine("cpp", home, model)
        mode = "source checkout build" if engine_mod.source_cpp_ready(home, environ) else "installed bundle"
        result["cpp"] = {"available": True, "reason": mode}
    except SystemExit as exc:
        result["cpp"] = {"available": False, "reason": str(exc).removeprefix("veystrix: ")}
    return result


def describe(home: Path | None = None) -> dict:
    """Selected vs active engine, model and availability: the ``veystrix engine`` report."""
    home = home or home_dir()
    config = Config(home)
    model = config.get("model")
    selected = engine_for(model, os.environ.get("VEYSTRIX_ENGINE"), home)
    state = status(home)
    return {
        "selected": selected,
        "active": state.get("engine") if state.get("status") in ("running", "unhealthy") else None,
        "running": state.get("status") == "running",
        "model": model,
        "available": availability(home, model),
    }


def engine_for(model: str | None, explicit: str | None = None, home: Path | None = None) -> str:
    """The runtime that serves ``model``. Users never choose it: the C++ engine whenever it
    supports the model, the Python engine only for a model C++ cannot run (yet), or when this
    install has no usable C++ engine and the model also runs on Python. ``explicit``
    (``--engine`` / ``VEYSTRIX_ENGINE``, developer use) still wins."""
    if explicit:
        return engine_mod.resolve_engine(explicit, environ={}, config={})
    ident = model_identity.IDENTITIES.get(model or "")
    if ident is None:
        return "cpp"
    if "cpp" not in ident.engines:
        return ident.engines[0]
    if "python" in ident.engines:
        try:
            cpp = availability(home or home_dir(), model).get("cpp") or {}
        except Exception:
            cpp = {}
        if cpp.get("available") is False and "port-check" not in str(cpp.get("reason") or ""):
            return "python"                    # no C++ engine in this install: stay usable
    return "cpp"


def swap(home: Path | None = None, *, engine: str | None = None, model: str | None = None,
         restart: bool = True, timeout: float = 600.0, echo=None) -> dict:
    """Change engine and/or model; restart a running engine so it takes effect.

    Validation happens first. Nothing is stopped or written when the target cannot run.
    """
    home = home or home_dir()
    config = Config(home)
    echo = echo or (lambda _message: None)
    if engine is None and model is None:
        raise ValueError("nothing to swap: give an engine and/or a model")
    if engine is not None:
        engine = engine_mod.resolve_engine(engine, environ={}, config={})
    if model is not None and model_identity.is_locked(model):
        raise ValueError(f"{model} is locked and unavailable in this build; use: {', '.join(MODELS)}")
    if model is not None and model not in MODELS:
        raise ValueError(f"unknown model {model!r}; use: {', '.join(MODELS)}")
    new_model = model or config.get("model")
    if model_identity.is_locked(new_model):
        raise ValueError(f"{new_model} is locked and unavailable in this build; use: {', '.join(MODELS)}")
    new_engine = engine or engine_for(new_model, os.environ.get("VEYSTRIX_ENGINE"), home)
    engine_mod.check_engine(new_engine, home, new_model)     # raises SystemExit with a clear refusal
    if model is not None:
        refusal = checkpoint_refusal(new_model)
        if refusal:
            raise SystemExit(f"veystrix: cannot switch to {new_model}: {refusal}")

    previous = state = status(home)
    was_running = state.get("status") in ("running", "unhealthy")
    changed = (new_engine != config.get("engine")) or (new_model != config.get("model"))
    config.values["engine"] = new_engine
    config.values["model"] = new_model
    if changed:
        config.save()
    outcome = {"engine": new_engine, "model": new_model, "changed": changed,
               "restarted": False, "was_running": was_running}
    if not (restart and was_running):
        return outcome
    active_matches = state.get("engine") == new_engine and state.get("model") in (new_model, None)
    if active_matches and not changed:
        return outcome
    port = state.get("port") or config.get("port")
    echo(f"stopping the {previous.get('engine') or 'running'} engine ...")
    stop(home)
    echo(f"starting {new_engine} / {new_model} (loading the model can take a while) ...")
    try:
        start(StartOptions(home=home, port=port, engine=new_engine, model=new_model, timeout=timeout,
                           checkpoint=default_checkpoint(new_model)))
    except SystemExit:
        # The new engine failed to come up: go back to the previous selection instead of
        # leaving the user with nothing running.
        config.values["engine"] = previous.get("configured_engine") or config.get("engine")
        config.values["model"] = previous.get("model") or new_model
        config.save()
        raise
    outcome["restarted"] = True
    return outcome


def ensure_engine(home: Path | None = None, *, port: int | None = None, timeout: float = 600.0,
                  echo=None, model: str | None = None, explicit_model: bool = False,
                  interactive: bool = False) -> dict:
    """Start the configured engine when nothing answers; return what is running.

    ``started`` is True only when this call launched it, so the caller can decide
    whether to stop it again on exit.
    """
    home = home or home_dir()
    config = Config(home)
    echo = echo or (lambda _message: None)
    port = port or config.get("port")
    key = LocalKeyStore(home).load()
    report = health(port, api_key=key)
    if report is not None:
        if report.get("ready", True) and report.get("status") != "fault":
            return {"started": False, "port": port, "engine": report.get("engine") or config.get("engine"),
                    "model": report.get("model")}
        state = status(home) or {}
        st_name = state.get("status")
        guarded = st_name == "guard_stopped" or (st_name in ("unhealthy", "stopped", None) and guard_stop(logs(home, 40)))
        if guarded:
            remaining = _guard_gate(home) - time.time()
            if remaining > 0:
                raise ValueError(
                    f"engine was stopped by the memory safety guard; retry in {int(remaining) + 1}s "
                    "(lower /context or choose /kv sparse-paged first)")
            _guard_gate(home, record=True)
            echo("previous engine was stopped by the memory safety guard; not restarting it automatically")
            raise ValueError(
                "engine was stopped by the memory safety guard; lower /context or choose /kv sparse-paged, "
                "then start it explicitly")
        echo("previous engine faulted; restarting ...")
        stop(home)
    target_model = model or config.get("model")
    if model_identity.is_locked(target_model):
        target_model = MODELS[0]
    target_model, checkpoint = resolve_model_and_checkpoint(
        target_model,
        explicit_model=explicit_model or (model is not None),
        interactive=interactive,
        echo=echo,
    )
    selected = engine_for(target_model, os.environ.get("VEYSTRIX_ENGINE"), home)
    engine_mod.check_engine(selected, home, target_model)
    echo(f"starting the {selected} engine ({target_model}); first load can take a while ...")
    state = start(StartOptions(home=home, port=port, engine=selected, model=target_model, timeout=timeout,
                               checkpoint=checkpoint))
    return {"started": True, "port": port, "engine": state.get("engine"), "model": state.get("model")}
