"""User configuration and the local API key for the terminal tool.

All state lives under ``~/.veystrix/`` (or ``VEYSTRIX_HOME``), is written atomically
with owner-only permissions, and never contains a prompt, a checkpoint payload or a
secret in clear text in the config file. The API key is stored in the macOS Keychain
through ``security`` where available, with an owner-only file fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
from pathlib import Path

from veystrix import kv_policy, model_identity
from veystrix.speed import KNOBS

KEYCHAIN_SERVICE = "io.github.joyboyreturns.veystrix"
KEYCHAIN_ACCOUNT = "local-api-key"

# One declaration per setting: default + validator, so there is nowhere to disagree.
EFFORTS = ("auto", "off", "low", "medium", "high", "xhigh")


def _valid_harness(v: str) -> bool:
    if v == "direct":
        return True
    try:
        from veystrix.tool.harnesses import is_locked
        return not is_locked(v)
    except Exception:
        return v in ("opencode", "pi", "hermes")


SCHEMA = {
    "port": (8080, lambda v: isinstance(v, int) and not isinstance(v, bool) and 1024 <= v <= 65535),
    "model": ("inkling2-mini", lambda v: v in model_identity.PUBLIC_MODEL_CHOICES),
    # KV precision for the generic C++ runtime; "auto" = chosen by the memory plan (veystrix/kv_policy.py).
    "kv": ("auto", lambda v: v in kv_policy.CHOICES),
    # Context ceiling the endpoint admits; 0 = the model's own ceiling (Ornith 262,144).
    "context": (0, lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 1_048_576),
    # Listen address: loopback (default) or 0.0.0.0 for the LAN, which requires require_api_key.
    "host": ("127.0.0.1", lambda v: v in ("127.0.0.1", "0.0.0.0")),
    "engine": ("python", lambda v: v in ("python", "cpp")),
    # Agent harness the branded UI drives (`veystrix tui`, /settings -> Harness).
    # Reasoning effort stamped on every request by the router, whatever harness sent it.
    # "auto" leaves the harness's own choice (or the server default: thinking off) in place.
    "effort": ("medium", lambda v: v in EFFORTS),
    # Show the model's reasoning live in the TUI (dimmed) instead of a collapsed summary.
    "show_reasoning": (True, lambda v: isinstance(v, bool)),
    "harness": ("opencode", _valid_harness),
    # No hardcoded default model (owner decision 2026-11-18). Empty means "not chosen yet".
    "default_model": ("", lambda v: isinstance(v, str)),
    "cache_quota_gb": (16, lambda v: isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 2000),
    "minimum_battery_percent": (25, lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100),
    "cache_directory": ("", lambda v: isinstance(v, str)),
    "supervised_restarts": (True, lambda v: isinstance(v, bool)),
    "require_api_key": (False, lambda v: isinstance(v, bool)),
    # `veystrix` starts the engine for you; keep it running after the UI exits (default: stop it).
    "keep_engine_running": (False, lambda v: isinstance(v, bool)),
    # Measured maximum-speed switches for the C++ engine ("max" is the default:
    # Flash-Next's benchmark recipe knobs and Ornith's 256 MiB MLX cache cap).
    # "recipe" uses the recipe's own defaults, "custom" uses speed_knobs below.
    "speed_mode": ("max", lambda v: v in ("max", "recipe", "custom")),
    "speed_knobs": ({}, lambda v: isinstance(v, dict) and all(
        isinstance(k, str) and k in KNOBS and isinstance(val, (int, bool)) and not isinstance(val, float)
        for k, val in v.items())),
}
DEFAULTS = {key: default for key, (default, _) in SCHEMA.items()}

# Friendly CLI names -> schema keys.
ALIASES = {
    "default-model": "default_model",
    "cache-quota-gb": "cache_quota_gb",
    "minimum-battery-percent": "minimum_battery_percent",
    "battery": "minimum_battery_percent",
    "cache-directory": "cache_directory",
    "supervised-restarts": "supervised_restarts",
    "require-api-key": "require_api_key",
    "keep-engine-running": "keep_engine_running",
    "speed-mode": "speed_mode",
    "speed-knobs": "speed_knobs",
}


def home_dir() -> Path:
    """The tool's home: VEYSTRIX_HOME overrides HOME so tests never touch a real config."""
    return Path(os.environ.get("VEYSTRIX_HOME") or Path.home()).expanduser()


def tool_home(home: Path | None = None) -> Path:
    return (home or home_dir()) / ".veystrix"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        tmp.write_text(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Config:
    def __init__(self, home: Path | None = None):
        self.home = home or home_dir()
        self.path = tool_home(self.home) / "config.json"
        self.values = self._load()

    def _load(self) -> dict:
        values = dict(DEFAULTS)
        try:
            saved = json.loads(self.path.read_text())
        except (OSError, TypeError, ValueError):
            return values
        if isinstance(saved, dict):
            for key, value in saved.items():
                if key in SCHEMA and SCHEMA[key][1](value):
                    values[key] = value
        # Safe migration: if saved harness points to a locked/unsupported harness, fall back cleanly
        if values.get("harness") != "direct" and not _valid_harness(str(values.get("harness", ""))):
            values["harness"] = DEFAULTS["harness"]
        return values

    def save(self) -> None:
        _atomic_write(self.path, json.dumps(self.values, indent=2, sort_keys=True) + "\n")

    def get(self, key: str):
        key = ALIASES.get(key, key)
        if key not in SCHEMA:
            raise ValueError(f"unknown setting {key!r}; use: {', '.join(sorted(ALIASES) | set(SCHEMA))}")
        return self.values[key]

    def set(self, key: str, raw: str):
        key = ALIASES.get(key, key)
        if key not in SCHEMA:
            raise ValueError(f"unknown setting {key!r}; use: {', '.join(sorted(ALIASES) | set(SCHEMA))}")
        default, valid = SCHEMA[key]
        if isinstance(default, bool):
            value = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}.get(raw.lower())
        elif isinstance(default, int):
            try:
                value = int(raw)
            except ValueError:
                value = None
        else:
            value = raw
        if value is None or not valid(value):
            raise ValueError(f"invalid value {raw!r} for {key}")
        self.values[key] = value
        self.save()
        return value


class LocalKeyStore:
    """Load/store the local engine key: Keychain via ``security``, file fallback."""

    def __init__(self, home: Path | None = None, *, runner=None, allow_keychain: bool | None = None):
        self.home = home or home_dir()
        self.path = tool_home(self.home) / "local-api-key"
        self.runner = runner or subprocess.run
        if allow_keychain is None:
            allow_keychain = os.environ.get("VEYSTRIX_NO_KEYCHAIN", "") != "1" and sys_is_macos()
        self.allow_keychain = allow_keychain

    def load(self) -> str | None:
        if self.allow_keychain:
            result = self._security("find-generic-password", "-s", KEYCHAIN_SERVICE,
                                    "-a", KEYCHAIN_ACCOUNT, "-w", input=None)
            if result is not None and result.stdout.strip():
                return result.stdout.strip()
        try:
            text = self.path.read_text().strip()
        except OSError:
            return None
        return text or None

    def store(self, key: str) -> str:
        key = key.strip()
        if not key or "\n" in key or "\r" in key:
            raise ValueError("API key must be one non-empty line")
        if self.allow_keychain:
            # -w with no value reads the secret from stdin, keeping it out of argv.
            result = self._security("add-generic-password", "-U", "-s", KEYCHAIN_SERVICE,
                                    "-a", KEYCHAIN_ACCOUNT, "-w", input=key)
            if result is not None and result.returncode == 0:
                self.path.unlink(missing_ok=True)
                return "keychain"
        _atomic_write(self.path, key + "\n")
        return "file"

    def rotate(self) -> tuple[str, str]:
        key = "vx-" + secrets.token_hex(32)
        return key, self.store(key)

    def fingerprint(self) -> str | None:
        key = self.load()
        return hashlib.sha256(key.encode()).hexdigest()[:12] if key else None

    def _security(self, *args: str, input: str | None):
        try:
            return self.runner(["security", *args], input=input, capture_output=True,
                               text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None


def sys_is_macos() -> bool:
    import sys
    return sys.platform == "darwin"


def installed_model_ids() -> list[str]:
    """Installed unlocked model IDs from metadata only; no weights are opened."""
    try:
        from veystrix.model_registry import discover
        return sorted({record.family for record in discover() if not model_identity.is_locked(record.family)})
    except Exception:
        return []


def resolve_default_model(config: Config, explicit: str | None = None,
                          *, input_fn=None, interactive: bool = True) -> str:
    """Choose and PERSIST the default model; never hardcode one.

    If both families are installed and nothing was chosen, the user is asked explicitly.
    A single installed family is stored without a prompt. With nothing installed, an
    explicit id is still accepted (offline tests use a synthetic id).
    """
    if explicit:
        config.set("default_model", explicit)
        return explicit
    saved = config.get("default_model")
    if saved:
        return saved
    families = installed_model_ids()
    if len(families) > 1:
        if not interactive:
            raise SystemExit("veystrix: choose a default model first: veystrix config set default-model "
                             + " | ".join(families))
        ask = input_fn or input
        print("Both Qwen and Nemotron are installed. Choose the default model (stored in config, reversible):")
        for index, family in enumerate(families, 1):
            print(f"  {index}. {family}")
        answer = (ask(f"Default model [1-{len(families)}]: ") or "").strip()
        try:
            chosen = families[int(answer) - 1]
        except (ValueError, IndexError):
            raise SystemExit("veystrix: invalid default-model choice") from None
        config.set("default_model", chosen)
        return chosen
    if len(families) == 1:
        config.set("default_model", families[0])
        return families[0]
    raise SystemExit("veystrix: no model is installed; place a checkpoint, then set "
                     "default-model with 'veystrix config set default-model <id>'")
