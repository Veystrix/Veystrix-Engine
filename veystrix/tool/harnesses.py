"""Harness adapters: ``veystrix connect`` / ``veystrix run``.

Every key below was verified against the harness's CURRENT official documentation on
2026-11-18 (URLs cited in each adapter). Nothing is guessed. Config writes go through
the existing backup + atomic + idempotent writer. ``run`` never writes config: it only
sets the endpoint environment for the harness process.

Cline publishes UI fields but no stable settings.json key names in its current docs, so
we generate a copy-paste snippet instead of inventing keys, and document that limitation.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path

from veystrix.harness_configure import _write
from veystrix.harness_models import (
    deepseek_config, kimi_config, opencode_config, qwen_config,
)
from veystrix.pi_extension import EXTENSION_SOURCE
from veystrix.tool.config import home_dir
from veystrix.tool._formats import toml_dumps, toml_loads, yaml_dumps, yaml_upsert

from dataclasses import dataclass

# URLs and the date each one was checked. Update the date whenever a key is re-verified.
DOC = {
    "opencode": ("https://opencode.ai/docs/providers", "2026-11-18"),
    "claude-code": ("https://code.claude.com/docs/en/env-vars", "2026-11-18"),
    "codex": ("https://developers.openai.com/codex/config-reference", "2026-11-18"),
    "qwen-code": ("https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/", "2026-11-18"),
    "deepseek": ("https://github.com/deepseek-ai/DeepSeek-TUI", "2026-11-20"),
    "kimi": ("https://moonshotai.github.io/kimi-code/en/configuration/providers", "2026-11-18"),
    "hermes": ("https://hermes-agent.nousresearch.com/docs/integrations/providers", "2026-11-18"),
    "pi": ("https://pi.dev/docs/latest/custom-provider", "2026-11-18"),
    "aider": ("https://aider.chat/docs/llms/openai-compat.html", "2026-11-18"),
    "cline": ("https://docs.cline.bot/provider-config/openai-compatible", "2026-11-18"),
    "continue": ("https://docs.continue.dev/customize/model-providers/top-level/openai", "2026-11-18"),
}
NAMES = tuple(DOC)
HARNESS_CHOICES: tuple[str, ...] = tuple(DOC)

@dataclass
class HarnessIdentity:
    name: str
    doc_url: str
    doc_checked: str
    locked: bool = False

OPENCODE = HarnessIdentity("opencode", DOC["opencode"][0], DOC["opencode"][1], locked=False)
PI = HarnessIdentity("pi", DOC["pi"][0], DOC["pi"][1], locked=False)
HERMES = HarnessIdentity("hermes", DOC["hermes"][0], DOC["hermes"][1], locked=False)
CLAUDE_CODE = HarnessIdentity("claude-code", DOC["claude-code"][0], DOC["claude-code"][1], locked=True)
CODEX = HarnessIdentity("codex", DOC["codex"][0], DOC["codex"][1], locked=True)
QWEN_CODE = HarnessIdentity("qwen-code", DOC["qwen-code"][0], DOC["qwen-code"][1], locked=True)
DEEPSEEK = HarnessIdentity("deepseek", DOC["deepseek"][0], DOC["deepseek"][1], locked=True)
KIMI = HarnessIdentity("kimi", DOC["kimi"][0], DOC["kimi"][1], locked=True)
AIDER = HarnessIdentity("aider", DOC["aider"][0], DOC["aider"][1], locked=True)
CLINE = HarnessIdentity("cline", DOC["cline"][0], DOC["cline"][1], locked=True)
CONTINUE = HarnessIdentity("continue", DOC["continue"][0], DOC["continue"][1], locked=True)

HARNESS_IDENTITIES: dict[str, HarnessIdentity] = {
    "opencode": OPENCODE,
    "pi": PI,
    "hermes": HERMES,
    "claude-code": CLAUDE_CODE,
    "codex": CODEX,
    "qwen-code": QWEN_CODE,
    "deepseek": DEEPSEEK,
    "kimi": KIMI,
    "aider": AIDER,
    "cline": CLINE,
    "continue": CONTINUE,
}

# Unlocked harnesses exposed to end users in the terminal tool, CLI, and documentation.
PUBLIC_HARNESS_CHOICES: tuple[str, ...] = ("opencode", "pi", "hermes")

HARNESS_ALIASES: dict[str, str] = {
    "claude": "claude-code",
    "qwen": "qwen-code",
    "kimi-code": "kimi",
}

def canonical_harness_name(name: str | None) -> str:
    norm = (name or "").strip()
    return HARNESS_ALIASES.get(norm, norm)

def is_locked(name: str | None) -> bool:
    """True when the harness is locked and hidden from public CLI/TUI surfaces."""
    canonical = canonical_harness_name(name)
    if canonical not in PUBLIC_HARNESS_CHOICES:
        return True
    ident = HARNESS_IDENTITIES.get(canonical)
    if ident is not None:
        return bool(ident.locked)
    return False

# Where `connect` writes each harness's provider config, relative to the user's home.
CONFIG_FILES = {
    "opencode": (".config", "opencode", "opencode.json"),
    "claude-code": (".claude", "settings.json"),
    "codex": (".codex", "config.toml"),
    "qwen-code": (".qwen", "settings.json"),
    "deepseek": (".dsh", "profiles", "headless", "cordis.patch.yml"),
    "kimi": (".kimi-code", "config.toml"),
    "hermes": (".hermes", "profiles", "veystrix", "config.yaml"),
    "pi": (".pi", "agent", "extensions", "veystrix.ts"),
    "aider": (".aider.conf.yml",),
    "cline": (".veystrix", "harness-snippets", "cline.md"),
    "continue": (".continue", "config.yaml"),
}


def config_path(name: str, home: Path | None = None) -> Path:
    """The file `configure(name)` writes (no I/O)."""
    if name not in CONFIG_FILES:
        raise ValueError(f"unknown harness {name!r}; use: {', '.join(NAMES)}")
    return Path(home or home_dir()).joinpath(*CONFIG_FILES[name])


def is_connected(name: str, home: Path | None = None) -> bool:
    """True when the harness config written by `connect` exists and points at Veystrix."""
    try:
        path = config_path(name, home)
        with open(path, encoding="utf-8", errors="replace") as handle:
            return "veystrix" in handle.read(512 * 1024).lower()
    except (OSError, ValueError):
        return False


def base_url(port: int) -> str:
    """OpenAI-compatible endpoint used by most harnesses."""
    return f"http://127.0.0.1:{port}/v1"


def anthropic_base_url(port: int) -> str:
    """Anthropic endpoint origin: Claude Code appends `/v1/messages` itself."""
    return f"http://127.0.0.1:{port}"


def resolve_model_id(port: int, configured: str | None, api_key: str | None = None,
                     explicit: bool = False) -> str:
    """Resolve the exact model id the server accepts (Sonnet findings 16/4).

    An explicit CLI --model is always honoured. A config-family default (e.g.
    "flashnext") is refined to the installed id from the live /health report
    when one is available, since that is what /v1/models actually accepts.
    """
    def family_id(value: str) -> str:
        value = value.strip()
        return value if value.startswith("veystrix-") else f"veystrix-{value}"

    if explicit and configured:
        return family_id(configured)
    report_model_id = None
    try:
        from veystrix.tool.supervisor import health
        report = health(port, api_key=api_key)
        if report and report.get("installed_model_id"):
            report_model_id = str(report["installed_model_id"])
    except Exception:
        pass
    if report_model_id:
        return report_model_id
    return family_id(configured) if configured else "veystrix-inkling2-mini"


def default_models() -> list:
    """Metadata-only inventory, or one synthetic id so offline tests need no checkpoint."""
    from veystrix.model_registry import discover
    models = discover()
    if models:
        return models
    from veystrix.model_registry import InstalledModel
    return [InstalledModel("veystrix-inkling2-mini", "inkling2-mini", "Inkling 2 Mini",
                           "(no installed checkpoint)", "inkling2_loop", {}, 1048576, "offline")]


def _dump_json(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _configure_opencode(home: Path, port: int, models) -> tuple[Path, str, str]:
    # opencode v2 custom provider: providers.<id>.package / settings.baseURL / models.<id>.limit.
    # Docs: https://opencode.ai/docs/providers (checked 2026-11-18). We keep the long-standing
    # `settings.apiKey` literal from the tested transform: on loopback any value is accepted,
    # and `veystrix run` additionally exports VEYSTRIX_API_KEY for builds that read `env`.
    path = config_path("opencode", home)
    current = json.loads(path.read_text()) if path.exists() else {}
    return path, _dump_json(opencode_config(current, models, port)), "opencode.json"


def _configure_claude_code(home: Path, port: int, models, key: str) -> tuple[Path, str, str]:
    # Claude Code reads ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN and the model aliases from
    # the `env` block of ~/.claude/settings.json.
    # Docs: https://code.claude.com/docs/en/env-vars (checked 2026-11-18).
    path = config_path("claude-code", home)
    current = json.loads(path.read_text()) if path.exists() else {}
    env = current.setdefault("env", {})
    if not isinstance(env, dict):
        raise ValueError("Claude Code settings.env is not an object")
    model_id = models[0].id
    env.update({
        "ANTHROPIC_BASE_URL": anthropic_base_url(port),  # Claude Code adds /v1/messages
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_MODEL": model_id,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id,
        "CLAUDE_CODE_SUBAGENT_MODEL": model_id,
    })
    # Claude Code does not know a veystrix-* id and assumes 200k tokens; tell it the real window
    # (docs: model-config, "Correct the window for a gateway or custom model id", checked 2026-11-21).
    from veystrix.model_registry import catalog_context
    env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(catalog_context(models[0]))
    return path, _dump_json(current), "settings.json"


def _configure_codex(home: Path, port: int, models) -> tuple[Path, str, str]:
    # Codex: top-level model_provider/model plus [model_providers.veystrix] with name,
    # base_url, env_key and wire_api. env_key names an environment variable, never the key.
    # Docs: https://developers.openai.com/codex/config-reference (checked 2026-11-18).
    path = config_path("codex", home)
    document = toml_loads(path.read_text()) if path.exists() else {}
    model_id = models[0].id
    document["model"] = model_id
    document["model_provider"] = "veystrix"
    providers = document.setdefault("model_providers", {})
    provider = providers.setdefault("veystrix", {})
    provider["name"] = "Veystrix (local)"
    provider["base_url"] = base_url(port)
    provider["env_key"] = "VEYSTRIX_API_KEY"
    provider["wire_api"] = "responses"
    return path, toml_dumps(document), "config.toml"


def _configure_qwen(home: Path, port: int, models) -> tuple[Path, str, str]:
    # Qwen Code: modelProviders.openai[] entries carry id/baseUrl/envKey/generationConfig;
    # security.auth.selectedType selects the protocol. Credentials are read from envKey.
    # Docs: https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/ (2026-11-18).
    path = config_path("qwen-code", home)
    current = json.loads(path.read_text()) if path.exists() else {}
    result = qwen_config(current, models, port)
    result.setdefault("security", {}).setdefault("auth", {})["selectedType"] = "openai"
    result.setdefault("model", {})["name"] = models[0].id
    return path, _dump_json(result), "settings.json"


def _configure_kimi(home: Path, port: int, models) -> tuple[Path, str, str]:
    # Kimi Code: [providers.<id>] type="openai", base_url, api_key; [models.<id>] max_context_size.
    # Docs: https://moonshotai.github.io/kimi-code/en/configuration/providers (checked 2026-11-18).
    path = config_path("kimi", home)
    converted = kimi_config(toml_loads(path.read_text()) if path.exists() else {}, models, port)
    return path, toml_dumps(converted), "config.toml"


def _configure_hermes(home: Path, port: int, models, key: str) -> tuple[Path, str, str]:
    # Hermes: legacy custom_providers[] entries (name, base_url, key_env, api_mode) still work
    # alongside the v12 `providers:` dict; model.provider uses the `custom:` prefix.
    # Docs: https://hermes-agent.nousresearch.com/docs/integrations/providers (checked 2026-11-18).
    from veystrix.model_registry import catalog_context
    path = config_path("hermes", home)
    provider = {
        "name": "veystrix", "base_url": base_url(port), "key_env": "VEYSTRIX_API_KEY",
        "api_mode": "chat_completions",
        "models": {m.id: {"name": m.display_name, "context_length": catalog_context(m)} for m in models},
    }
    model = {"default": models[0].id, "provider": "custom:veystrix", "base_url": base_url(port)}
    existing = path.read_text() if path.exists() else ""
    try:
        import yaml
        from veystrix.tool._formats import strip_managed
        # Our old managed block is replaced anyway; dropping it first keeps the user's own copy
        # of any key it duplicated (a YAML parser silently keeps only the last duplicate).
        unmanaged = strip_managed(existing)
        current = yaml.safe_load(unmanaged) if unmanaged.strip() else {}
    except Exception:                       # no PyYAML or an unreadable file: line-level upsert
        current = None
    if not isinstance(current, dict):
        data = {"custom_providers": [provider], "model": model, "fallback_providers": []}
        return path, yaml_upsert(existing, yaml_dumps(data)), "config.yaml"
    # Real merge: keep every other provider and setting, replace only Veystrix's own entry
    # (any spelling) and point the default model at it. No cloud fallback behind a local model.
    others = [p for p in (current.get("custom_providers") or [])
              if not (isinstance(p, dict) and str(p.get("name", "")).lower() == "veystrix")]
    current["custom_providers"] = [*others, provider]
    merged_model = current.get("model") if isinstance(current.get("model"), dict) else {}
    merged_model.update(model)
    current["model"] = merged_model
    current["fallback_providers"] = []
    return path, yaml.safe_dump(current, sort_keys=False, allow_unicode=True), "config.yaml"


def _configure_deepseek(home: Path, port: int, models) -> tuple[Path, str, str]:
    """DeepSeek (dsh) headless profile: merge the Veystrix provider into its
    cordis.patch.yml. The desktop/web profiles are handled by
    `python -m veystrix.harness_configure`; this is the terminal-relevant one."""
    import yaml
    from veystrix.harness_configure import load_yaml12

    path = config_path("deepseek", home)
    if path.exists() and path.stat().st_size:
        patch = load_yaml12(path.read_text())
    else:
        patch = None
    if not isinstance(patch, list) or not patch:
        # A missing or empty profile yields a seed patch instead of a crash.
        patch = [{"id": "llm-pi-ai", "name": "@deepseek-ai/dsh-llm-pi-ai",
                  "config": {"providers": {}}}]
    output = yaml.safe_dump(deepseek_config(patch, models, port),
                            sort_keys=False, allow_unicode=True)
    return path, output, "cordis.patch.yml"


def _configure_pi(home: Path, port: int, models) -> tuple[Path, str, str]:
    # pi: provider extensions register with pi.registerProvider({baseUrl, apiKey, api, models}).
    # Docs: https://pi.dev/docs/latest/custom-provider (checked 2026-11-18).
    path = config_path("pi", home)
    # Bake the chosen port into the extension's default URL so `connect --port`
    # works without relying on the launch environment (Sonnet finding 18).
    source = EXTENSION_SOURCE.replace("127.0.0.1:8080/v1", f"127.0.0.1:{port}/v1")
    return path, source, "veystrix.ts"


def _configure_aider(home: Path, port: int, models) -> tuple[Path, str, str]:
    # Aider OpenAI-compatible: OPENAI_API_BASE / AIDER_OPENAI_API_BASE plus `openai/` model prefix.
    # Docs: https://aider.chat/docs/llms/openai-compat.html (checked 2026-11-18).
    path = config_path("aider", home)
    data = {"model": "openai/" + models[0].id, "openai-api-base": base_url(port)}
    existing = path.read_text() if path.exists() else ""
    return path, yaml_upsert(existing, yaml_dumps(data)), ".aider.conf.yml"


def _configure_cline(home: Path, port: int, models) -> tuple[Path, str, str]:
    # Cline's current docs describe UI fields (API Provider "OpenAI Compatible", Base URL,
    # API Key, Model ID) but publish no stable settings.json keys, so we do NOT guess them.
    # Docs: https://docs.cline.bot/provider-config/openai-compatible (checked 2026-11-18).
    path = config_path("cline", home)
    text = ("# Cline — Veystrix (manual UI setup)\n\n"
            "Current docs document UI fields, not stable settings.json keys, so Veystrix does not edit them.\n"
            f"- API Provider: **OpenAI Compatible**\n- Base URL: `{base_url(port)}`\n"
            f"- API Key: any non-empty value (loopback)\n- Model ID: `{models[0].id}`\n\n"
            "Source: https://docs.cline.bot/provider-config/openai-compatible (checked 2026-11-18)\n")
    return path, text, "cline.md"


def _configure_continue(home: Path, port: int, models) -> tuple[Path, str, str]:
    # Continue config.yaml: models[] entries need name/provider/model; apiBase points at the
    # OpenAI-compatible root; capabilities: [tool_use] enables agent mode.
    # Docs: https://docs.continue.dev/customize/model-providers/top-level/openai (checked 2026-11-18).
    path = config_path("continue", home)
    veystrix_models = [{"name": "Veystrix · " + m.display_name, "provider": "openai",
                        "model": m.id, "apiBase": base_url(port), "apiKey": "local",
                        "capabilities": ["tool_use"], "roles": ["chat", "edit", "apply"]}
                       for m in models]
    existing = path.read_text() if path.exists() else ""
    # Only consider USER lines: a `models:` key inside our own managed block
    # does not count (otherwise the second `connect` would rewrite it and
    # break idempotency).
    from veystrix.tool._formats import BEGIN as _BEGIN, END as _END
    user_lines = []
    skipping = False
    for line in existing.splitlines():
        if line.strip() == _BEGIN:
            skipping = True
        elif line.strip() == _END:
            skipping = False
        elif not skipping:
            user_lines.append(line)
    user_text = "\n".join(user_lines)
    # If the user already has a `models:` list, MERGE ours by model id rather
    # than appending a second top-level key that last-wins YAML would silently
    # replace (Sonnet finding 21).
    if user_text and re.search(r"^models\s*:", user_text, re.MULTILINE):
        import yaml
        try:
            document = yaml.safe_load(existing) or {}
        except yaml.YAMLError:
            document = {}
        if isinstance(document, dict) and isinstance(document.get("models"), list):
            current = document["models"]
            ids = {e.get("model") for e in current if isinstance(e, dict)}
            document["models"] = current + [m for m in veystrix_models
                                             if m["model"] not in ids]
            return path, yaml.safe_dump(document, sort_keys=False,
                                        allow_unicode=True), "config.yaml"
    data = {
        "name": "Veystrix local", "version": "0.0.1", "schema": "v1",
        "models": veystrix_models,
    }
    return path, yaml_upsert(existing, yaml_dumps(data)), "config.yaml"


def _ensure_hermes_profile(home: Path) -> None:
    """Hermes only loads profiles it created itself (`hermes -p veystrix` fails with 'does not exist'
    when just a config.yaml is dropped into the folder), so create it first when hermes is installed."""
    import shutil
    import subprocess
    if (home / ".hermes" / "profiles" / "veystrix").exists() or not shutil.which("hermes"):
        return
    try:
        subprocess.run(["hermes", "profile", "create", "veystrix", "--no-alias", "--no-skills"],
                       env={**os.environ, "HOME": str(home)}, stdin=subprocess.DEVNULL,
                       capture_output=True, timeout=180, check=False)
    except (OSError, subprocess.SubprocessError):
        pass                      # the config is still written; hermes will say what it needs


def configure(name: str, home: Path | None = None, port: int = 8080,
              model_id: str | None = None, key: str = "local", apply: bool = True) -> dict:
    """Write one harness's provider config with backup + idempotent rewrite."""
    if name not in DOC:
        raise ValueError(f"unknown harness {name!r}; use: {', '.join(NAMES)}")
    home = home or home_dir()
    models = default_models()
    if model_id:
        from veystrix import model_identity
        family = next((f for f, i in model_identity.IDENTITIES.items()
                       if model_id in (i.served_id, f) or model_id.startswith(f"veystrix-{f}-")), "inkling2-mini")
        identity = model_identity.for_family(family)
        selected = next((m for m in models if m.id == model_id), None)
        if selected is None:
            selected = type(models[0])(model_id, family, identity.display_name, "(offline)",
                                       identity.model_type or "inkling2_loop", {},
                                       identity.context_limit, "offline")
            models = [selected] + [m for m in models if m.id != model_id]
        else:
            models = [selected] + [m for m in models if m.id != model_id]
    builders = {
        "opencode": lambda: _configure_opencode(home, port, models),
        "claude-code": lambda: _configure_claude_code(home, port, models, key),
        "codex": lambda: _configure_codex(home, port, models),
        "qwen-code": lambda: _configure_qwen(home, port, models),
        "deepseek": lambda: _configure_deepseek(home, port, models),
        "kimi": lambda: _configure_kimi(home, port, models),
        "hermes": lambda: _configure_hermes(home, port, models, key),
        "pi": lambda: _configure_pi(home, port, models),
        "aider": lambda: _configure_aider(home, port, models),
        "cline": lambda: _configure_cline(home, port, models),
        "continue": lambda: _configure_continue(home, port, models),
    }
    path, content, _label = builders[name]()
    if apply and name == "hermes":
        _ensure_hermes_profile(home)
    result = {"harness": name, "path": str(path), "doc_url": DOC[name][0], "doc_checked": DOC[name][1]}
    if apply:
        result["result"] = _write(path, content)
    else:
        result["result"] = "ready to update"
    return result


def family_of(model_id: str | None) -> str | None:
    """Model family for a family key, alias (``veystrix-ornith``) or installation id."""
    from veystrix import model_identity
    value = (model_id or "").strip()
    for family, ident in sorted(model_identity.IDENTITIES.items(), key=lambda kv: -len(kv[1].served_id)):
        if value in (family, ident.served_id) or value.startswith(ident.served_id + "-"):
            return family
    return None


def ensure_opencode_models(home: Path | None = None, port: int = 8080) -> bool:
    """Make OpenCode's Veystrix provider list every family alias (``veystrix-<family>``), so the
    branded UI can pin a session to whichever model the engine serves. The router resolves an
    alias to the single installation of that family. Existing entries are kept; the user's
    default ``model`` is not touched. Writes (backup + atomic) only when something is missing.
    Returns True when the file was changed."""
    from veystrix import model_identity
    from veystrix.model_registry import InstalledModel
    path = config_path("opencode", home)
    current = json.loads(path.read_text()) if path.exists() else {}
    provider = ((current.get("providers") or {}).get("veystrix") or {}) if isinstance(current, dict) else {}
    have = provider.get("models") if isinstance(provider.get("models"), dict) else {}
    aliases = [i for i in model_identity.IDENTITIES.values() if i.served_id]
    base = ((provider.get("settings") or {}).get("baseURL") if isinstance(provider.get("settings"), dict) else None)
    if all(i.served_id in have for i in aliases) and base == base_url(port):
        return False
    models = [InstalledModel(i.served_id, i.family, i.display_name, "(family alias)",
                             i.model_type or "qwen4_exp", {}, i.context_limit, "alias") for i in aliases]
    result = opencode_config(current, models, port)
    merged = dict(have)
    merged.update(result["providers"]["veystrix"]["models"])
    result["providers"]["veystrix"]["models"] = merged
    _write(path, _dump_json(result))
    return True


def sync_hermes_model(home: Path | None, port: int, alias: str) -> bool:
    """Point a connected Hermes profile at the model the engine serves (``veystrix-<family>``),
    listing every family alias so a later /model switch needs no reconnect. Writes (with a
    backup) only when the file would change. Returns True when it was rewritten."""
    from veystrix import model_identity
    from veystrix.model_registry import InstalledModel
    path = config_path("hermes", home)
    if not is_connected("hermes", home):
        return False
    idents = sorted((i for i in model_identity.IDENTITIES.values() if i.served_id),
                    key=lambda i: i.served_id != alias)
    models = [InstalledModel(i.served_id, i.family, i.display_name, "(family alias)",
                             i.model_type or "qwen4_exp", {}, i.context_limit, "alias") for i in idents]
    _path, content, _label = _configure_hermes(home or home_dir(), port, models, "local")
    if path.read_text() == content:
        return False
    _write(path, content)
    return True


def run_environment(name: str, port: int, model_id: str, key: str = "local", context: int | None = None) -> dict:
    """Environment for `veystrix run`: endpoint only, no config file writes.

    ``context`` is the running server's window; Claude Code gets it as CLAUDE_CODE_MAX_CONTEXT_TOKENS (otherwise it
    assumes 200k tokens for an id it does not recognise)."""
    url = anthropic_base_url(port) if name == "claude-code" else base_url(port)
    if name == "claude-code":
        env = {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_MODEL": model_id,
               "ANTHROPIC_DEFAULT_OPUS_MODEL": model_id, "ANTHROPIC_DEFAULT_SONNET_MODEL": model_id,
               "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_id, "CLAUDE_CODE_SUBAGENT_MODEL": model_id}
        if isinstance(context, int) and not isinstance(context, bool) and context > 0:
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(context)
        return env
    if name == "codex":
        return {"OPENAI_API_KEY": key, "VEYSTRIX_API_KEY": key, "OPENAI_BASE_URL": url}
    if name in ("opencode", "hermes", "pi"):
        return {"VEYSTRIX_BASE_URL": url, "VEYSTRIX_API_KEY": key}
    if name == "deepseek":
        return {"OPENAI_API_KEY": key, "OPENAI_BASE_URL": url,
                "VEYSTRIX_API_KEY": key, "VEYSTRIX_BASE_URL": url,
                "DSH_PROFILE": "headless"}
    if name in ("aider", "cline", "continue"):
        return {"OPENAI_API_BASE": url, "OPENAI_BASE_URL": url, "OPENAI_API_KEY": key,
                "VEYSTRIX_API_KEY": key}
    return {"OPENAI_API_KEY": key, "OPENAI_BASE_URL": url, "OPENAI_MODEL": model_id,
            "VEYSTRIX_API_KEY": key, "VEYSTRIX_BASE_URL": url}


def run_command(name: str, model_id: str) -> list[str]:
    binaries = {
        "opencode": ["opencode"], "claude-code": ["claude"], "codex": ["codex"],
        "qwen-code": ["qwen"], "deepseek": ["dsh"], "kimi": ["kimi"],
        # hermes only sees the provider that `connect` wrote through its `veystrix` profile; pi would
        # otherwise start on the first model its extension discovers, which can be the one that is not loaded.
        "hermes": ["hermes", "-p", "veystrix"], "pi": ["pi", "--model", f"veystrix/{model_id}"],
        "aider": ["aider", "--model", "openai/" + model_id],
    }
    if name in binaries:
        return list(binaries[name])
    raise ValueError(f"{name} has no documented standalone CLI launch; use the generated config snippet")
