"""``veystrix endpoint``: everything a client needs to talk to the local endpoint.

Read-only and works whether or not the endpoint is running. It prints base URL(s), where the
API key lives (and the key itself when one is required), the exact served model id, the engine
and runtime (python, cpp custom or cpp generic), the model, KV mode, context ceiling, a
copy-paste ``curl`` smoke test, and ready-to-paste connection snippets for every harness in
the matrix (flagging the ones that are not installed). ``--json`` is for scripts.

``veystrix start`` prints the same block (without the harness snippets) once /health is ready.
Probes (health, /v1/models, tailscale, LAN address, PATH lookups) are injectable for tests.
"""
from __future__ import annotations

import json
import shlex
import shutil
import socket
import subprocess
from pathlib import Path

from veystrix import model_identity
from veystrix.tool.config import Config, LocalKeyStore, home_dir
from veystrix.tool.harnesses import (
    DOC, PUBLIC_HARNESS_CHOICES, anthropic_base_url, base_url, run_command, run_environment,
)

MATRIX = PUBLIC_HARNESS_CHOICES


def _memory_tier_name() -> str:
    from veystrix.memory_tiers import current_tier
    return current_tier().name


def _get_json(url: str, key: str | None, timeout: float = 1.0) -> dict | None:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url)
    if key:
        request.add_header("Authorization", "Bearer " + key)
    try:
        with opener.open(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def lan_address() -> str | None:
    """This Mac's primary LAN IPv4 (no packets are sent: UDP connect only picks a route)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            address = sock.getsockname()[0]
            return None if address.startswith("127.") else address
    except OSError:
        return None


def tailscale_url(port: int, runner=None) -> str | None:
    """https://<mac>.<tailnet>.ts.net when Tailscale Serve forwards to this port."""
    binary = shutil.which("tailscale")
    if binary is None and runner is None:
        return None
    runner = runner or (lambda argv: subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False).stdout)
    try:
        status = json.loads(runner([binary or "tailscale", "status", "--json"]) or "{}")
        serve = runner([binary or "tailscale", "serve", "status", "--json"]) or "{}"
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    name = str((status.get("Self") or {}).get("DNSName") or "").rstrip(".")
    if not name or f"127.0.0.1:{port}" not in serve and f"localhost:{port}" not in serve:
        return None
    return f"https://{name}"


def _served_model_id(models: dict | None, family: str, checkpoint: str | None) -> str:
    if isinstance(models, dict):
        for entry in models.get("data") or []:
            if isinstance(entry, dict) and entry.get("id"):
                return str(entry["id"])
    if checkpoint:
        from veystrix.model_registry import installation_id
        return installation_id(family, checkpoint)
    return model_identity.served_id(family) or f"veystrix-{family}"


def _running_family(health: dict | None) -> str | None:
    """The family a running server serves, from its /health alias or installation id (not the config default)."""
    if not isinstance(health, dict):
        return None
    alias = str(health.get("model") or "")
    installed = str(health.get("installed_model_id") or "")
    for family, identity in model_identity.IDENTITIES.items():
        if identity.served_id and (alias == identity.served_id or installed.startswith(identity.served_id + "-")):
            return family
    return None


def _checkpoint_for(family: str) -> str | None:
    from veystrix import checkpoints
    if family not in model_identity.IDENTITIES:
        return None
    try:
        return str(checkpoints.resolve_for_family(family, None))
    except Exception:
        return None


def describe(home: Path | None = None, *, port: int | None = None, probe=None, which=None,
             lan=None, tailscale=None) -> dict:
    """The endpoint facts as a dict (``veystrix endpoint --json``)."""
    from veystrix.tool.supervisor import _read_state
    home = home or home_dir()
    config = Config(home)
    state = _read_state(home) or {}
    port = port or state.get("port") or config.get("port")
    key = LocalKeyStore(home).load()
    probe = probe or (lambda path: _get_json(f"http://127.0.0.1:{port}{path}", key))
    which = which or shutil.which
    health = probe("/health")
    running = health is not None
    family = str((health or {}).get("model_family") or _running_family(health)
                 or state.get("model") or config.get("model"))
    identity = model_identity.for_family(family)
    models = probe("/v1/models") if running else None
    checkpoint = None if running else _checkpoint_for(family)
    served = (str(health["installed_model_id"]) if health and health.get("installed_model_id")
              else _served_model_id(models, family, checkpoint))
    runtime = (health or {}).get("runtime") or {}
    engine = (health or {}).get("engine") or state.get("engine") or config.get("engine")
    if engine == "python" and "python" not in identity.engines:
        engine = identity.engines[0]
    runtime_name = runtime.get("runtime") or ("python" if engine == "python" else identity.cpp_runtime)
    kv = runtime.get("kv") or {}
    kv_mode = kv.get("mode") or state.get("kv") or config.get("kv")
    if not kv and identity.cpp_runtime != "generic":
        kv_mode = "engine default (KV policy applies to the generic runtime)"
    host = state.get("host") or config.get("host")
    urls = {"local": base_url(port), "local_anthropic": anthropic_base_url(port)}
    if host == "0.0.0.0":
        address = (lan or lan_address)()
        if address:
            urls["lan"] = f"http://{address}:{port}/v1"
            urls["lan_anthropic"] = f"http://{address}:{port}"
    remote = (tailscale or (lambda: tailscale_url(port)))()
    if remote:
        urls["tailscale"] = remote + "/v1"
        urls["tailscale_anthropic"] = remote
    required = bool(config.get("require_api_key"))
    api_key = key if (required and key) else "local"
    context = (health or {}).get("context_limit") or state.get("context") or config.get("context") \
        or identity.context_limit
    info = {
        "running": running,
        "ready": bool(health and health.get("ready", True)),
        "port": port, "host": host, "urls": urls,
        "api_key": api_key,
        "api_key_required": required,
        "api_key_location": ("macOS Keychain (veystrix config get api-key)" if key
                             else "none set: any value works"),
        "model": family, "display_name": identity.display_name, "served_model_id": served,
        "alias": identity.served_id,
        "engine": engine, "runtime": runtime_name,
        "kv_mode": kv_mode, "kv_plan": kv.get("plan") or runtime.get("kv_plan"),
        "kv_bytes_per_token": runtime.get("kv_bytes_per_token"),
        "context_ceiling": context, "context_in_use": runtime.get("context_in_use"),
        "memory_tier": _memory_tier_name(),
        "reasoning": identity.reasoning, "tool_format": identity.tool_format,
        "sampling_defaults": model_identity.sampling_defaults(family),
    }
    info["curl"] = smoke_curl(urls["local"], api_key, served)
    info["harnesses"] = harness_snippets(port, served, api_key, which, context=context if running else None)
    info["generic_env"] = {
        "openai": {"OPENAI_BASE_URL": urls["local"], "OPENAI_API_KEY": api_key, "OPENAI_MODEL": served},
        "anthropic": {"ANTHROPIC_BASE_URL": urls["local_anthropic"], "ANTHROPIC_API_KEY": api_key,
                      "ANTHROPIC_MODEL": served},
    }
    return info


def smoke_curl(url: str, key: str, model: str) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "Say hello."}],
                       "max_tokens": 64, "stream": False})
    return (f"curl -s {url}/chat/completions -H 'Content-Type: application/json' "
            f"-H 'Authorization: Bearer {key}' -d {shlex.quote(body)}")


def harness_snippets(port: int, model_id: str, key: str, which=None, context: int | None = None) -> dict:
    which = which or shutil.which
    out = {}
    for name in MATRIX:
        env = run_environment(name, port, model_id, key, context=context)
        try:
            command = run_command(name, model_id)
        except ValueError:
            command = None
        installed = bool(command and which(command[0]))
        exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
        out[name] = {
            "installed": installed,
            "configure_once": f"veystrix connect {name}",
            "launch_prewired": f"veystrix run {name}",
            "direct": (f"{exports} {shlex.join(command)}" if command else None),
            "environment": env,
            "doc": DOC[name][0],
        }
    return out


def render(info: dict, *, harnesses: bool = True) -> str:
    urls = info["urls"]
    state = ("ready" if info["ready"] else "starting") if info["running"] else "not running"
    lines = [f"Veystrix endpoint ({state})",
             f"  OpenAI base URL:     {urls['local']}",
             f"  Anthropic base URL:  {urls['local_anthropic']}"]
    if "lan" in urls:
        lines.append(f"  LAN:                 {urls['lan']}  (Anthropic: {urls['lan_anthropic']})")
    if "tailscale" in urls:
        lines.append(f"  Tailscale:           {urls['tailscale']}  (Anthropic: {urls['tailscale_anthropic']})")
    lines += [
        f"  API key:             {info['api_key']}"
        + ("  (required)" if info["api_key_required"] else "  (not required; any value works)")
        + f"  [{info['api_key_location']}]",
        f"  model id:            {info['served_model_id']}   (alias {info['alias']})",
        f"  model:               {info['display_name']}",
        f"  KV:                  {info['kv_mode']}"
        + (f"  ({info['kv_bytes_per_token']:,} B/token)" if info.get("kv_bytes_per_token") else ""),
        f"  context ceiling:     {info['context_ceiling']:,}"
        + (f"  (in use {info['context_in_use']:,})" if info.get("context_in_use") else ""),
        f"  memory tier:         {info.get('memory_tier') or _memory_tier_name()}",
    ]
    if info.get("sampling_defaults"):
        s = info["sampling_defaults"]
        lines.append("  sampling defaults:   " + ", ".join(f"{k} {v:g}" for k, v in s.items()))
    if info.get("reasoning"):
        lines.append("  reasoning:           <think>...</think> is returned as reasoning content, "
                     f"tool calls parsed ({info['tool_format']})")
    lines += ["", "  smoke test:", "    " + info["curl"]]
    if not info["running"]:
        lines += ["", "  start it with: veystrix start" + (f" --model {info['model']}" if info["model"] else "")]
    if harnesses:
        lines += ["", "Harnesses (configure once with `veystrix connect NAME`, then run the harness yourself;",
                  "or `veystrix run NAME` launches it pre-wired; neither needs the TUI):"]
        for name, item in info["harnesses"].items():
            flag = "installed" if item["installed"] else "NOT INSTALLED"
            lines.append(f"  {name:<12} [{flag}]  {item['configure_once']}  |  {item['launch_prewired']}")
            if item["direct"]:
                lines.append(f"      direct: {item['direct']}")
        g = info["generic_env"]
        lines += ["", "  any OpenAI client:",
                  "    " + " ".join(f"export {k}={shlex.quote(str(v))};" for k, v in g["openai"].items()),
                  "  any Anthropic client:",
                  "    " + " ".join(f"export {k}={shlex.quote(str(v))};" for k, v in g["anthropic"].items())]
    return "\n".join(lines)


def main(argv: list[str], home: Path | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="veystrix endpoint")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    info = describe(home, port=args.port)
    print(json.dumps(info, indent=2, sort_keys=True) if args.json else render(info))
    return 0
