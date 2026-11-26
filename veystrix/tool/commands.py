"""Offline front-end commands: models, config, cache, doctor, diagnostics."""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from veystrix.tool import engine as engine_mod
from veystrix.tool.config import Config, LocalKeyStore, home_dir
from veystrix.tool.sanitizer import clean
from veystrix.tool.supervisor import health, log_path, logs, memory_state, status


def _json(data) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))


def cmd_install(argv: list[str], home: Path, config: Config, *, apply: bool = False) -> int:
    """Fetch and verify a release bundle into ~/.veystrix. No network in dry-run."""
    from veystrix.tool import engine as engine_mod

    version = ""
    if argv and not argv[0].startswith("--"):
        version = argv[0]
    asset = f"veystrix-engine-{version or 'latest'}-arm64.tar.zst"
    base = "https://github.com/Veystrix/veystrix-engine/releases/"
    tag = "v" + version.lstrip("v") if version else ""
    url = base + (f"download/{tag}/" if tag else "latest/download/") + asset
    plan = {"url": url, "artifact": asset, "target": str(engine_mod.bundle_root(home)),
            "verify": ["minisign -V -p tools/minisign.pub -m SHA256SUMS.txt -x SHA256SUMS.txt.minisig",
                       "shasum -a 256 -c SHA256SUMS.txt"], "applied": False}
    if not apply:
        plan["result"] = "dry-run (pass --apply after the owner's go)"
        _json(plan)
        return 0
    if shutil.which("minisign") is None:
        raise SystemExit("veystrix: minisign is required to verify an install; see public tools/README.md")
    raise SystemExit("veystrix: install requires an explicit owner-approved download; "
                     "use install.sh from the public repo after the owner's go")


def cmd_uninstall(argv: list[str], home: Path, *, apply: bool = False) -> int:
    """Remove tool/engine binaries but keep model folders and the cache."""
    base = home / ".veystrix"
    targets = [base / "bin", base / "engine"]
    plan = {"remove": [str(path) for path in targets],
            "keep": [str(home / "models"), str(home / ".cache" / "veystrix")],
            "applied": False}
    if not apply:
        plan["result"] = "dry-run"
        _json(plan)
        return 0
    for path in targets:
        shutil.rmtree(path, ignore_errors=True)
    plan["applied"] = True
    _json(plan)
    return 0


def cmd_models(argv: list[str], home: Path, config: Config) -> int:
    """Discover, validate, inspect and port-check checkpoints. Only `models pull` downloads, and only when asked."""
    from veystrix import checkpoints, model_registry

    root = Path(os.environ.get("VEYSTRIX_MODEL_ROOT", "~/models")).expanduser()
    if argv and argv[0] == "pull":
        from veystrix.model_pull import PullError, pull
        names = [a for a in argv[1:] if not a.startswith("--")]
        if len(names) != 1:
            raise SystemExit("veystrix: models pull <model> [--dry-run] [--json]   (e.g. models pull inkling2-mini)")
        try:
            report = pull(names[0], root, dry_run="--dry-run" in argv,
                          echo=lambda m: print("veystrix: " + m, file=sys.stderr))
        except PullError as exc:
            raise SystemExit(f"veystrix: {exc}") from exc
        if "--json" in argv:
            _json(report)
        else:
            inspect_text = report.pop("inspect", None)
            print(", ".join(f"{k}={v}" for k, v in report.items()))
            if inspect_text:
                print(inspect_text)
        return 0
    if argv and argv[0] == "port-check":
        from veystrix import port_check
        return port_check.main(argv[1:])
    if argv and argv[0] == "inspect":
        from veystrix import model_inspect
        if len(argv) < 2:
            raise SystemExit("veystrix: models inspect <directory> [--json] [--emit-recipe [FILE]]")
        if "--emit-recipe" in argv:
            from veystrix.recipe import RecipeError, emit_recipe
            at = argv.index("--emit-recipe")
            out = argv[at + 1] if at + 1 < len(argv) and not argv[at + 1].startswith("--") else None
            try:
                recipe = emit_recipe(Path(argv[1]).expanduser())
            except RecipeError as exc:
                raise SystemExit(f"veystrix: cannot build a recipe: {exc}") from exc
            text = json.dumps(recipe, indent=1, sort_keys=True)
            if out:
                Path(out).expanduser().write_text(text + "\n")
                print(f"veystrix: recipe written to {out}")
            else:
                print(text)
            return 0
        report = model_inspect.inspect(argv[1])
        if "--json" in argv:
            _json(report.as_dict())
        else:
            print(model_inspect.render(report))
        return 0
    if argv and argv[0] == "add":
        if len(argv) < 2:
            raise SystemExit("veystrix: models add <directory>")
        path = Path(argv[1]).expanduser()
        if not path.is_dir():
            raise SystemExit(f"veystrix: no such checkpoint folder: {path}")
        family = None
        notes: list[str] = []
        from veystrix import model_identity as _mi
        for fam, ident in _mi.IDENTITIES.items():
            if checkpoints.generic_compatible(path, ident):
                family = fam
                break
        if not family:
            raise SystemExit(f"veystrix: {path} is not a complete supported checkpoint (nothing was downloaded).")
        if _mi.is_locked(family):
            raise SystemExit(f"veystrix: {family} is locked and unavailable in this build.")
        _json({"path": str(path.resolve()), "family": family, "usable": True, "notes": notes})
        return 0
    from veystrix import model_identity as _mi
    found = [record.as_dict() for record in model_registry.discover(root) if not _mi.is_locked(record.family)]
    _json({"object": "list", "data": found})
    return 0


def cmd_config(argv: list[str], home: Path, config: Config, keystore: LocalKeyStore) -> int:
    if not argv or argv[0] == "get":
        key = argv[1] if len(argv) > 1 else None
        if key in ("api-key", "api_key"):
            _json({"api-key": "set" if keystore.fingerprint() else "unset",
                   "fingerprint": keystore.fingerprint()})
            return 0
        if key is None:
            _json(config.values)
            return 0
        _json({key: config.get(key)})
        return 0
    if argv[0] == "set":
        if len(argv) != 3:
            raise SystemExit("veystrix: config set <key> <value>")
        key, value = argv[1], argv[2]
        if key in ("api-key", "api_key"):
            if value == "generate":
                key_value, where = keystore.rotate()
                _json({"api-key": "generated", "storage": where, "fingerprint": keystore.fingerprint()})
            else:
                where = keystore.store(value)
                _json({"api-key": "set", "storage": where, "fingerprint": keystore.fingerprint()})
            return 0
        _json({key: config.set(key, value)})
        return 0
    raise SystemExit("veystrix: config get [<key>] | config set <key> <value>")


def cmd_cache(argv: list[str], home: Path, config: Config) -> int:
    """Show usage/quota and purge via the existing cache manager (no model load)."""
    action = argv[0] if argv else "status"
    allowed = ("status", "preview", "reclaim", "clear", "quota", "compact", "artifacts")
    if action not in allowed:
        raise SystemExit(f"veystrix: cache action must be one of {', '.join(allowed)}")
    command = [sys.executable, "-m", "veystrix.universal_cache"]
    directory = config.get("cache_directory")
    # Tests set VEYSTRIX_HOME: keep every cache operation inside that home.
    if not directory and os.environ.get("VEYSTRIX_HOME"):
        directory = str(home / ".cache" / "veystrix")
    if directory:
        command += ["--directory", directory]
    command.append(action)
    if len(argv) > 1:
        command.append(argv[1])
    env = dict(os.environ)
    env["VEYSTRIX_CACHE_GB"] = str(config.get("cache_quota_gb"))
    completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[2]),
                               env=env, capture_output=True, text=True)
    if completed.returncode:
        raise SystemExit("veystrix: cache command failed: " + (completed.stderr.strip() or "unknown error"))
    print(completed.stdout.strip())
    return 0


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def cmd_doctor(argv: list[str], home: Path, config: Config, selected_engine: str) -> int:
    from veystrix import model_registry

    rows = []

    def row(name, ok, detail, label="MEASURED"):
        rows.append({"check": name, "status": "ok" if ok else "warn", "detail": detail, "label": label})

    row("python", sys.version_info >= (3, 12), platform.python_version(), "MEASURED")
    mac_version = platform.mac_ver()[0]
    row("macOS >= 26.2 (MLX 0.32.2)", not mac_version or tuple(map(int, mac_version.split(".")[:2])) >= (26, 2),
        mac_version or "not available here", "CITED (brief 2026-11-18)")
    row("Apple silicon", platform.machine() in ("arm64", "aarch64") or not sys.platform == "darwin",
        platform.machine() or "unknown", "MEASURED")
    disk = shutil.disk_usage(home)
    row("free disk >= 20 GB", disk.free >= 20 * 1024**3, f"{disk.free / 1024**3:.1f} GiB", "MEASURED")
    available = memory_state().get("available_bytes")
    row("available memory >= 3 GiB", available is None or available >= 3 * 1024**3,
        f"{available / 1024**3:.1f} GiB" if available else "not readable here", "MEASURED")
    port = config.get("port")
    row("port free", _port_free(port) or health(port) is not None, f"127.0.0.1:{port}", "MEASURED")
    try:
        checkpoints = model_registry.discover()
        row("checkpoint present", bool(checkpoints), f"{len(checkpoints)} installed", "MEASURED")
    except Exception as exc:                                  # never let doctor crash on a damaged checkpoint
        row("checkpoint present", False, str(exc)[:120], "MEASURED")
    if engine_mod.manifest_path(home).is_file():
        problems = engine_mod.bundle_integrity(home)
        row("bundle integrity", not problems, "; ".join(problems) if problems else "manifest and files verified", "MEASURED")
    else:
        row("bundle integrity", True, "no installed bundle (source mode)", "MEASURED")
    report = health(port)
    row("engine selector", True, f"selected={selected_engine}, active={engine_mod.report_engine(report, selected_engine)}", "MEASURED")
    for item in rows:
        print(f"{item['status']:>4}  {item['check']}: {item['detail']} [{item['label']}]")
    # Warnings are advisory, not failures: doctor is a diagnostic, not a gate.
    return 0


RELEASES_URL = "https://api.github.com/repos/Veystrix/veystrix-engine/releases/latest"


def cmd_update(argv: list[str], home: Path, config: Config, *, url: str = RELEASES_URL) -> int:
    """`veystrix update` installs the newest signed build; `veystrix update --check` only reports.
    Nothing runs in the background: updates happen only when the user asks."""
    from veystrix.tool.updater import update
    if argv and argv[0] not in ("check", "--check", "install", "--yes", ""):
        raise SystemExit("veystrix: usage: veystrix update [--check]")
    return update(home, check_only=bool(argv) and argv[0] in ("check", "--check"))


def cmd_diagnostics(argv: list[str], home: Path, config: Config, keystore: LocalKeyStore) -> int:
    """Write a sanitized bundle: no prompts, no home paths, no keys."""
    report = status(home)
    raw = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": dict(config.values),
        "engine": report,
        "key_fingerprint": keystore.fingerprint(),
        "logs": logs(home, 200),
    }
    text = clean(json.dumps(raw, indent=2, sort_keys=True), home=str(home),
                 extra_secrets=[keystore.load()] if keystore.load() else [])
    directory = home / ".veystrix" / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / ("diagnostics-" + time.strftime("%Y%m%dT%H%M%S") + ".txt")
    path.write_text(text + "\n")
    os.chmod(path, 0o600)
    print(str(path))
    return 0
