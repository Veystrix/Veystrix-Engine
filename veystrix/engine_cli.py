"""Engine-side CLI contract for the front end (M4).

``veystrix-engine --version --json`` returns the version, protocol version, the engines
present and the cpp_parity map. ``veystrix-engine serve …`` starts the Python engine on
loopback. This is compiled/obfuscated into the bundle alongside the engine; no CLI source
is public. All engine paths stay behind the --engine selector.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROTOCOL_VERSION = "1"


def project_version() -> str:
    # Sealed bundle: the manifest sits at the engine root (…/engine/lib/veystrix/engine_cli ->
    # …/engine/BUNDLE-MANIFEST.json). Source checkout: fall back to pyproject.toml.
    try:
        import json
        manifest = Path(__file__).resolve().parents[2] / "BUNDLE-MANIFEST.json"
        version = str(json.loads(manifest.read_text()).get("version") or "")
        if version:
            return version
    except (OSError, KeyError, ValueError, TypeError):
        pass
    try:
        import tomllib
        root = Path(__file__).resolve().parents[1]
        return tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    except (OSError, KeyError, ValueError, TypeError):
        return "unknown"


def version_payload(home: Path | None = None) -> dict:
    engines = ["python"]
    cpp_parity: dict = {}
    if home is not None:
        from veystrix.tool.engine import load_manifest
        manifest = load_manifest(home)
        if "cpp" in (manifest.get("engines") or []):
            engines.append("cpp")
        if isinstance(manifest.get("cpp_parity"), dict):
            cpp_parity = manifest["cpp_parity"]
    return {
        "version": project_version(),
        "protocol_version": PROTOCOL_VERSION,
        "engines": engines,
        "cpp_parity": cpp_parity,
    }


def serve(argv: list[str]) -> int:
    # The front end launches the installed bundle as
    #   engine/bin/veystrix-engine serve --port ... --model ... --engine python|cpp
    # ``--engine`` is a front-end/engine selector, not a model_router flag (the
    # router's argparse is strict). Consume it here and mirror it into the
    # environment, which is what actually selects CppEngineBackend through
    # veystrix.session.install_cpp_selector(). The default python path is
    # unaffected: no flag, no environment change.
    import os
    argv = list(argv)
    engine = None
    # Support both "--engine cpp" and "--engine=cpp" forms; reject unknown
    # values against the selector's choices rather than passing a bogus value.
    kept: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--engine":
            if i + 1 >= len(argv):
                raise SystemExit("veystrix-engine: --engine needs a value")
            engine = str(argv[i + 1]).strip().lower()
            i += 2
            continue
        if arg.startswith("--engine="):
            engine = arg.split("=", 1)[1].strip().lower()
            i += 1
            continue
        kept.append(arg)
        i += 1
    if engine is not None:
        from veystrix.tool.engine import ENGINE_CHOICES
        if engine not in ENGINE_CHOICES:
            raise SystemExit(f"veystrix-engine: unknown engine {engine!r}; use {', '.join(ENGINE_CHOICES)}")
        os.environ["VEYSTRIX_ENGINE"] = engine
    from veystrix.model_router import main as router_main
    return router_main(["--host", "127.0.0.1", *kept])


def _split_serve(argv: list[str]) -> tuple[list[str], list[str]] | None:
    """Split ``[global opts] serve [router args]`` without letting argparse see the router args.

    argparse.REMAINDER does not capture a remainder whose first token is an option
    (``serve --port 1234`` fails with "unrecognized arguments: --port"), so the router
    arguments are cut off here and handed to serve() verbatim.
    """
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "serve":
            return argv[:i], argv[i + 1:]
        if arg == "--home":
            # Consume its value so a home path spelled "serve" is not mistaken
            # for the subcommand. A missing value is left to argparse to report.
            if i + 1 >= len(argv):
                return None
            i += 2
            continue
        if arg.startswith("--home="):
            i += 1
            continue
        if not arg.startswith("-"):
            return None
        i += 1
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = _split_serve(argv)
    if split is not None:
        head, rest = split
        if any(a in ("-h", "--help") for a in head):
            split = None
        else:
            return serve(rest)
    parser = argparse.ArgumentParser(prog="veystrix-engine")
    parser.add_argument("--version", action="store_true", help="print version information")
    parser.add_argument("--json", action="store_true", help="JSON output for --version")
    parser.add_argument("--home", type=Path, default=None, help="tool home used to read the bundle manifest")
    sub = parser.add_subparsers(dest="command")
    serve_parser = sub.add_parser("serve", help="start the engine on loopback")
    serve_parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.version:
        payload = version_payload(args.home)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(" ".join(f"{k}={v}" for k, v in payload.items()))
        return 0
    if args.command == "serve":
        return serve(args.rest)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
