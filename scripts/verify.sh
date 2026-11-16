#!/usr/bin/env bash
# One-command verification. Mirrors CI as closely as the local toolchain allows.
# Usage: ./scripts/verify.sh [--fast | --python-only | --artifacts]
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
mode=${1:-full}

run() { printf '\n== %s ==\n' "$1"; shift; "$@"; }

case "$mode" in
  --fast)
    run "Shared tests without MLX" "$PY" scripts/ci/linux_unit_tests.py
    run "Secret scan" "$PY" scripts/security/secret_scan.py
    ;;
  --artifacts)
    run "Built artifact guardrails" "$PY" scripts/verify_artifacts.py
    ;;
  --python-only)
    run "Shared runtime tests" "$PY" -m unittest discover -s veystrix/tests -t . -p 'test_*.py' -q
    run "Flash-Next runtime tests" "$PY" -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py' -q
    run "Nemotron runtime tests" "$PY" -m unittest discover -s models/nemotron35/tests/unit -t . -p 'test_*.py' -q
    run "Secret scan" "$PY" scripts/security/secret_scan.py
    ;;
  full)
    run "Shared runtime tests" "$PY" -m unittest discover -s veystrix/tests -t . -p 'test_*.py' -q
    run "Flash-Next runtime tests" "$PY" -m unittest discover -s models/flashnext/tests/unit -t . -p 'test_*.py' -q
    run "Nemotron runtime tests" "$PY" -m unittest discover -s models/nemotron35/tests/unit -t . -p 'test_*.py' -q
    run "Secret scan" "$PY" scripts/security/secret_scan.py
    run "macOS app build" swift build --package-path apps/macos/Veystrix
    run "macOS app logic tests" swift test --package-path apps/macos/Veystrix
    ;;
  *)
    echo "usage: $0 [--fast | --python-only | --artifacts]" >&2
    exit 2
    ;;
esac

printf '\nAll verification steps passed.\n'
