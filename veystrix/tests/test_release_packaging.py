"""Offline release packaging and security scanner tests.

Covers secret/exposure scanners, engine-bundle import closure, and leak-scan hardening.
No network, no signing keys, no model.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ScannerTests(unittest.TestCase):
    def git_repo(self, tmp: str, files: dict[str, str]) -> str:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        for name, text in files.items():
            p = Path(tmp) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        subprocess.run(["git", "-C", tmp, "add", "-A"], check=True)
        subprocess.run(["git", "-C", tmp, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "t"], check=True)
        return tmp

    def test_secret_scan_flags_a_token_and_passes_clean_text(self):
        scan = ROOT / "scripts" / "security" / "secret_scan.py"
        with tempfile.TemporaryDirectory() as dirty, tempfile.TemporaryDirectory() as clean:
            self.git_repo(dirty, {"a.txt": "token = " + "ghp_" + "A" * 36})
            self.git_repo(clean, {"a.txt": "nothing to see"})
            self.assertEqual(subprocess.run([sys.executable, str(scan), "--repo", dirty], capture_output=True).returncode, 1)
            self.assertEqual(subprocess.run([sys.executable, str(scan), "--repo", dirty, "--history"], capture_output=True).returncode, 1)
            self.assertEqual(subprocess.run([sys.executable, str(scan), "--repo", clean, "--history"], capture_output=True).returncode, 0)

    def test_exposure_audit_detects_shared_source_and_private_paths(self):
        audit = ROOT / "scripts" / "security" / "public_exposure_audit.py"
        with tempfile.TemporaryDirectory() as core, tempfile.TemporaryDirectory() as good, tempfile.TemporaryDirectory() as bad:
            self.git_repo(core, {"engine.py": "print('secret engine')\n", "LICENSE": "MIT text\n"})
            self.git_repo(good, {"README.md": "hello\n", "LICENSE": "MIT text\n"})
            self.git_repo(bad, {"engine.py": "print('secret engine')\n", "notes.md": "see /Users/someone/work\n"})
            ok = subprocess.run([sys.executable, str(audit), "--public", good, "--core", core], capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            nope = subprocess.run([sys.executable, str(audit), "--public", bad, "--core", core], capture_output=True, text=True)
            self.assertEqual(nope.returncode, 1)
            self.assertIn("identical to private-repo content", nope.stdout)
            self.assertIn("private paths", nope.stdout)


class EngineBundleClosureTests(unittest.TestCase):
    """The bundle script removes every ``tests`` and ``bench`` directory and copies a named few files back.

    An engine module that imports a removed module at load makes the packaged engine unable to start a
    model while every source-tree test still passes. Found 2026-11-14: ``veystrix/bench`` was removed
    although ``expert_cache`` and both backends import ``veystrix.bench.telemetry`` at load.
    """

    SCRIPT = ROOT / "packaging" / "engine" / "build_engine_bundle.sh"
    PACKAGES = ("veystrix", "models/flashnext", "models/nemotron35", "models/shared")

    def shipped(self):
        import re

        kept = set()
        for folder, names in re.findall(r'cp "\$ROOT/([\w/]+)/"\{([^}]+)\} "\$OUT/lib/', self.SCRIPT.read_text()):
            kept.update(f"{folder}/{name}" for name in names.split(","))
        for single in re.findall(r'cp "\$ROOT/([\w/.]+\.py)" "\$OUT/lib/', self.SCRIPT.read_text()):
            kept.add(single)
        return kept

    def engine_modules(self):
        for package in self.PACKAGES:
            for path in sorted((ROOT / package).rglob("*.py")):
                relative = path.relative_to(ROOT)
                pruned = {"tests", "bench"} & set(relative.parts[:-1])
                if not pruned and not (relative.parts[:2] == ("models", "nemotron35") and path.name.startswith("test_")):
                    yield relative, path
        for kept in self.shipped():
            yield Path(kept), ROOT / kept

    def test_copied_back_files_exist(self):
        kept = self.shipped()
        self.assertIn("veystrix/bench/telemetry.py", kept)
        for name in kept:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_load_time_imports_of_shipped_modules_stay_inside_the_bundle(self):
        import ast

        kept = self.shipped()
        missing = []
        for relative, path in self.engine_modules():
            package = list(relative.parts[:-1])
            for node in ast.parse(path.read_text()).body:          # module level only: lazy imports are optional paths
                if isinstance(node, ast.ImportFrom):
                    base = package[:len(package) - node.level + 1] if node.level else []
                    targets = [".".join(base + (node.module.split(".") if node.module else []))]
                    targets += [targets[0] + "." + alias.name for alias in node.names]
                elif isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                else:
                    continue
                for target in targets:
                    parts = target.split(".")
                    if parts[0] not in ("veystrix", "models") or not {"tests", "bench"} & set(parts):
                        continue
                    as_file, as_package = "/".join(parts) + ".py", "/".join(parts) + "/__init__.py"
                    exists = (ROOT / as_file).is_file() or (ROOT / as_package).is_file()
                    if exists and as_file not in kept and as_package not in kept:
                        missing.append(f"{relative} imports {target} at load, which the bundle removes")
        self.assertEqual(missing, [])


class LeakScanHardeningTests(unittest.TestCase):
    SCAN = ROOT / "packaging" / "engine" / "leak_scan.py"
    CYTHON_LIST = ROOT / "packaging" / "engine" / "cython_modules.txt"

    def test_rejects_uncompiled_protected_pyc_and_binary_private_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            (bundle / "engine" / "lib" / "veystrix" / "tool").mkdir(parents=True)
            # Protected module shipped as .pyc must fail under --require-compiled
            (bundle / "engine" / "lib" / "veystrix" / "recipe.pyc").write_bytes(b"pyc-bytes")
            (bundle / "engine" / "lib" / "veystrix" / "tool" / "tui.pyc").write_bytes(b"pyc-bytes")
            # Binary >2 MB with embedded /Users/ path must fail
            (bundle / "engine" / "lib" / "fake.dylib").write_bytes(b"\x00" * 2_100_000 + b"/Users/alice/secret")
            res = subprocess.run(
                [sys.executable, str(self.SCAN), "--bundle", str(bundle), "--require-compiled"],
                capture_output=True, text=True,
            )
            self.assertEqual(res.returncode, 1)
            self.assertIn("recipe.pyc: uncompiled bytecode for protected module", res.stderr)
            self.assertIn("tui.pyc: uncompiled bytecode for protected module", res.stderr)
            self.assertIn("fake.dylib: contains /Users/ private path", res.stderr)

    def test_accepts_compiled_so_and_hardened_libveystrix_c(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            (bundle / "engine" / "lib" / "veystrix").mkdir(parents=True)
            (bundle / "engine" / "lib" / "veystrix" / "recipe.cpython-312-darwin.so").write_bytes(b"compiled")
            dylib = ROOT / "cpp" / "engine" / "build" / "libveystrix_c.dylib"
            if dylib.is_file():
                import shutil as _shutil
                dst = bundle / "engine" / "lib" / "libveystrix_c.dylib"
                _shutil.copy2(dylib, dst)
                # Simulate release packaging rpath rewrite (dev BUILD_RPATH -> @loader_path)
                if _shutil.which("otool") and _shutil.which("install_name_tool"):
                    out = subprocess.run(["otool", "-l", str(dst)], capture_output=True, text=True).stdout
                    for line in out.splitlines():
                        line = line.strip()
                        if line.startswith("path /Users/"):
                            rpath = line.split()[1]
                            subprocess.run(["install_name_tool", "-delete_rpath", rpath, str(dst)], check=True)
            res = subprocess.run(
                [sys.executable, str(self.SCAN), "--bundle", str(bundle), "--require-compiled"],
                capture_output=True, text=True,
            )
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)


if __name__ == "__main__":
    unittest.main()
