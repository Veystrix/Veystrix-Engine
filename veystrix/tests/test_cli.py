from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import io
import json

from veystrix import cli


class LauncherTests(unittest.TestCase):
    def test_help_and_version_do_not_require_checkpoint(self):
        for flag in ("--help", "-h", "--version", "-V"):
            with patch.object(cli.sys, "argv", ["veystrix", flag]), \
                 patch("veystrix.cli.installed_checkpoints",
                       side_effect=AssertionError("no discovery on --help")), \
                 patch("veystrix.cli.os.execvpe",
                       side_effect=AssertionError("no exec on --help")):
                out = io.StringIO()
                with patch("sys.stdout", out):
                    self.assertEqual(cli.main(), 0)
                self.assertIn("veystrix", out.getvalue())

    def test_main_probe_does_not_misparse_mode_as_model(self):
        args = ["veystrix", "--mode", "agentic", "--model", "inkling2-mini", "--checkpoint", "/models/test"]
        with patch.object(cli.sys, "argv", args), patch("veystrix.cli.branch_sync_warning", return_value=""), \
             patch("veystrix.cli._default_checkpoint_args", side_effect=lambda x: x), \
             patch("veystrix.cli.command", return_value=(["python", "-m", "test"], {})) as command, \
             patch("veystrix.cli.os.execvpe") as run:
            self.assertEqual(cli.main(), 0)
        command.assert_called_once_with(args[1:])
        run.assert_called_once()

    def test_default_launch_uses_the_only_installed_checkpoint(self):
        checkpoint = Path("/models/only")
        with patch("veystrix.cli.installed_checkpoints", return_value=[
            ("inkling2-mini", checkpoint),
        ]):
            self.assertEqual(
                cli._default_checkpoint_args(["--profile", "plain"]),
                [
                    "--profile", "plain", "--model", "inkling2-mini",
                    "--checkpoint", str(checkpoint),
                ],
            )

    def test_default_launch_asks_when_multiple_checkpoints_exist(self):
        choices = [
            ("inkling2-mini", Path("/models/one")),
            ("inkling2-mini", Path("/models/two")),
        ]
        with patch("veystrix.cli.installed_checkpoints", return_value=choices), \
                patch.object(cli.sys.stdin, "isatty", return_value=True), \
                patch("builtins.input", return_value="2"):
            selected = cli._default_checkpoint_args([])
        self.assertEqual(
            selected,
            ["--model", "inkling2-mini", "--checkpoint", "/models/two"],
        )

    def test_first_launch_prepares_the_managed_environment_and_retries(self):
        with patch("veystrix.cli._supports_python", side_effect=(False, True)), \
                patch("veystrix.cli.setup_environment") as setup:
            self.assertEqual(cli._interpreter("inkling2-mini"), cli.MANAGED_PYTHON)
        setup.assert_called_once_with(["--venv", str(cli.MANAGED_ENV)])

    def test_repeated_launch_reuses_the_managed_environment(self):
        with patch("veystrix.cli._supports_python", return_value=True), \
                patch("veystrix.cli.setup_environment") as setup:
            self.assertEqual(cli._interpreter("inkling2-mini"), cli.MANAGED_PYTHON)
        setup.assert_not_called()

    def test_explicit_override_is_rejected_when_its_runtime_is_wrong(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root, "python")
            python.touch()
            with patch.dict(
                os.environ, {"VEYSTRIX_PYTHON": str(python)}, clear=False
            ), patch("veystrix.cli._supports_python", return_value=False), \
                    patch("veystrix.cli._python_runtime_error", return_value="mlx 0.31"):
                with self.assertRaisesRegex(SystemExit, "VEYSTRIX_PYTHON.*mlx 0.31"):
                    cli._interpreter("inkling2-mini")

    def test_packaged_runtime_skips_git_probe(self):
        with tempfile.TemporaryDirectory() as root, patch("veystrix.cli.subprocess.run") as run:
            self.assertEqual(cli.branch_sync_warning(Path(root)), "")
            run.assert_not_called()

    def test_warns_when_branch_does_not_include_known_main(self):
        stale = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        branch = subprocess.CompletedProcess(
            [], 0, stdout="feature/test\n", stderr=""
        )
        with tempfile.TemporaryDirectory() as root, patch("veystrix.cli.subprocess.run", side_effect=(stale, branch)):
            (Path(root) / ".git").mkdir()
            warning = cli.branch_sync_warning(Path(root))
        self.assertIn("feature/test", warning)
        self.assertIn("origin/main", warning)

    def test_skips_warning_when_branch_includes_known_main(self):
        current = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as root, patch("veystrix.cli.subprocess.run", return_value=current) as run:
            (Path(root) / ".git").mkdir()
            warning = cli.branch_sync_warning(Path(root))
        self.assertEqual(warning, "")
        run.assert_called_once()

    def test_setup_installs_the_shared_runtime_in_a_local_environment(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root, ".venv")
            with patch("veystrix.cli.subprocess.check_call") as check_call:
                cli.setup_environment(["--venv", str(target)])
        commands = check_call.call_args_list
        self.assertEqual(commands[0].args[0][-2:], ["venv", str(target.resolve())])
        self.assertEqual(commands[-1].args[0][-1], str(cli.ROOT))

    def test_removed_positional_model_argument_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "unexpected argument"):
            cli.command(["qwen27b"])

    def test_removed_models_are_not_selectable(self):
        with patch("sys.stderr", new_callable=io.StringIO):
            for name in ("flashnext", "nemotron35-lightning", "ornith", "bonsai2", "k2-horizon", "qwen27b"):
                with self.subTest(model=name), self.assertRaises(SystemExit):
                    cli.command(["--model", name])


if __name__ == "__main__":
    unittest.main()
