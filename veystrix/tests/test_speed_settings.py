"""Startup-workspace resilience and measured speed settings."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veystrix import speed as speed_mod
from veystrix.preferences import load as load_preferences
from veystrix.tool.config import Config
from veystrix.tool.supervisor import engine_env_for
from veystrix.tool.tui import slash


class SpeedSettingsTest(unittest.TestCase):
    def test_measured_presets_and_env_precedence(self):
        env = speed_mod.speed_env("max", {}, "inkling2-mini")
        self.assertEqual(env["VEYSTRIX_GENERIC_PIPE"], "0")
        self.assertEqual(env["VEYSTRIX_GENERIC_GATE"], "1")
        self.assertEqual(env["VEYSTRIX_CPP_KEEPWARM"], "1")
        self.assertEqual(env["VEYSTRIX_GENERIC_SPEC_FETCH"], "4")
        self.assertEqual(env["VEYSTRIX_GENERIC_ROW_ORDER"], "1")
        self.assertEqual(env["VEYSTRIX_CPP_MLX_CACHE_BYTES"], str(256 << 20))
        self.assertEqual(env["VEYSTRIX_CPP_RDAHEAD"], "0")
        self.assertEqual(speed_mod.speed_env("recipe", {}, "inkling2-mini"), {})
        custom = speed_mod.speed_env("custom", {"gate_tail": 2}, "inkling2-mini")
        self.assertEqual(custom["VEYSTRIX_CPP_GATE_TAIL"], "2")
        # An operator-set value always wins.
        with_env = speed_mod.speed_env("max", {}, "inkling2-mini",
                                       {"VEYSTRIX_GENERIC_GATE": "0"})
        self.assertNotIn("VEYSTRIX_GENERIC_GATE", with_env)

    def test_supervisor_carries_speed_env_to_the_watcher(self):
        env = {"VEYSTRIX_CPP_GATE_TAIL": "3", "VEYSTRIX_KV_MODE": "int8"}
        carried = engine_env_for("cpp", env)
        self.assertEqual(carried["VEYSTRIX_CPP_GATE_TAIL"], "3")
        self.assertEqual(carried["VEYSTRIX_KV_MODE"], "int8")

    def test_slash_speed_switches_mode_and_knob(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            lines = slash("/speed", home=home, port=1)
            self.assertIn("speed=max", "\n".join(lines))
            slash("/speed custom", home=home, port=1)
            self.assertEqual(Config(home).get("speed_mode"), "custom")
            slash("/speed gate_tail 2", home=home, port=1)
            self.assertEqual(Config(home).get("speed_knobs"), {"gate_tail": 2})
            self.assertIn("gate_tail=2", "\n".join(slash("/speed", home=home, port=1)))


class WorkspaceResilienceTest(unittest.TestCase):
    def test_missing_saved_workspace_falls_back_to_a_live_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preferences.json"
            path.write_text(json.dumps({"workspace": str(Path(tmp) / "deleted-project")}))
            with patch.dict(os.environ, {"VEYSTRIX_HOME": tmp}):
                prefs = load_preferences(path)
            self.assertTrue(Path(prefs["workspace"]).is_dir())

    def test_builtin_adapter_survives_a_missing_cwd(self):
        from veystrix.tool.builtin_adapter import BuiltinAdapter
        adapter = BuiltinAdapter(port=1, cwd="/no/such/directory")
        self.assertTrue(Path(adapter.cwd).is_dir())


if __name__ == "__main__":
    unittest.main()
