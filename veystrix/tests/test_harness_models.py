import unittest
from veystrix.harness_models import qwen_config, deepseek_config, opencode_config, kimi_config, model_context
from veystrix.model_registry import InstalledModel


def m(x, ctx=None):
    return InstalledModel(x, "inkling2-mini", x, "/" + x, "inkling2_loop", {}, ctx, "test")


def unknown_m(x):
    return InstalledModel(x, "unknown-family", x, "/" + x, "unknown", {}, None, "test")


class HarnessConfigTests(unittest.TestCase):
    def test_catalog_uses_native_window_with_a_safe_fallback(self):
        full = m("full", ctx=262144)
        small = m("small", ctx=8192)
        entries = qwen_config({}, [full, small, unknown_m("unknown")])["modelProviders"]["openai"]
        self.assertEqual([x["generationConfig"]["contextWindowSize"] for x in entries], [262144, 8192, 32768])
        patch = [{"id": "llm-pi-ai"}]
        models = deepseek_config(patch, [full, small, unknown_m("unknown")])[0]["config"]["providers"]["veystrix"]["models"]
        self.assertEqual([x["contextWindow"] for x in models], [266240, 12288, 36864])
        self.assertEqual([x["maxTokens"] for x in models], [262144, 8192, 32768])
        self.assertEqual(models[0]["reasoningEfforts"]["xhigh"], "xhigh")
        self.assertEqual(models[0]["reasoningEfforts"]["max"], "max")

    def test_inkling_catalog_advertises_the_configured_one_million_window(self):
        ink = m("ink")
        self.assertEqual(model_context(ink), 1_048_576)
        entries = qwen_config({}, [unknown_m("u"), ink])["modelProviders"]["openai"]
        self.assertEqual([x["generationConfig"]["contextWindowSize"] for x in entries], [32768, 1_048_576])
        models = deepseek_config([{"id": "llm-pi-ai"}], [unknown_m("u"), ink])[0]["config"]["providers"]["veystrix"]["models"]
        self.assertEqual([x["contextWindow"] for x in models], [36864, 1_052_672])
        self.assertEqual([x["maxTokens"] for x in models], [32768, 1_048_576])

    def test_opencode_merge_writes_window_limited_models_and_is_idempotent(self):
        full = m("full", ctx=262144)
        ink = m("ink")
        result = opencode_config({}, [full, ink])
        block = result["providers"]["veystrix"]
        self.assertEqual(block["package"], "@opencode/ai/providers/openai-compatible")
        self.assertEqual(block["settings"]["baseURL"], "http://127.0.0.1:8080/v1")
        self.assertEqual(block["models"]["full"]["limit"], {"context": 262144, "output": 262144})
        self.assertEqual(block["models"]["ink"]["limit"], {"context": 1_048_576, "output": 1_048_576})
        self.assertTrue(block["models"]["full"]["capabilities"]["tools"])
        self.assertEqual(opencode_config(result, [full, ink]), result)

    def test_opencode_preserves_foreign_providers_and_rejects_foreign_endpoint(self):
        original = {"providers": {"other": {"name": "Other"}, "veystrix": {"name": "V", "extra": True}}}
        result = opencode_config(original, [m("a")])
        self.assertEqual(result["providers"]["other"], original["providers"]["other"])
        self.assertEqual(result["providers"]["veystrix"]["extra"], True)
        self.assertEqual(opencode_config(result, [m("a")]), result)
        with self.assertRaises(ValueError):
            opencode_config({"providers": {"veystrix": {"settings": {"baseURL": "https://elsewhere"}}}}, [m("a")])

    def test_kimi_merge_writes_provider_and_models_and_is_idempotent(self):
        original = {
            "default_model": "kimi-code/k3",
            "providers": {"managed:kimi-code": {"type": "kimi"}},
            "models": {"kimi-code/k3": {"provider": "managed:kimi-code", "model": "k3", "max_context_size": 1048576}},
        }
        result = kimi_config(original, [unknown_m("q"), m("ink")])
        self.assertEqual(result["default_model"], "kimi-code/k3")
        self.assertEqual(result["providers"]["veystrix"],
                         {"type": "openai", "base_url": "http://127.0.0.1:8080/v1", "api_key": "local"})
        self.assertEqual(result["models"]["q"]["max_context_size"], 32768)
        self.assertEqual(result["models"]["ink"]["max_context_size"], 1_048_576)
        self.assertNotIn("max_output_size", result["models"]["q"])
        self.assertEqual(kimi_config(result, [unknown_m("q"), m("ink")]), result)

    def test_kimi_rejects_bad_shapes_and_foreign_endpoint(self):
        with self.assertRaises(ValueError): kimi_config([], [m("a")])
        with self.assertRaises(ValueError): kimi_config({"providers": "x"}, [m("a")])
        with self.assertRaises(ValueError): kimi_config({"models": "x"}, [m("a")])
        with self.assertRaises(ValueError):
            kimi_config({"providers": {"veystrix": {"base_url": "https://elsewhere"}}}, [m("a")])
        with self.assertRaises(ValueError):
            kimi_config({"providers": {"a": {"base_url": "http://127.0.0.1:8080/v1"},
                                       "b": {"base_url": "http://127.0.0.1:8080/v1"}}}, [m("a")])

    def test_kimi_merges_into_the_users_same_endpoint_provider(self):
        full = m("veystrix-inkling2-mini-x", ctx=262144)
        original = {
            "default_model": "Veystrix/veystrix-inkling2-mini-x",
            "providers": {"Veystrix": {"base_url": "http://127.0.0.1:8080/v1",
                                       "type": "openai", "api_key": "apple"}},
            "models": {"Veystrix/veystrix-inkling2-mini-x": {
                "provider": "Veystrix", "model": "veystrix-inkling2-mini-x",
                "max_context_size": 256000, "capabilities": ["tool_use", "thinking"],
                "display_name": "Inkling", "adaptive_thinking": True}},
        }
        result = kimi_config(original, [full, m("ink")])
        self.assertNotIn("veystrix", result["providers"])
        theirs = result["providers"]["Veystrix"]
        self.assertEqual(theirs["api_key"], "apple")
        self.assertEqual(result["default_model"], "Veystrix/veystrix-inkling2-mini-x")
        updated = result["models"]["Veystrix/veystrix-inkling2-mini-x"]
        self.assertEqual(updated["max_context_size"], 262144)
        self.assertEqual(updated["capabilities"], ["tool_use", "thinking"])
        self.assertEqual(updated["display_name"], "Inkling")
        self.assertEqual(result["models"]["ink"]["provider"], "Veystrix")
        self.assertEqual(result["models"]["ink"]["max_context_size"], 1_048_576)
        self.assertEqual(kimi_config(result, [full, m("ink")]), result)

    def test_qwen_merge_preserves_unrelated_providers_and_is_idempotent(self):
        original = {"modelProviders": {"openai": [{"id": "other", "baseUrl": "https://example.test"}], "other": [{"id": "x"}]}, "model": {"name": "other"}}
        result = qwen_config(original, [m("a"), m("b")])
        self.assertEqual(result["model"], original["model"])
        self.assertEqual(result["modelProviders"]["other"], original["modelProviders"]["other"])
        self.assertEqual([x["id"] for x in result["modelProviders"]["openai"]], ["other", "a", "b"])
        self.assertEqual(qwen_config(result, [m("a"), m("b")]), result)
        self.assertEqual(len(original["modelProviders"]["openai"]), 1)

    def test_qwen_does_not_replace_foreign_entry(self):
        with self.assertRaises(ValueError): qwen_config({"modelProviders": {"openai": [{"id": "a", "baseUrl": "https://other"}]}}, [m("a")])

    def test_deepseek_preserves_other_plugins_and_provider(self):
        old = [{"id": "llm-pi-ai", "config": {"providers": {"other": {"api": "other"}, "veystrix": {"baseURL": "http://127.0.0.1:8080/v1", "extra": True}}}}, {"id": "ui", "config": {"x": 1}}]
        new = deepseek_config(old, [m("a"), m("b")])
        self.assertEqual(new[1], old[1])
        self.assertEqual(new[0]["config"]["providers"]["other"], {"api": "other"})
        self.assertTrue(new[0]["config"]["providers"]["veystrix"]["extra"])
        self.assertEqual([x["id"] for x in new[0]["config"]["providers"]["veystrix"]["models"]], ["a", "b"])
        self.assertEqual(deepseek_config(new, [m("a"), m("b")]), new)

    def test_deepseek_rejects_ambiguous_plugins_and_foreign_provider(self):
        with self.assertRaises(ValueError): deepseek_config([], [m("a")])
        with self.assertRaises(ValueError): deepseek_config([{"id": "llm-pi-ai", "config": {"providers": {"veystrix": {"baseURL": "https://elsewhere"}}}}], [m("a")])


class HarnessWriteTests(unittest.TestCase):
    def test_configure_creates_backup_and_preserves_unrelated_settings(self):
        import json, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from veystrix.harness_configure import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / ".qwen" / "settings.json"; path.parent.mkdir()
            original = '{"model":{"name":"other"},"other":{"keep":true}}\n'; path.write_text(original)
            with patch("veystrix.harness_configure.discover", return_value=[m("a"), m("b")]):
                result = run(home=root, apply=True)
                self.assertIn("updated", result["Qwen Code"])
                backups = list(path.parent.glob("settings.json.veystrix-backup-*"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), original)
                self.assertEqual(json.loads(path.read_text())["other"], {"keep": True})
                self.assertEqual(run(home=root, apply=True)["Qwen Code"], "unchanged")

    def test_hermes_refresh_uses_dedicated_profile_and_preserves_main_config(self):
        import tempfile, yaml
        from pathlib import Path
        from unittest.mock import patch
        from veystrix.harness_configure import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); home = root / ".hermes"; home.mkdir()
            main = home / "config.yaml"; main.write_text("model: external\n")
            with patch("veystrix.harness_configure.discover", return_value=[m("a"), m("b")]):
                result = run(home=root, apply=True)
                self.assertIn("created", result["Hermes"])
                profile = yaml.safe_load((home / "profiles/veystrix/config.yaml").read_text())
                self.assertEqual(set(profile["custom_providers"][0]["models"]), {"a", "b"})
                self.assertEqual(main.read_text(), "model: external\n")
                self.assertIn("unchanged", run(home=root, apply=True)["Hermes"])

    def test_opencode_kimi_and_pi_are_created_with_backups_and_stay_unchanged(self):
        import json, tempfile, tomllib
        from pathlib import Path
        from unittest.mock import patch
        from veystrix.harness_configure import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("veystrix.harness_configure.discover", return_value=[m("a"), m("b")]):
                first = run(home=root, apply=True)
                self.assertIn("created", first["OpenCode"])
                self.assertIn("created", first["Kimi Code"])
                self.assertEqual(first["pi"], "not installed")
                opencode = json.loads((root / ".config/opencode/opencode.json").read_text())
                self.assertEqual(opencode["providers"]["veystrix"]["models"]["b"]["limit"]["context"], 1_048_576)
                kimi = tomllib.loads((root / ".kimi-code/config.toml").read_text())
                self.assertEqual(kimi["providers"]["veystrix"]["base_url"], "http://127.0.0.1:8080/v1")
                self.assertEqual(kimi["models"]["b"]["max_context_size"], 1_048_576)
                (root / ".pi" / "agent").mkdir(parents=True)
                second = run(home=root, apply=True)
                self.assertIn("created", second["pi"])
                self.assertEqual((root / ".pi" / "agent" / "extensions" / "veystrix.ts").read_text(),
                                 __import__("veystrix.pi_extension", fromlist=["EXTENSION_SOURCE"]).EXTENSION_SOURCE)
                self.assertEqual(run(home=root, apply=True)["OpenCode"], "unchanged")
                self.assertEqual(run(home=root, apply=True)["Kimi Code"], "unchanged")
                self.assertEqual(run(home=root, apply=True)["pi"], "unchanged")

    def test_deepseek_yaml_off_key_stays_a_string(self):
        from veystrix.harness_configure import load_yaml12
        parsed = load_yaml12("reasoningEfforts:\n  off: none\n  enabled: false\n")
        self.assertEqual(parsed["reasoningEfforts"]["off"], "none")
        self.assertIs(parsed["reasoningEfforts"]["enabled"], False)
