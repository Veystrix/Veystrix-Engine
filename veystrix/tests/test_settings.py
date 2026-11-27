"""What `/settings` shows about the next turn."""
from __future__ import annotations

import unittest
from unittest.mock import patch


class DecodingInSettingsTests(unittest.TestCase):
    """Shared decoding controls belong to Session, backend settings to registry."""

    def test_model_settings_renders_shared_then_backend(self):
        class FakeBackend:
            def configure(self, argument):
                return "backend settings\n  public-value  1"

        session = object.__new__(__import__("veystrix.session", fromlist=["Session"]).Session)
        session.backend = FakeBackend()
        session.profile = "plain"
        session.preferences = {
            "thinking_enabled": False, "effort": "xhigh", "temperature": 1.0,
            "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "max_tokens": -1, "think_budget": -1,
        }
        text = session.model_settings("")
        self.assertLess(text.index("Chat settings"), text.index("backend settings"))
        self.assertIn("reasoning shares it", text)
