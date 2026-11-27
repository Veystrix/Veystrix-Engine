"""Per-request reasoning controls: the resolver, the wire contract Qwen Code 0.22.0 produces, and the real chat template."""
import os
import unittest

from veystrix import preferences, reasoning as R
from veystrix.conversation import EXTRA_REASONING
from veystrix.server import ModelService, RequestError, _check_chat_fields

LOW = "Reasoning effort is set to low."
XHIGH = "Reasoning effort is set to xhigh."


def resolve(payload, thinking=False, effort="medium"):
    return R.resolve(payload, thinking, effort)


class ResolverTests(unittest.TestCase):
    def test_qwen_code_slash_effort_wire_shape_for_each_tier(self):
        """Measured from the installed client: /effort low|medium|high|xhigh -> reasoning.effort; max is clamped to xhigh by it."""
        for tier in ("low", "medium", "high", "xhigh"):
            r = resolve({"reasoning": {"effort": tier}, "max_tokens": 4000})
            self.assertEqual((r.thinking, r.effort, r.source), (True, tier, "reasoning.effort"), tier)
            self.assertEqual(r.wire, {"reasoning": {"effort": tier}})

    def test_no_reasoning_field_means_the_server_default_and_says_so(self):
        r = resolve({})
        self.assertEqual((r.thinking, r.effort, r.source), (False, "medium", "server-default"))
        r = resolve({}, thinking=True, effort="high")
        self.assertEqual((r.thinking, r.effort, r.source), (True, "high", "server-default"))

    def test_max_is_clamped_to_xhigh_and_reported(self):
        r = resolve({"reasoning": {"effort": "max"}})
        self.assertEqual((r.thinking, r.effort, r.requested), (True, "xhigh", "xhigh"))
        self.assertTrue(any("max has no distinct level" in n for n in r.notes))

    def test_every_off_spelling_turns_thinking_off(self):
        for payload in ({"reasoning": {"effort": "none"}}, {"reasoning_effort": "none"}, {"reasoning": False},
                        {"reasoning": {"enabled": False}}, {"chat_template_kwargs": {"enable_thinking": False}},
                        {"enable_thinking": False}, {"reasoning": {"effort": "off"}}):
            r = resolve(payload, thinking=True)
            self.assertFalse(r.thinking, payload)
            self.assertNotEqual(r.source, "server-default", payload)

    def test_flat_openai_field_and_native_template_switch_are_honoured(self):
        self.assertEqual(resolve({"reasoning_effort": "low"}).effort, "low")
        r = resolve({"chat_template_kwargs": {"enable_thinking": True}}, effort="high")
        self.assertEqual((r.thinking, r.effort, r.source), (True, "high", "chat_template_kwargs.enable_thinking"))
        self.assertTrue(resolve({"enable_thinking": True}).thinking)

    def test_an_explicit_disable_beats_an_effort_in_the_same_request_and_is_reported(self):
        r = resolve({"reasoning": {"effort": "medium"}, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertFalse(r.thinking)
        self.assertTrue(any("explicit thinking disable" in n for n in r.notes))

    def test_unknown_levels_and_unsupported_controls_are_errors_not_ignored(self):
        for payload in ({"reasoning": {"effort": "ludicrous"}}, {"reasoning_effort": "turbo"},
                        {"reasoning": {"max_tokens": 2000}}, {"chat_template_kwargs": {"preserve_thinking": True}},
                        {"chat_template_kwargs": "x"}, {"reasoning": ["high"]}, {"reasoning": {"verbosity": "high"}}):
            with self.assertRaises(R.ReasoningError, msg=payload):
                resolve(payload)

    def test_the_four_levels_are_distinct_after_translation(self):
        from veystrix.conversation import reasoning_system_text
        seen = {tier: reasoning_system_text("SYSTEM", tier) for tier in R.LEVELS}
        self.assertEqual(len(set(seen.values())), 4)
        self.assertEqual(seen["high"][1], "medium")              # high rides on the medium template plus its own sentence
        self.assertIn(EXTRA_REASONING["high"], seen["high"][0])
        self.assertEqual(seen["low"], ("SYSTEM", "low"))

    def test_server_accepts_the_fields_instead_of_refusing_them(self):
        ignored = _check_chat_fields({"model": "m", "messages": [], "reasoning": {"effort": "low"}, "reasoning_effort": "low",
                                      "chat_template_kwargs": {"enable_thinking": True}, "enable_thinking": True})
        self.assertEqual(ignored, [])
        with self.assertRaises(RequestError):
            _check_chat_fields({"model": "m", "messages": [], "thinking_budget": 100})


if __name__ == "__main__":
    unittest.main()


class ResponsesSummaryField(unittest.TestCase):
    def test_codex_reasoning_summary_is_accepted(self):
        from veystrix import reasoning as R
        result = R.resolve({"reasoning": {"effort": "medium", "summary": "auto"}}, False, "medium")
        self.assertEqual(result.effort, "medium")


class AnthropicThinkingField(unittest.TestCase):
    def test_anthropic_thinking_type_switches_the_template(self):
        """Claude Code and other Messages clients send thinking={"type": ...}; it used to be ignored (thinking off)."""
        on = R.resolve({"thinking": {"type": "enabled", "budget_tokens": 1024}}, False, "medium")
        self.assertEqual((on.thinking, on.source), (True, "thinking.type"))
        self.assertTrue(R.resolve({"thinking": {"type": "adaptive"}}, False, "medium").thinking)
        self.assertFalse(R.resolve({"thinking": {"type": "disabled"}}, True, "medium").thinking)
        # an explicit template switch wins
        both = R.resolve({"thinking": {"type": "enabled"}, "chat_template_kwargs": {"enable_thinking": False}},
                         False, "medium")
        self.assertFalse(both.thinking)


class AnthropicThinkingHistory(unittest.TestCase):
    def test_assistant_thinking_blocks_become_reasoning_content(self):
        from veystrix.server import _anthropic_messages
        out = _anthropic_messages({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "plan", "signature": ""},
                                              {"type": "text", "text": "hello"}]},
            {"role": "user", "content": "again"}]})
        self.assertEqual(out[1]["reasoning_content"], "plan")
        self.assertEqual(out[1]["content"], "hello")
