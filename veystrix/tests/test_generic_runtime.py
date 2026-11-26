"""Generic recipe-driven C++ runtime: conformance, KV policy, Ornith wiring, endpoint, pull.

Offline only: tiny synthetic qwen3_5_moe checkpoints (built in a temp dir, a few MB), mocked
Hub client, mock HTTP server. No real checkpoint, no download, no harness config touched.
The C++ tests skip when the dylib is not built.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "cpp" / "engine" / "build" / "libveystrix_c.dylib"
GIB = 1024**3


def _mlx_lm_available() -> bool:
    try:
        import mlx_lm.models.qwen3_5_moe  # noqa: F401
        return True
    except Exception:
        return False


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Home(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        env = {"VEYSTRIX_HOME": str(self.home), "VEYSTRIX_MODEL_ROOT": str(self.home / "models"),
               "VEYSTRIX_TEST_POWER": json.dumps({"on_ac": True, "percent": 100}),
               "VEYSTRIX_TEST_MEMORY": json.dumps({"available_bytes": 16 * GIB}),
               "VEYSTRIX_NO_KEYCHAIN": "1"}
        self._patch = patch.dict(os.environ, env, clear=False)
        self._patch.start()
        for name in ("VEYSTRIX_ENGINE", "VEYSTRIX_SOURCE_CPP", "VEYSTRIX_KV_MODE", "VEYSTRIX_CONTEXT_LIMIT",
                     "VEYSTRIX_KV_INT4_GATE"):
            os.environ.pop(name, None)

    def tearDown(self):
        from veystrix.tool.supervisor import stop
        stop(self.home)
        self._patch.stop()
        self._tmp.cleanup()


TINY_TOKENIZER = {"version": "1.0", "added_tokens": [], "model": {"type": "WordLevel", "vocab": {}}}


def _tiny_ornith(root: Path, tokenizer: bool = True) -> Path:
    from veystrix.port_check import build_tiny
    path = build_tiny(root / "Ornith-tiny")
    if tokenizer:
        (path / "tokenizer.json").write_text(json.dumps(TINY_TOKENIZER))
        (path / "tokenizer_config.json").write_text("{}")
    return path


# --------------------------------------------------------------------------- #
# C++ conformance (the port-check) and the Python backend over the C ABI      #
# --------------------------------------------------------------------------- #

@unittest.skipUnless(LIB.is_file() and _mlx_lm_available(), "C++ dylib not built or mlx_lm missing")
class PortCheckTests(unittest.TestCase):
    def test_quick_port_check_is_bit_identical(self):
        from veystrix.port_check import run_tiny
        with tempfile.TemporaryDirectory() as tmp:
            checks = run_tiny(str(LIB), quick=True, workdir=tmp, modes=["int8", "tiered", "tiered4+"])
        failed = [f"{c.name}: {c.detail}" for c in checks if not c.ok]
        self.assertEqual(failed, [])
        names = {c.name for c in checks}
        for required in ("bf16 step_batch N=4", "bf16 fork after 3 steps", "bf16 two turns (APPEND after decode)",
                         "kv=tiered4+ snapshot/restore across tiers", "kv=tiered inside window == bf16 (C++)",
                         "snapshot from another KV plan is refused", "norm_shift variant (mtp tensors present)"):
            self.assertIn(required, names)


    def test_quick_qwen4_port_check_is_bit_identical(self):
        from veystrix.port_check import run_tiny_qwen4
        with tempfile.TemporaryDirectory() as tmp:
            checks = run_tiny_qwen4(str(LIB), quick=True, workdir=tmp)
        self.assertEqual([f"{c.name}: {c.detail}" for c in checks if not c.ok], [])
        self.assertIn("qwen4 snapshot(120)/restore + APPEND(80) (PLE + GDN + KV state)", {c.name for c in checks})

    def test_inspect_reports_generic_blocks(self):
        from veystrix.model_inspect import _library_block_types, generic_report
        have = _library_block_types(str(LIB))
        for block in ("linear.gdn", "attn.gated_gqa", "moe.softmax_topk", "linear.gdn_qwen4", "attn.qsa_gated",
                      "pre.ple_ngram"):
            self.assertIn(block, have)
        with tempfile.TemporaryDirectory() as tmp:
            report = generic_report(_tiny_ornith(Path(tmp)), lib=str(LIB))
            self.assertEqual(report["recipe"], "ok")
            self.assertEqual(report["missing_blocks"], [])
            (Path(tmp) / "x").mkdir()
            (Path(tmp) / "x" / "config.json").write_text(json.dumps({"model_type": "not_a_model"}))
            self.assertIn("no recipe", generic_report(Path(tmp) / "x", lib=str(LIB))["recipe"])

    def test_blocked_quantized_attention_matches_the_reference(self):
        """Force tiny score tiles so prefill takes the query-blocked path; it must stay identical to RefTieredKV."""
        import subprocess
        env = dict(os.environ, VEYSTRIX_KV_SCORE_TILE="20000", PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        out = subprocess.run([sys.executable, "-m", "veystrix.port_check", "--tiny", "--no-record", "--model-type",
                              "qwen3_5_moe", "--modes", "tiered4+,tiered-k8v4", "--lib", str(LIB)],
                             env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:] + out.stderr[-2000:])
        from veystrix.port_check import attention_query_block
        self.assertEqual(attention_query_block(2048, 2048, 16), 2048)
        self.assertEqual(attention_query_block(2048, 131072, 16), 32)
        self.assertEqual(attention_query_block(1, 262144, 16), 1)
        from veystrix.port_check import query_spans
        self.assertEqual(query_spans(23, 16), [(0, 23)])                 # a 7-row tail never gets its own kernel
        self.assertEqual(query_spans(40, 16), [(0, 16), (16, 40)])
        self.assertEqual(query_spans(64, 16), [(0, 16), (16, 32), (32, 48), (48, 64)])

    def test_flashnext_runs_on_the_generic_runtime(self):
        from veystrix import model_identity
        self.assertEqual(model_identity.INKLING2_MINI.cpp_runtime, "generic")
        self.assertEqual({i.family for i in model_identity.IDENTITIES.values() if i.cpp_runtime == "custom"},
                         set())

    def test_recipe_speed_knobs(self):
        from veystrix.backends.cpp_engine import CtypesVxBinding
        from veystrix.recipe import emit_recipe
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = _tiny_ornith(Path(tmp), tokenizer=False)
            recipe = emit_recipe(ckpt)
            recipe["speed"] = {"read_lanes": 8}  # the code default: no effect on later tests in this process
            b = CtypesVxBinding(str(LIB))
            eng = b.open_engine(str(ckpt), json.dumps({"recipe": json.dumps(recipe)}))
            self.assertIn("read_lanes=", b.capabilities(eng).get("speed", ""))
            b.close_engine(eng)
            recipe["speed"] = {"not_a_knob": 1}
            with self.assertRaises(Exception):
                b.open_engine(str(ckpt), json.dumps({"recipe": json.dumps(recipe)}))


@unittest.skipUnless(_mlx_lm_available(), "mlx_lm missing")
class StreamingOracleTests(unittest.TestCase):
    def test_streaming_oracle_is_bit_identical_to_full_load(self):
        from veystrix.port_check import Oracle, StreamingOracle, _prompt, build_tiny, oracle_run
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = build_tiny(Path(tmp) / "t")
            full, streamed = Oracle(ckpt), StreamingOracle(ckpt)
            for n in (16, 120):
                p = _prompt(n, n, 512)
                self.assertEqual(oracle_run(full, [p], 5, turns=[[3, 4, 5]])[:2],
                                 oracle_run(streamed, [p], 5, turns=[[3, 4, 5]])[:2])
            self.assertGreater(streamed.bytes_read, 0)
            streamed.index.close()


class _StubTokenizer:
    eos_token_id = 511
    unk_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [1 + (ord(c) % 500) for c in text]

    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 510, "<|endoftext|>": 509}.get(token)


@unittest.skipUnless(LIB.is_file() and _mlx_lm_available(), "C++ dylib not built or mlx_lm missing")
class GenericBackendTests(_Home):
    def test_backend_two_turns_match_mlx_lm(self):
        from argparse import Namespace
        from veystrix.backends.cpp_generic import build_generic_backend
        from veystrix.port_check import Oracle, oracle_run
        from veystrix.sampling import Sampling
        ckpt = _tiny_ornith(self.home / "models")
        os.environ["VEYSTRIX_SESSION_DIR"] = str(self.home / "sessions")
        os.environ["VEYSTRIX_CACHE_ROOT"] = str(self.home / "cache")
        os.environ["VEYSTRIX_PREFIX_CACHE"] = "0"
        os.environ["VEYSTRIX_CONTEXT_LIMIT"] = "4096"
        with patch("veystrix.recipe.recipe_cache_path", lambda c, r=None: self.home / "recipe.json"):
            backend = build_generic_backend("inkling2-mini", Namespace(model_path=str(ckpt), session_dir=None),
                                            {}, tokenizer=_StubTokenizer())
        try:
            self.assertEqual(backend.context_limit, 4096)
            self.assertEqual(backend.sampling.temperature, 0.6)       # model-card defaults applied
            self.assertEqual(backend.sampling.top_k, 20)
            backend.sampling = Sampling.greedy_settings()
            info = backend.runtime_info()
            self.assertEqual(info["runtime"], "generic")
            self.assertEqual(info["kv"]["mode"], "bf16")              # 4K ceiling: bf16 fits the plan
            first = list(range(3, 43))
            backend.append_tokens(first)
            backend.generate(6)
            turn1 = list(backend.tape[len(first):])
            second = list(range(50, 61))
            backend.append_tokens(second)
            backend.generate(6)
            turn2 = list(backend.tape[len(first) + len(turn1) + len(second):])
        finally:
            backend.close()
        oracle = Oracle(ckpt, "bf16")
        ref = oracle_run(oracle, [first], 6, turns=[second])[0]
        self.assertEqual(turn1 + turn2, [t for t in ref if t not in (509, 510, 511)][:len(turn1 + turn2)])
        self.assertEqual(turn1, ref[:len(turn1)])


@unittest.skipUnless(LIB.is_file() and _mlx_lm_available(), "C++ dylib not built or mlx_lm missing")
class KVGateTests(unittest.TestCase):
    def test_probe_returns_greedy_and_appends_forced(self):
        from veystrix.port_check import Cpp, build_tiny
        from veystrix.recipe import emit_recipe
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = build_tiny(Path(tmp) / "t")
            recipe = Path(tmp) / "r.json"
            recipe.write_text(json.dumps(emit_recipe(ckpt)))
            c = Cpp(str(LIB), ckpt, recipe)
            a, b = c.session(), c.session()
            prompt = list(range(5, 45))
            c.b.prefill(a, prompt)
            c.b.prefill(b, prompt)
            greedy = c.b.step(a, '{"greedy": true}')
            probed = c.b.step(b, json.dumps({"force": 7, "probe": True}))
            self.assertEqual(probed, greedy)                 # returns what greedy would have picked
            c.b.prefill_ex(a, [7], 1)                        # a: greedy token, then 7 (as APPEND)
            ref = c.b.step(a, '{"greedy": true}')
            c.b.close_session(a)
            # b already holds 7 in place of the greedy token: next prediction differs in general,
            # but forcing must have appended exactly one token
            self.assertEqual(c.b.tokens(b), len(prompt) + 1)
            del ref
            c.b.close_session(b)
            c.close()

    def test_probe_logprob_is_the_forced_tokens_log_softmax(self):
        from veystrix import kv_gate
        from veystrix.port_check import Cpp, build_tiny
        from veystrix.recipe import emit_recipe
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = build_tiny(Path(tmp) / "t")
            recipe = Path(tmp) / "r.json"
            recipe.write_text(json.dumps(emit_recipe(ckpt)))
            c = Cpp(str(LIB), ckpt, recipe)
            a, b = c.session(), c.session()
            prompt = list(range(5, 45))
            c.b.prefill(a, prompt)
            c.b.prefill(b, prompt)
            top = c.b.step(a, json.dumps({"force": 3, "probe": True, "logprob": True}))
            lp_other, lp_top = (float(x.split("=")[1]) for x in c.b.last_digest(a).split())
            self.assertLessEqual(lp_other, lp_top)
            self.assertLessEqual(lp_top, 0.0)
            c.b.step(b, json.dumps({"force": top, "probe": True, "logprob": True}))   # forcing the argmax
            lp_forced, lp_top2 = (float(x.split("=")[1]) for x in c.b.last_digest(b).split())
            self.assertEqual(lp_forced, lp_top2)
            self.assertEqual(lp_top2, lp_top)
            self.assertEqual(kv_gate._lp(c.b.last_digest(b)), lp_forced)
            with self.assertRaises(Exception):
                c.b.step(a, json.dumps({"greedy": True, "logprob": True}))           # needs probe
            for x in (a, b):
                c.b.close_session(x)
            c.close()

    def test_gate_scores_tiered_inside_window_as_identical(self):
        from veystrix import kv_gate
        from veystrix.port_check import build_tiny
        from veystrix.recipe import emit_recipe
        tok = _StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = build_tiny(Path(tmp) / "t")
            recipe = Path(tmp) / "r.json"
            recipe.write_text(json.dumps(emit_recipe(ckpt)))
            ids, facts = kv_gate.build(tok, 1200, 4, 128, 256)
            self.assertTrue(any(f["region"] == "middle" for f in facts))
            ref = kv_gate.run_mode(str(LIB), ckpt, recipe, "bf16", 128, 256, ids, facts, 40, 4, tok)
            same = kv_gate.run_mode(str(LIB), ckpt, recipe, "tiered", 100000, 256, ids, facts, 40, 4, tok,
                                    ref_answers=[n["answer_ids"] for n in ref["needles"]])
            sc = kv_gate.score(ref, same)
            self.assertEqual((sc["top1_continuation"], sc["top1_probes"]), (1.0, 1.0))
            low = kv_gate.run_mode(str(LIB), ckpt, recipe, "tiered4+", 128, 256, ids, facts, 40, 4, tok,
                                   ref_answers=[n["answer_ids"] for n in ref["needles"]])
            self.assertEqual(len(low["continuation_argmax"]), 40)


class KVGateOfflineTests(unittest.TestCase):
    def test_tool_check_uses_the_server_parser(self):
        from veystrix import kv_gate
        tasks = kv_gate.tool_tasks([{"key": "1234", "value": "AMBER-12"}], 4)
        good = ("<tool_call>\n<function=record_code>\n<parameter=vault>\n1234\n</parameter>\n"
                "<parameter=code>\nAMBER-12\n</parameter>\n</function>\n</tool_call>")
        self.assertEqual(kv_gate.check_tool(good, tasks[0])["correct"], True)
        wrong = good.replace("AMBER-12", "AMBER-13")
        self.assertEqual({k: kv_gate.check_tool(wrong, tasks[0])[k] for k in ("valid", "correct")},
                         {"valid": True, "correct": False})
        self.assertFalse(kv_gate.check_tool("The code is AMBER-12.", tasks[0])["valid"])
        limit = ("<tool_call>\n<function=lookup_history>\n<parameter=vault>\n1234\n</parameter>\n"
                 f"<parameter=limit>\n{tasks[3]['expect']['limit']}\n</parameter>\n</function>\n</tool_call>")
        self.assertTrue(kv_gate.check_tool(limit, tasks[3])["correct"])

    def test_record_fails_a_mode_that_failed_at_any_size(self):
        from veystrix import kv_gate
        from veystrix.kv_policy import gate_passed
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VEYSTRIX_HOME": tmp}):
            os.environ.pop("VEYSTRIX_KV_INT4_GATE", None)
            base = {"window": 8192, "prefix": 1024, "modes": {}}
            kv_gate.record("m", 8192, {**base, "verdicts": {"tiered4+": "pass", "int8": "pass"}})
            self.assertTrue(gate_passed("m"))
            kv_gate.record("m", 32768, {**base, "verdicts": {"tiered4+": "fail", "int8": "pass"}})
            self.assertFalse(gate_passed("m"))
            data = json.loads((Path(tmp) / ".veystrix" / "kv-gate.json").read_text())
            self.assertEqual(data["m"]["int8"], "pass")
            self.assertTrue((Path(tmp) / ".veystrix" / "parity" / "kv-gate-m.json").is_file())

    def test_verdict_bars(self):
        from veystrix import kv_gate
        sc = {"top1_continuation": 0.995, "middle_probe_tokens": 10, "top1_middle_probes": 1.0, "needles_lost": [],
              "ppl_ratio": 1.004, "tools": 24, "tools_valid": 23, "tools_correct": 21, "ref_tools_valid": 23,
              "ref_tools_correct": 21, "tools_regressed": 1}
        self.assertEqual(kv_gate.verdict(sc), "pass")
        self.assertEqual(kv_gate.verdict({**sc, "ppl_ratio": 1.02}), "fail")
        self.assertEqual(kv_gate.verdict({**sc, "tools_valid": 22}), "fail")
        self.assertEqual(kv_gate.verdict({**sc, "tools_regressed": 2}), "fail")
        self.assertEqual(kv_gate.verdict({**sc, "needles_lost": ["1234"]}), "fail")
        self.assertEqual(kv_gate.parse_mode("tiered@49152"), ("tiered", 49152))
        self.assertEqual((kv_gate.parse_mode("tiered@32768:h8/4"), kv_gate.parse_bits("tiered@32768:h8/4")),
                         (("tiered", 32768), (8, 4)))
        self.assertIsNone(kv_gate.parse_bits("int8"))
        self.assertEqual((kv_gate.parse_mode("bf16#c1024"), kv_gate.parse_chunk("bf16#c1024"),
                          kv_gate.parse_chunk("tiered4+")), (("bf16", None), 1024, 2048))
        # a low raw top-1 from near-ties passes when the confident positions agree
        self.assertEqual(kv_gate.verdict({**sc, "top1_continuation": 0.88, "top1_confident": 0.995}), "pass")
        self.assertEqual(kv_gate.verdict({**sc, "top1_continuation": 0.99, "top1_confident": 0.97}), "fail")
        ref = {"continuation_argmax": [1, 2, 3, 4], "continuation_top_lp": [-0.1, -2.0, -0.2, -3.0], "needles": []}
        got = {"continuation_argmax": [1, 9, 3, 9], "needles": []}
        sc2 = kv_gate.score(ref, got)
        self.assertEqual((sc2["top1_continuation"], sc2["top1_confident"], sc2["confident_tokens"]), (0.5, 1.0, 2))
        self.assertEqual(kv_gate._lp("lp=-1.5 top=-0.25", "top"), -0.25)

    def test_noise_relative_verdict(self):
        from veystrix import kv_gate
        noise = {"top1_continuation": 0.83, "top1_confident": 0.95, "middle_probe_tokens": 40,
                 "top1_middle_probes": 1.0, "tools_regressed": 1}
        sc = {"top1_continuation": 0.86, "top1_confident": 0.94, "top1_p90": 1.0, "middle_probe_tokens": 40,
              "top1_middle_probes": 0.99, "needles_lost": [], "ppl_ratio": 1.01, "ppl_ratio_ci95": [0.95, 1.07],
              "tools": 24, "tools_valid": 19, "ref_tools_valid": 19, "tools_regressed": 2}
        self.assertEqual(kv_gate.verdict(sc), "fail")                 # absolute bar
        self.assertEqual(kv_gate.verdict(sc, noise), "pass")          # within the bf16 noise floor
        self.assertEqual(kv_gate.verdict({**sc, "top1_confident": 0.92}, noise), "fail")
        self.assertEqual(kv_gate.verdict({**sc, "top1_p90": 0.98}, noise), "fail")
        self.assertEqual(kv_gate.verdict({**sc, "ppl_ratio_ci95": [1.002, 1.03]}, noise), "fail")  # measurably worse
        self.assertEqual(kv_gate.verdict({**sc, "tools_regressed": 3}, noise), "fail")
        self.assertTrue(kv_gate.is_noise_arm("bf16#c1024") and not kv_gate.is_noise_arm("bf16"))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VEYSTRIX_HOME": tmp}):
            kv_gate.record("m", 32768, {"window": 8192, "prefix": 1024, "modes": {},
                                        "verdicts": {"bf16#c1024": "noise-floor", "int8": "pass"}})
            data = json.loads((Path(tmp) / ".veystrix" / "kv-gate.json").read_text())
            self.assertNotIn("bf16#c1024", data["m"])


class SelectorTests(unittest.TestCase):
    def test_install_generic_routes_only_generic_families(self):
        import types
        from veystrix.backends import cpp_generic
        def build_backend_cpp(name, args, prefs):
            return ("custom", name)
        module = types.SimpleNamespace(build_backend=build_backend_cpp)
        cpp_generic.install_generic(module)
        cpp_generic.install_generic(module)                       # idempotent
        self.assertEqual(module.build_backend.__name__, "build_backend_cpp")
        self.assertEqual(module.build_backend("nemotron35-lightning", None, {}), ("custom", "nemotron35-lightning"))
        with patch.object(cpp_generic, "build_generic_backend", return_value="generic") as built:
            self.assertEqual(module.build_backend("inkling2-mini", None, {}), "generic")
        built.assert_called_once()

    def test_generic_backend_caps_the_mlx_cache(self):
        # The server's memory budget exports VEYSTRIX_CPP_RESIDENT, which alone makes the C++ loader pick a 2 GiB
        # MLX cache; the generic route must ship the same 256 MiB cap the custom factory does (Ornith server gap).
        from argparse import Namespace
        from unittest.mock import MagicMock
        from veystrix.backends import cpp_generic
        args = Namespace(model_path="/nonexistent/ckpt", session_dir=None)
        stub = (lambda *a, **k: ({}, MagicMock(), "recipe.json"))
        for preset, expected in ((None, str(256 << 20)), ("1048576", "1048576")):
            env = {k: v for k, v in os.environ.items() if k != cpp_generic.MLX_CACHE_ENV}
            if preset is not None:
                env[cpp_generic.MLX_CACHE_ENV] = preset
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(cpp_generic, "generic_profile", stub), \
                    patch.object(cpp_generic, "_stop_tokens", return_value=[1]), \
                    patch.object(cpp_generic, "apply_sampling_defaults"), \
                    patch.object(cpp_generic, "GenericCppBackend") as backend:
                cpp_generic.build_generic_backend("inkling2-mini", args, {}, tokenizer=object())
                self.assertEqual(os.environ[cpp_generic.MLX_CACHE_ENV], expected)
                backend.assert_called_once()


# --------------------------------------------------------------------------- #
# KV policy                                                                    #
# --------------------------------------------------------------------------- #

class KVPolicyTests(unittest.TestCase):
    def setUp(self):
        from veystrix.kv_policy import KVGeometry
        self.g = KVGeometry(10, 2, 256)          # Ornith: 10 attention layers, 2 KV heads, head dim 256

    def test_ornith_kv_sizes(self):
        from veystrix.kv_policy import bytes_per_token, kv_bytes
        self.assertEqual(bytes_per_token(self.g, "bf16"), 20480)
        self.assertAlmostEqual(kv_bytes(self.g, "bf16", 262144) / GIB, 5.0, places=2)
        self.assertLess(kv_bytes(self.g, "tiered4+", 262144), kv_bytes(self.g, "tiered-k8v4", 262144))
        self.assertLess(kv_bytes(self.g, "tiered-k8v4", 262144), kv_bytes(self.g, "int8", 262144))

    def test_geometry_from_recipe(self):
        from veystrix.kv_policy import KVGeometry
        recipe = {"blocks": {"g": {"type": "linear.gdn"},
                             "a": {"type": "attn.gated_gqa", "num_kv_heads": 2, "head_dim": 256}},
                  "layers": {"count": 40, "pattern": [{"mixer": "g"}] * 3 + [{"mixer": "a"}]}}
        self.assertEqual(KVGeometry.from_recipe(recipe), self.g)

    def test_auto_plan_prefers_exact_then_window_then_compressed(self):
        from veystrix.kv_policy import plan
        self.assertEqual(plan(self.g, 32768, 3 * GIB).mode, "bf16")
        p = plan(self.g, 262144, int(3.5 * GIB))
        self.assertEqual(p.mode, "tiered")
        self.assertGreaterEqual(p.window, 8192)
        self.assertLessEqual(p.bytes_at_context, int(3.5 * GIB))
        self.assertEqual(plan(self.g, 262144, int(2.2 * GIB)).mode, "tiered-k8v4")
        # INT4 history only after the quality gate passed
        self.assertNotEqual(plan(self.g, 262144, int(1.6 * GIB)).mode, "tiered4+")
        self.assertEqual(plan(self.g, 262144, int(1.6 * GIB), int4_allowed=True).mode, "tiered4+")
        self.assertIn("lower the context", plan(self.g, 262144, GIB // 5).reason)

    def test_requested_mode_and_gate_record(self):
        from veystrix.kv_policy import gate_passed, plan
        self.assertEqual(plan(self.g, 4096, GIB, requested="int8").mode, "int8")
        with self.assertRaises(ValueError):
            plan(self.g, 4096, GIB, requested="fp3")
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "gate.json"
            record.write_text(json.dumps({"qwen3_5_moe": {"tiered4+": "pass"}}))
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VEYSTRIX_KV_INT4_GATE", None)
                self.assertTrue(gate_passed("qwen3_5_moe", record))
                self.assertFalse(gate_passed("qwen4_exp", record))


# --------------------------------------------------------------------------- #
# Ornith identity, discovery, inspect, recipe                                  #
# --------------------------------------------------------------------------- #

@unittest.skipUnless(_mlx_lm_available(), "mlx_lm missing")
class OrnithDiscoveryTests(_Home):
    def test_discovery_registry_inspect_and_recipe(self):
        from veystrix import model_inspect
        from veystrix.recipe import emit_recipe
        ckpt = _tiny_ornith(self.home / "models")
        report = model_inspect.inspect(ckpt)
        self.assertEqual(report.runtime, "generic")
        self.assertIn("affine-q8", report.needs)
        self.assertEqual(report.kv_bytes_per_token["attention_layers"], 2)
        self.assertIn("generic recipe runtime", model_inspect.render(report))
        recipe = emit_recipe(ckpt)
        self.assertEqual(recipe["model_type"], "qwen3_5_moe")
        self.assertEqual(recipe["blocks"]["moe"]["top_k"], 4)
        self.assertTrue(any(v.get("bits") == 8 for v in recipe["quant"]["overrides"].values()))

    def test_emit_recipe_command(self):
        from veystrix.tool.commands import cmd_models
        from veystrix.tool.config import Config
        ckpt = _tiny_ornith(self.home / "models")
        out = self.home / "r.json"
        self.assertEqual(cmd_models(["inspect", str(ckpt), "--emit-recipe", str(out)], self.home,
                                    Config(self.home)), 0)
        self.assertEqual(json.loads(out.read_text())["format"], "veystrix-recipe/3")


# --------------------------------------------------------------------------- #
# Engine selection, parity gate, swap                                          #
# --------------------------------------------------------------------------- #

class EngineGateTests(_Home):
    def test_python_refused_for_inkling2_mini(self):
        from veystrix.tool import engine
        with self.assertRaises(SystemExit) as caught:
            engine.check_engine("python", self.home, "inkling2-mini")
        self.assertIn("no python engine", str(caught.exception))

    def test_generic_parity_record_gates_cpp_and_tracks_rebuilds(self):
        from veystrix import model_identity
        from veystrix.tool import engine
        lib = self.home / "libveystrix_c.dylib"
        lib.write_bytes(b"build-1")
        inkling = model_identity.INKLING2_MINI
        self.assertIn("port-check", engine.generic_parity_refusal(self.home, inkling, lib))
        record = engine.parity_record_path(self.home)
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"inkling2_loop": {"status": "pass", "lib_sha256": engine.file_sha256(lib)}}))
        self.assertIsNone(engine.generic_parity_refusal(self.home, inkling, lib))
        lib.write_bytes(b"build-2")
        self.assertIn("rebuilt", engine.generic_parity_refusal(self.home, inkling, lib))

    def test_bundle_refusal_does_not_suggest_python_for_inkling2_mini(self):
        from veystrix.tool import engine
        with self.assertRaises(SystemExit) as caught:
            engine.check_engine("cpp", self.home, "inkling2-mini")
        self.assertNotIn("--engine python", str(caught.exception))
        self.assertIn("no Python engine", str(caught.exception))

    def test_model_swap_moves_to_cpp_and_requires_a_checkpoint(self):
        from veystrix.tool import swap
        self.assertIn("inkling2-mini", swap.MODELS)
        with patch("veystrix.tool.engine.check_engine"):
            with self.assertRaises(SystemExit) as caught:
                swap.swap(self.home, model="inkling2-mini", restart=False)
            self.assertIn("no complete Inkling 2 Mini checkpoint", str(caught.exception))
            with patch("veystrix.tool.swap.checkpoint_refusal", return_value=None):
                outcome = swap.swap(self.home, model="inkling2-mini", restart=False)
        self.assertEqual((outcome["engine"], outcome["model"]), ("cpp", "inkling2-mini"))
        report = swap.availability(self.home, "inkling2-mini")
        self.assertFalse(report["python"]["available"])


# --------------------------------------------------------------------------- #
# Endpoint block, start flags, harness snippets                                #
# --------------------------------------------------------------------------- #

class EndpointTests(_Home):
    HEALTH = {"ready": True, "engine": "cpp", "installed_model_id": "veystrix-ornith-abc", "context_limit": 262144,
              "runtime": {"runtime": "generic", "kv": {"mode": "tiered", "plan": "tiered:p0b16:w40960b16:h8/8:g64"},
                          "kv_bytes_per_token": 10880, "context_in_use": 1234}}

    def _probe(self, path):
        return self.HEALTH if path == "/health" else {"data": [{"id": "veystrix-ornith-abc"}]}

    def test_running_endpoint_block(self):
        from veystrix.tool.config import Config
        from veystrix.tool.endpoint import describe, render
        config = Config(self.home)
        config.values["model"] = "ornith"
        config.save()
        info = describe(self.home, port=9123, probe=self._probe, which=lambda b: b == "opencode",
                        lan=lambda: "192.168.1.20", tailscale=lambda: None)
        self.assertTrue(info["running"])
        self.assertEqual(info["served_model_id"], "veystrix-ornith-abc")
        self.assertEqual((info["engine"], info["runtime"], info["kv_mode"]), ("cpp", "generic", "tiered"))
        self.assertEqual(info["context_ceiling"], 262144)
        self.assertEqual(info["urls"]["local"], "http://127.0.0.1:9123/v1")
        self.assertNotIn("lan", info["urls"])                     # loopback unless --host 0.0.0.0
        self.assertTrue(info["harnesses"]["opencode"]["installed"])
        self.assertEqual(set(info["harnesses"]), {"opencode", "pi", "hermes"})
        self.assertEqual(info["generic_env"]["anthropic"]["ANTHROPIC_MODEL"], "veystrix-ornith-abc")
        self.assertEqual(info["sampling_defaults"], {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        text = render(info)
        for needle in ("veystrix-ornith-abc", "KV:                  tiered", "10,880 B/token",
                       "NOT INSTALLED", "curl -s http://127.0.0.1:9123/v1/chat/completions",
                       "ANTHROPIC_BASE_URL=http://127.0.0.1:9123"):
            self.assertIn(needle, text)
        self.assertNotIn("cpp / generic", text)
        self.assertNotIn("engine / runtime", text)

    def test_running_server_family_beats_config_default(self):
        # The config says Flash-Next (the default) but the running server serves Ornith (21 Nov harness matrix).
        from veystrix.tool.config import Config
        from veystrix.tool.endpoint import describe
        config = Config(self.home)
        config.values["model"] = "flashnext"
        config.save()
        health = dict(self.HEALTH, model="veystrix-inkling2-mini")
        info = describe(self.home, port=9125, probe=lambda p: health if p == "/health" else {"data": []},
                        which=lambda b: None, lan=lambda: None, tailscale=lambda: None)
        self.assertEqual((info["model"], info["alias"]), ("inkling2-mini", "veystrix-inkling2-mini"))

    def test_lan_and_tailscale_urls(self):
        from veystrix.tool.config import Config
        from veystrix.tool.endpoint import describe
        config = Config(self.home)
        config.values["host"] = "0.0.0.0"
        config.save()
        info = describe(self.home, port=9124, probe=lambda p: None, which=lambda b: None,
                        lan=lambda: "10.0.0.7", tailscale=lambda: "https://mac.tail.ts.net")
        self.assertFalse(info["running"])
        self.assertEqual(info["urls"]["lan"], "http://10.0.0.7:9124/v1")
        self.assertEqual(info["urls"]["tailscale_anthropic"], "https://mac.tail.ts.net")

    def test_tailscale_url_only_when_serving_this_port(self):
        from veystrix.tool.endpoint import tailscale_url
        outputs = {"status": json.dumps({"Self": {"DNSName": "mac.tail.ts.net."}}),
                   "serve": json.dumps({"Web": {"mac:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8080"}}}}})}
        runner = lambda argv: outputs["serve" if "serve" in argv else "status"]
        self.assertEqual(tailscale_url(8080, runner), "https://mac.tail.ts.net")
        self.assertIsNone(tailscale_url(9000, runner))

    def test_endpoint_json_command(self):
        import io
        from contextlib import redirect_stdout
        from veystrix.tool.cli import dispatch_tool
        buffer = io.StringIO()
        with redirect_stdout(buffer), patch("veystrix.tool.endpoint._get_json", return_value=None):
            self.assertEqual(dispatch_tool(["endpoint", "--json", "--port", "9125"]), 0)
        data = json.loads(buffer.getvalue())
        self.assertFalse(data["running"])
        self.assertIn("curl", data)

    def test_lan_host_requires_api_key(self):
        from veystrix.tool.supervisor import StartOptions, start
        with self.assertRaises(SystemExit) as caught:
            start(StartOptions(home=self.home, port=_free_port(), host="0.0.0.0",
                               command=[sys.executable, "-c", "pass"], timeout=2))
        self.assertIn("API key", str(caught.exception))

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true" and sys.platform == "darwin",
                     "detached watcher loopback probe times out on macOS GitHub runners")
    def test_start_carries_kv_context_and_announces_ready(self):
        from veystrix.tool.supervisor import StartOptions, _read_state, build_command, engine_env_for, start, stop
        port = _free_port()
        seen = []
        state = start(StartOptions(home=self.home, port=port, kv="tiered4+", context=65536,
                                   command=[sys.executable, "-m", "integrations.hermes.mock_veystrix_server",
                                            "--port", str(port)], timeout=30, on_ready=seen.append))
        self.assertEqual(state["status"], "running")
        self.assertEqual(len(seen), 1)
        saved = _read_state(self.home)
        self.assertEqual((saved["kv"], saved["context"], saved["host"]), ("tiered4+", 65536, "127.0.0.1"))
        stop(self.home)
        env = engine_env_for("cpp", {"VEYSTRIX_KV_MODE": "int8", "VEYSTRIX_CONTEXT_LIMIT": "8192"})
        self.assertEqual(env["VEYSTRIX_KV_MODE"], "int8")
        command = build_command(port=1, model="ornith", checkpoint="/m", environ={}, host="0.0.0.0")
        self.assertIn("0.0.0.0", command)
        self.assertIn("ornith", command)

    def test_cli_start_parses_new_flags_and_picks_cpp_for_inkling2_mini(self):
        from veystrix.tool import cli
        captured = {}
        with patch("veystrix.tool.cli.start", side_effect=lambda o: captured.setdefault("o", o) and {}), \
                patch("veystrix.tool.engine.check_engine"), \
                patch("veystrix.tool.cli._default_checkpoint", return_value="/models/Inkling"):
            cli.dispatch_tool(["start", "--model", "inkling2-mini", "--kv", "int8", "--context", "32768", "--json"])
        options = captured["o"]
        self.assertEqual((options.engine, options.model, options.kv, options.context, options.checkpoint),
                         ("cpp", "inkling2-mini", "int8", 32768, "/models/Inkling"))

    def test_kv_command_and_slash(self):
        import io
        from contextlib import redirect_stdout
        from veystrix.tool.cli import dispatch_tool
        from veystrix.tool.config import Config
        from veystrix.tool.tui import slash
        with redirect_stdout(io.StringIO()):
            dispatch_tool(["kv", "tiered-k8v4"])
        self.assertEqual(Config(self.home).get("kv"), "tiered-k8v4")
        lines = slash("/kv int8", home=self.home, port=_free_port())
        self.assertIn("saved: int8", lines[0])
        self.assertEqual(Config(self.home).get("kv"), "int8")
        self.assertIn("unknown KV mode", slash("/kv fp2", home=self.home, port=1)[0])

    def test_status_line_shows_generic_runtime_and_kv(self):
        from veystrix.tool import renderer as R
        from veystrix.tool.tui import apply_engine_status
        status = R.Status()
        apply_engine_status(status, {"engine": "cpp", "model": "veystrix-inkling2-mini-abc",
                                     "runtime": {"runtime": "generic", "kv": {"mode": "tiered4+"}}})
        line = R.status_line(status, R.Theme(color=False, ascii_only=True), width=200)
        self.assertEqual(status.runtime, "generic")                 # tracked, never displayed
        self.assertNotIn("cpp", line)
        self.assertIn("Inkling 2 Mini", line)                       # friendly name, not the served id
        self.assertIn("KV tiered4+", line)

    def test_harness_connect_offline_uses_inkling2_identity(self):
        from veystrix.tool.harnesses import configure
        result = configure("opencode", home=self.home, port=8080, model_id="veystrix-inkling2-mini", apply=False)
        self.assertEqual(result["result"], "ready to update")


# --------------------------------------------------------------------------- #
# models pull (mocked Hub)                                                     #
# --------------------------------------------------------------------------- #

class _FakeHub:
    def __init__(self, files: dict[str, bytes], corrupt: set[str] = frozenset()):
        import hashlib
        from veystrix.model_pull import RemoteFile
        self.files = files
        self.corrupt = corrupt
        self.downloads = []
        self.remote = [RemoteFile(n, len(b), hashlib.sha256(b).hexdigest() if n.endswith(".safetensors") else None)
                       for n, b in files.items()]

    def list_files(self, repo):
        self.repo = repo
        return self.remote

    def download(self, repo, filename, dest):
        self.downloads.append(filename)
        data = self.files[filename]
        if filename in self.corrupt:
            data = b"x" * len(data)
        (dest / filename).write_bytes(data)
        return dest / filename


class ModelPullTests(unittest.TestCase):
    FILES = {"config.json": b"{}", "model-00001.safetensors": b"weights" * 100}

    def test_pull_resume_integrity_and_space(self):
        from veystrix.model_pull import MARGIN_BYTES, PullError, pull
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub = _FakeHub(self.FILES)
            with self.assertRaises(PullError) as caught:
                pull("inkling2-mini", root, client=hub, free_bytes=lambda d: 10, inspect_after=False)
            self.assertIn("not enough disk space", str(caught.exception))
            self.assertEqual(hub.downloads, [])
            report = pull("inkling2-mini", root, client=hub, free_bytes=lambda d: MARGIN_BYTES * 2, inspect_after=False)
            self.assertEqual(hub.repo, "mlx-community/Inkling-2-Mini-oQ4e-mtp")
            self.assertEqual(report["destination"], str(root / "Inkling-2-Mini-oQ4e-mtp"))
            self.assertTrue(report["verified"])
            self.assertEqual(hub.downloads, ["config.json", "model-00001.safetensors"])     # small first
            again = pull("inkling2-mini", root, client=hub, free_bytes=lambda d: MARGIN_BYTES * 2, inspect_after=False)
            self.assertEqual(again["to_download"], 0)                                     # resume: nothing left
            self.assertEqual(len(hub.downloads), 2)

    def test_existing_wrong_file_is_never_overwritten(self):
        from veystrix.model_pull import MARGIN_BYTES, PullError, pull
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "Inkling-2-Mini-oQ4e-mtp"
            dest.mkdir()
            (dest / "model-00001.safetensors").write_bytes(b"y" * len(self.FILES["model-00001.safetensors"]))
            hub = _FakeHub(self.FILES)
            with self.assertRaises(PullError) as caught:
                pull("inkling2-mini", Path(tmp), client=hub, free_bytes=lambda d: MARGIN_BYTES * 2, inspect_after=False)
            self.assertIn("left untouched", str(caught.exception))
            self.assertEqual((dest / "model-00001.safetensors").read_bytes()[:1], b"y")
            self.assertEqual(hub.downloads, [])

    def test_corrupt_download_fails_integrity(self):
        from veystrix.model_pull import MARGIN_BYTES, PullError, pull
        with tempfile.TemporaryDirectory() as tmp:
            hub = _FakeHub(self.FILES, corrupt={"model-00001.safetensors"})
            with self.assertRaises(PullError) as caught:
                pull("inkling2-mini", Path(tmp), client=hub, free_bytes=lambda d: MARGIN_BYTES * 2, inspect_after=False)
            self.assertIn("integrity check failed", str(caught.exception))

    def test_unknown_name_and_dry_run(self):
        from veystrix.model_pull import MARGIN_BYTES, PullError, pull
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PullError):
                pull("not-a-model", Path(tmp), client=_FakeHub(self.FILES))
            hub = _FakeHub(self.FILES)
            report = pull("inkling2-mini", Path(tmp), client=hub, free_bytes=lambda d: MARGIN_BYTES * 2, dry_run=True)
            self.assertTrue(report["dry_run"])
            self.assertEqual(hub.downloads, [])


if __name__ == "__main__":
    unittest.main()
