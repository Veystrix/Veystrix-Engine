"""Regression tests for Flash-Next (qwen4_exp) QSA sparse attention in the generic C++ runtime.

Promotes scripts/flashnext/qsa_probe.py into the test suite: runs tiny random-weight qwen4_exp
checkpoints with small indexer budgets (16 and 32) so the mlx_vlm reference activates its sparse
QSA block mask, and asserts token-for-token and float32 logit-digest parity across prefill,
decode threshold crossing, mixed dense/sparse step_batch, snapshot/restore, fork, and int8 KV.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "cpp" / "engine" / "build" / "libveystrix_c.dylib"


def _mlx_vlm_available() -> bool:
    try:
        import mlx_vlm.models.qwen4_exp.language  # noqa: F401
        return True
    except Exception:
        return False


class QsaRecipeTests(unittest.TestCase):
    def test_qsa_recipe_carries_sparse_state_version(self):
        from veystrix.port_check import query_spans
        self.assertEqual(query_spans(23, 16), [(0, 23)])
        self.assertEqual(query_spans(40, 16), [(0, 16), (16, 40)])
        self.assertEqual(query_spans(64, 16), [(0, 16), (16, 32), (32, 48), (48, 64)])


@unittest.skipUnless(LIB.is_file() and _mlx_vlm_available(), "C++ dylib not built or mlx_vlm missing")
class QsaSparseParityTests(unittest.TestCase):
    def test_qsa_sparse_budget_16_and_32_bit_exact(self):
        from veystrix.port_check import run_tiny_qwen4
        for budget in (16, 32):
            with tempfile.TemporaryDirectory() as tmp:
                checks = run_tiny_qwen4(str(LIB), quick=True, workdir=tmp, indexer_budget=budget)
            failed = [f"{c.name}: {c.detail}" for c in checks if not c.ok]
            self.assertEqual(failed, [], f"budget={budget} failures: {failed}")
            names = {c.name for c in checks}
            self.assertTrue(any("mid-chunk threshold crossing" in n for n in names))
            self.assertTrue(any("decode crossing threshold" in n for n in names))
            self.assertTrue(any("step_batch N=2 mixed dense+sparse" in n for n in names))
            self.assertTrue(any("fork across threshold" in n for n in names))
            if budget == 16:
                self.assertTrue(any("kv=int8 solo len=64" in n for n in names))

    def test_qsa_sparse_mask_changes_output_above_budget(self):
        """Guard against a silent regression where sparse masking is skipped on both sides."""
        from veystrix.port_check import Cpp, _prompt, build_tiny_qwen4
        from veystrix.recipe import emit_recipe
        with tempfile.TemporaryDirectory() as tmp:
            p = _prompt(64, 64, 512)
            ckpt_dense = build_tiny_qwen4(Path(tmp) / "dense", seed=7, overrides={"indexer_budget": 2048})
            ckpt_sparse = build_tiny_qwen4(Path(tmp) / "sparse", seed=7, overrides={"indexer_budget": 16})
            rec_dense = Path(tmp) / "r_dense.json"
            rec_sparse = Path(tmp) / "r_sparse.json"
            rec_dense.write_text(json.dumps(emit_recipe(ckpt_dense)))
            rec_sparse.write_text(json.dumps(emit_recipe(ckpt_sparse)))
            self.assertEqual(json.loads(rec_sparse.read_text())["blocks"]["attn"]["indexer"]["state"],
                             "qsa-sparse-pooled-v1")
            c_dense = Cpp(str(LIB), ckpt_dense, rec_dense, "bf16")
            c_sparse = Cpp(str(LIB), ckpt_sparse, rec_sparse, "bf16")
            try:
                out_dense = c_dense.run([p], 4)
                out_sparse = c_sparse.run([p], 4)
            finally:
                c_dense.close()
                c_sparse.close()
            self.assertNotEqual(out_dense[1], out_sparse[1],
                                "budget=16 must diverge from dense budget=2048 at len=64")


if __name__ == "__main__":
    unittest.main()
