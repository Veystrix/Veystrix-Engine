import os
import unittest
from unittest.mock import patch

import mlx.core as mx
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpAttention, Qwen4ExpQSAIndexer

from models.flashnext import qsa_chunk
from models.flashnext.tests.unit.test_prefix_cache import build_model, cache_bits


class PrefillQSATileTests(unittest.TestCase):
    def test_evaluating_tiles_preserves_logits_and_attention_state(self):
        model = build_model()
        ids = mx.arange(5, 165)[None]
        with (
            patch.object(Qwen4ExpAttention, "__call__", qsa_chunk._chunked_call),
            patch.object(qsa_chunk, "_ORIGINAL_CALL", Qwen4ExpAttention.__call__),
            patch.object(qsa_chunk, "_ORIGINAL_INDEXER_CALL", Qwen4ExpQSAIndexer.__call__),
            patch.object(qsa_chunk, "QSA_CHUNK_THRESHOLD", 32),
            patch.object(qsa_chunk, "QSA_QUERY_CHUNK", 32),
            patch.dict(os.environ, {qsa_chunk.FLAG_KERNEL: "reference"}),
        ):
            def run(value):
                cache = model.make_cache()
                model._position_ids = None
                model._rope_deltas = None
                with patch.dict(os.environ, {qsa_chunk.FLAG_PREFILL_TILE_EVAL: value}):
                    logits = model(ids, cache=cache).logits
                    mx.eval(logits, [entry.state for entry in cache])
                return logits, cache_bits(cache)

            baseline, baseline_state = run("0")
            candidate, candidate_state = run("1")
        self.assertTrue(mx.array_equal(baseline, candidate).item())
        self.assertEqual(baseline_state, candidate_state)
