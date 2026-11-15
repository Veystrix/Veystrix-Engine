"""A tiny Qwen4-Exp language model whose MoE layers are real ``StreamingSwitchGLU`` layers over a synthetic checkpoint.

Shared by the route-union and prefill-read-stat tests. Nothing here loads a real checkpoint: the expert rows are random
quantized tensors written to a temporary safetensors directory (hidden 128, expert width 32, Q4/G32, ``EXPERTS`` experts),
and the attention, hyper-connection, PLE and router weights are random. The layer kinds are those of the real model
(gated-delta-net linear attention, the QSA full-attention layer, one PLE layer).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from models.flashnext import expert_cache, expert_pool, prefill_ring
from models.flashnext.store import SafeTensorStore
from models.flashnext.tests.unit.test_prefill_ring import ENV, TEST_WORKING_SET_LIMIT

EXPERTS = 96
TOP_K = 2
HIDDEN = 128
INTER = 32
LAYERS = 8
VOCAB = 512


def layer_prefix(layer: int) -> str:
    return f"language_model.model.layers.{layer}.mlp.switch_mlp"


def build_checkpoint(directory: str) -> None:
    mx.random.seed(21)
    tensors = {}
    shapes = {"gate_proj": (INTER, HIDDEN // 8, HIDDEN // 32), "up_proj": (INTER, HIDDEN // 8, HIDDEN // 32),
              "down_proj": (HIDDEN, INTER // 8, INTER // 32)}
    for layer in range(LAYERS):
        for projection, (rows, packed, groups) in shapes.items():
            base = f"{layer_prefix(layer)}.{projection}"
            tensors[f"{base}.weight"] = mx.random.randint(0, 2**31 - 1, (EXPERTS, rows, packed), dtype=mx.uint32)
            tensors[f"{base}.scales"] = mx.random.uniform(0.001, 0.004, (EXPERTS, rows, groups)).astype(mx.bfloat16)
            tensors[f"{base}.biases"] = mx.random.uniform(-0.004, 0.004, (EXPERTS, rows, groups)).astype(mx.bfloat16)
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(os.path.join(directory, shard), tensors)
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
        json.dump({"weight_map": {key: shard for key in tensors}}, out)


def build_stock_model():
    """The tiny language model with its stock (resident) SwitchGLU experts."""
    from mlx_vlm.models.qwen4_exp.config import TextConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from types import SimpleNamespace

    mx.random.seed(3)
    config = TextConfig(
        model_type="qwen4_exp_text", hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=4,
        linear_num_value_heads=8, linear_num_key_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_experts=EXPERTS, num_experts_per_tok=TOP_K, shared_expert_intermediate_size=32,
        moe_intermediate_size=INTER, rms_norm_eps=1e-6, vocab_size=VOCAB, num_key_value_heads=2,
        max_position_embeddings=65536, head_dim=64, full_attention_interval=4, indexer_budget=64,
        indexer_compress_ratio=4, indexer_head_dim=32, indexer_n_heads=2, hc_lowrank=32, ple_layer_ids=[2],
        ple_embed_dim=128, eos_token_id=1, ngram_vocab_size_base=1000, split_ngram_parts=4, heads_per_ngram=2)
    stub = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=-1,
                           video_token_id=-2, vision_start_token_id=-3, text_config=config)
    model = LanguageModel(config, stub)
    # ``patch_rmsnorm.apply()`` replaces Qwen4ExpRMSNorm.__call__ for the whole process and some tests (for example
    # test_cooperative_disconnect) never undo it. The patched norm reads one-centered gains, so this zero-centered random
    # model would get gain 0, collapse to a constant hidden state and route every token to the same 2 experts, which makes
    # every ring-capacity test vacuous. The convention lives on each norm instance; pinning it here gives the stock
    # arithmetic (and dense routing) whether or not the patch is installed. The stock class ignores the attribute.
    from models.flashnext import patch_rmsnorm

    patch_rmsnorm.configure(model, one_centered=False)
    mx.eval(model.parameters())
    return model


def stream_experts(model, store):
    """Swap every layer's resident experts for streamed ones, the way ``loader._swap_experts`` does."""
    for index, layer in enumerate(model.model.layers):
        block = layer.mlp
        block._flashnext_layer_id = index
        block.switch_mlp = expert_cache.StreamingSwitchGLU(
            store, layer_prefix(index), 32, 4, "affine", block.switch_mlp.activation, layer_id=index)
    return model


class StreamingCase(unittest.TestCase):
    """A checkpoint, a store, a streamed tiny model and (optionally) a main expert pool per test."""

    ring_experts = 0                    # FLASHNEXT_PREFILL_RING_EXPERTS for the test (0: no ring)
    pool = None

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        build_checkpoint(cls.directory.name)
        cls.stock = build_stock_model()
        mx.random.seed(17)
        cls.ids = [int(t) for t in mx.random.randint(2, VOCAB, (900,)).tolist()]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        env = {**ENV, "FLASHNEXT_EXPERT_POOL_GB": "0.15"}
        if self.ring_experts:
            env["FLASHNEXT_PREFILL_RING_EXPERTS"] = str(self.ring_experts)
        else:
            env["FLASHNEXT_PREFILL_RING_EXPERTS"] = "0"
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("FLASHNEXT_PREFILL_ORDER", "FLASHNEXT_PREFILL_ROUTE_UNION", "FLASHNEXT_PREFILL_READ_STATS",
                     "FLASHNEXT_PREFILL_LAYER_PREFETCH", "FLASHNEXT_PREFILL_SEED_DECODE"):
            os.environ.pop(name, None)
        limit = mock.patch.object(prefill_ring, "metal_working_set_limit", return_value=TEST_WORKING_SET_LIMIT)
        limit.start()
        self.addCleanup(limit.stop)
        self.store = SafeTensorStore(self.directory.name)
        self.addCleanup(self.store.close)
        self.model = stream_experts(self.fresh_model(), self.store)
        self.addCleanup(expert_cache.set_prefill_ring, None)
        self.addCleanup(expert_cache.set_prefill_progress, None)
        self.addCleanup(expert_cache.set_route_trace, None)
        if self.ring_experts:
            self.pool = expert_pool.for_store(self.store, 32)
            self.addCleanup(self.pool.close)
            self.free_before = sorted(self.pool._free)

    def fresh_model(self):
        """A model with the stock model's weights but its own layer objects (the expert swap mutates the blocks)."""
        model = build_stock_model()
        return model

    def assert_restored(self):
        self.assertIsNone(expert_cache._PREFILL_RING[0])
        self.assertIsNone(expert_cache._LAYER_WINDOW[0])
        if self.pool is not None:
            self.assertEqual(sorted(self.pool._free), self.free_before)


def route_sets(model, runner):
    """Run ``runner()`` and return ``{layer: [set of routed experts per MoE call]}`` from the route trace."""
    seen = {}

    def trace(layer, routed):
        seen.setdefault(layer, []).append(set(routed))

    expert_cache.set_route_trace(trace)
    try:
        runner()
    finally:
        expert_cache.set_route_trace(None)
    return seen
