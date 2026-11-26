"""Conformance harness for the generic C++ runtime: ``veystrix models port-check``.

Tiny synthetic pass (seconds, no SSD, no big model):

1. build a shrunken random-weight checkpoint from ``mlx_lm.models.<type>.ModelArgs``
   with the same layer pattern and the real checkpoint's quantization scheme
   (affine 4-bit group 64, 8-bit router and shared-expert gate for qwen3_5_moe);
2. run ``mlx_lm`` (the oracle) and the generic C++ runtime through the C ABI on
   the same scenarios: prompt lengths 16/64/256/1024, solo and N=2..4 sessions
   with ``vx_session_step_batch``, unchunked vs chunked prefill (RESET + APPEND
   and the runtime's internal chunking), snapshot/restore of a strict prefix,
   fork, a two-turn conversation, and every KV mode;
3. require identical greedy tokens and identical float32 logit SHA-256 digests.

KV modes: ``bf16`` is checked against mlx_lm ``KVCache``, ``int8`` against mlx_lm
``QuantizedKVCache`` (its quantized SDPA), the tiered/k8v4 modes against an
independent Python reference (``RefTieredKV``) that re-quantizes the original
activations at read time. For each quantized mode the bounded difference to
bf16 (max |logit diff|, top-1 agreement) is recorded, and bf16-vs-``tiered``
(bf16 recent window) must be bit-identical while the context fits the window.

Usage::

    veystrix models port-check --tiny [--lib PATH] [--json] [--quick]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

PAGE = 256

# --------------------------------------------------------------------------- #
# tiny checkpoint                                                              #
# --------------------------------------------------------------------------- #

TINY_QWEN3_5_MOE = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 128,
        "num_hidden_layers": 8,              # 2 x (3 GDN + 1 gated attention)
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 512,
        "max_position_embeddings": 262144,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "num_experts": 16,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "rope_parameters": {"type": "default", "mrope_section": [11, 11, 10],
                            "rope_theta": 10000000, "partial_rotary_factor": 0.25},
    },
}


def build_tiny(out_dir: str | Path, model_type: str = "qwen3_5_moe", seed: int = 7,
               variant: str = "mlx", text_overrides: dict | None = None) -> Path:
    """Write a tiny random checkpoint quantized like the real one.

    variant "mlx": names/layout as mlx_lm saves them (no norm shift on load).
    variant "mtp": additionally carries an ``mtp.`` tensor, so mlx_lm's sanitize
    shifts every 1-D norm weight by +1 on load; the recipe must do the same.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.models import qwen3_5_moe
    from mlx_lm.utils import quantize_model, save_config, save_model

    if model_type != "qwen3_5_moe":
        raise ValueError(f"no tiny builder for {model_type}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(TINY_QWEN3_5_MOE))
    config["text_config"].update(text_overrides or {})
    mx.random.seed(seed)
    model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(config))
    # Non-trivial norms/gains so a bug in any of them changes the logits.
    def perturb(path, value):
        if path.endswith("norm.weight") or path.endswith("layernorm.weight"):
            return (1.0 + 0.2 * mx.random.normal(value.shape)).astype(value.dtype)
        if path.endswith("dt_bias"):
            return mx.random.normal(value.shape) * 0.5
        return value
    params = dict(tree_flatten(model.parameters()))
    model.load_weights([(k, perturb(k, v)) for k, v in params.items()], strict=True)
    model.set_dtype(mx.bfloat16, lambda x: mx.issubdtype(x, mx.floating))
    # A_log stays float32 (cast_predicate in mlx_lm qwen3_5).
    model.load_weights([(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())
                        if k.endswith("A_log")], strict=False)
    model, qconfig = quantize_model(model, config, 64, 4, "affine", quant_predicate=model.quant_predicate)
    save_model(out, model, donate_model=False)
    save_config(qconfig, out / "config.json")
    if variant == "mtp":
        import mlx.core as mx2
        # Un-shift: mlx_lm will add +1 on load, so store weight-1 to keep the same model.
        weights = dict(mx2.load(str(out / "model.safetensors")))
        mx2.eval(list(weights.values()))
        for k in list(weights):
            if any(k.endswith(s) for s in (".input_layernorm.weight", ".post_attention_layernorm.weight",
                                           "model.norm.weight", ".q_norm.weight", ".k_norm.weight")):
                weights[k] = weights[k] - 1.0
        weights["mtp.fc.weight"] = mx2.zeros((4, 4), dtype=mx2.bfloat16)
        mx2.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
        index = out / "model.safetensors.index.json"
        if index.exists():
            data = json.loads(index.read_text())
            data["weight_map"] = {k: "model.safetensors" for k in weights}
            index.write_text(json.dumps(data))
    return out


# --------------------------------------------------------------------------- #
# tiny qwen4_exp (Qwen3.8 Flash-Next / Qwen 4 preview)                         #
# --------------------------------------------------------------------------- #

TINY_QWEN4_EXP = {
    "model_type": "qwen4_exp_text", "hidden_size": 128, "num_hidden_layers": 8, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 64, "rms_norm_eps": 1e-6, "vocab_size": 512,
    "max_position_embeddings": 262144, "linear_num_value_heads": 6, "linear_num_key_heads": 2,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4, "num_experts": 32, "num_experts_per_tok": 4, "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64, "norm_topk_prob": True, "tie_word_embeddings": False,
    "hc_count": 4, "hc_lowrank": 32, "ple_layer_ids": [2], "ple_embed_dim": 128, "ple_conv_kernel_size": 4,
    "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 1000, "split_ngram_parts": 4, "seed": 1234,
    "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 32, "indexer_budget": 2048,
    "indexer_compress_ratio": 4, "output_gate_type": "sigmoid", "hidden_act": "silu", "eos_token_id": 1,
    "rope_parameters": {"type": "default", "mrope_section": [11, 11, 10], "mrope_interleaved": True,
                        "rope_theta": 10000000, "partial_rotary_factor": 0.25},
}


def _qwen4_language_model(text: dict):
    from types import SimpleNamespace
    from mlx_vlm.models.qwen4_exp.config import TextConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    config = TextConfig.from_dict(text)
    stub = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=-1,
                           video_token_id=-2, vision_start_token_id=-3, text_config=config)
    return LanguageModel(config, stub)


def build_tiny_qwen4(out_dir: str | Path, seed: int = 7, overrides: dict | None = None) -> Path:
    """Tiny random qwen4_exp checkpoint laid out and quantized like Qwen3.8 Flash-Next (affine 4-bit group 32,
    dense router / norm gains / conv weights, quantized n-gram embedding shards), one-centered norm gains."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = json.loads(json.dumps(dict(TINY_QWEN4_EXP, **(overrides or {}))))
    mx.random.seed(seed)
    lm = _qwen4_language_model(text)
    # One-centered gains (the shipped convention): gain = w, so draw w around 1.
    def perturb(path, value):
        if path.endswith("norm.weight") or path.endswith("layernorm.weight") or ".norm_" in path:
            return (1.0 + 0.2 * mx.random.normal(value.shape)).astype(value.dtype)
        if path.endswith("dt_bias"):
            return mx.random.normal(value.shape) * 0.5
        if "hyper_connection" in path and path.endswith("weight"):
            return value * 4.0  # keep the stream mix away from a flat 0.5
        return value
    params = dict(tree_flatten(lm.parameters()))
    lm.load_weights([(k, perturb(k, v)) for k, v in params.items()], strict=False)
    lm.set_dtype(mx.bfloat16, lambda x: mx.issubdtype(x, mx.floating))

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.endswith("mlp.gate") or module.weight.shape[-1] % 32:
            return False
        return {"group_size": 32, "bits": 4, "mode": "affine"}
    nn.quantize(lm, group_size=32, bits=4, mode="affine", class_predicate=predicate)
    import re
    # Checkpoint layout: n-gram embedding shards are stored as ``shard_K`` (mlx_vlm's module path is ``shards.K``).
    weights = {f"language_model.{re.sub(r'ngram_embedding[.]shards[.](\d+)', r'ngram_embedding.shard_\1', k)}": v
               for k, v in tree_flatten(lm.parameters())}
    mx.eval(list(weights.values()))
    mx.save_safetensors(str(out / "model.safetensors"), weights, metadata={"format": "mlx"})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}}))
    config = {"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
              "text_config": text, "vision_config": {}, "tie_word_embeddings": False,
              "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
              "norm_convention": "one-centered"}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    return out


class Qwen4Oracle:
    """Production Flash-Next numerics on a tiny checkpoint: mlx_vlm qwen4_exp + models/flashnext patch_rmsnorm
    (one-centered gains) + adaptive_topk at threshold 1.0 (softmax top-k renormalised by the top-k mass)."""

    def __init__(self, path: Path, kv_mode: str = "bf16"):
        import mlx.core as mx
        import mlx.nn as nn
        from models.flashnext import adaptive_topk, patch_rmsnorm
        patch_rmsnorm.apply()
        adaptive_topk.apply()
        self.kv_mode = kv_mode
        config = json.loads((Path(path) / "config.json").read_text())
        self.lm = _qwen4_language_model(config["text_config"])
        import re
        weights = {re.sub(r"ngram_embedding[.]shard_(\d+)", r"ngram_embedding.shards.\1", k[len("language_model."):]): v
                   for k, v in mx.load(str(Path(path) / "model.safetensors")).items()}

        def predicate(p, module):
            return hasattr(module, "to_quantized") and f"{p}.scales" in weights
        nn.quantize(self.lm, group_size=32, bits=4, mode="affine", class_predicate=predicate)
        self.lm.load_weights(list(weights.items()), strict=True)
        patch_rmsnorm.configure(self.lm, one_centered=True)
        mx.eval(self.lm.parameters())
        self.lm.eval()

    def new(self):
        caches = self.lm.make_cache()
        if self.kv_mode == "int8":
            from mlx_vlm.models.qwen4_exp.language import QSAKVCache, QSAQuantizedKVCache
            for i, c in enumerate(caches):
                if isinstance(c, QSAKVCache):
                    caches[i] = QSAQuantizedKVCache(group_size=64, bits=8)
        elif self.kv_mode != "bf16":
            raise ValueError(f"Qwen4Oracle: unsupported kv_mode {self.kv_mode!r}")
        return caches

    def forward(self, cache, tokens):
        import mlx.core as mx
        h = self.lm.model(mx.array(tokens, dtype=mx.int32)[None], cache=cache)
        logits = self.lm.lm_head(h[:, -1:, :]).reshape(1, -1)
        mx.eval(logits)
        return logits


def run_tiny_qwen4(lib: str | None = None, quick: bool = False, workdir: str | None = None,
                   verbose: bool = False, indexer_budget: int | None = None) -> list[Check]:
    """qwen4_exp blocks (hyper-connection residual, Qwen4 GDN, gated attention, n-gram PLE, top-k MoE with a
    shared expert) against the production Flash-Next numerics on a tiny random checkpoint.

    ``indexer_budget`` (e.g. 16 or 32) shrinks the QSA budget so scenarios above ~budget tokens run the
    model's SPARSE attention (indexer top-k blocks + tail) in the reference; a C++ attention that went dense
    again fails these checks from step 0 (the real budget 2048 is above every length here)."""
    from veystrix.recipe import emit_recipe
    tmp = Path(workdir or tempfile.mkdtemp(prefix="vx-portcheck-q4-"))
    checks: list[Check] = _Live(verbose)
    label = "qwen4" if indexer_budget is None else f"qwen4 QSA-sparse(budget {indexer_budget})"
    ckpt = build_tiny_qwen4(tmp / ("tiny-qwen4" if indexer_budget is None else f"tiny-qwen4-b{indexer_budget}"),
                            overrides=None if indexer_budget is None else {"indexer_budget": indexer_budget})
    recipe_path = tmp / f"recipe-{ckpt.name}.json"
    recipe_path.write_text(json.dumps(emit_recipe(ckpt)))
    vocab = TINY_QWEN4_EXP["vocab_size"]
    gen = 6 if quick else 8
    oracle = Qwen4Oracle(ckpt)
    cpp = Cpp(lib, ckpt, recipe_path, "bf16")
    b = cpp.b
    for n in ([16, 64, 256] if quick else [16, 64, 256, 1024]):
        p = _prompt(n, n, vocab)
        checks.append(_cmp(f"{label} solo len={n}", oracle_run(oracle, [p], gen), cpp.run([p], gen)))
    p = _prompt(300, 11, vocab)
    plan = [100, 37, 128]
    checks.append(_cmp(f"{label} chunked RESET+APPEND [100,37,128,35]",
                       oracle_run(oracle, _chunks(p, plan), gen), cpp.run(_chunks(p, plan), gen)))
    p1, p2 = _prompt(50, 21, vocab), _prompt(23, 22, vocab)
    checks.append(_cmp(f"{label} two turns (APPEND after decode)",
                       oracle_run(oracle, [p1], gen, turns=[p2])[:2], cpp.run([p1], gen, turns=[p2])))
    prompts = [_prompt(40 + 17 * i, 100 + i, vocab) for i in range(2)]
    sess = [cpp.session() for _ in prompts]
    for s, q in zip(sess, prompts):
        b.prefill(s, q)
    got, dig = [[], []], [[], []]
    for _ in range(gen):
        for i, t in enumerate(b.step_batch(sess, ['{"greedy": true}'] * 2)):
            got[i].append(t)
            dig[i].append(b.last_digest(sess[i]))
    ok = all(oracle_run(oracle, [q], gen)[:2] == (got[i], dig[i]) for i, q in enumerate(prompts))
    for s in sess:
        b.close_session(s)
    checks.append(Check(f"{label} step_batch N=2", ok, "each lane equals its solo oracle" if ok else "lane mismatch"))
    full = _prompt(200, 31, vocab)
    snap = tmp / f"snap-{ckpt.name}"
    s = cpp.session()
    b.prefill(s, full[:120])
    b.save(s, str(snap))
    b.close_session(s)
    s = cpp.session()
    b.restore(s, str(snap))
    b.prefill_ex(s, full[120:], 1)
    toks, digs = [], []
    for _ in range(gen):
        toks.append(b.step(s, '{"greedy": true}'))
        digs.append(b.last_digest(s))
    b.close_session(s)
    checks.append(_cmp(f"{label} snapshot(120)/restore + APPEND(80) (PLE + GDN + KV state)",
                       oracle_run(oracle, [full[:120], full[120:]], gen), (toks, digs)))
    if indexer_budget is not None:
        # Threshold-crossing coverage: sparse activates once complete_blocks > budget // compress_ratio.
        # 1. Multi-chunk prefill crossing the threshold in the middle of chunk 1.
        c0 = max(6, indexer_budget - 6)
        p_mid = _prompt(c0 + 20 + 24, 51 + indexer_budget, vocab)
        mid_chunks = _chunks(p_mid, [c0, 20])
        checks.append(_cmp(f"{label} mid-chunk threshold crossing [{c0},20,24]",
                           oracle_run(oracle, mid_chunks, gen), cpp.run(mid_chunks, gen)))
        # 2. Decode crossing the threshold (prefill stays dense at budget+2; decode crosses budget+4).
        p_dec = _prompt(indexer_budget + 2, 61 + indexer_budget, vocab)
        checks.append(_cmp(f"{label} decode crossing threshold (prefill={len(p_dec)})",
                           oracle_run(oracle, [p_dec], gen), cpp.run([p_dec], gen)))
        # 3. step_batch N=2 with one session strictly below threshold (dense) and one above (sparse).
        mixed_prompts = [_prompt(max(8, indexer_budget - 8), 71 + indexer_budget, vocab),
                         _prompt(indexer_budget + 44, 72 + indexer_budget, vocab)]
        msess = [cpp.session() for _ in mixed_prompts]
        for s, q in zip(msess, mixed_prompts):
            b.prefill(s, q)
        mgot, mdig = [[], []], [[], []]
        for _ in range(gen):
            for i, t in enumerate(b.step_batch(msess, ['{"greedy": true}'] * 2)):
                mgot[i].append(t)
                mdig[i].append(b.last_digest(msess[i]))
        mok = all(oracle_run(oracle, [q], gen)[:2] == (mgot[i], mdig[i]) for i, q in enumerate(mixed_prompts))
        for s in msess:
            b.close_session(s)
        checks.append(Check(f"{label} step_batch N=2 mixed dense+sparse", mok,
                            "each lane equals its solo oracle" if mok else "lane mismatch"))
        # 4. Snapshot below threshold (with pending indexer keys) -> restore -> APPEND across threshold.
        pre_len = indexer_budget - 3
        p_snap = _prompt(pre_len + 40, 81 + indexer_budget, vocab)
        snap_pre = tmp / f"snap-pre-{ckpt.name}"
        s = cpp.session()
        b.prefill(s, p_snap[:pre_len])
        b.save(s, str(snap_pre))
        b.close_session(s)
        s = cpp.session()
        b.restore(s, str(snap_pre))
        b.prefill_ex(s, p_snap[pre_len:], 1)
        toks, digs = [], []
        for _ in range(gen):
            toks.append(b.step(s, '{"greedy": true}'))
            digs.append(b.last_digest(s))
        b.close_session(s)
        checks.append(_cmp(f"{label} snapshot({pre_len})/restore + APPEND(40) across threshold",
                           oracle_run(oracle, [p_snap[:pre_len], p_snap[pre_len:]], gen), (toks, digs)))
        # 5. Fork right at the threshold and decode both branches across it.
        p_fork = _prompt(indexer_budget + 2, 91 + indexer_budget, vocab)
        src = cpp.session()
        b.prefill(src, p_fork)
        head_t, head_d = [], []
        for _ in range(2):
            head_t.append(b.step(src, '{"greedy": true}'))
            head_d.append(b.last_digest(src))
        dst = cpp.session()
        b.fork(src, dst)
        ta, da, tb, db = [], [], [], []
        for _ in range(gen - 2):
            ta.append(b.step(src, '{"greedy": true}'))
            da.append(b.last_digest(src))
            tb.append(b.step(dst, '{"greedy": true}'))
            db.append(b.last_digest(dst))
        b.close_session(src)
        b.close_session(dst)
        ref_fork = oracle_run(oracle, [p_fork], gen)[:2]
        fok = (head_t + ta, head_d + da) == ref_fork and (head_t + tb, head_d + db) == ref_fork
        checks.append(Check(f"{label} fork across threshold", fok,
                            "both branches match oracle" if fok else "branch mismatch"))
        # 6. INT8 KV with sparse QSA mask vs mlx_vlm QSAQuantizedKVCache(bits=8, group_size=64).
        if indexer_budget == 16:
            orc8 = Qwen4Oracle(ckpt, kv_mode="int8")
            cpp8 = Cpp(lib, ckpt, recipe_path, "int8")
            p8 = _prompt(64, 141, vocab)
            checks.append(_cmp(f"{label} kv=int8 solo len=64",
                               oracle_run(orc8, [p8], gen), cpp8.run([p8], gen)))
            checks.append(_cmp(f"{label} kv=int8 mid-chunk [{c0},20,24]",
                               oracle_run(orc8, mid_chunks, gen), cpp8.run(mid_chunks, gen)))
            cpp8.close()
    cpp.close()
    if workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return checks


# --------------------------------------------------------------------------- #
# reference KV caches for the oracle                                           #
# --------------------------------------------------------------------------- #

KV_MODES = {
    # name: (prefix_tokens, prefix_bits, recent_tokens(-1 unbounded), recent_bits, hist_k, hist_v)
    "bf16": (0, 16, -1, 16, 16, 16),
    "int8": (0, 16, 0, 16, 8, 8),
    "k8v4": (0, 16, 0, 16, 8, 4),
    "int4": (0, 16, 0, 16, 4, 4),
    "tiered": (0, 16, 8192, 16, 8, 8),
    "tiered4": (0, 16, 8192, 8, 4, 4),
    "tiered4+": (1024, 8, 8192, 8, 4, 4),
    "tiered-k8v4": (1024, 8, 8192, 8, 8, 4),
}


def kv_policy(mode: str, window: int | None = None, prefix: int | None = None) -> tuple:
    p, pb, w, rb, hk, hv = KV_MODES[mode]
    if window is not None:
        w = window
    if prefix is not None:
        p = prefix
    return p, pb, w, rb, hk, hv


SCORE_TILE = 1 << 26      # cpp/engine/src/generic/kv_store.hpp kScoreTile


def attention_query_block(L: int, kL: int, Hq: int) -> int:
    """Query rows per block of the quantised attention (kv_store.cpp attention_query_block)."""
    tile = int(os.environ.get("VEYSTRIX_KV_SCORE_TILE") or 0) or SCORE_TILE
    qb = tile // (Hq * max(kL, 1))
    if qb >= L:
        return L
    return min(max(16, (qb // 16) * 16), L)


def query_spans(L: int, qb: int) -> list[tuple[int, int]]:
    """kv_store.cpp query_spans: blocks of qb rows; a remainder under 16 rows joins the block before it."""
    out, a, qb = [], 0, max(qb, 1)
    while a < L:
        b = min(L, a + qb)
        if 0 < L - b < 16:
            b = L
        out.append((a, b))
        a = b
    return out


class RefTieredKV:
    """Independent Python reference of the C++ KVStore regions (see kv_store.hpp).

    Keeps the ORIGINAL bf16 keys/values and quantizes the needed slices at read
    time (per-token groups along D, so slicing commutes with quantization).
    """

    def __init__(self, policy: tuple, group_size: int = 64):
        self.P, self.pb, self.W, self.rb, self.hk, self.hv = policy
        self.gs = group_size
        self.keys = None
        self.values = None
        self.offset = 0
        self.lengths = None

    # mlx_lm cache protocol -------------------------------------------------
    def make_mask(self, N, return_array=False, window_size=None):
        return None if N == 1 else "causal"

    def update_and_fetch(self, keys, values):
        import mlx.core as mx
        self.keys = keys if self.keys is None else mx.concatenate([self.keys, keys], axis=2)
        self.values = values if self.values is None else mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self, self

    @property
    def state(self):
        return self.keys, self.values

    def is_trimmable(self):
        return False

    # regions -----------------------------------------------------------------
    def history_end(self) -> int:
        n, P, W = self.offset, self.P, self.W
        if W < 0:
            return min(n, P)
        if W == 0:
            return n
        h = ((n - W) // PAGE) * PAGE if n > W else 0
        return max(min(n, P), h)

    def parts(self):
        n = self.offset
        H = self.history_end()
        out = []
        pre = min(n, self.P)
        if pre > 0:
            out.append((0, pre, self.pb, self.pb))
        if self.W >= 0 and H > self.P:
            out.append((self.P, H, self.hk, self.hv))
        if self.W != 0 and n > max(H, self.P):
            out.append((max(H, self.P), n, self.rb, self.rb))
        return out

    def attend(self, queries, scale, mask):
        import mlx.core as mx
        parts = self.parts()
        if len(parts) == 1 and parts[0][2] == 16 and parts[0][3] == 16:
            a, b = parts[0][:2]
            return mx.fast.scaled_dot_product_attention(
                queries, self.keys[..., a:b, :], self.values[..., a:b, :], scale=scale, mask=mask)
        B, Hq, L, D = queries.shape
        Hkv = self.keys.shape[1]
        rep = Hq // Hkv
        q = queries * scale
        if rep > 1:
            q = mx.reshape(q, (B, Hkv, rep, L, D))
        keys, vals = [], []
        for a, b, kb, vb in parts:
            k = self.keys[..., a:b, :]
            v = self.values[..., a:b, :]
            kq = (k,) if kb == 16 else tuple(mx.quantize(k, group_size=self.gs, bits=kb))
            vq = (v,) if vb == 16 else tuple(mx.quantize(v, group_size=self.gs, bits=vb))
            if rep > 1:
                kq = tuple(mx.expand_dims(x, axis=-3) for x in kq)
                vq = tuple(mx.expand_dims(x, axis=-3) for x in vq)
            keys.append((kq, kb))
            vals.append((vq, vb, b - a))
        kL = sum(n for _, _, n in vals)

        def rows(a, b):
            qa = q if b - a == L else q[..., a:b, :]
            scores = [qa @ mx.swapaxes(kq[0], -1, -2) if kb == 16 else
                      mx.quantized_matmul(qa, *kq, transpose=True, group_size=self.gs, bits=kb) for kq, kb in keys]
            s = scores[0] if len(scores) == 1 else mx.concatenate(scores, axis=-1)
            if mask is not None:
                qi = mx.arange(kL - L + a, kL - L + b)
                ki = mx.arange(kL)
                m = qi[:, None] >= ki[None]
                s = mx.where(m, s, mx.finfo(s.dtype).min)
            s = mx.softmax(s, axis=-1, precise=True)
            out = None
            at = 0
            for vq, vb, n in vals:
                p = s if len(vals) == 1 else s[..., at:at + n]
                at += n
                o = p @ vq[0] if vb == 16 else mx.quantized_matmul(p, *vq, transpose=False,
                                                                    group_size=self.gs, bits=vb)
                out = o if out is None else out + o
            return out

        spans = query_spans(L, attention_query_block(L, kL, Hq))   # the C++ KVStore's score-tile blocks
        out = rows(*spans[0])
        if len(spans) > 1:
            outs = [out]
            for a, b in spans[1:]:
                mx.eval(outs[-1])
                outs.append(rows(a, b))
            out = mx.concatenate(outs, axis=-2)
        if rep > 1:
            out = mx.reshape(out, (B, Hq, L, D))
        return out


_PATCHED = False


def _patch_attention():
    """Route mlx_lm's attention through RefTieredKV when that cache is used."""
    global _PATCHED
    if _PATCHED:
        return
    from mlx_lm.models import qwen3_next
    original = qwen3_next.scaled_dot_product_attention

    def sdpa(queries, keys, values, cache, scale, mask, **kw):
        if isinstance(cache, RefTieredKV):
            return cache.attend(queries, scale, mask)
        return original(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)

    qwen3_next.scaled_dot_product_attention = sdpa
    _PATCHED = True


# --------------------------------------------------------------------------- #
# oracle (mlx_lm) and C++ runners                                              #
# --------------------------------------------------------------------------- #

def digest(logits) -> str:
    import mlx.core as mx
    import numpy as np
    return hashlib.sha256(np.array(logits.astype(mx.float32)).reshape(-1).tobytes()).hexdigest()


class Oracle:
    def __init__(self, path: Path, kv_mode: str = "bf16", window: int | None = None,
                 prefix: int | None = None):
        from mlx_lm.models.cache import KVCache, QuantizedKVCache
        from mlx_lm.utils import load_model
        self.model, _ = load_model(Path(path))
        self.lm = self.model.language_model
        self.kv_mode = kv_mode
        self.policy = kv_policy(kv_mode, window, prefix)
        self._KVCache, self._QKV = KVCache, QuantizedKVCache
        if kv_mode not in ("bf16", "int8"):
            _patch_attention()

    def new(self):
        caches = self.lm.make_cache()
        for i, c in enumerate(caches):
            if isinstance(c, self._KVCache):
                if self.kv_mode == "int8":
                    caches[i] = self._QKV(group_size=64, bits=8)
                elif self.kv_mode != "bf16":
                    caches[i] = RefTieredKV(self.policy)
        return caches

    def forward(self, cache, tokens):
        import mlx.core as mx
        h = self.lm.model(mx.array(tokens, dtype=mx.int32)[None], cache)
        head = getattr(self.lm, "lm_head", None)
        last = h[:, -1:, :]
        logits = (head(last) if head is not None else self.lm.model.embed_tokens.as_linear(last)).reshape(1, -1)
        mx.eval(logits)
        return logits


# --------------------------------------------------------------------------- #
# streaming oracle for real checkpoints (experts read per layer, like the C++) #
# --------------------------------------------------------------------------- #

class _ShardIndex:
    """safetensors headers -> (path, absolute offset, nbytes, dtype, shape) per tensor; no data read."""

    def __init__(self, root: Path):
        import struct
        self.tensors = {}
        for shard in sorted(Path(root).glob("*.safetensors")):
            with open(shard, "rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                start, end = meta["data_offsets"]
                self.tensors[name] = (shard, 8 + length + start, end - start, meta["dtype"], meta["shape"])
        self._fds = {}

    def find(self, suffix: str) -> str:
        hits = [n for n in self.tensors if n.endswith(suffix) and "mtp." not in n]
        if len(hits) != 1:
            raise KeyError(f"expected one tensor ending with {suffix!r}, found {hits}")
        return hits[0]

    def rows(self, name: str, wanted: list[int]):
        """Rows ``wanted`` (sorted expert ids) of a stacked tensor as one mx.array, by pread."""
        import mlx.core as mx
        import numpy as np
        shard, offset, nbytes, dtype, shape = self.tensors[name]
        fd = self._fds.get(shard)
        if fd is None:
            fd = self._fds[shard] = os.open(shard, os.O_RDONLY)
        row = nbytes // shape[0]
        buf = np.empty(row * len(wanted), dtype=np.uint8)
        view = memoryview(buf)
        at = 0
        i = 0
        while i < len(wanted):                       # coalesce consecutive experts into one read
            j = i + 1
            while j < len(wanted) and wanted[j] == wanted[j - 1] + 1:
                j += 1
            length = row * (j - i)
            got = os.preadv(fd, [view[at:at + length]], offset + wanted[i] * row)
            if got != length:
                raise OSError(f"short read in {name}")
            at += length
            i = j
        np_dtype = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}[dtype]
        arr = mx.array(buf.view(np_dtype).reshape([len(wanted), *shape[1:]]))
        del buf, view
        return arr.view(mx.bfloat16) if dtype == "BF16" else arr

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds = {}


def _EMPTY():
    import mlx.core as mx
    return mx.zeros((1,), dtype=mx.uint32)


class StreamingSwitchGLU:
    """Drop-in for an mlx_lm SwitchGLU whose expert stacks stay on disk.

    Each call reads only the experts the router picked (union over the batch), builds the
    compact stacks and runs mlx_lm's own SwitchGLU code on remapped indices. The remap is
    monotonic, so the sort order and therefore every kernel input is the same as with the
    full stack (bit-identical; checked on the tiny model)."""

    def __init__(self, template, index: _ShardIndex, layer: int, num_experts: int):
        self.template = template
        self.index = index
        self.layer = layer
        self.num_experts = num_experts
        self.names = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for kind in ("weight", "scales", "biases"):
                try:
                    self.names[(proj, kind)] = index.find(f"layers.{layer}.mlp.switch_mlp.{proj}.{kind}")
                except KeyError:
                    if kind != "biases":
                        raise
        self.bytes_read = 0

    def __call__(self, x, indices):
        import mlx.core as mx
        mx.eval(indices)
        wanted = sorted(set(int(v) for v in indices.reshape(-1).tolist()))
        remap = [0] * self.num_experts
        for row, expert in enumerate(wanted):
            remap[expert] = row
        for (proj, kind), name in self.names.items():
            arr = self.index.rows(name, wanted)
            self.bytes_read += arr.nbytes
            setattr(getattr(self.template, proj), kind, arr)
        local = mx.take(mx.array(remap, dtype=mx.uint32), indices.astype(mx.uint32))
        out = self.template(x, local)
        # Bound memory: finish this layer now and drop the compact stacks, so at most one layer's
        # experts are alive (evaluation boundaries do not change any kernel's result).
        mx.eval(out)
        for (proj, kind) in self.names:
            setattr(getattr(self.template, proj), kind, _EMPTY())
        mx.clear_cache()
        return out


class StreamingOracle(Oracle):
    """mlx_lm on a checkpoint larger than RAM: dense weights loaded, experts streamed per layer."""

    def __init__(self, path: Path, kv_mode: str = "bf16", window: int | None = None,
                 prefix: int | None = None):
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from mlx_lm.models.cache import KVCache, QuantizedKVCache
        mx.set_cache_limit(256 << 20)
        self.model, _ = load_model(Path(path), lazy=True)
        self.lm = self.model.language_model
        for name in ("vision_tower", "visual", "mtp"):          # never evaluated by the text oracle
            for owner in (self.model, self.lm, getattr(self.lm, "model", None)):
                if owner is not None and name in owner:
                    owner.pop(name)
        self.index = _ShardIndex(Path(path))
        self.streams = []
        for i, layer in enumerate(self.lm.model.layers):
            mlp = getattr(layer, "mlp", None)
            switch = getattr(mlp, "switch_mlp", None)
            if switch is None:
                continue
            for proj in ("gate_proj", "up_proj", "down_proj"):       # drop the lazy full stacks unread
                for kind in ("weight", "scales", "biases"):
                    if kind in getattr(switch, proj):
                        setattr(getattr(switch, proj), kind, mx.zeros((1,), dtype=mx.uint32))
            stream = StreamingSwitchGLU(switch, self.index, i, int(mlp.num_experts))
            mlp.switch_mlp = stream
            self.streams.append(stream)
        mx.eval(self.model.parameters())
        self.kv_mode = kv_mode
        self.policy = kv_policy(kv_mode, window, prefix)
        self._KVCache, self._QKV = KVCache, QuantizedKVCache
        if kv_mode not in ("bf16", "int8"):
            _patch_attention()

    @property
    def bytes_read(self) -> int:
        return sum(s.bytes_read for s in self.streams)


def oracle_run(oracle: Oracle, chunks: list[list[int]], gen: int, turns: list[list[int]] | None = None):
    """Prefill in the given chunks, greedy-decode `gen`, then each extra turn:
    commit the pending token, APPEND the turn's tokens, decode `gen` again."""
    import mlx.core as mx
    cache = oracle.new()
    logits = None
    for ch in chunks:
        logits = oracle.forward(cache, ch)
    toks, digs = [], []
    def decode(logits):
        pending = None
        for i in range(gen):
            if pending is not None:
                logits = oracle.forward(cache, [pending])
            t = int(mx.argmax(logits.reshape(-1)).item())
            toks.append(t)
            digs.append(digest(logits))
            pending = t
        return pending
    pending = decode(logits)
    for turn in turns or []:
        oracle.forward(cache, [pending])
        logits = oracle.forward(cache, turn)
        pending = decode(logits)
    return toks, digs, logits


class Cpp:
    def __init__(self, lib: str | None, ckpt: Path, recipe: Path | None, kv: str = "bf16",
                 prefill_chunk: int = 2048, window: int | None = None, prefix: int | None = None,
                 profile: dict | None = None, hist_bits: tuple[int, int] | None = None):
        from veystrix.backends.cpp_engine import CtypesVxBinding
        self.b = CtypesVxBinding(lib)
        if profile is None:
            profile = {"recipe": str(recipe), "kv": kv, "digests": True, "prefill_chunk": prefill_chunk}
            if window is not None:
                profile["kv_window"] = window
            if prefix is not None:
                profile["kv_prefix"] = prefix
            if hist_bits is not None:
                profile["kv_hist_k_bits"], profile["kv_hist_v_bits"] = hist_bits
        self.engine = self.b.open_engine(str(ckpt), json.dumps(profile))

    def close(self):
        self.b.close_engine(self.engine)

    def session(self):
        return self.b.open_session(self.engine)

    def run(self, chunks, gen, turns=None, session=None):
        s = session or self.session()
        self.b.prefill(s, chunks[0])
        for ch in chunks[1:]:
            self.b.prefill_ex(s, ch, 1)
        toks, digs = [], []
        def decode():
            for _ in range(gen):
                toks.append(self.b.step(s, '{"greedy": true}'))
                digs.append(self.b.last_digest(s))
        decode()
        for turn in turns or []:
            self.b.prefill_ex(s, turn, 1)
            decode()
        if session is None:
            self.b.close_session(s)
        return toks, digs


# --------------------------------------------------------------------------- #
# scenarios                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    extra: dict = field(default_factory=dict)


def _prompt(n: int, seed: int, vocab: int) -> list[int]:
    import random
    rng = random.Random(seed)
    return [rng.randrange(1, vocab) for _ in range(n)]


def _chunks(tokens, plan):
    out, at = [], 0
    for size in plan:
        out.append(tokens[at:at + size])
        at += size
    if at < len(tokens):
        out.append(tokens[at:])
    return [c for c in out if c]


def _cmp(name, a, b, extra=None) -> Check:
    ta, da = a[0], a[1]
    tb, db = b[0], b[1]
    ok = ta == tb and da == db
    first = next((i for i, (x, y) in enumerate(zip(da, db)) if x != y), None)
    detail = "tokens+digests identical" if ok else (
        f"tokens {'same' if ta == tb else 'DIFFER'}; first digest mismatch at step {first}")
    return Check(name, ok, detail, extra or {})


class _Live(list):
    """Check list that prints each result as it lands (a crash keeps the evidence)."""

    def __init__(self, verbose: bool):
        super().__init__()
        self.verbose = verbose

    def append(self, c):
        super().append(c)
        if self.verbose:
            print(f"{'PASS' if c.ok else 'FAIL'}  {c.name}: {c.detail}", flush=True)


def run_tiny(lib: str | None = None, quick: bool = False, workdir: str | None = None,
             modes: list[str] | None = None, verbose: bool = False) -> list[Check]:
    from veystrix.recipe import emit_recipe
    import mlx.core as mx

    tmp = Path(workdir or tempfile.mkdtemp(prefix="vx-portcheck-"))
    checks: list[Check] = _Live(verbose)
    ckpt = build_tiny(tmp / "tiny-mlx")
    recipe_path = tmp / "recipe.json"
    recipe_path.write_text(json.dumps(emit_recipe(ckpt)))
    vocab = TINY_QWEN3_5_MOE["text_config"]["vocab_size"]
    gen = 6 if quick else 8
    lengths = [16, 64, 256] if quick else [16, 64, 256, 1024]

    # ---- bf16: solo, chunking, sessions, snapshot, fork, two turns ----------
    oracle = Oracle(ckpt, "bf16")
    cpp = Cpp(lib, ckpt, recipe_path, "bf16")
    for n in lengths:
        p = _prompt(n, n, vocab)
        checks.append(_cmp(f"bf16 solo len={n}", oracle_run(oracle, [p], gen), cpp.run([p], gen)))
    # chunked: RESET + APPENDs with uneven chunks (oracle uses the same plan)
    p = _prompt(300, 11, vocab)
    plan = [100, 37, 128]
    checks.append(_cmp("bf16 chunked RESET+APPEND [100,37,128,35]",
                       oracle_run(oracle, _chunks(p, plan), gen), cpp.run(_chunks(p, plan), gen)))
    cpp.close()
    # internal chunking (prefill_chunk=96) == oracle plan of 96s
    cpp96 = Cpp(lib, ckpt, recipe_path, "bf16", prefill_chunk=96)
    p = _prompt(300, 12, vocab)
    checks.append(_cmp("bf16 internal chunk=96",
                       oracle_run(oracle, _chunks(p, [96, 96, 96]), gen), cpp96.run([p], gen)))
    # unchunked vs chunked differ only by kernel shapes: record, do not require
    un = oracle_run(oracle, [p], gen)
    ch = oracle_run(oracle, _chunks(p, [96, 96, 96]), gen)
    checks.append(Check("oracle chunked vs unchunked (info)", True,
                        f"tokens {'same' if un[0] == ch[0] else 'differ'} "
                        f"(chunk plans are part of the contract; C++ follows the caller's plan)"))
    cpp96.close()

    cpp = Cpp(lib, ckpt, recipe_path, "bf16")
    b = cpp.b
    # N=2..4 sessions via step_batch, each must equal its solo oracle run
    for nsess in (2, 3, 4):
        prompts = [_prompt(40 + 17 * i, 100 + i, vocab) for i in range(nsess)]
        sess = [cpp.session() for _ in prompts]
        for s, p in zip(sess, prompts):
            b.prefill(s, p)
        got = [[] for _ in sess]
        dig = [[] for _ in sess]
        for _ in range(gen):
            out = b.step_batch(sess, ['{"greedy": true}'] * nsess)
            for i, t in enumerate(out):
                got[i].append(t)
                dig[i].append(b.last_digest(sess[i]))
        ok = True
        for i, p in enumerate(prompts):
            ref = oracle_run(oracle, [p], gen)
            ok = ok and ref[0] == got[i] and ref[1] == dig[i]
        for s in sess:
            b.close_session(s)
        checks.append(Check(f"bf16 step_batch N={nsess}", ok,
                            "each lane equals its solo oracle" if ok else "lane mismatch"))
    # two-turn conversation
    p1, p2 = _prompt(50, 21, vocab), _prompt(23, 22, vocab)
    checks.append(_cmp("bf16 two turns (APPEND after decode)",
                       oracle_run(oracle, [p1], gen, turns=[p2])[:2], cpp.run([p1], gen, turns=[p2])))
    # snapshot of a strict prefix, restore into a fresh session, append rest
    full = _prompt(200, 31, vocab)
    snap = tmp / "snap"
    s = cpp.session()
    b.prefill(s, full[:120])
    b.save(s, str(snap))
    b.close_session(s)
    s2 = cpp.session()
    b.restore(s2, str(snap))
    b.prefill_ex(s2, full[120:], 1)
    toks, digs = [], []
    for _ in range(gen):
        toks.append(b.step(s2, '{"greedy": true}'))
        digs.append(b.last_digest(s2))
    b.close_session(s2)
    checks.append(_cmp("bf16 snapshot(120)/restore + APPEND(80)",
                       oracle_run(oracle, [full[:120], full[120:]], gen), (toks, digs)))
    # snapshot right after restore without append: decode from stored logits
    s3 = cpp.session()
    b.restore(s3, str(snap))
    t3 = [b.step(s3, '{"greedy": true}') for _ in range(gen)]
    b.close_session(s3)
    ref = oracle_run(oracle, [full[:120]], gen)
    checks.append(Check("bf16 restore then decode", t3 == ref[0], "tokens identical" if t3 == ref[0] else "differ"))
    # fork mid-decode: both branches continue identically to the oracle
    p = _prompt(70, 41, vocab)
    src = cpp.session()
    b.prefill(src, p)
    head = [b.step(src, '{"greedy": true}') for _ in range(3)]
    dst = cpp.session()
    b.fork(src, dst)
    ta = [b.step(src, '{"greedy": true}') for _ in range(gen - 3)]
    tb = [b.step(dst, '{"greedy": true}') for _ in range(gen - 3)]
    ref = oracle_run(oracle, [p], gen)[0]
    ok = head + ta == ref and head + tb == ref
    b.close_session(src)
    b.close_session(dst)
    checks.append(Check("bf16 fork after 3 steps", ok, "both branches equal the oracle" if ok else "differ"))
    # seeded non-greedy: same seed -> same tokens (solo and batched), cancel
    samp = json.dumps({"greedy": False, "temperature": 0.6, "top_p": 0.95, "top_k": 20})
    runs = []
    for _ in range(2):
        s = cpp.session()
        b.seed(s, 1234)
        b.prefill(s, p)
        runs.append([b.step(s, samp) for _ in range(gen)])
        b.close_session(s)
    checks.append(Check("seeded non-greedy reproducible", runs[0] == runs[1], str(runs[0])))
    caps = b.capabilities(cpp.engine)
    checks.append(Check("capabilities report generic engine + kv", caps.get("engine") == "generic"
                        and caps.get("kv", {}).get("mode") == "bf16", json.dumps(caps)[:300]))
    cpp.close()

    # ---- KV modes ----------------------------------------------------------------
    ref_logits = {}
    for mode in (modes or ["int8", "k8v4", "tiered", "tiered4", "tiered4+", "tiered-k8v4", "int4"]):
        # Small windows so the tiny run crosses every boundary: prefix 256,
        # window 512 (page 256): lengths below hit prefix-only, prefix+recent,
        # and prefix+middle+recent with the window edge moving page by page.
        window = None if mode in ("int8", "k8v4", "int4") else 512
        prefix = 256 if mode in ("tiered4+", "tiered-k8v4") else None
        orc = Oracle(ckpt, mode, window=window, prefix=prefix)
        c = Cpp(lib, ckpt, recipe_path, mode, window=window, prefix=prefix)
        for n in ([200, 764] if quick else [200, 512, 764, 1100]):
            p = _prompt(n, 500 + n, vocab)
            # decode across the window edge: gen tokens appended one by one
            a = oracle_run(orc, _chunks(p, [128] * 16), gen)
            got = c.run(_chunks(p, [128] * 16), gen)
            chk = _cmp(f"kv={mode} len={n} vs reference", a[:2], got)
            # bounded difference to bf16 on the same plan
            bref = oracle_run(oracle, _chunks(p, [128] * 16), gen)
            lb, lq = bref[2].astype(mx.float32), a[2].astype(mx.float32)
            diff = float(mx.max(mx.abs(lb - lq)).item())
            agree = sum(int(x == y) for x, y in zip(bref[0], a[0])) / len(a[0])
            chk.extra = {"max_abs_logit_diff_vs_bf16_last_step": round(diff, 4),
                         "greedy_agreement_vs_bf16": agree}
            chk.detail += f"; vs bf16: max|dlogit|={diff:.4f} agree={agree:.2f}"
            checks.append(chk)
        # inside the exact window, `tiered` (bf16 recent) must equal bf16 bit-for-bit
        if mode == "tiered":
            p = _prompt(300, 77, vocab)
            same = oracle_run(oracle, [p], gen)[:2]
            checks.append(_cmp("kv=tiered inside window == bf16 (C++)", same, c.run([p], gen)))
        # snapshot/restore into the same plan; restore into another plan refused
        p = _prompt(800, 88, vocab)
        s = c.session()
        b2 = c.b
        b2.prefill(s, p[:128])
        for ch in _chunks(p[128:600], [128] * 8):
            b2.prefill_ex(s, ch, 1)
        snapdir = tmp / f"snap-{mode}"
        b2.save(s, str(snapdir))
        b2.close_session(s)
        s = c.session()
        b2.restore(s, str(snapdir))
        b2.prefill_ex(s, p[600:], 1)
        toks = []
        digs = []
        for _ in range(gen):
            toks.append(b2.step(s, '{"greedy": true}'))
            digs.append(b2.last_digest(s))
        b2.close_session(s)
        checks.append(_cmp(f"kv={mode} snapshot/restore across tiers",
                           oracle_run(orc, _chunks(p[:600], [128] * 8) + [p[600:]], gen)[:2], (toks, digs)))
        c.close()
    # restoring a bf16 snapshot into an int8 engine must fail closed
    c8 = Cpp(lib, ckpt, recipe_path, "int8")
    s = c8.session()
    try:
        c8.b.restore(s, str(snap))
        refused = False
    except Exception:
        refused = True
    c8.b.close_session(s)
    c8.close()
    checks.append(Check("snapshot from another KV plan is refused", refused, ""))

    # ---- norm-shift variant (checkpoint still carrying MTP weights) ----------
    ck2 = build_tiny(tmp / "tiny-mtp", variant="mtp")
    r2 = tmp / "recipe-mtp.json"
    rec2 = emit_recipe(ck2)
    r2.write_text(json.dumps(rec2))
    o2 = Oracle(ck2, "bf16")
    c2 = Cpp(lib, ck2, r2, "bf16")
    p = _prompt(64, 5, vocab)
    checks.append(_cmp("norm_shift variant (mtp tensors present)", oracle_run(o2, [p], gen)[:2], c2.run([p], gen)))
    checks.append(Check("recipe derived norm_shift=true for mtp variant", rec2["norm_shift"] is True, ""))
    c2.close()
    if workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return checks


REAL_TEXT = (
    "The engine streams mixture-of-experts weights from the SSD and keeps the dense core resident. "
    "Each token activates a few experts per layer; the operating system page cache is the expert cache. "
    "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n"
    "Explain what this function does, its complexity, and one edge case. "
    "Then list three ways to verify that a numerical port of a neural network is exact, "
    "and why per-step logit digests catch errors that comparing generated text misses. "
)


def _real_prompt(tokenizer, n: int, salt: int) -> list[int]:
    """A fresh chat prompt of exactly n tokens (template + text, truncated/padded deterministically)."""
    messages = [{"role": "user", "content": f"[{salt}] " + REAL_TEXT * (1 + n // 120)}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if len(ids) >= n:
        return ids[-n:]
    return (ids * (n // len(ids) + 1))[:n]


def real_scenarios(tokenizer, lengths=(16, 64, 256, 1024), gen: int = 16, kv_modes=()) -> list[dict]:
    """The real-pass scenarios, built identically in the oracle and the engine process."""
    out = []
    for n in lengths:
        out.append({"name": f"real solo len={n}", "kind": "solo", "kv": "bf16",
                    "prompts": [_real_prompt(tokenizer, n, n)], "gen": gen})
    out.append({"name": "real step_batch N=2", "kind": "batch", "kv": "bf16",
                "prompts": [_real_prompt(tokenizer, 200 + 37 * i, 900 + i) for i in range(2)], "gen": gen})
    out.append({"name": "real two turns", "kind": "turns", "kv": "bf16",
                "prompts": [_real_prompt(tokenizer, 300, 1), _real_prompt(tokenizer, 40, 2)], "gen": gen})
    for mode in kv_modes:
        # window 512 / prefix 256 so a 1100-token prompt crosses every region edge
        out.append({"name": f"real kv={mode} len=1100", "kind": "solo", "kv": mode,
                    "window": None if mode in ("int8", "k8v4", "int4") else 512,
                    "prefix": 256 if mode in ("tiered4+", "tiered-k8v4") else None,
                    "prompts": [_real_prompt(tokenizer, 1100, 77)], "gen": gen})
    return out


def _plan2048(p):
    return _chunks(p, [2048] * (len(p) // 2048 + 1))


def _footprint() -> dict:
    import resource
    import mlx.core as mx
    return {"max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3, 2),
            "mlx_peak_gib": round(mx.get_peak_memory() / 1024**3, 2)}


def run_real_oracle(checkpoint: str, out_path: str, lengths, gen: int, kv_modes=()) -> int:
    """Process 1: streaming mlx_lm oracle; writes tokens + per-step float32 logit digests (JSON, per scenario)."""
    from transformers import AutoTokenizer
    ckpt = Path(checkpoint).expanduser()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    t0 = time.time()
    oracles = {}
    result = {"checkpoint": str(ckpt), "scenarios": {}}
    for sc in real_scenarios(tok, lengths, gen, kv_modes):
        key = (sc["kv"], sc.get("window"), sc.get("prefix"))
        if key not in oracles:
            oracles.clear()
            import mlx.core as mx
            mx.clear_cache()
            oracles[key] = StreamingOracle(ckpt, sc["kv"], window=sc.get("window"), prefix=sc.get("prefix"))
            print(f"oracle {key} loaded in {time.time() - t0:.1f}s {_footprint()}", flush=True)
        orc = oracles[key]
        ta = time.time()
        if sc["kind"] == "turns":
            runs = [oracle_run(orc, _plan2048(sc["prompts"][0]), sc["gen"], turns=[sc["prompts"][1]])[:2]]
        else:
            runs = [oracle_run(orc, _plan2048(p), sc["gen"])[:2] for p in sc["prompts"]]
        result["scenarios"][sc["name"]] = {"runs": runs, "seconds": round(time.time() - ta, 1),
                                           "expert_bytes": orc.bytes_read, **_footprint()}
        Path(out_path).write_text(json.dumps(result))
        print(f"oracle {sc['name']}: {time.time() - ta:.1f}s {_footprint()}", flush=True)
    return 0


def _exec_scenario(cpp: "Cpp", sc: dict) -> list:
    """Run one real scenario on a C ABI engine; returns [(tokens, digests)] per prompt."""
    b = cpp.b
    if sc["kind"] == "solo":
        return [cpp.run([sc["prompts"][0]], sc["gen"])]
    if sc["kind"] == "turns":
        return [cpp.run([sc["prompts"][0]], sc["gen"], turns=[sc["prompts"][1]])]
    sess = [cpp.session() for _ in sc["prompts"]]
    for s_, p in zip(sess, sc["prompts"]):
        b.prefill(s_, p)
    got = [([], []) for _ in sess]
    for _ in range(sc["gen"]):
        out = b.step_batch(sess, ['{"greedy": true}'] * len(sess))
        for i, t in enumerate(out):
            got[i][0].append(t)
            got[i][1].append(b.last_digest(sess[i]))
    for s_ in sess:
        b.close_session(s_)
    return got


def _model_type(ckpt: Path) -> str:
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    return cfg.get("model_type") or cfg.get("text_config", {}).get("model_type", "")


def run_real_engine(checkpoint: str, lib: str, ref_path: str, lengths, gen: int, kv_modes=(),
                    verbose: bool = True) -> list[Check]:
    """Process 2: the generic C++ runtime on the same scenarios, compared with the oracle's file."""
    from transformers import AutoTokenizer
    from veystrix.recipe import ensure_recipe
    checks: list[Check] = _Live(verbose)
    ckpt = Path(checkpoint).expanduser()
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    recipe = ensure_recipe(ckpt)
    ref = json.loads(Path(ref_path).read_text())["scenarios"]
    engines = {}
    for sc in real_scenarios(tok, lengths, gen, kv_modes):
        if sc["name"] not in ref:
            checks.append(Check(sc["name"], False, "no oracle result"))
            continue
        key = (sc["kv"], sc.get("window"), sc.get("prefix"))
        if key not in engines:
            for e in engines.values():
                e.close()
            engines.clear()
            engines[key] = Cpp(lib, ckpt, recipe, sc["kv"], prefill_chunk=2048,
                               window=sc.get("window"), prefix=sc.get("prefix"))
        cpp = engines[key]
        ta = time.time()
        got = _exec_scenario(cpp, sc)
        want = [tuple(r) for r in ref[sc["name"]]["runs"]]
        ok = all((list(g[0]), list(g[1])) == (list(w[0]), list(w[1])) for g, w in zip(got, want))
        first = None
        if not ok:
            for g, w in zip(got, want):
                first = next((k for k, (x, y) in enumerate(zip(g[1], w[1])) if x != y), first)
        detail = ("tokens+digests identical" if ok else
                  f"MISMATCH first digest step {first}; cpp tokens {[list(g[0]) for g in got]} "
                  f"oracle {[list(w[0]) for w in want]}")
        took = ref[sc["name"]].get("seconds")             # absent in a frozen reference
        checks.append(Check(sc["name"], ok, detail + f" (C++ {time.time() - ta:.1f}s, "
                                                   + (f"oracle {took}s)" if took is not None else "frozen reference)")))
    for cpp in engines.values():
        try:
            stats = cpp.b.stats(cpp.engine)
            checks.append(Check("engine stats", True, json.dumps({k: stats.get(k) for k in (
                "expert_bytes_read", "resident_bytes", "streamed_bytes") if k in stats}) + " " + json.dumps(_footprint())))
        except Exception:
            pass
        cpp.close()
    return checks


def run_real(checkpoint: str, lib: str, lengths=(16, 64, 256, 1024), gen: int = 16,
             verbose: bool = True, kv_modes=(), workdir: str | None = None) -> list[Check]:
    """Real-checkpoint pass in TWO processes (never both models in one): the oracle writes its tokens and
    digests to a file, then the engine process compares. Run only under real_run_guard.py."""
    import subprocess
    work = Path(workdir or tempfile.mkdtemp(prefix="vx-real-"))
    work.mkdir(parents=True, exist_ok=True)
    ref = work / "oracle.json"
    common = ["--checkpoint", checkpoint, "--lengths", ",".join(map(str, lengths)), "--gen", str(gen)]
    if kv_modes:
        common += ["--kv-modes", ",".join(kv_modes)]
    env = dict(os.environ, MallocLargeCache="0")
    if not ref.is_file() and _model_type(Path(checkpoint).expanduser()) == "qwen4_exp":
        # qwen4_exp has no mlx_lm oracle. Its reference is the frozen output of the retired custom Qwen C++
        # runtime (validated against production Flash-Next), recorded for this exact checkpoint config.
        golden = frozen_reference(Path(checkpoint).expanduser(), lengths, gen)
        if golden is None:
            return [Check("frozen reference", False,
                          f"no frozen qwen4_exp reference in {FROZEN_REFS} matches this checkpoint's config.json "
                          f"and lengths={list(lengths)} gen={gen}")]
        ref.write_text(golden.read_text())
    if not ref.is_file():
        code = subprocess.run([sys.executable, "-m", "veystrix.port_check", *common, "--role", "oracle",
                               "--out", str(ref)], stdin=subprocess.DEVNULL, env=env).returncode
        if code != 0 and not ref.is_file():
            return [Check("oracle process", False, f"exit {code}")]
    out = work / "engine.json"
    subprocess.run([sys.executable, "-m", "veystrix.port_check", *common, "--role", "engine", "--ref", str(ref),
                    "--lib", lib, "--json"], stdin=subprocess.DEVNULL, env=env, stdout=open(out, "w"))
    try:
        data = json.loads(out.read_text())
    except ValueError:
        return [Check("engine process", False, out.read_text()[-500:])]
    checks = [Check(c["name"], c["ok"], c["detail"]) for c in data["checks"]]
    if verbose:
        for c in checks:
            print(f"{'PASS' if c.ok else 'FAIL'}  {c.name}: {c.detail}", flush=True)
    return checks


MODEL_TYPES_COVERED = ("qwen3_5_moe", "qwen4_exp")

# Frozen real-checkpoint references (tokens + float32 logit digests per scenario) for model types without an
# mlx_lm oracle. qwen4_exp's came from the custom Qwen C++ runtime at its final defaults on 21 Nov 2026, the
# last day it existed (the generic runtime matched it 6/6; docs/performance/ENGINE_AB_2026-11-20.md).
FROZEN_REFS = Path(__file__).resolve().parent / "parity_refs"


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_reference(ckpt: Path, lengths, gen: int) -> Path | None:
    """The frozen reference recorded for this checkpoint (same config.json bytes, lengths and gen), if any."""
    want = _sha256_file(ckpt / "config.json")
    for path in sorted(FROZEN_REFS.glob("*.json")):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (meta.get("config_sha256") == want and meta.get("lengths") == list(lengths)
                and meta.get("gen") == gen):
            return path
    return None


def record_result(lib: str, checks, seconds: float, home: Path | None = None) -> Path:
    """Write the conformance record ``veystrix engine cpp`` checks (tied to this exact dylib)."""
    from veystrix.tool.config import home_dir
    from veystrix.tool.engine import file_sha256, parity_record_path
    path = parity_record_path(home or home_dir())
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        record = {}
    # checks: a list (every covered model type) or {model_type: list}.
    per_type = checks if isinstance(checks, dict) else {m: checks for m in MODEL_TYPES_COVERED}
    for model_type, these in per_type.items():
        ok = all(c.ok for c in these)
        record[model_type] = {"status": "pass" if ok else "fail", "checks": len(these),
                              "failed": [c.name for c in these if not c.ok],
                              "lib": str(lib), "lib_sha256": file_sha256(Path(lib)),
                              "seconds": round(seconds, 1),
                              "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="veystrix models port-check")
    ap.add_argument("--tiny", action="store_true", help="synthetic tiny-model pass (default)")
    ap.add_argument("--lib", default=None, help="libveystrix_c dylib (default: the build tree)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", default=None, help="keep the tiny checkpoint in this directory")
    ap.add_argument("--modes", default=None, help="comma list of KV modes to check")
    ap.add_argument("--model-type", default=None, choices=MODEL_TYPES_COVERED,
                    help="tiny pass for one model type only (default: every covered type)")
    ap.add_argument("--no-record", action="store_true",
                    help="do not record the result (a full pass unlocks --engine cpp for the model type)")
    ap.add_argument("--checkpoint", default=None,
                    help="REAL pass on this checkpoint (streaming mlx_lm oracle vs C++); run it only under "
                         "scripts/flashnext/real_run_guard.py")
    ap.add_argument("--lengths", default="16,64,256,1024")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--kv-modes", default="", help="real pass: extra KV modes to check (e.g. int8,tiered4+)")
    ap.add_argument("--role", choices=("both", "oracle", "engine"), default="both")
    ap.add_argument("--out", help="oracle role: where to write tokens + digests")
    ap.add_argument("--ref", help="engine role: the oracle's file")
    args = ap.parse_args(argv)
    lib = args.lib or os.environ.get("VEYSTRIX_CPP_LIB")
    if lib is None:
        lib = str(Path(__file__).resolve().parents[1] / "cpp" / "engine" / "build" / "libveystrix_c.dylib")
    t0 = time.time()
    if args.checkpoint:
        lengths = tuple(int(x) for x in args.lengths.split(","))
        kv_modes = tuple(m for m in (args.kv_modes or "").split(",") if m)
        if args.role == "oracle":
            return run_real_oracle(args.checkpoint, args.out, lengths, args.gen, kv_modes)
        if args.role == "engine":
            checks = run_real_engine(args.checkpoint, lib, args.ref, lengths, args.gen, kv_modes,
                                     verbose=not args.json)
        else:
            checks = run_real(args.checkpoint, lib, lengths, args.gen, verbose=not args.json,
                              kv_modes=kv_modes, workdir=args.keep)
        failed = [c for c in checks if not c.ok]
        if args.json:
            print(json.dumps({"seconds": round(time.time() - t0, 1), "passed": not failed,
                              "checkpoint": args.checkpoint, "checks": [c.__dict__ for c in checks]}, indent=1))
        else:
            print(f"{len(checks) - len(failed)}/{len(checks)} passed in {time.time() - t0:.1f}s")
        return 0 if not failed else 1
    only = args.model_type
    by_type = {}
    if only in (None, "qwen3_5_moe"):
        by_type["qwen3_5_moe"] = run_tiny(lib, quick=args.quick, workdir=args.keep,
                                          modes=args.modes.split(",") if args.modes else None,
                                          verbose=not args.json)
    if only in (None, "qwen4_exp"):
        by_type["qwen4_exp"] = (run_tiny_qwen4(lib, quick=args.quick, workdir=args.keep, verbose=not args.json)
                                + run_tiny_qwen4(lib, quick=args.quick, workdir=args.keep, verbose=not args.json,
                                                 indexer_budget=16)
                                + run_tiny_qwen4(lib, quick=args.quick, workdir=args.keep, verbose=not args.json,
                                                 indexer_budget=32))
    checks = [c for these in by_type.values() for c in these]
    failed = [c for c in checks if not c.ok]
    full = not args.quick and args.modes is None
    if full and not args.no_record:
        record_result(lib, by_type, time.time() - t0)
    if args.json:
        print(json.dumps({"seconds": round(time.time() - t0, 1), "passed": not failed,
                          "checks": [c.__dict__ for c in checks]}, indent=1))
    else:
        print(f"{len(checks) - len(failed)}/{len(checks)} passed in {time.time() - t0:.1f}s")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
