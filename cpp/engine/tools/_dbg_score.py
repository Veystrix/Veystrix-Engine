#!/usr/bin/env python3
import sys, json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mlx.core as mx

sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpQSAIndexer, QSAKVCache
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

d = Path("cpp/engine/tests/vectors_qsa_full/case1_select")
m = json.loads((d / "meta.json").read_text())
L = lambda n: mx.load(str(d / (n + ".npy")))
x, pw, qw, kw = L("x"), L("proj_weight"), L("q_weight"), L("k_weight")
pk, pp, pos = L("past_index_keys"), L("past_index_positions"), L("position_ids")

rot = Qwen3_5RotaryEmbedding(m["rotary_dim"], max_position_embeddings=m["max_position_embeddings"],
                             base=int(m["rope_base"]), mrope_section=m["mrope_section"])
rot.attention_scaling = m["attention_scaling"]; rot.fused_apply = False
cfg = SimpleNamespace(indexer_n_heads=m["n_heads"], indexer_kv_heads=m["kv_heads"],
    indexer_head_dim=m["head_dim"], indexer_budget=m["token_budget"],
    indexer_compress_ratio=m["compress_ratio"], hidden_size=m["H"], rms_norm_eps=m["eps"])
idx = Qwen4ExpQSAIndexer(cfg, rot)
idx.index_qk_proj.weight = pw; idx.q_layernorm.weight = qw; idx.k_layernorm.weight = kw

def save(n, a):
    mx.eval(a); mx.save("/tmp/dbg_s_" + n + ".npy", a)

B,S,_ = x.shape; Hq,Hk = cfg.indexer_n_heads, cfg.indexer_kv_heads
D, ratio = cfg.indexer_head_dim, cfg.indexer_compress_ratio
qk = (x @ pw.T).reshape(B,S,Hq+Hk,D)
query = qk[:,:,:Hq]; raw = qk[:,:,Hq:].squeeze(2)
query = idx.q_layernorm(query).transpose(0,2,1,3)
save("query_pre", query)

cache = QSAKVCache(); cache.index_keys = pk; cache.index_position_ids = pp; cache.offset = m["past_len"]
raw, full_pos = cache.update_indexer(raw, pos)
key_len = raw.shape[1]; blocks = key_len // ratio; clen = blocks * ratio
pooled = raw[:,:clen].reshape(B,blocks,ratio,D)
pooled = mx.expand_dims(idx.k_layernorm(mx.mean(pooled.astype(mx.float32),axis=2).astype(raw.dtype)), axis=1)
save("pooled_pre", pooled)

query = idx._apply_rope(query, pos); save("query_roped", query)
bs = mx.arange(blocks)*ratio
bp = full_pos[..., bs]
pooled = idx._apply_rope(pooled, bp); save("pooled_roped", pooled)

scores = query @ pooled.transpose(0,1,3,2)
save("scores_matmul", scores)
scores = mx.sum(mx.maximum(scores.astype(mx.float32),0),axis=1)
save("scores_sum", scores)
scores = scores / math.sqrt(D)
save("scores_scaled", scores)

past_len = m["past_len"]
qe = past_len + mx.arange(S) + 1
cc = qe // ratio
valid = mx.arange(blocks)[None,None,:] < cc[None,:,None]
scores = mx.where(valid, scores, -mx.inf)
save("scores_masked", scores)
sel = mx.argpartition(scores, kth=-m["block_topk"], axis=-1)[..., -m["block_topk"]:]
save("selected_blocks", sel)
token_indices = mx.arange(key_len)
token_blocks = token_indices // ratio
sel_tokens = mx.any(token_blocks[None,None,None,:] == sel[...,None], axis=2)
save("sel_any", sel_tokens)
tail_starts = cc * ratio
tail = (token_indices[None,None,:] >= tail_starts[None,:,None]) & (token_indices[None,None,:] < qe[None,:,None])
causal = token_indices[None,None,:] < qe[None,:,None]
use_sparse = cc > m["block_topk"]
sel_tokens = mx.where(use_sparse[None,:,None], sel_tokens | tail, causal)
save("sel_final", sel_tokens)
print("cc", np.asarray(cc), "scores_masked", np.asarray(scores))
print("selected_blocks", np.asarray(sel))
print("sel_final", np.asarray(sel_tokens))
