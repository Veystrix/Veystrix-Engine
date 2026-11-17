#!/usr/bin/env python3
"""Confirm selection test vectors have distinct top-k boundary scores."""
import sys, json, math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
sys.path.insert(0, ".venv/lib/python3.12/site-packages")
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpQSAIndexer, QSAKVCache
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

root = Path("cpp/engine/tests/vectors_qsa_full")
cases = json.loads((root / "cases.json").read_text())
for c in cases:
    if c["returned_none"]:
        continue
    d = root / c["dir"]
    L = lambda n: mx.load(str(d / (n + ".npy")))
    x, pw, qw, kw = L("x"), L("proj_weight"), L("q_weight"), L("k_weight")
    pk, pp, pos = L("past_index_keys"), L("past_index_positions"), L("position_ids")
    rot = Qwen3_5RotaryEmbedding(c["rotary_dim"], max_position_embeddings=c["max_position_embeddings"],
        base=int(c["rope_base"]), mrope_section=c["mrope_section"])
    rot.fused_apply = False
    cfg = SimpleNamespace(indexer_n_heads=c["n_heads"], indexer_kv_heads=c["kv_heads"],
        indexer_head_dim=c["head_dim"], indexer_budget=c["token_budget"],
        indexer_compress_ratio=c["compress_ratio"], hidden_size=c["H"], rms_norm_eps=c["eps"])
    idx = Qwen4ExpQSAIndexer(cfg, rot)
    idx.index_qk_proj.weight = pw; idx.q_layernorm.weight = qw; idx.k_layernorm.weight = kw
    B,S,_=x.shape; Hq,Hk=c["n_heads"],c["kv_heads"]; D=c["head_dim"]; ratio=c["compress_ratio"]
    qk=(x@pw.T).reshape(B,S,Hq+Hk,D); q=qk[:,:,:Hq]; raw=qk[:,:,Hq:].squeeze(2)
    q=idx.q_layernorm(q).transpose(0,2,1,3)
    cache=QSAKVCache(); cache.index_keys=pk; cache.index_position_ids=pp; cache.offset=c["past_len"]
    raw,full=cache.update_indexer(raw,pos)
    kl=raw.shape[1]; blocks=kl//ratio; clen=blocks*ratio
    pooled=raw[:,:clen].reshape(B,blocks,ratio,D)
    pooled=mx.expand_dims(idx.k_layernorm(mx.mean(pooled.astype(mx.float32),axis=2).astype(raw.dtype)),axis=1)
    q=idx._apply_rope(q,pos); pooled=idx._apply_rope(pooled,full[...,mx.arange(blocks)*ratio])
    sc=q@pooled.transpose(0,1,3,2)
    sc=mx.sum(mx.maximum(sc.astype(mx.float32),0),axis=1)/math.sqrt(D)
    s=np.asarray(sc).ravel()
    k=c["block_topk"]
    order=np.sort(s)[::-1]
    gap=order[k-1]-order[k]
    print(f"{c['dir']:22s} scores={np.round(s,6)}  top{k}th={order[k-1]:.6f} next={order[k]:.6f} gap={gap:.6f} distinct={len(set(s.tolist()))==len(s)}")
