# MTP (multi-token prediction) draft head + verify — C++ port (foundation stage)

**Status (measured):** bit-exact C++ ↔ Python parity for the MTP draft-head
forward and the point-mass verify/accept/reject primitive, on a tiny synthetic
config, CPU, fixed seeds. Full checkpoint integration is **not** done — see
"Remaining work".

All parity numbers below are **measured** on this machine (mlx 0.32.2, Apple
Silicon, `mlx 0.32.2` from `/Users/vincent/Documents/Veystrix/.venv`), using raw
byte comparison (uint32 view) against golden vectors. Python behavior claims
about the reference modules are **cited** to `models/flashnext/mtp.py`,
`models/flashnext/settings/setting_mtp.py`, and `veystrix/sampling.py`. No real
checkpoint and no benchmark were used anywhere.

> **Quality caveat (cited):** MTP is **off by default** in Python
> (`setting_mtp.py`: `native-mtp` default `"off"`, warning "experimental, exact
> greedy only …"). It carries known quality risk (repetition loops on some
> checkpoints). This port implements the reference exactly; it does **not**
> "improve" or enable it.

## Files added / changed

| File | Role |
|---|---|
| `cpp/engine/src/mtp.hpp` / `.cpp` | `MTPConfig`, `MTPWeights`, `mtp_fuse_inputs`, `mtp_draft_forward`, `mtp_verify` |
| `cpp/engine/tools/gen_mtp_vectors.py` | synthetic, CPU-only golden vectors from the **real** Python `Qwen4ExpMTPModule` + `Sampler.verify_point_mass` (tiny config, no checkpoint) |
| `cpp/engine/tests/test_mtp.cpp` | byte-for-byte comparison of fused inputs, draft logits, and verify accept/token |
| `cpp/engine/CMakeLists.txt` | add `mtp.cpp` to lib, `test_mtp` target, guarded `add_test(NAME mtp ...)` |
| `cpp/engine/tests/vectors_mtp/` | generated golden vectors (regenerate with the tool above) |

## Feasibility check (why the full draft path was not pivoted away)

The task allowed a fallback to verify-only if the real Python MTP path could
not be constructed without a checkpoint. **Measured:** a tiny synthetic
`TextConfig` (H=32, HC=2, 4 attention heads, head_dim=8, 4 experts, top_k=2,
vocab=64, …) constructs `Qwen4ExpMTPModule` and runs a full forward with a
fresh `QSAKVCache` — no checkpoint and no SSD expert pool needed. The draft
head is therefore validated end-to-end, not just the verify logic.

## Dataflow — `mtp_draft_forward` (one fresh-cache draft step)

```
next_token_ids [B,S] ── embed_tokens.weight ──> token_emb [B,S,H]
hidden_states  [B,S,HC*H]  (previous hidden stream)
        │
        ├─ mtp_fuse_inputs()                         (MTP-specific primitive)
        │    pre_fc_norm_embedding -> fc_embedding   -> proj_emb [B,S,H]
        │    pre_fc_norm_hidden    -> reshape[...,HC,H] -> fc_hidden -> proj_hidden
        │    fused = (proj_emb[...,None,:] + proj_hidden).reshape[...,HC*H]
        │
        ├─ attn_hyper_connection (gated residual, one-centered hc_norm)
        │    -> mixed [B,S,H], hyper [B,S,HC*H], injection [B,S,HC]
        ├─ dense_attention()  (fresh cache, S=1 -> QSA indexer returns None)
        │    q/k/v proj, one-centered q/k norm, MRoPE @ position 0,
        │    dense SDPA (single key => out = v), o_proj(out*sigmoid(gate))
        │    -> branch [B,S,H]; inject_residual -> h1 [B,S,HC*H]
        ├─ mlp_hyper_connection (gated residual) -> mixed2, hyper2, injection2
        ├─ dense_moe()  (router softmax(precise)+argpartition+renorm,
        │    dense SwitchGLU experts via gather_mm, dense shared expert)
        │    -> branch2 [B,S,H]; inject_residual -> h2 [B,S,HC*H]
        ├─ hyper_connection_mixer (gated residual, use_combine=False)
        │    -> mixed3 [B,S,H]
        └─ tied LM head: logits = mixed3 @ embed_tokens.weight.T   [B,S,V]
```

Notes on exactness (measured / cited):
- **Measured** — `fuse_inputs` output and full draft logits are byte-identical
  to the real Python module on the synthetic config.
- **Cited** — `mtp.py::fuse_inputs` uses one-centered `Qwen4ExpRMSNorm` gains
  (`1 + weight`); `qsa_rms_norm` mirrors this exactly.
- **Cited** — with `make_cache()` (fresh `QSAKVCache`) and `position_ids=None`,
  the QSA indexer sees 1 token / 0 complete blocks and returns `None`
  (`qsa_full.cpp` reviewer note + `mlx_vlm` indexer), so attention is a dense
  1-token softmax; MRoPE at position 0 is the identity (cos=1, sin=0).

## Dataflow — `mtp_verify` (lossless speculative sampling)

Mirrors `veystrix/sampling.py::Sampler.verify_point_mass` one-for-one:
```
for i, draft in enumerate(drafts):
    kept, masked = distribution(rows[i])        # sampling_distribution()
    p = softmax(masked); hit = kept==draft | arange==draft
    chance = sum(where(hit, p, 0)); draw = random.uniform(); eval(chance, draw)
    if draw < chance or chance > 0.999999:      # accept
        observe(draft)  (presence-penalty history); continue
    residual = where(hit, -inf, masked)         # reject -> redraw w/o draft
    return i, categorical(residual)[via kept remap]
kept, masked = distribution(rows[len(drafts)])
return len(drafts), categorical(masked)[via kept remap]
```
MLX Python and C++ share the PRNG; seeding with the same `seed` reproduces every
uniform/categorical draw bit-for-bit (**measured** across all 11 cases).

## Validation (exact test output)

Command:
```
cpp/engine/build/test_mtp cpp/engine/tests/vectors_mtp
```
Measured output:
```
  fuse_inputs OK (256 bytes bit-identical)
  draft logits OK (256 bytes bit-identical)
  [case0_accept_all] OK (accepted=0 token=22)
  [case1_reject_first] OK (accepted=0 token=10)
  [case2_reject_later] OK (accepted=0 token=21)
  [case3_point_mass] OK (accepted=1 token=8)
  [case4_topk] OK (accepted=1 token=44)
  [case5_topp_minp] OK (accepted=0 token=46)
  [case6_presence] OK (accepted=0 token=55)
  [case7_argmax_chain] OK (accepted=0 token=29)
  [case8_argmax_topk] OK (accepted=1 token=41)
  [case9_argmax_presence] OK (accepted=0 token=59)
  [case10_peaked_chain] OK (accepted=2 token=30)
test_mtp OK (fuse_inputs + draft logits bit-identical; 11 verify cases)
```
`ctest -R '^mtp$'` also passes. The 11 verify cases cover: reject-at-0,
accept-then-reject at index 1 and index 2, the `chance>0.999999` auto-accept,
top-k survivor remap, top-p/min-p masking, and presence-penalty history.

## Honest scope / what is *not* validated here

- **Cited / not measured:** real-checkpoint weights, quantized/bf16 expert
  pools (`StreamingSwitchGLU`), SSD I/O, and the real 48-layer backbone.
- **Not implemented:** the QSA *sparse* indexer/mask branch (multi-token or
  prepopulated indexer cache), MRoPE positions other than the fresh-cache
  position 0, and the untied-LM-head branch (`tie_word_embeddings=False`). The
  synthetic golden vector exercises the dense 1-token fresh-cache draft with a
  tied LM head, which is what the native draft step's default path uses.
- **Not integrated:** `run_step` / `real_checkpoint_generate.cpp` do not yet
  call the MTP head; no speculative tree/draft buffer is wired up.

## Remaining work for full integration

1. Load MTP weights from `language_model.mtp.*` checkpoint keys into
   `MTPWeights` (the real experts are quantized `SwitchGLU`; substitute
   `q4_projection`/the existing dense-vs-Q4 MoE path for the dense synthetic
   experts used here — the combine math is identical).
2. Generalize `dense_attention` to the QSA sparse mask branch via
   `full_attention_forward` / `qsa_full_indexer` for draft steps with a
   prepopulated cache or S>1, and thread the real draft position into
   `MTPConfig::position` (MRoPE).
3. Implement the draft-token loop: call `mtp_draft_forward` per draft position,
   feed `mtp_verify` the target logit rows for the verified prefix, and commit
   the accepted prefix + redraw token into the decode stream / KV cache.
4. Keep MTP gated off by default (mirror `setting_mtp.py`) and respect the
   "exact greedy only" caveat until quality gates pass.
5. Add a real-checkpoint parity test once (1) is done; until then this
   foundation is validated only against synthetic golden vectors.
