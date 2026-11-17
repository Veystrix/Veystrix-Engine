# Standalone C++ greedy generation (`veystrix_native_generate`)

**Decisive milestone: the decode engine runs end-to-end in C++, from weights on
disk, with no Python in the generation loop, and its token ids + logits are
bit-identical to Python's greedy decode of the real `mlx_vlm` model.**

## What was built

| File | Role |
|---|---|
| `cpp/engine/tools/veystrix_native_generate.cpp` | `main()`: loads `.npy` weights via `mx::load`, parses a flat config into `DecoderConfig` / `GDNConfig`, initializes one **zeroed** cache per layer, and for N tokens runs `run_decoder_step_from_tokens` → `sample_greedy`, feeding each next token back. Prints `step=<i> in=<t> out=<t> sha256=<hex>` per step. SHA-256 is self-contained (no OpenSSL/CommonCrypto link) over the float32-LE logit row, matching `models/shared/logit_reference.py`. |
| `cpp/engine/tools/gen_native_generate_vectors.py` | `--out` mode builds a tiny **real** Qwen4-Exp stack (mlx_vlm modules, `eval()`, seeded), serializes every weight to `.npy`, writes `config.txt` (flat key=value), and runs Python greedy for N steps from a prompt token. At every step it asserts the REAL modules equal a C++-mirror primitive formula bit-for-bit, then records the real logits' greedy token id + SHA-256. `--check` mode runs the binary and requires ids + digests to match, printing `test_native_generate OK`. |
| `cpp/engine/tests/test_native_generate.cpp` | Optional in-process C++ parity test: same loop, compares every step's logits **bit-exactly** to `expected_logits_step{k}.npy` and token ids to `expected_token_ids.npy`. |
| `cpp/engine/tests/vectors_native_generate/` | Generated vectors (weights/, config.txt, expected.json, expected_*). Not source; regenerate with the tool. |

Shared files (`decoder*`, `engine.*`, `model_adapter.hpp`, `CMakeLists.txt`,
existing tests) were NOT modified.

## Why the all-linear stack (honest scope)

`run_decoder_step_full` (decoder_full.cpp) is the verified mixed GDN+QSA loop,
but its full-attention branch requires a QSA indexer cache **prepopulated past
the block budget** to emit a mask — that is how `gen_decoder_full_vectors.py`
seeds a single-step test. A from-scratch multi-token generation must start from
**zeroed** caches, where the indexer returns no mask (dense path, not composed).
So this milestone uses the **all-linear** configuration (`full_attention_interval`
huge), which is exactly the config `run_decoder_step` is verified bit-exact for
and which supports zero caches. Everything else is the real Qwen4-Exp path:
hyper-connections (`Qwen4ExpGatedResidual`, twice per layer + mixer), GDN
(`Qwen4ExpGatedDeltaNet` with the fused gated-delta kernel), router + Q4 SwiGLU
MoE (`Qwen3_5MoeSparseMoeBlock`, shared expert zeroed so routed-only == C++
`moe_block`), `nn.Embedding` and `lm_head`. Extending to the mixed stack with a
prepopulated QSA cache is a straightforward follow-up; no primitive is missing.

## Commands + output (from repo root)

```bash
cd /Users/vincent/Documents/Veystrix

# 1. Generate weights + Python ground truth (real modules, 3 steps, seed 20261117)
.venv/bin/python cpp/engine/tools/gen_native_generate_vectors.py \
    --out cpp/engine/tests/vectors_native_generate --steps 3 --seed 20261117
# python step 0: 7 -> 33  sha256=60bf66cccc9f2ff768884d743fef8e1dbb33c85824e3e4ca5a92f43291aec1e4
# python step 1: 33 -> 33  sha256=12ba4366d7beaa7093104b238d0b0eafac51ad8490d2b80353f96548ae23a4f2
# python step 2: 33 -> 29  sha256=a9929bd25e624adebb1bc0be7dae33aff8fe589ce8aee4c8300e93a52ca66a7c

# 2. Build the standalone executable (exact VERIFY command)
MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp cpp/engine/src/gdn_conv.cpp \
  cpp/engine/src/gdn_layer.cpp cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
  cpp/engine/src/decoder.cpp cpp/engine/src/decoder_full.cpp \
  cpp/engine/tools/veystrix_native_generate.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/native_generate

# 3. Run C++ generation (NO Python involved)
/tmp/native_generate cpp/engine/tests/vectors_native_generate/weights \
  cpp/engine/tests/vectors_native_generate/config.txt 3
# step=0 in=7 out=33 sha256=60bf66cccc9f2ff768884d743fef8e1dbb33c85824e3e4ca5a92f43291aec1e4
# step=1 in=33 out=33 sha256=12ba4366d7beaa7093104b238d0b0eafac51ad8490d2b80353f96548ae23a4f2
# step=2 in=33 out=29 sha256=a9929bd25e624adebb1bc0be7dae33aff8fe589ce8aee4c8300e93a52ca66a7c

# 4. Automated comparison (token ids AND logit digests, byte-for-byte)
.venv/bin/python cpp/engine/tools/gen_native_generate_vectors.py --check \
  --out cpp/engine/tests/vectors_native_generate --binary /tmp/native_generate --steps 3
# test_native_generate OK (3 steps, token ids + logit digests bit-exact)

# 5. Optional in-process C++ bit-exact test
MLXROOT=.venv/lib/python3.12/site-packages/mlx && \
xcrun clang++ -std=c++20 -O2 -I"$MLXROOT/include" -Icpp/engine/src \
  cpp/engine/src/engine.cpp cpp/engine/src/gdn_kernel.cpp cpp/engine/src/gdn_conv.cpp \
  cpp/engine/src/gdn_layer.cpp cpp/engine/src/qsa.cpp cpp/engine/src/qsa_full.cpp \
  cpp/engine/src/decoder.cpp cpp/engine/src/decoder_full.cpp \
  cpp/engine/tests/test_native_generate.cpp \
  -L"$MLXROOT/lib" -lmlx -Wl,-rpath,"$(pwd)/$MLXROOT/lib" -o /tmp/test_native_generate && \
/tmp/test_native_generate cpp/engine/tests/vectors_native_generate
# test_native_generate OK (bit-exact, 3 steps, 2 layers)
```

Step 3's digests are identical to step 1's Python digests, character for
character, and the `out=` ids are identical to Python's greedy `->` ids. This
is the proof: **the same tokens and the same logits**, generated entirely in
C++.

## Robustness

Re-generated with `--seed 777 --steps 6` into a different directory: all 6
token ids and all 6 digests matched between the real-module Python greedy loop
and `/tmp/native_generate`; `--check` printed `test_native_generate OK`.  The
generator asserts real-modules == primitive-formula at every step, so the
agreement is not a coincidence of one seed.

## Config / weight naming contract

`config.txt` is `key=value` lines (`#` comments ignored). Scalars consumed:
`n_layers, hidden_size, hc_count, hc_lowrank, vocab, top_k, group_size, bits,
eps, prompt_token, key_dim, value_dim, num_k_heads, num_v_heads, head_k_dim,
head_v_dim, conv_kernel`. Weight file names (all relative to `<weights-dir>`):
global `embed_w, lm_head_w, mix_norm_w, mix_down_w, mix_up_w`; per layer `L{l}_`
prefixed `attn_norm_w, attn_down_w, attn_up_w, attn_inject_w, gdn_{qkv,z,b,a,
conv,A_log,dt_bias,norm,out}, mlp_norm_w, mlp_down_w, mlp_up_w, mlp_inject_w,
gate_w, expert_{gw,gs,gb,uw,us,ub,dw,ds,db}`. Initial GDN caches are zeros of
shape `[1, K-1, 2*key_dim+value_dim]` (conv) and `[1, Hv, Dv, Dk]` (recurrent),
dtype float32.

## Integration

To productionize: (1) replace `mx::load` with the native safetensors loader
(`expert_store.hpp::load_projection` / `load_layer`) for real checkpoints;
(2) for the mixed 3:1 GDN/QSA stack, seed each full layer's `FullAttentionCache`
with the checkpoint's KV/indexer state and call `run_decoder_step_full` instead
of `run_decoder_step` (the loop structure and head/sample code are unchanged);
(3) wire `run_generation` into a server loop with KV-cache paging. No changes to
the verified primitives are needed.
