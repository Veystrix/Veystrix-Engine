#pragma once

// Fail-closed snapshot/restore for Nemotron session state (D3).
//
// Mirrors models/nemotron35/prefix_cache.py: a snapshot is usable only when its
// tokens are a STRICT prefix of the continuing prompt (at least one new token
// is fed to produce logits). No logits are stored. Every layer's mutable state
// is captured byte-exactly:
//   * Mamba-2: conv_state [B,K-1,conv_dim], ssm_state [B,H,Dh,Ds] f32
//   * attention (bf16): K/V + token offset
//   * attention (INT8, production default): packed K/V blocks + scales/biases
//     + block offsets
// MoE has no state. No arithmetic is changed: this module only moves bytes.

#include "nemotron_runtime.hpp"
#include "tensor_blob.hpp"

#include <string>
#include <vector>

namespace veystrix::engine {

struct NemoMambaSnapshot {
  TensorBlob conv, ssm;
};
struct NemoAttnSnapshot {
  bool int8 = false;
  int offset = 0;
  int block_tokens = 4096;
  // bf16 path: one block of K/V (arrays already hold full history).
  TensorBlob keys, values;
  // INT8 path: each block stores six tensors.
  struct Block {
    TensorBlob packed_k, scales_k, biases_k;
    TensorBlob packed_v, scales_v, biases_v;
    int offset = 0;
  };
  std::vector<Block> blocks;
};
struct NemoSnapshot {
  std::string chain_key;
  std::vector<int32_t> tokens;
  std::vector<NemoMambaSnapshot> mamba;
  std::vector<NemoAttnSnapshot> attn;
  bool int8 = false;
  std::string fingerprint;
};

NemoSnapshot nemo_snapshot_capture(const nemotron_runtime::SessionCacheState& state,
                                   const std::vector<int32_t>& tokens,
                                   const std::string& fingerprint,
                                   bool int8);

void nemo_snapshot_save(const NemoSnapshot& snap, const std::string& directory);

// Loads and verifies chain key, tokens, shapes and dtypes, then rebuilds a
// fresh nemotron_runtime::SessionCacheState. Any corruption throws (fail closed).
nemotron_runtime::SessionCacheState nemo_snapshot_restore(const std::string& directory,
                                        const std::string& fingerprint,
                                        const std::vector<int32_t>& tokens);

}  // namespace veystrix::engine
