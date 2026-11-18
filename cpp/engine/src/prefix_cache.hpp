// Exact prefix-state cache for the C++ engine (Phase 1 of the full port).
//
// Mirrors the contract of models/flashnext/prefix_cache.py: a snapshot after N
// tokens stores the FULL live cache state (GDN conv/recurrent, QSA indexer
// keys/positions, full-attention KV) with NO lossy codec, so a restored cache
// is bit-identical to the captured one. Arrays handed back by restore() are
// fresh objects: continuing generation cannot mutate a stored snapshot
// (copy-on-write at the array level).
//
// Entries are keyed by a SHA-256 chain over an engine fingerprint and the exact
// token ids. On load the token ids are re-checked; a fingerprint mismatch or a
// corrupt/truncated file fails closed (throws). This module is NON-NUMERIC:
// it never changes arithmetic; it only saves/restores bytes.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "decoder_full.hpp"  // DecoderFullLayerCache

namespace veystrix::engine {

// Raw bytes + shape + dtype of one cache array. Stored little-endian and
// reinterpreted verbatim on restore (no numeric conversion).
struct TensorBlob {
  std::vector<int> shape;
  int dtype = 0;               // mx::Dtype::Val int
  std::vector<uint8_t> bytes;  // raw host bytes
};

// One layer's cache snapshot. Present fields match which branches that layer
// uses (linear layers: gdn + qsa; full-attention layers: full + qsa).
struct LayerSnapshot {
  bool has_gdn = false;
  TensorBlob gdn_conv, gdn_recurrent;
  bool has_full = false;
  TensorBlob full_keys, full_values;
  int full_offset = 0;
  bool has_qsa = false;
  TensorBlob qsa_keys, qsa_positions;
  int qsa_offset = 0;
};

struct PrefixSnapshot {
  std::string chain_key;        // SHA-256 fingerprint+token chain (hex)
  std::vector<int32_t> tokens;  // exact token ids, re-checked on load
  std::vector<LayerSnapshot> layers;
};

// Compute the SHA-256 chain key over a canonical JSON of fingerprint and the
// little-endian token ids (identical scheme to Python prefix_cache.chain_key).
std::string prefix_chain_key(const std::string& fingerprint,
                             const std::vector<int32_t>& tokens);

// Capture a bit-exact snapshot of `caches` for `tokens`.
PrefixSnapshot prefix_capture(const std::vector<DecoderFullLayerCache>& caches,
                              const std::vector<int32_t>& tokens,
                              const std::string& fingerprint);

// Restore a fresh cache vector from a snapshot. Throws on shape/dtype/token
// mismatch or corrupt data. The returned arrays are independent copies.
std::vector<DecoderFullLayerCache> prefix_restore(
    const PrefixSnapshot& snap, int n_layers);

// Persist / load a snapshot to a directory (one file per tensor + meta.json).
// load() verifies chain_key and tokens; failures throw.
void prefix_save(const PrefixSnapshot& snap, const std::string& directory);
PrefixSnapshot prefix_load(const std::string& directory,
                           const std::string& fingerprint,
                           const std::vector<int32_t>& tokens);

}  // namespace veystrix::engine
