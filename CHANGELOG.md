# Changelog

## Veystrix 0.1.0

Initial public release of Veystrix for Apple Silicon Macs (`arm64`, macOS 14+):

- **Generic recipe-driven C++20 + MLX Metal inference engine (`cpp/engine/`):**
  - Recipe-driven execution supporting **Inkling 2 Mini** (`Inkling-2-Mini-oQ4e-mtp`) in native 4-bit precision with zero quality loss, 48-visit looped MoE graph, 0.29 GiB native DSpark speculative drafter, and day-1 extensibility for future architectures.
  - Stage-major looped-graph prefill (150+ tok/s on 16 GB M4 Air; up to 1,900+ tok/s on M-Max/Ultra) with dynamic core residency and strict 5 GiB swap ceiling enforcement.
- **VXKV2 compressed KV-cache disk storage (`kv_snapshot_store.{hpp,cpp}`, `kv_store.cpp`, `veystrix/kv_policy.py`):**
  - Bit-exact 256-token page-level SHA-256 content-addressed storage (CAS) deduplicating shared pages across sessions, turns, and branches with automatic garbage collection.
  - Byte-plane and bit-plane shuffling with Apple `libcompression` (LZFSE / LZ4) and 4-thread parallel zero-temp-file restore (`< 35 ms` cached prompt TTFT).
  - Extreme low-bit and asymmetric KV modes (`k4v2`, `k3v2`, `int2`, `sparse-paged-k4v2`, `sparse-paged-k2v2`) alongside 1M-token Quest `sparse-paged` KV cache.
- **Curated terminal coding agent & harness integrations (`veystrix/tool/`):**
  - Built-in terminal TUI (`veystrix`), auto-launching wrappers for `opencode`, `pi`, and `hermes` (`veystrix run <harness>`), and local OpenAI/Anthropic loopback APIs.
  - SSD prompt-cache storage manager (`veystrix storage`, `veystrix storage claim`, `veystrix storage clear`) with CAS-aware accounting and LRU eviction.
