#ifndef VEYSTRIX_C_H
#define VEYSTRIX_C_H

/*
 * Narrow C ABI for the Veystrix C++ engine (Phase 4 / A2).
 *
 * This header is C89-clean: plain C types, opaque handles, integer status
 * codes.  No C++ type, exception, or STL object may cross this boundary.
 * Every entry point catches all exceptions, records a thread-local message
 * retrievable with vx_last_error(), and returns a status code.
 *
 * Phase G: the production build (VEYSTRIX_BUILD_CAPI=ON, which defines
 * VEYSTRIX_CAPI_LIVE and links veystrix_engine) drives the real Qwen or
 * Nemotron runtime, selected from model_dir or profile {"model":"nemotron"}:
 * vx_engine_open() loads the checkpoint, prefill/step post jobs to ONE driver
 * thread, and vx_mode() reports "live".  vx_engine_stats reports the selected
 * engine.  The standalone CPU-only build in capi/CMakeLists.txt has no MLX
 * and still reports "stub" (VX_ERR_NOT_WIRED).
 * Setting VEYSTRIX_C_CPP_FAKE=1 selects a deterministic CPU-only fake in both
 * builds (used by the ctypes round-trip tests).  See docs/engineering/capi_design.md
 * and docs/engineering/phase_g_capi_live.md.
 */

#include <stdint.h>

#if defined(_WIN32)
#  if defined(VEYSTRIX_C_BUILD)
#    define VX_API __declspec(dllexport)
#  else
#    define VX_API __declspec(dllimport)
#  endif
#else
#  define VX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Stable ABI version; bump only on incompatible changes. */
VX_API int vx_abi_version(void);

/* "live" (production build), "stub" (standalone CPU build) or "fake"
 * (VEYSTRIX_C_CPP_FAKE=1).  Lifetime is static. */
VX_API const char* vx_mode(void);

/* Message for the most recent failed call on this thread, or "". */
VX_API const char* vx_last_error(void);

typedef enum vx_status {
    VX_OK                   =  0,
    VX_ERR_INVALID_ARGUMENT = -1,
    VX_ERR_NOT_WIRED        = -2,
    VX_ERR_CANCELLED        = -3,
    VX_ERR_INTERNAL         = -4,
    VX_ERR_CONTEXT_LIMIT    = -5
} vx_status;

typedef struct vx_engine  vx_engine;
typedef struct vx_session vx_session;

/* Ownership: engine must be closed after its sessions. */
VX_API vx_engine* vx_engine_open(const char* model_dir, const char* profile_json);
VX_API void       vx_engine_close(vx_engine* engine);

VX_API vx_session* vx_session_open(vx_engine* engine);
VX_API void        vx_session_close(vx_session* session);

/* Feed prompt token ids into a session. */
VX_API int vx_session_prefill(vx_session* session, const uint32_t* tokens, int n);

/* One decode step.  sampler_json is a UTF-8 JSON string; out_token is written
 * on VX_OK only.  Fake mode honours {"fake_tokens": [ids...]} if present. */
VX_API int vx_session_step(vx_session* session, const char* sampler_json,
                           uint32_t* out_token);

/* Batched decode: one step for n sessions that share ONE engine, driven on that
 * engine's single MLX thread. sampler_jsons[i] applies to sessions[i]; out_tokens[i]
 * is written on VX_OK. Sessions must belong to the same engine. Each session's
 * sampler (greedy or non-greedy) is honoured.
 * Returns VX_ERR_INVALID_ARGUMENT on n<1, null arrays, or mixed engines. */
VX_API int vx_session_step_batch(vx_session** sessions, int n,
                                  const char** sampler_jsons,
                                  uint32_t* out_tokens);

/* Snapshot / restore session state under an existing or creatable directory. */
VX_API int vx_snapshot_save(vx_session* session, const char* dir);
VX_API int vx_snapshot_restore(vx_session* session, const char* dir);

/* Cooperative cancellation: a following vx_session_step returns VX_ERR_CANCELLED. */
VX_API int vx_cancel(vx_session* session);

/* Write a short JSON stats object into buf (cap bytes, NUL-terminated). */
VX_API int vx_engine_stats(vx_engine* engine, char* buf, int cap);

/* ---- ABI v2 (additive; old dylibs lack these symbols and must fail closed) ----
 * profile_json optional keys (absent = legacy behaviour):
 *   context_window, max_cold_prefill, max_incremental_prefill (integers, >0),
 *   digests (bool), kv ("bf16"|"int8"), expert_pool_policy (string).
 * Unknown keys are refused by name at vx_engine_open. */
#define VX_PREFILL_RESET 0u
#define VX_PREFILL_APPEND 1u

/* Capabilities JSON, e.g. {"abi":2,"sampling":["greedy"],
 * "prefill_append":false,"snapshot":true,"context_window":0,
 * "logit_digest":false,"refused":{...}}. */
VX_API int vx_engine_capabilities(vx_engine* e, char* buf, int cap);

/* Prefill with explicit semantics. VX_PREFILL_RESET (0) is identical to
 * vx_session_prefill. VX_PREFILL_APPEND (1) extends the live caches by one
 * chunk (one Python-anchored chunk per call). */
VX_API int vx_session_prefill_ex(vx_session* s, const uint32_t* tokens, int n,
                                 uint32_t flags);

/* Seed this session's own PRNG key sequence. */
VX_API int vx_session_seed(vx_session* s, uint64_t seed);

/* SHA-256 hex of the float32-LE logits from the LAST step/prefill. Available
 * only when profile_json has "digests": true; otherwise VX_ERR_NOT_WIRED. */
VX_API int vx_session_last_digest(vx_session* s, char* buf, int cap);

/* In-RAM copy-on-write fork of a session's caches (MLX arrays are immutable). */
VX_API int vx_session_fork(vx_session* src, vx_session* dst);

/* Number of cached tokens (prompt + emitted) in the session. */
VX_API int vx_session_tokens(vx_session* s);

/* Exact prefix truncation: rewind the session's KV caches and context
 * to target_tokens (0 <= target_tokens <= vx_session_tokens(s)).
 * Fast in-place operation that preserves valid prefix state without re-prefill.
 * Returns VX_OK on success, VX_ERR_INVALID_ARGUMENT if target_tokens is out of range. */
VX_API int vx_session_truncate(vx_session* s, uint32_t target_tokens);

/* Comma-separated block types the generic runtime of this build implements
 * (mixers, MLPs, pre-layer blocks), e.g. "attn.gated_gqa,linear.gdn,...".
 * Returns the string length, or VX_ERR_NOT_WIRED in a stub build. Lets
 * `veystrix models inspect` report which recipe blocks a checkpoint needs
 * that this build lacks. Additive: older libraries do not export it. */
VX_API int vx_generic_block_types(char* buf, int cap);

/* Test-only: deliberately throws a C++ exception inside the guarded surface.
 * A correct binding receives VX_ERR_INTERNAL, never a crash. */
VX_API int vx_selftest_throw(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* VEYSTRIX_C_H */
