// Implementation of the narrow C ABI declared in veystrix_c.h.
//
// STATUS (Phase G):
//   * Production build (VEYSTRIX_CAPI_LIVE): vx_engine_open() selects the
//     recipe-driven GenericEngineRuntime (every Qwen-family checkpoint) or the
//     NemotronEngineRuntime from the model_dir / profile JSON ("engine" or
//     "model" key). Each owns ONE MLX-driving thread. prefill/step post jobs
//     to that thread via a mutex+CV and block; ctypes releases the GIL.
//   * Standalone CPU-only build (capi/CMakeLists.txt, no MLX): default "stub"
//     mode returns VX_ERR_NOT_WIRED so a cutover bug cannot hide behind a
//     silent Python fallback.
//   * VEYSTRIX_C_CPP_FAKE=1 selects the deterministic CPU-only fake engine in
//     either build. It touches no GPU and no checkpoint.
//
// No C++ exception ever escapes an exported function. Every entry point is
// wrapped in try/catch(...) that records a thread-local message and returns a
// status code (or nullptr for the two open calls).

#include "veystrix_c.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef VEYSTRIX_CAPI_LIVE
#include "nemotron_engine_runtime.hpp"
#include "generic/generic_engine_runtime.hpp"
#include "generic/generic_model.hpp"
#endif

namespace {

thread_local std::string g_error;

void set_error(const char* message) {
    g_error = message ? message : "";
}

void clear_error() {
    g_error.clear();
}

bool fake_enabled() {
    const char* value = std::getenv("VEYSTRIX_C_CPP_FAKE");
    return value && (std::strcmp(value, "1") == 0 ||
                     std::strcmp(value, "true") == 0);
}

const char* const kStubMode = "stub";
const char* const kFakeMode = "fake";
#ifdef VEYSTRIX_CAPI_LIVE
const char* const kLiveMode = "live";
#endif

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 2);
    for (unsigned char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xf];
                    out += hex[c & 0xf];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Minimal reader for the fake-only {"fake_tokens": [12, 34, ...]} hint.
// A real sampler gets its parameters from a C++ parser in a later phase;
// this deliberately needs no JSON dependency.
std::vector<uint32_t> fake_tokens_from_sampler(const char* sampler_json) {
    std::vector<uint32_t> result;
    if (!sampler_json) return result;
    std::string text(sampler_json);
    std::size_t key = text.find("\"fake_tokens\"");
    if (key == std::string::npos) return result;
    std::size_t open = text.find('[', key + 13);
    if (open == std::string::npos) return result;
    std::size_t i = open + 1;
    while (i < text.size() && text[i] != ']') {
        if (text[i] >= '0' && text[i] <= '9') {
            unsigned long long value = 0;
            while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                value = value * 10ULL + static_cast<unsigned>(text[i] - '0');
                ++i;
                if (value > 0xffffffffULL)
                    throw std::runtime_error("fake_tokens value exceeds uint32");
            }
            result.push_back(static_cast<uint32_t>(value));
        } else {
            ++i;
        }
    }
    return result;
}

// Minimal JSON string reader for the optional model selector. A bare profile
// (flat-config path) has no keys and falls through to the model_dir heuristic.
std::string json_string_field(const std::string& text, const char* key) {
    std::string needle = std::string("\"") + key + "\"";
    std::size_t at = text.find(needle);
    if (at == std::string::npos) return "";
    std::size_t colon = text.find(':', at + needle.size());
    if (colon == std::string::npos) return "";
    std::size_t q1 = text.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    std::size_t q2 = text.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return text.substr(q1 + 1, q2 - q1 - 1);
}

std::string ascii_lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

// ---- ABI v2 helpers: flat JSON key validation and a dependency-free SHA-256 ----
// Profiles are flat objects; nested values are deliberately unsupported.
std::vector<std::string> json_top_keys(const std::string& text) {
    std::vector<std::string> keys;
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] != '"') { ++i; continue; }
        std::size_t q2 = text.find('"', i + 1);
        if (q2 == std::string::npos) break;
        std::string name = text.substr(i + 1, q2 - i - 1);
        std::size_t j = q2 + 1;
        while (j < text.size() && (text[j] == ' ' || text[j] == '\t')) ++j;
        if (j < text.size() && text[j] == ':') keys.push_back(name);
        i = q2 + 1;
    }
    return keys;
}

bool json_number_field(const std::string& text, const char* key, long long& out) {
    std::string needle = std::string("\"") + key + "\"";
    std::size_t at = text.find(needle);
    if (at == std::string::npos) return false;
    std::size_t colon = text.find(':', at + needle.size());
    if (colon == std::string::npos) return false;
    std::size_t i = colon + 1;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
    bool negative = false;
    if (i < text.size() && text[i] == '-') { negative = true; ++i; }
    if (i >= text.size() || text[i] < '0' || text[i] > '9') return false;
    long long value = 0;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
        value = value * 10 + (text[i] - '0');
        ++i;
    }
    out = negative ? -value : value;
    return true;
}

bool json_bool_field(const std::string& text, const char* key, bool& out) {
    std::string needle = std::string("\"") + key + "\"";
    std::size_t at = text.find(needle);
    if (at == std::string::npos) return false;
    std::size_t colon = text.find(':', at + needle.size());
    if (colon == std::string::npos) return false;
    std::size_t i = colon + 1;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
    if (text.compare(i, 4, "true") == 0) { out = true; return true; }
    if (text.compare(i, 5, "false") == 0) { out = false; return true; }
    return false;
}

struct Sha256 {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint64_t total = 0;
    unsigned char block[64] = {};
    std::size_t used = 0;

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void compress(const unsigned char* p) {
        static const uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4*i]) << 24) | (uint32_t(p[4*i+1]) << 16) |
                   (uint32_t(p[4*i+2]) << 8) | uint32_t(p[4*i+3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }

    void update(const void* data, std::size_t len) {
        const unsigned char* p = static_cast<const unsigned char*>(data);
        total += len;
        while (len) {
            std::size_t take = 64 - used;
            if (take > len) take = len;
            std::memcpy(block + used, p, take);
            used += take; p += take; len -= take;
            if (used == 64) { compress(block); used = 0; }
        }
    }

    std::string hex() {
        uint64_t bits = total * 8;
        unsigned char pad = 0x80;
        update(&pad, 1);
        unsigned char zero = 0;
        while (used != 56) update(&zero, 1);
        unsigned char len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<unsigned char>(bits >> (56 - 8*i));
        update(len, 8);  // updates total again, but we already emitted the length
        static const char* digits = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (uint32_t v : h)
            for (int b = 3; b >= 0; --b) {
                unsigned char byte = static_cast<unsigned char>(v >> (8*b));
                out += digits[(byte >> 4) & 0xf];
                out += digits[byte & 0xf];
            }
        return out;
    }
};

std::string sha256_hex_bytes(const void* data, std::size_t len) {
    Sha256 s;
    s.update(data, len);
    return s.hex();
}

// Explicit profile key ("model" or "engine") wins; otherwise infer from the
// checkpoint directory name. Unknown explicit values fail closed.
std::string engine_name_for_open(const char* model_dir, const char* profile_json) {
    std::string prof = profile_json ? profile_json : "";
    // "engine" first: an explicit {"engine": "generic"} must win over a family name in "model"
    // (the generic backend sends both, e.g. {"engine": "generic", "model": "flashnext"}).
    const char* keys[] = {"engine", "model"};
    for (const char* key : keys) {
        std::string value = ascii_lower(json_string_field(prof, key));
        if (value.empty()) continue;
        // Generic recipe-driven runtime (universal engine): every Qwen family
        // (qwen3_5_moe / Ornith, qwen4_exp / Flash-Next). The hand-written Qwen
        // runtime was retired on 21 Nov 2026 (ENGINE_AB_2026-11-20.md).
        if (value == "generic" || value.find("ornith") != std::string::npos ||
            value.find("qwen") != std::string::npos || value.find("flashnext") != std::string::npos ||
            value.find("inkling") != std::string::npos)
            return "generic";
        if (value.find("nemotron") != std::string::npos) return "nemotron";
        throw std::runtime_error(std::string("vx_engine_open: unsupported engine '") +
                                 value + "'");
    }
#ifdef VEYSTRIX_CAPI_LIVE
    // A checkpoint whose config.json model_type the generic runtime serves.
    if (model_dir && veystrix::engine::capi_live::generic_model_type_supported(model_dir))
        return "generic";
    if (json_string_field(prof, "recipe").size() > 0) return "generic";
#endif
    std::string dir = ascii_lower(model_dir ? model_dir : "");
    if (dir.find("nemotron") != std::string::npos) return "nemotron";
    return "generic";  // fails with an actionable "no recipe" error if none is given
}

#ifdef VEYSTRIX_CAPI_LIVE
namespace live = veystrix::engine::capi_live;

// Narrow dispatch wrapper so every call site below stays identical for both
// runtimes. Exactly one of the two is non-null after open().
struct LiveRuntime {
    std::unique_ptr<live::NemotronEngineRuntime> nemotron;
    std::unique_ptr<live::GenericEngineRuntime> generic;

    void open(const std::string& model_dir, const std::string& profile_json,
              const std::string& engine) {
        if (engine == "generic") {
            auto r = std::unique_ptr<live::GenericEngineRuntime>(
                new live::GenericEngineRuntime());
            r->open(model_dir, profile_json);
            generic = std::move(r);
        } else {
            auto r = std::unique_ptr<live::NemotronEngineRuntime>(
                new live::NemotronEngineRuntime());
            r->open(model_dir, profile_json);  // all MLX on its driver thread
            nemotron = std::move(r);
        }
    }
    int new_session() {
        if (generic) return generic->new_session();
        if (nemotron) return nemotron->new_session();
        throw std::runtime_error("vx_engine: runtime not open");
    }
    void close_session(int sid) {
        if (generic) { generic->close_session(sid); return; }
        if (nemotron) nemotron->close_session(sid);
    }
    void prefill(int sid, const std::vector<uint32_t>& tokens) {
        if (generic) { generic->prefill(sid, tokens); return; }
        if (nemotron) nemotron->prefill(sid, tokens);
    }
    void prefill_append(int sid, const std::vector<uint32_t>& tokens) {
        if (generic) { generic->prefill_append(sid, tokens); return; }
        if (nemotron) nemotron->prefill_append(sid, tokens);
        else throw std::runtime_error("vx_engine: runtime not open");
    }
    uint32_t step(int sid, const std::string& sampler_json) {
        if (generic) return generic->step(sid, sampler_json);
        return nemotron->step(sid, sampler_json);
    }
    // Batched step on the one driver thread: both runtimes support N sessions.
    std::vector<uint32_t> step_batch(const std::vector<int>& sids,
                                     const std::vector<std::string>& samplers) {
        if (generic) return generic->step_batch(sids, samplers);
        if (nemotron) return nemotron->step_batch(sids, samplers);
        throw std::runtime_error("vx_session_step_batch: runtime not open");
    }
    void snapshot_save(int sid, const std::string& dir) {
        if (generic) { generic->snapshot_save(sid, dir); return; }
        nemotron->snapshot_save(sid, dir);
    }
    void snapshot_restore(int sid, const std::string& dir) {
        if (generic) { generic->snapshot_restore(sid, dir); return; }
        nemotron->snapshot_restore(sid, dir);
    }
    void session_truncate(int sid, uint32_t target_tokens) {
        if (generic) { generic->session_truncate(sid, target_tokens); return; }
    }
    void fork_session(int src_sid, int dst_sid) {
        if (generic) { generic->fork_session(src_sid, dst_sid); return; }
        if (nemotron) nemotron->fork_session(src_sid, dst_sid);
        else throw std::runtime_error("vx_engine: runtime not open");
    }
    void seed(int sid, uint64_t seed) {
        if (generic) { generic->seed(sid, seed); return; }
        if (nemotron) nemotron->seed(sid, seed);
        else throw std::runtime_error("vx_engine: runtime not open");
    }
    std::string last_digest(int sid) {
        if (generic) return generic->last_digest(sid);
        if (nemotron) return nemotron->last_digest(sid);
        throw std::runtime_error("vx_engine: runtime not open");
    }
    void cancel_session(int sid) {
        if (generic) { generic->cancel_session(sid); return; }
        if (nemotron) nemotron->cancel_session(sid);
        else throw std::runtime_error("vx_engine: runtime not open");
    }
    void cancel() {
        if (generic) { generic->cancel(); return; }
        if (nemotron) nemotron->cancel();
    }
    bool consume_cancel() {
        if (generic) return generic->consume_cancel();
        return nemotron->consume_cancel();
    }
    std::string driver_stats() const {
        if (generic) return generic->driver_stats();
        return nemotron->driver_stats();
    }
    std::string capabilities_extra() const {
        if (generic) return generic->capabilities_extra();
        return ",\"engine\":\"nemotron\"";
    }
};
#endif

}  // namespace

struct vx_engine {
    bool fake = false;
    std::string model_dir;
    std::string profile_json;
    int sessions = 0;
    std::atomic<unsigned long long> prefill_tokens{0};
    std::atomic<unsigned long long> steps{0};
    std::atomic<unsigned long long> session_tokens{0};  // prompt + emitted now
    // ABI v2 profile options (absent = legacy unlimited / no digests).
    long long context_window = 0;
    long long max_cold_prefill = 0;
    long long max_incremental_prefill = 0;
    bool digests = false;
    std::string kv;
    std::string expert_pool_policy;
    // Dynamic-memory profile keys (all optional; see src/memory_budget.hpp).
    // The live runtimes resolve the SAME keys from the profile JSON (profile
    // key > VEYSTRIX_CPP_* env), so these copies only drive ABI-level
    // validation and the explicit max_sessions cap.
    long long expert_pool_bytes = 0;
    long long nemotron_slots = 0;
    bool resident = false;
    long long max_sessions = 0;  // 0 = no profile cap (env value is advisory)
#ifdef VEYSTRIX_CAPI_LIVE
    std::string engine_name;
    std::unique_ptr<LiveRuntime> runtime;
#endif
};

struct vx_session {
    vx_engine* engine = nullptr;
    std::vector<uint32_t> prompt;
    std::size_t step_index = 0;
    uint32_t last_token = 0;
    std::atomic<bool> cancelled{false};
    int runtime_sid = 0;  // valid in live mode
    uint64_t seed = 0;
    bool seeded = false;
    std::string last_digest;
};

extern "C" {

int vx_abi_version(void) {
    return 2;  // ABI v2: prefill_ex/seed/last_digest/fork/tokens/capabilities
}

int vx_generic_block_types(char* buf, int cap) {
#ifdef VEYSTRIX_CAPI_LIVE
    try {
        std::string out;
        for (const std::string& t : veystrix::engine::generic::registered_blocks())
            out += (out.empty() ? "" : ",") + t;
        if (buf && cap > 0) {
            const size_t n = std::min(out.size(), static_cast<size_t>(cap - 1));
            std::memcpy(buf, out.data(), n);
            buf[n] = '\0';
        }
        return static_cast<int>(out.size());
    } catch (...) {
        return VX_ERR_INTERNAL;
    }
#else
    (void)buf;
    (void)cap;
    return VX_ERR_NOT_WIRED;
#endif
}

const char* vx_mode(void) {
    if (fake_enabled()) return kFakeMode;  // never throws
#ifdef VEYSTRIX_CAPI_LIVE
    return kLiveMode;
#else
    return kStubMode;
#endif
}

const char* vx_last_error(void) {
    return g_error.c_str();  // never throws
}

vx_engine* vx_engine_open(const char* model_dir, const char* profile_json) {
    clear_error();
    try {
        if (!model_dir || !profile_json) {
            set_error("vx_engine_open: model_dir and profile_json must be non-null");
            return nullptr;
        }
        vx_engine* engine = new vx_engine();
        // ABI v2: profiles are flat objects. Refuse unknown keys by NAME — a
        // silent ignore would be exactly the "different behaviour" the gate
        // forbids. A bare flat-config path (no '{') has no keys and is legal.
        std::string prof(profile_json);
        if (!prof.empty() && prof[0] == '{') {
            static const char* const allowed[] = {
                "model", "engine", "config_path", "kv", "context_window",
                "max_cold_prefill", "max_incremental_prefill", "digests",
                "expert_pool_policy", "expert_pool_bytes", "nemotron_slots",
                "resident", "max_sessions",
                // generic (recipe-driven) runtime keys
                "recipe", "kv_window", "kv_prefix", "kv_hist_k_bits",
                "kv_hist_v_bits", "prefill_chunk"
            };
            for (const std::string& key : json_top_keys(prof)) {
                bool found = false;
                for (const char* a : allowed)
                    if (key == a) { found = true; break; }
                if (!found) {
                    delete engine;
                    set_error((std::string("vx_engine_open: profile key '") + key +
                               "' is not supported by this C++ engine").c_str());
                    return nullptr;
                }
            }
            long long number = 0;
            bool flag = false;
            if (json_number_field(prof, "context_window", number)) engine->context_window = number;
            if (json_number_field(prof, "max_cold_prefill", number)) engine->max_cold_prefill = number;
            if (json_number_field(prof, "max_incremental_prefill", number)) engine->max_incremental_prefill = number;
            if (json_bool_field(prof, "digests", flag)) engine->digests = flag;
            {
                // Memory keys: non-negative integers; resident is a bool or 0/1.
                // A malformed value is refused by NAME (fail closed), never
                // silently defaulted.
                struct NumKey { const char* name; long long* dst; };
                const NumKey nums[] = {
                    {"expert_pool_bytes", &engine->expert_pool_bytes},
                    {"nemotron_slots", &engine->nemotron_slots},
                    {"max_sessions", &engine->max_sessions},
                };
                for (const NumKey& k : nums) {
                    long long v = 0;
                    if (json_number_field(prof, k.name, v)) {
                        if (v < 0) {
                            delete engine;
                            set_error((std::string("vx_engine_open: profile key '") + k.name +
                                       "' must be a non-negative integer").c_str());
                            return nullptr;
                        }
                        *k.dst = v;
                    } else if (prof.find(std::string("\"") + k.name + "\"") != std::string::npos) {
                        delete engine;
                        set_error((std::string("vx_engine_open: profile key '") + k.name +
                                   "' must be a non-negative integer").c_str());
                        return nullptr;
                    }
                }
                long long rv = 0;
                bool rb = false;
                if (json_bool_field(prof, "resident", rb)) {
                    engine->resident = rb;
                } else if (json_number_field(prof, "resident", rv) && (rv == 0 || rv == 1)) {
                    engine->resident = rv == 1;
                } else if (prof.find("\"resident\"") != std::string::npos) {
                    delete engine;
                    set_error("vx_engine_open: profile key 'resident' must be true/false or 0/1");
                    return nullptr;
                }
            }
            engine->kv = json_string_field(prof, "kv");
            engine->expert_pool_policy = json_string_field(prof, "expert_pool_policy");
        }
        engine->model_dir = model_dir;
        engine->profile_json = profile_json;
        if (fake_enabled()) {
            engine->fake = true;
            return engine;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        engine->engine_name = engine_name_for_open(model_dir, profile_json);
        {
            auto runtime = std::unique_ptr<LiveRuntime>(new LiveRuntime());
            runtime->open(model_dir, profile_json, engine->engine_name);
            engine->runtime = std::move(runtime);
        }
        return engine;
#else
        delete engine;
        set_error("not wired: this standalone C ABI build has no MLX engine "
                  "(build with -DVEYSTRIX_BUILD_CAPI=ON, or set VEYSTRIX_C_CPP_FAKE=1)");
        return nullptr;
#endif
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("vx_engine_open: unknown C++ exception");
        return nullptr;
    }
}

void vx_engine_close(vx_engine* engine) {
    clear_error();
    try {
        delete engine;  // nullptr-safe
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("vx_engine_close: unknown C++ exception");
    }
}

vx_session* vx_session_open(vx_engine* engine) {
    clear_error();
    try {
        if (!engine) {
            set_error("vx_session_open: engine is null");
            return nullptr;
        }
        if (engine->max_sessions > 0 && engine->sessions >= engine->max_sessions) {
            set_error((std::string("vx_session_open: max_sessions=") +
                       std::to_string(engine->max_sessions) +
                       " reached (memory profile cap)").c_str());
            return nullptr;
        }
        vx_session* session = new vx_session();
        session->engine = engine;
#ifdef VEYSTRIX_CAPI_LIVE
        if (engine->runtime) session->runtime_sid = engine->runtime->new_session();
#endif
        ++engine->sessions;
        return session;
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    } catch (...) {
        set_error("vx_session_open: unknown C++ exception");
        return nullptr;
    }
}

void vx_session_close(vx_session* session) {
    clear_error();
    try {
        if (session) {
#ifdef VEYSTRIX_CAPI_LIVE
            if (session->engine && session->engine->runtime && session->runtime_sid)
                session->engine->runtime->close_session(session->runtime_sid);
#endif
            if (session->engine) {
                if (session->engine->sessions > 0)
                    --session->engine->sessions;
                unsigned long long held = static_cast<unsigned long long>(session->prompt.size())
                                        + static_cast<unsigned long long>(session->step_index);
                if (session->engine->session_tokens >= held)
                    session->engine->session_tokens -= held;
            }
            delete session;
        }
    } catch (const std::exception& e) {
        set_error(e.what());
    } catch (...) {
        set_error("vx_session_close: unknown C++ exception");
    }
}

int vx_session_prefill(vx_session* session, const uint32_t* tokens, int n) {
    clear_error();
    try {
        if (!session || !tokens || n < 0) {
            set_error("vx_session_prefill: need a session, tokens and n >= 0");
            return VX_ERR_INVALID_ARGUMENT;
        }
        long long window = session->engine->context_window;
        unsigned long long cached = static_cast<unsigned long long>(session->prompt.size())
                                  + session->step_index;
        if (window > 0 && cached + static_cast<unsigned long long>(n) >
                          static_cast<unsigned long long>(window)) {
            set_error(("context window exceeded: " + std::to_string(cached) + " cached + " +
                       std::to_string(n) + " new > window " + std::to_string(window)).c_str());
            return VX_ERR_CONTEXT_LIMIT;
        }
        if (session->engine->max_cold_prefill > 0 &&
            static_cast<long long>(n) > session->engine->max_cold_prefill) {
            set_error(("max cold prefill exceeded: " + std::to_string(n) + " > " +
                       std::to_string(session->engine->max_cold_prefill)).c_str());
            return VX_ERR_CONTEXT_LIMIT;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (session->engine->runtime) {
            session->engine->runtime->prefill(
                session->runtime_sid, std::vector<uint32_t>(tokens, tokens + n));
        }
#endif
        unsigned long long held = static_cast<unsigned long long>(session->prompt.size())
                                + static_cast<unsigned long long>(session->step_index);
        if (session->engine->session_tokens >= held)
            session->engine->session_tokens -= held;
        session->prompt.assign(tokens, tokens + n);
        session->step_index = 0;
        session->cancelled = false;
        session->engine->prefill_tokens += static_cast<unsigned long long>(n);
        session->engine->session_tokens += static_cast<unsigned long long>(n);
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_session_prefill: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_session_step(vx_session* session, const char* sampler_json,
                    uint32_t* out_token) {
    clear_error();
    try {
        if (!session || !out_token) {
            set_error("vx_session_step: need a session and out_token");
            return VX_ERR_INVALID_ARGUMENT;
        }
        if (session->cancelled) {
            set_error("vx_session_step: cancelled");
            return VX_ERR_CANCELLED;
        }
        long long window = session->engine->context_window;
        unsigned long long cached = static_cast<unsigned long long>(session->prompt.size())
                                  + session->step_index;
        if (window > 0 && cached >= static_cast<unsigned long long>(window)) {
            set_error(("context window exceeded: " + std::to_string(cached) +
                       " cached >= window " + std::to_string(window)).c_str());
            return VX_ERR_CONTEXT_LIMIT;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (session->engine->runtime) {
            if (session->engine->runtime->consume_cancel()) {
                set_error("vx_session_step: cancelled");
                return VX_ERR_CANCELLED;
            }
            try {
                *out_token = session->engine->runtime->step(
                    session->runtime_sid, sampler_json ? sampler_json : "");
            } catch (const veystrix::engine::capi_live::CancelledError&) {
                set_error("vx_session_step: cancelled");
                return VX_ERR_CANCELLED;
            }
            session->last_token = *out_token;
            ++session->step_index;
            ++session->engine->steps;
            ++session->engine->session_tokens;
            return VX_OK;
        }
#endif
        std::vector<uint32_t> fake = fake_tokens_from_sampler(sampler_json);
        uint32_t token;
        if (!fake.empty()) {
            token = fake[session->step_index % fake.size()];
        } else {
            uint64_t hash = 1469598103934665603ULL;  // FNV-1a offset basis
            for (uint32_t id : session->prompt) {
                hash ^= id;
                hash *= 1099511628211ULL;
            }
            hash ^= session->step_index + 1;
            hash *= 1099511628211ULL;
            token = static_cast<uint32_t>(hash % 50000ULL) + 1U;
        }
        if (session->engine->digests)
            session->last_digest = sha256_hex_bytes(&token, sizeof(token));
        *out_token = token;
        session->last_token = token;
        ++session->step_index;
        ++session->engine->steps;
        ++session->engine->session_tokens;
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_session_step: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_session_step_batch(vx_session** sessions, int n,
                          const char** sampler_jsons, uint32_t* out_tokens) {
    clear_error();
    try {
        if (n < 1 || !sessions || !sampler_jsons || !out_tokens) {
            set_error("vx_session_step_batch: need sessions, samplers, out_tokens and n>=1");
            return VX_ERR_INVALID_ARGUMENT;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        // Check ALL pointers before dereferencing any (Sonnet finding 16).
        for (int i = 0; i < n; ++i) {
            if (!sessions[i]) {
                set_error("vx_session_step_batch: null session");
                return VX_ERR_INVALID_ARGUMENT;
            }
        }
        // All sessions must share ONE engine (one driver thread, one pooled read).
        vx_engine* eng = sessions[0]->engine;
        if (!eng) {
            set_error("vx_session_step_batch: session has no engine");
            return VX_ERR_INVALID_ARGUMENT;
        }
        for (int i = 1; i < n; ++i)
            if (sessions[i]->engine != eng) {
                set_error("vx_session_step_batch: sessions must share one engine");
                return VX_ERR_INVALID_ARGUMENT;
            }
        if (eng->runtime) {
            if (eng->runtime->consume_cancel()) {
                set_error("vx_session_step_batch: cancelled");
                return VX_ERR_CANCELLED;
            }
            std::vector<int> sids;
            std::vector<std::string> samplers;
            sids.reserve(n); samplers.reserve(n);
            for (int i = 0; i < n; ++i) {
                if (!sessions[i]) {
                    set_error("vx_session_step_batch: null session");
                    return VX_ERR_INVALID_ARGUMENT;
                }
                if (sessions[i]->cancelled) {
                    set_error("vx_session_step_batch: cancelled");
                    return VX_ERR_CANCELLED;
                }
                sids.push_back(sessions[i]->runtime_sid);
                samplers.push_back(sampler_jsons[i] ? sampler_jsons[i] : "");
            }
            std::vector<uint32_t> out;
            try {
                out = eng->runtime->step_batch(sids, samplers);
            } catch (const veystrix::engine::capi_live::CancelledError&) {
                set_error("vx_session_step_batch: cancelled");
                return VX_ERR_CANCELLED;
            }
            for (int i = 0; i < n; ++i) {
                out_tokens[i] = out[i];
                sessions[i]->last_token = out[i];
                ++sessions[i]->step_index;
                ++eng->steps;
                ++eng->session_tokens;
            }
            return VX_OK;
        }
#endif
        // Fake/stub mode: independent deterministic steps per session.
        for (int i = 0; i < n; ++i) {
            vx_session* s = sessions[i];
            if (!s) {
                set_error("vx_session_step_batch: null session");
                return VX_ERR_INVALID_ARGUMENT;
            }
            std::vector<uint32_t> fake = fake_tokens_from_sampler(sampler_jsons[i]);
            uint32_t token;
            if (!fake.empty()) token = fake[s->step_index % fake.size()];
            else {
                uint64_t hash = 1469598103934665603ULL;
                for (uint32_t id : s->prompt) { hash ^= id; hash *= 1099511628211ULL; }
                hash ^= s->step_index + 1; hash *= 1099511628211ULL;
                token = static_cast<uint32_t>(hash % 50000ULL) + 1U;
            }
            out_tokens[i] = token;
            s->last_token = token;
            ++s->step_index;
            ++s->engine->steps;
        }
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_session_step_batch: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_snapshot_save(vx_session* session, const char* dir) {
    clear_error();
    try {
        if (!session || !dir || !*dir) {
            set_error("vx_snapshot_save: need a session and a directory");
            return VX_ERR_INVALID_ARGUMENT;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (session->engine->runtime) {
            session->engine->runtime->snapshot_save(session->runtime_sid, dir);
            return VX_OK;
        }
#endif
        std::filesystem::create_directories(dir);
        std::filesystem::path file = std::filesystem::path(dir) / "vx_snapshot.txt";
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("vx_snapshot_save: cannot open " + file.string());
        out << "1\n" << session->prompt.size() << "\n";
        for (uint32_t id : session->prompt) out << id << " ";
        out << "\n" << session->step_index << "\n" << session->last_token << "\n";
        if (!out) throw std::runtime_error("vx_snapshot_save: write failed");
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_snapshot_save: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_snapshot_restore(vx_session* session, const char* dir) {
    clear_error();
    try {
        if (!session || !dir || !*dir) {
            set_error("vx_snapshot_restore: need a session and a directory");
            return VX_ERR_INVALID_ARGUMENT;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (session->engine->runtime) {
            session->engine->runtime->snapshot_restore(session->runtime_sid, dir);
            return VX_OK;
        }
#endif
        std::filesystem::path file = std::filesystem::path(dir) / "vx_snapshot.txt";
        std::ifstream in(file, std::ios::binary);
        if (!in) throw std::runtime_error("vx_snapshot_restore: cannot open " + file.string());
        int version = 0;
        std::size_t count = 0;
        std::size_t steps = 0;
        uint32_t last = 0;
        in >> version >> count;
        if (!in || version != 1 || count > 1000000)
            throw std::runtime_error("vx_snapshot_restore: bad snapshot header");
        std::vector<uint32_t> prompt(count);
        for (std::size_t i = 0; i < count; ++i) {
            in >> prompt[i];
            if (!in) throw std::runtime_error("vx_snapshot_restore: truncated token list");
        }
        in >> steps >> last;
        if (!in) throw std::runtime_error("vx_snapshot_restore: truncated footer");
        unsigned long long held = static_cast<unsigned long long>(session->prompt.size())
                                + static_cast<unsigned long long>(session->step_index);
        unsigned long long restored = static_cast<unsigned long long>(prompt.size()) + steps;
        if (session->engine->session_tokens >= held)
            session->engine->session_tokens -= held;
        session->prompt = std::move(prompt);
        session->step_index = steps;
        session->last_token = last;
        session->cancelled = false;
        session->engine->session_tokens += restored;
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_snapshot_restore: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_cancel(vx_session* session) {
    clear_error();
    try {
        if (!session) {
            set_error("vx_cancel: session is null");
            return VX_ERR_INVALID_ARGUMENT;
        }
        session->cancelled = true;
#ifdef VEYSTRIX_CAPI_LIVE
        if (session->engine && session->engine->runtime)
            session->engine->runtime->cancel_session(session->runtime_sid);
#endif
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_cancel: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_engine_stats(vx_engine* engine, char* buf, int cap) {
    clear_error();
    try {
        if (!engine || !buf || cap <= 0) {
            set_error("vx_engine_stats: need an engine, a buffer and cap > 0");
            return VX_ERR_INVALID_ARGUMENT;
        }
        const char* mode = engine->fake ? kFakeMode :
#ifdef VEYSTRIX_CAPI_LIVE
            (engine->runtime ? kLiveMode : kStubMode);
#else
            kStubMode;
#endif
        std::string extra;
#ifdef VEYSTRIX_CAPI_LIVE
        if (engine->runtime) extra = engine->runtime->driver_stats();
#endif
        std::ostringstream out;
        out << "{\"mode\":\"" << mode << "\"" << extra
            << ",\"abi_version\":" << vx_abi_version()
            << ",\"model_dir\":\"" << json_escape(engine->model_dir) << "\""
            << ",\"profile_bytes\":" << engine->profile_json.size()
            << ",\"sessions\":" << engine->sessions
            << ",\"prefill_tokens\":" << engine->prefill_tokens
            << ",\"steps\":" << engine->steps
            << ",\"session_tokens\":" << engine->session_tokens << "}";
        std::string text = out.str();
        if (static_cast<int>(text.size()) + 1 > cap) {
            set_error("vx_engine_stats: buffer too small");
            return VX_ERR_INVALID_ARGUMENT;
        }
        std::memcpy(buf, text.c_str(), text.size() + 1);
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_engine_stats: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_engine_capabilities(vx_engine* e, char* buf, int cap) {
    clear_error();
    try {
        if (!e || !buf || cap <= 0) {
            set_error("vx_engine_capabilities: need an engine, a buffer and cap > 0");
            return VX_ERR_INVALID_ARGUMENT;
        }
        const char* mode = e->fake ? kFakeMode :
#ifdef VEYSTRIX_CAPI_LIVE
            (e->runtime ? kLiveMode : kStubMode);
#else
            kStubMode;
#endif
        // APPEND is wired for fake/stub and all live runtimes (P3).
        bool append = true;
        std::string extra;
#ifdef VEYSTRIX_CAPI_LIVE
        if (e->runtime) extra = e->runtime->capabilities_extra();
#endif
        std::ostringstream out;
        out << "{\"abi\":2"
            << ",\"mode\":\"" << mode << "\""
            << ",\"sampling\":[\"greedy\",\"non_greedy\"]"
            << ",\"prefill_append\":" << (append ? "true" : "false")
            << ",\"snapshot\":true"
            << ",\"context_window\":" << e->context_window
            << ",\"logit_digest\":" << (e->digests ? "true" : "false")
            << extra
            << ",\"refused\":{"
            << "\"mtp\":\"measured slower, kept off\","
            << "\"lookup\":\"measured slower, kept off\"}}";
        std::string text = out.str();
        if (static_cast<int>(text.size()) + 1 > cap) {
            set_error("vx_engine_capabilities: buffer too small");
            return VX_ERR_INVALID_ARGUMENT;
        }
        std::memcpy(buf, text.c_str(), text.size() + 1);
        return VX_OK;
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_engine_capabilities: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_session_prefill_ex(vx_session* session, const uint32_t* tokens, int n,
                          uint32_t flags) {
    clear_error();
    try {
        if (!session || !tokens || n < 0) {
            set_error("vx_session_prefill_ex: need a session, tokens and n >= 0");
            return VX_ERR_INVALID_ARGUMENT;
        }
        if (flags != VX_PREFILL_RESET && flags != VX_PREFILL_APPEND) {
            set_error("vx_session_prefill_ex: unknown flags (use VX_PREFILL_RESET or VX_PREFILL_APPEND)");
            return VX_ERR_INVALID_ARGUMENT;
        }
        if (flags == VX_PREFILL_RESET)
            return vx_session_prefill(session, tokens, n);

        // APPEND: one caller-supplied, Python-anchored chunk. Limits apply to
        // both live and fake engines, before any runtime work is issued.
        if (session->engine->max_incremental_prefill > 0 &&
            static_cast<long long>(n) > session->engine->max_incremental_prefill) {
            set_error(("max incremental prefill exceeded: " + std::to_string(n) + " > " +
                       std::to_string(session->engine->max_incremental_prefill)).c_str());
            return VX_ERR_CONTEXT_LIMIT;
        }
        long long window = session->engine->context_window;
        unsigned long long cached = static_cast<unsigned long long>(session->prompt.size())
                                  + session->step_index;
        if (window > 0 && cached + static_cast<unsigned long long>(n) >
                          static_cast<unsigned long long>(window)) {
            set_error(("context window exceeded: " + std::to_string(cached) + " cached + " +
                       std::to_string(n) + " new > window " + std::to_string(window)).c_str());
            return VX_ERR_CONTEXT_LIMIT;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (!session->engine->fake && session->engine->runtime)
            session->engine->runtime->prefill_append(
                session->runtime_sid, std::vector<uint32_t>(tokens, tokens + n));
#endif
        // Extend the C ABI bookkeeping for LIVE mode too: vx_session_tokens and
        // the context-window check use this tape, so leaving it stale made both
        // undercount after an APPEND (Sonnet P0 finding 2).
        session->prompt.insert(session->prompt.end(), tokens, tokens + n);
        session->cancelled = false;
        session->engine->prefill_tokens += static_cast<unsigned long long>(n);
        session->engine->session_tokens += static_cast<unsigned long long>(n);
        if (session->engine->digests && n > 0)
            session->last_digest = sha256_hex_bytes(&tokens[n - 1], sizeof(uint32_t));
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_session_prefill_ex: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_session_seed(vx_session* session, uint64_t seed) {
    clear_error();
    try {
        if (!session) {
            set_error("vx_session_seed: session is null");
            return VX_ERR_INVALID_ARGUMENT;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (!session->engine->fake && session->engine->runtime)
            session->engine->runtime->seed(session->runtime_sid, seed);
#endif
        session->seed = seed;
        session->seeded = true;
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_session_seed: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_session_last_digest(vx_session* session, char* buf, int cap) {
    clear_error();
    try {
        if (!session || !buf || cap <= 0) {
            set_error("vx_session_last_digest: need a session, a buffer and cap > 0");
            return VX_ERR_INVALID_ARGUMENT;
        }
        if (!session->engine->digests) {
            set_error("vx_session_last_digest: profile has digests=false; enable it to record logit digests");
            return VX_ERR_NOT_WIRED;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (!session->engine->fake && session->engine->runtime) {
            std::string live = session->engine->runtime->last_digest(session->runtime_sid);
            if (!live.empty()) session->last_digest = live;
        }
#endif
        if (session->last_digest.empty()) {
            set_error("vx_session_last_digest: no step or prefill has been recorded yet");
            return VX_ERR_NOT_WIRED;
        }
        if (static_cast<int>(session->last_digest.size()) + 1 > cap) {
            set_error("vx_session_last_digest: buffer too small");
            return VX_ERR_INVALID_ARGUMENT;
        }
        std::memcpy(buf, session->last_digest.c_str(), session->last_digest.size() + 1);
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_session_last_digest: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_session_fork(vx_session* src, vx_session* dst) {
    clear_error();
    try {
        if (!src || !dst || src == dst) {
            set_error("vx_session_fork: need two distinct sessions");
            return VX_ERR_INVALID_ARGUMENT;
        }
        if (src->engine != dst->engine) {
            set_error("vx_session_fork: sessions must share one engine");
            return VX_ERR_INVALID_ARGUMENT;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (!src->engine->fake && src->engine->runtime)
            src->engine->runtime->fork_session(src->runtime_sid, dst->runtime_sid);
#endif
        // Fake/stub COW: MLX arrays are immutable; copying the token tape and
        // cursor here is the same ownership model the live fork will use.
        dst->prompt = src->prompt;
        dst->step_index = src->step_index;
        dst->last_token = src->last_token;
        dst->seed = src->seed;
        dst->seeded = src->seeded;
        dst->last_digest = src->last_digest;
        dst->cancelled = false;
        src->engine->session_tokens += static_cast<unsigned long long>(dst->prompt.size())
                                     + dst->step_index;
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_session_fork: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_session_tokens(vx_session* session) {
    clear_error();
    try {
        if (!session) {
            set_error("vx_session_tokens: session is null");
            return -1;
        }
        return static_cast<int>(session->prompt.size() + session->step_index);
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("vx_session_tokens: unknown C++ exception");
        return -1;
    }
}

int vx_session_truncate(vx_session* session, uint32_t target_tokens) {
    clear_error();
    try {
        if (!session) {
            set_error("vx_session_truncate: session is null");
            return VX_ERR_INVALID_ARGUMENT;
        }
        const size_t total = session->prompt.size() + session->step_index;
        if (target_tokens > total) {
            set_error("vx_session_truncate: target_tokens exceeds session tokens");
            return VX_ERR_INVALID_ARGUMENT;
        }
        if (target_tokens == total) {
            return VX_OK;
        }
#ifdef VEYSTRIX_CAPI_LIVE
        if (!session->engine->fake && session->engine->runtime) {
            session->engine->runtime->session_truncate(session->runtime_sid, target_tokens);
        }
#endif
        const size_t diff = total - target_tokens;
        if (session->engine->session_tokens >= diff)
            session->engine->session_tokens -= diff;
        if (target_tokens <= session->prompt.size()) {
            session->prompt.resize(target_tokens);
            session->step_index = 0;
            session->last_token = target_tokens > 0 ? session->prompt.back() : 0;
        } else {
            session->step_index = target_tokens - session->prompt.size();
        }
        session->cancelled = false;
        return VX_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_session_truncate: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

int vx_selftest_throw(void) {
    clear_error();
    try {
        throw std::runtime_error("vx_selftest_throw: deliberate C++ exception");
    } catch (const std::exception& e) {
        set_error(e.what());
        return VX_ERR_INTERNAL;
    } catch (...) {
        set_error("vx_selftest_throw: unknown C++ exception");
        return VX_ERR_INTERNAL;
    }
}

}  // extern "C"
