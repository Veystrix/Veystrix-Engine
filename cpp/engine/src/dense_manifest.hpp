// Parser for the dense manifest produced by tools/dump_dense_manifest.py.
// Header-only so both the tool and a pure-CPU unit test can use the exact same
// code. The manifest is the authoritative record of (kind, group_size, bits,
// has_bias) for every non-streamed parameter path in the Python model; the C++
// loader cross-checks its own choice against it and fails closed on mismatch.
#pragma once
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace veystrix::engine::dmanifest {

struct MInfo {
  std::string kind;
  int group_size = -1, bits = -1;
  bool has_bias = false;
};

inline std::string json_str(const std::string& s, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return "";
  size_t colon = s.find(':', p + pat.size());
  size_t b = s.find('"', colon);
  if (b == std::string::npos) return "";
  size_t e = s.find('"', b + 1);
  return e == std::string::npos ? "" : s.substr(b + 1, e - b - 1);
}

inline int json_int(const std::string& s, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return -1;
  size_t colon = s.find(':', p + pat.size());
  size_t b = colon + 1;
  while (b < s.size() && (s[b] == ' ' || s[b] == '\t')) ++b;
  size_t e = b;
  if (e < s.size() && s[e] == '-') ++e;
  while (e < s.size() && s[e] >= '0' && s[e] <= '9') ++e;
  if (e == b || (e == b + 1 && s[b] == '-')) return -1;
  return std::atoi(s.substr(b, e - b).c_str());
}

inline bool json_bool(const std::string& s, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return false;
  size_t colon = s.find(':', p + pat.size());
  size_t t = s.find("true", colon);
  size_t f = s.find("false", colon);
  return t != std::string::npos && (f == std::string::npos || t < f) && t < colon + 10;
}

// Stream the JSON looking for the pattern  "<path>": { ... }  where <path>
// contains a '.'. Uses balanced-brace matching so it works whether the entries
// are top-level or nested (dump_dense_manifest.py wraps them under "manifest").
// A value like "mlx.core.uint32" is never followed by ':'-then-'{', so it is
// not mistaken for a key.
inline std::map<std::string, MInfo> parse(const std::string& txt) {
  std::map<std::string, MInfo> out;
  size_t pos = 0;
  while (true) {
    size_t q1 = txt.find('"', pos);
    if (q1 == std::string::npos) break;
    size_t q2 = txt.find('"', q1 + 1);
    if (q2 == std::string::npos) break;
    std::string key = txt.substr(q1 + 1, q2 - q1 - 1);
    size_t c = q2 + 1;
    while (c < txt.size() && (txt[c] == ' ' || txt[c] == '\t' || txt[c] == '\n' || txt[c] == '\r')) ++c;
    if (c < txt.size() && txt[c] == ':') {
      ++c;
      while (c < txt.size() && (txt[c] == ' ' || txt[c] == '\t' || txt[c] == '\n' || txt[c] == '\r')) ++c;
      if (c < txt.size() && txt[c] == '{') {
        // Balanced brace matching, skipping strings (brace chars inside a
        // quoted value must not change the depth).
        int depth = 0;
        bool in_str = false;
        size_t close = std::string::npos;
        for (size_t i = c; i < txt.size(); ++i) {
          char ch = txt[i];
          if (ch == '"' && (i == 0 || txt[i - 1] != '\\')) in_str = !in_str;
          if (in_str) continue;
          if (ch == '{') ++depth;
          else if (ch == '}') {
            --depth;
            if (depth == 0) { close = i; break; }
          }
        }
        if (close == std::string::npos) break;
        std::string obj = txt.substr(c, close - c + 1);
        if (key.find('.') != std::string::npos) {
          MInfo mi;
          mi.kind = json_str(obj, "kind");
          mi.group_size = json_int(obj, "group_size");
          mi.bits = json_int(obj, "bits");
          mi.has_bias = json_bool(obj, "has_bias");
          out[key] = mi;
          pos = close + 1;  // consume a leaf entry object
        } else {
          pos = c + 1;  // a wrapper object (e.g. "manifest"): scan inside it
        }
        continue;
      }
    }
    pos = q2 + 1;
  }
  return out;
}

inline std::map<std::string, MInfo> load_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  std::stringstream ss; ss << f.rdbuf();
  return parse(ss.str());
}

// True if `observed` matches the manifest entry `m` for a quantized Linear.
inline bool matches(const MInfo& m, int group_size, int bits,
                    const std::string& expected_kind = "quantized_linear") {
  return m.kind == expected_kind && m.group_size == group_size && m.bits == bits;
}

}  // namespace veystrix::engine::dmanifest
