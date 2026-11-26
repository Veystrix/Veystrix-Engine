#include "generic/json_lite.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace veystrix::engine::generic {

struct JsonParser {
  const std::string& t;
  size_t i = 0;
  explicit JsonParser(const std::string& text) : t(text) {}

  [[noreturn]] void fail(const char* msg) const {
    throw std::runtime_error(std::string("json: ") + msg + " at offset " + std::to_string(i));
  }
  void ws() {
    while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) ++i;
  }
  bool lit(const char* w) {
    size_t n = std::char_traits<char>::length(w);
    if (t.compare(i, n, w) == 0) { i += n; return true; }
    return false;
  }
  static void utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  uint32_t hex4() {
    if (i + 4 > t.size()) fail("short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = t[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad \\u escape");
    }
    return v;
  }
  std::string string() {
    if (t[i] != '"') fail("expected string");
    ++i;
    std::string out;
    while (true) {
      if (i >= t.size()) fail("unterminated string");
      char c = t[i++];
      if (c == '"') break;
      if (c != '\\') { out += c; continue; }
      if (i >= t.size()) fail("bad escape");
      char e = t[i++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= t.size() && t[i] == '\\' && t[i + 1] == 'u') {
            i += 2;
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(out, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return out;
  }
  Json value() {
    ws();
    if (i >= t.size()) fail("unexpected end");
    Json v;
    char c = t[i];
    if (c == '{') {
      v.kind_ = Json::Kind::Object;
      ++i;
      ws();
      if (i < t.size() && t[i] == '}') { ++i; return v; }
      while (true) {
        ws();
        std::string k = string();
        ws();
        if (i >= t.size() || t[i] != ':') fail("expected ':'");
        ++i;
        v.o_.emplace_back(std::move(k), value());
        ws();
        if (i < t.size() && t[i] == ',') { ++i; continue; }
        if (i < t.size() && t[i] == '}') { ++i; break; }
        fail("expected ',' or '}'");
      }
    } else if (c == '[') {
      v.kind_ = Json::Kind::Array;
      ++i;
      ws();
      if (i < t.size() && t[i] == ']') { ++i; return v; }
      while (true) {
        v.a_.push_back(value());
        ws();
        if (i < t.size() && t[i] == ',') { ++i; continue; }
        if (i < t.size() && t[i] == ']') { ++i; break; }
        fail("expected ',' or ']'");
      }
    } else if (c == '"') {
      v.kind_ = Json::Kind::String;
      v.s_ = string();
    } else if (lit("true")) {
      v.kind_ = Json::Kind::Bool; v.b_ = true;
    } else if (lit("false")) {
      v.kind_ = Json::Kind::Bool; v.b_ = false;
    } else if (lit("null")) {
      v.kind_ = Json::Kind::Null;
    } else if (lit("NaN")) {
      v.kind_ = Json::Kind::Number; v.n_ = NAN;
    } else if (lit("Infinity")) {
      v.kind_ = Json::Kind::Number; v.n_ = INFINITY;
    } else if (lit("-Infinity")) {
      v.kind_ = Json::Kind::Number; v.n_ = -INFINITY;
    } else {
      size_t s = i;
      bool integral = true;
      if (t[i] == '-') ++i;
      while (i < t.size()) {
        char d = t[i];
        if (d >= '0' && d <= '9') { ++i; continue; }
        if (d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') { integral = false; ++i; continue; }
        break;
      }
      if (s == i) fail("unexpected character");
      std::string num = t.substr(s, i - s);
      v.kind_ = Json::Kind::Number;
      v.n_ = std::strtod(num.c_str(), nullptr);
      v.integral_ = integral;
      if (integral) v.i_ = std::strtoll(num.c_str(), nullptr, 10);
      else v.i_ = static_cast<int64_t>(v.n_);
    }
    return v;
  }
};

Json Json::parse(const std::string& text) {
  JsonParser p(text);
  Json v = p.value();
  p.ws();
  if (p.i != text.size()) p.fail("trailing characters");
  return v;
}

static const char* kind_name(Json::Kind k) {
  switch (k) {
    case Json::Kind::Null: return "null";
    case Json::Kind::Bool: return "bool";
    case Json::Kind::Number: return "number";
    case Json::Kind::String: return "string";
    case Json::Kind::Array: return "array";
    case Json::Kind::Object: return "object";
  }
  return "?";
}

#define JSON_EXPECT(K)                                                                    \
  if (kind_ != K)                                                                         \
    throw std::runtime_error(std::string("json: ") + what + " is " + kind_name(kind_) +    \
                             ", expected " + kind_name(K));

const std::string& Json::str(const char* what) const { JSON_EXPECT(Kind::String) return s_; }
double Json::num(const char* what) const { JSON_EXPECT(Kind::Number) return n_; }
int64_t Json::i64(const char* what) const {
  JSON_EXPECT(Kind::Number)
  if (!integral_ && std::floor(n_) != n_)
    throw std::runtime_error(std::string("json: ") + what + " is not an integer");
  return integral_ ? i_ : static_cast<int64_t>(n_);
}
bool Json::boolean(const char* what) const { JSON_EXPECT(Kind::Bool) return b_; }
const std::vector<Json>& Json::arr(const char* what) const { JSON_EXPECT(Kind::Array) return a_; }
const std::vector<std::pair<std::string, Json>>& Json::obj(const char* what) const {
  JSON_EXPECT(Kind::Object) return o_;
}

const Json* Json::find(const std::string& key) const {
  if (kind_ != Kind::Object) return nullptr;
  for (const auto& kv : o_)
    if (kv.first == key) return &kv.second;
  return nullptr;
}
bool Json::has(const std::string& key) const { return find(key) != nullptr; }
const Json& Json::at(const std::string& key) const {
  const Json* v = find(key);
  if (!v) throw std::runtime_error("json: missing key '" + key + "'");
  return *v;
}
std::string Json::get_str(const std::string& key, const std::string& def) const {
  const Json* v = find(key);
  return (v && v->is_string()) ? v->s_ : def;
}
double Json::get_num(const std::string& key, double def) const {
  const Json* v = find(key);
  return (v && v->is_number()) ? v->n_ : def;
}
int64_t Json::get_i64(const std::string& key, int64_t def) const {
  const Json* v = find(key);
  return (v && v->is_number()) ? v->i64(key.c_str()) : def;
}
bool Json::get_bool(const std::string& key, bool def) const {
  const Json* v = find(key);
  return (v && v->is_bool()) ? v->b_ : def;
}

static void dump_str(std::ostringstream& o, const std::string& s) {
  o << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') o << '\\' << c;
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o << b; }
    else o << c;
  }
  o << '"';
}
static void dump_rec(std::ostringstream& o, const Json& v) {
  switch (v.kind()) {
    case Json::Kind::Null: o << "null"; break;
    case Json::Kind::Bool: o << (v.boolean() ? "true" : "false"); break;
    case Json::Kind::Number: {
      double d = v.num();
      if (std::floor(d) == d && std::fabs(d) < 9e15) o << static_cast<long long>(d);
      else { char b[40]; std::snprintf(b, sizeof b, "%.17g", d); o << b; }
      break;
    }
    case Json::Kind::String: dump_str(o, v.str()); break;
    case Json::Kind::Array: {
      o << '[';
      bool first = true;
      for (const auto& e : v.arr()) { if (!first) o << ','; first = false; dump_rec(o, e); }
      o << ']';
      break;
    }
    case Json::Kind::Object: {
      o << '{';
      bool first = true;
      for (const auto& kv : v.obj()) {
        if (!first) o << ',';
        first = false;
        dump_str(o, kv.first);
        o << ':';
        dump_rec(o, kv.second);
      }
      o << '}';
      break;
    }
  }
}
std::string Json::dump() const {
  std::ostringstream o;
  dump_rec(o, *this);
  return o.str();
}

}  // namespace veystrix::engine::generic
