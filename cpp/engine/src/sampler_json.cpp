#include "sampler_json.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace veystrix::engine {
namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("sampler_json: " + message);
}

class Parser {
 public:
  Parser(const std::string& text, const std::set<std::string>& allowed)
      : text_(text), allowed_(allowed) {}

  std::map<std::string, double> parse() {
    ws();
    expect('{');
    ws();
    if (peek() == '}') { advance(); return values_; }
    for (;;) {
      ws();
      std::string key = string();
      if (!keys_.insert(key).second) fail("duplicate key '" + key + "'");
      if (allowed_.count(key) == 0)
        fail("sampler key '" + key +
             "' is not supported by Veystrix (Python engine has no equivalent)");
      ws();
      expect(':');
      ws();
      if (starts("true")) { literal("true"); booleans_[key] = true; }
      else if (starts("false")) { literal("false"); booleans_[key] = false; }
      else values_[key] = number();
      ws();
      if (peek() == ',') { advance(); continue; }
      expect('}');
      break;
    }
    ws();
    if (pos_ != text_.size()) fail("trailing content after object");
    return values_;
  }

  bool boolean(const std::string& key, bool fallback = false) const {
    auto it = booleans_.find(key);
    if (it == booleans_.end()) return fallback;
    return it->second;
  }

  bool has_number(const std::string& key) const { return values_.count(key) != 0; }
  bool has_boolean(const std::string& key) const { return booleans_.count(key) != 0; }
  const std::set<std::string>& keys() const { return keys_; }

  double number_value(const std::string& key, double fallback) const {
    auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    return it->second;
  }

 private:
  const std::string& text_;
  const std::set<std::string>& allowed_;
  std::size_t pos_ = 0;
  std::map<std::string, double> values_;
  std::map<std::string, bool> booleans_;
  std::set<std::string> keys_;

  char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }
  void advance() { ++pos_; }
  void ws() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' ||
            text_[pos_] == '\r' || text_[pos_] == '\n')) ++pos_;
  }
  bool starts(const char* word) const {
    return text_.compare(pos_, std::string(word).size(), word) == 0;
  }
  void literal(const char* word) {
    if (!starts(word)) fail(std::string("expected ") + word);
    pos_ += std::string(word).size();
  }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    advance();
  }
  std::string string() {
    expect('"');
    std::string out;
    while (pos_ < text_.size() && text_[pos_] != '"') {
      if (text_[pos_] == '\\') {
        ++pos_;
        if (pos_ >= text_.size()) fail("unfinished escape");
        char escaped = text_[pos_++];
        switch (escaped) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          default: fail("unsupported escape in key");
        }
      } else {
        out += text_[pos_++];
      }
    }
    if (pos_ >= text_.size()) fail("unterminated key string");
    advance();  // closing quote
    return out;
  }
  double number() {
    if (pos_ >= text_.size()) fail("expected number");
    char* end = nullptr;
    double value = std::strtod(text_.c_str() + pos_, &end);
    if (end == text_.c_str() + pos_) fail("expected number");
    if (!std::isfinite(value)) fail("non-finite number");
    pos_ = static_cast<std::size_t>(end - text_.c_str());
    return value;
  }
};

}  // namespace

SamplerSettings parse_sampler_json(const std::string& text) {
  static const std::set<std::string> allowed = {
      "greedy", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
      "force", "probe", "logprob"};
  Parser parser(text, allowed);
  parser.parse();

  SamplerSettings out;
  if (parser.has_number("force")) {
    double forced = parser.number_value("force", -1.0);
    if (forced < 0 || forced > 2147483647.0 || forced != std::floor(forced))
      fail("force must be a non-negative integer token id");
    out.force_token = static_cast<int>(forced);
  }
  // Python: greedy == temperature <= 0. An explicit `greedy` overrides.
  double temperature = parser.number_value("temperature", out.temperature);
  out.temperature = static_cast<float>(temperature);
  out.top_p = static_cast<float>(parser.number_value("top_p", out.top_p));
  out.min_p = static_cast<float>(parser.number_value("min_p", out.min_p));
  out.presence_penalty =
      static_cast<float>(parser.number_value("presence_penalty", out.presence_penalty));
  double top_k = parser.number_value("top_k", out.top_k);
  if (top_k < 0 || top_k != std::floor(top_k))
    fail("top_k must be a non-negative integer");
  if (top_k > 2147483647.0) fail("top_k is too large");
  out.top_k = static_cast<int>(top_k);

  if (parser.has_boolean("probe")) {
    out.probe = parser.boolean("probe");
    if (out.probe && out.force_token < 0) fail("probe requires force");
  }
  if (parser.has_boolean("logprob")) {
    out.logprob = parser.boolean("logprob");
    if (out.logprob && !out.probe) fail("logprob requires probe");
  }
  if (parser.has_boolean("greedy")) {
    out.greedy = parser.boolean("greedy");
  } else if (parser.has_number("temperature")) {
    out.greedy = temperature <= 0.0;  // mirrors Sampling.greedy
  } else {
    out.greedy = true;  // legacy C ABI: no sampler settings => greedy
  }
  // Explicit greedy:true always wins; greedy:false with temperature<=0 would
  // divide by zero in Python's sampler, so refuse it rather than emit garbage.
  if (out.greedy && temperature <= 0.0) {
    // fine: greedy path ignores temperature
  } else if (!out.greedy && temperature <= 0.0) {
    fail("greedy:false requires temperature > 0");
  }
  return out;
}

}  // namespace veystrix::engine
