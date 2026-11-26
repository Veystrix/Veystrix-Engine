// Minimal JSON reader for recipes and safetensors headers (generic runtime).
//
// Dependency-free recursive-descent parser producing an immutable tree. Only
// what recipes and safetensors headers need: objects, arrays, strings (with the
// standard escapes; \u is decoded to UTF-8), numbers (as double + int64 when
// integral), true/false/null. Errors throw std::runtime_error with an offset.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace veystrix::engine::generic {

class Json {
 public:
  enum class Kind { Null, Bool, Number, String, Array, Object };

  Json() = default;
  static Json parse(const std::string& text);

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool is_object() const { return kind_ == Kind::Object; }
  bool is_array() const { return kind_ == Kind::Array; }
  bool is_string() const { return kind_ == Kind::String; }
  bool is_number() const { return kind_ == Kind::Number; }
  bool is_bool() const { return kind_ == Kind::Bool; }

  // Typed access; throws with `what` in the message on a kind mismatch.
  const std::string& str(const char* what = "value") const;
  double num(const char* what = "value") const;
  int64_t i64(const char* what = "value") const;
  int i32(const char* what = "value") const { return static_cast<int>(i64(what)); }
  bool boolean(const char* what = "value") const;
  const std::vector<Json>& arr(const char* what = "value") const;
  const std::vector<std::pair<std::string, Json>>& obj(const char* what = "value") const;

  // Object lookup. has() is false for non-objects. at() throws if absent.
  bool has(const std::string& key) const;
  const Json& at(const std::string& key) const;
  const Json* find(const std::string& key) const;

  // Convenience getters with defaults (absent or null -> default).
  std::string get_str(const std::string& key, const std::string& def) const;
  double get_num(const std::string& key, double def) const;
  int64_t get_i64(const std::string& key, int64_t def) const;
  bool get_bool(const std::string& key, bool def) const;

  std::string dump() const;  // compact re-serialisation (for fingerprints)

 private:
  Kind kind_ = Kind::Null;
  bool b_ = false;
  double n_ = 0.0;
  int64_t i_ = 0;
  bool integral_ = false;
  std::string s_;
  std::vector<Json> a_;
  std::vector<std::pair<std::string, Json>> o_;

  friend struct JsonParser;
};

}  // namespace veystrix::engine::generic
