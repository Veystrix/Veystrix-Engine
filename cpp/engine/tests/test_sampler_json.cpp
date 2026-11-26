// Offline gate for the strict sampler_json parser (P2). No MLX, no checkpoint.
#include "../src/sampler_json.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using veystrix::engine::SamplerSettings;
using veystrix::engine::parse_sampler_json;

namespace {
int failures = 0;

void check(bool ok, const std::string& name) {
  if (!ok) { std::cerr << "FAIL: " << name << "\n"; ++failures; }
}

void ok(const std::string& json, const std::string& name) {
  try { parse_sampler_json(json); }
  catch (const std::exception& e) {
    std::cerr << "FAIL: " << name << " threw: " << e.what() << "\n";
    ++failures;
  }
}

void refused(const std::string& json, const std::string& needle, const std::string& name) {
  try {
    parse_sampler_json(json);
    std::cerr << "FAIL: " << name << " did not refuse\n";
    ++failures;
  } catch (const std::exception& e) {
    if (needle.empty()) return;
    if (std::string(e.what()).find(needle) == std::string::npos) {
      std::cerr << "FAIL: " << name << " wrong message: " << e.what() << "\n";
      ++failures;
    }
  }
}
}  // namespace

int main() {
  // Python's two real presets round-trip exactly (float32).
  SamplerSettings thinking = parse_sampler_json(
      R"({"greedy":false,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0})");
  check(!thinking.greedy, "thinking non-greedy");
  check(thinking.temperature == 1.0f && thinking.top_p == 0.95f &&
        thinking.top_k == 20 && thinking.min_p == 0.0f &&
        thinking.presence_penalty == 0.0f, "thinking values");
  SamplerSettings instruct = parse_sampler_json(
      R"({"greedy":false,"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5})");
  check(instruct.temperature == 0.7f && instruct.top_p == 0.8f &&
        instruct.presence_penalty == 1.5f, "instruct values");

  // Greedy defaults and explicit forms.
  check(parse_sampler_json("{}").greedy, "empty greedy");
  check(parse_sampler_json(R"({"temperature":0})").greedy, "temperature 0 greedy");
  check(parse_sampler_json(R"({"greedy":true,"temperature":0.7})").greedy,
        "explicit greedy wins");
  check(!parse_sampler_json(R"({"temperature":0.7})").greedy,
        "temperature > 0 is non-greedy");
  ok(R"({ "greedy" : false , "temperature" : 1.0 })", "whitespace");

  // Unknown keys (the exact features Python does NOT have).
  refused(R"({"frequency_penalty":0.5})", "frequency_penalty", "frequency penalty");
  refused(R"({"repetition_penalty":1.1})", "repetition_penalty", "repetition penalty");
  refused(R"({"seed":7})", "seed", "seed");
  refused(R"({"stop":["END"]})", "stop", "stop strings");
  refused(R"({"grammar":"x"})", "grammar", "grammar");

  // Malformed / hostile input.
  refused("", "expected", "empty text");
  refused("{", "expected", "unclosed object");
  refused("[]", "expected", "array");
  refused(R"({"greedy":false,})", "expected", "trailing comma");
  refused(R"({"temperature":1.0} extra)", "trailing", "trailing content");
  refused(R"({"temperature":1.0,"temperature":0.5})", "duplicate", "duplicate key");
  refused(R"({"temperature":nan})", "non-finite", "nan");
  refused(R"({"temperature":1e999})", "non-finite", "infinity");
  refused(R"({"top_k":-1})", "non-negative integer", "negative top_k");
  refused(R"({"top_k":2.5})", "non-negative integer", "fractional top_k");
  refused(R"({"greedy":false,"temperature":0})", "temperature > 0", "false greedy at t=0");
  refused(R"({"greedy":"false"})", "expected", "string bool");
  refused(R"({"nested":{"a":1}})", "nested", "nested object");

  // force token (think-budget </think> override)
  SamplerSettings f = parse_sampler_json("{\"force\": 248046}");
  check(f.force_token == 248046, "force token parsed");
  SamplerSettings nf = parse_sampler_json("{}");
  check(nf.force_token == -1, "no force by default");
  refused("{\"force\": -1}", "force", "negative force");

  if (failures == 0) std::cout << "sampler_json: all cases passed\n";
  return failures == 0 ? 0 : 1;
}
