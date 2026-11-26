#pragma once

// Strict, dependency-free parser for the C ABI sampler_json object.
// Python's engine surface is EXACTLY five fields (veystrix/sampling.py:46-54):
// temperature, top_p, top_k, min_p, presence_penalty, plus `greedy`.
// Unknown keys are refused by NAME: silently ignoring a requested knob would
// be a different generation, which the bit-exact gate forbids.

#include <string>

namespace veystrix::engine {

struct SamplerSettings {
  bool greedy = true;
  float temperature = 1.0f;
  float top_p = 0.95f;
  int top_k = 20;
  float min_p = 0.0f;
  float presence_penalty = 0.0f;
  int force_token = -1;  // >=0 overrides sampling (forced </think> token)
  // With force: the step appends the forced token but RETURNS the greedy argmax of
  // the logits it replaced (teacher-forced quality probes; generic runtime only).
  bool probe = false;
  // With probe: also put "lp=<log p(forced)> top=<log p(argmax)>" (float32 log-softmax of the replaced
  // logits) in the session's last digest (KV quality gate: per-token NLL; generic runtime only).
  bool logprob = false;
};

// Throws std::runtime_error with a named message on malformed input, unknown
// keys, duplicate keys, non-finite numbers, or a non-integer/negative top_k.
SamplerSettings parse_sampler_json(const std::string& text);

}  // namespace veystrix::engine
