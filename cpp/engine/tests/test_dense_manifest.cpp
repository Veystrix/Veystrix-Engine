// Pure-CPU unit test for the dense manifest parser + fail-closed match check.
// No MLX, no I/O beyond an in-memory JSON string.
#include "dense_manifest.hpp"

#include <cstdio>
#include <string>

using namespace veystrix::engine::dmanifest;

static int failures = 0;
#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    if (!(cond)) {                                                          \
      std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);        \
      ++failures;                                                           \
    }                                                                       \
  } while (0)

int main() {
  // Shape mirroring dump_dense_manifest.py output.
  const std::string doc = R"({
  "language_model.model.layers.0.mlp.shared_expert.gate_proj": {
    "kind": "quantized_linear", "group_size": 32, "bits": 4,
    "has_bias": false, "weight_dtype": "mlx.core.uint32"
  },
  "language_model.model.layers.0.mlp.shared_expert_gate.weight": {
    "kind": "linear", "group_size": null, "bits": null,
    "has_bias": false, "weight_dtype": "mlx.core.float32"
  },
  "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": {
    "kind": "raw", "group_size": null, "bits": null, "has_bias": false
  }
})";
  auto m = parse(doc);
  CHECK(m.size() == 3, "three entries parsed");
  const std::string key = "language_model.model.layers.0.mlp.shared_expert.gate_proj";
  CHECK(m.count(key) == 1, "quantized entry present");
  CHECK(m[key].kind == "quantized_linear", "kind parsed");
  CHECK(m[key].group_size == 32, "group_size parsed");
  CHECK(m[key].bits == 4, "bits parsed");
  CHECK(m[key].has_bias == false, "has_bias parsed");

  // Correct match passes.
  CHECK(matches(m[key], 32, 4), "correct gs/bits matches manifest");
  // Wrong group size or bit width fails closed.
  CHECK(!matches(m[key], 64, 4), "wrong group size rejected");
  CHECK(!matches(m[key], 32, 8), "wrong bit width rejected");
  CHECK(!matches(m[key], 32, 4, "linear"), "wrong kind rejected");

  if (failures) { std::fprintf(stderr, "test_dense_manifest FAILED\n"); return 1; }
  std::printf("test_dense_manifest OK (all checks passed)\n");
  return 0;
}
