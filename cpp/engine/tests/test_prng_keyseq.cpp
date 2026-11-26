// G-NG-KEYSEQ: a per-session KeySequence must produce the same categorical
// draws as the process-wide default key after random::seed(s). This is what
// makes a C++ session independent of other sessions while staying bit-exact
// against Python's mx.random.categorical with no explicit key.
#include <mlx/mlx.h>
#include <mlx/random.h>

#include <cstdint>
#include <iostream>
#include <vector>

namespace mx = mlx::core;

int main() {
  int failures = 0;
  std::vector<float> row(64);
  for (std::size_t i = 0; i < row.size(); ++i)
    row[i] = static_cast<float>((i * 37 + 11) % 29) * 0.25f - 2.0f;
  mx::array logits(row.data(), {static_cast<int>(row.size())});

  for (uint64_t seed : {uint64_t(0), uint64_t(1), uint64_t(42),
                        uint64_t(20261117), uint64_t(0xdeadbeefULL)}) {
    mx::random::seed(seed);
    mx::random::KeySequence session_key(seed);
    for (int draw = 0; draw < 8; ++draw) {
      mx::array a = mx::random::categorical(logits, -1);  // default key
      mx::array b = mx::random::categorical(logits, -1, session_key.next());
      mx::eval(a);
      mx::eval(b);
      uint32_t av = a.data<uint32_t>()[0];
      uint32_t bv = b.data<uint32_t>()[0];
      if (av != bv) {
        std::cerr << "FAIL seed=" << seed << " draw=" << draw
                  << " default=" << av << " session=" << bv << "\n";
        ++failures;
      }
    }
  }
  if (failures == 0) std::cout << "prng_keyseq: all 40 draws match\n";
  return failures == 0 ? 0 : 1;
}
