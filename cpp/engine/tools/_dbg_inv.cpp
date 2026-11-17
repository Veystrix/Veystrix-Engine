#include "qsa_full.hpp"
#include <mlx/mlx.h>
namespace mx = mlx::core;
using namespace veystrix::engine;
int main() {
  QSARotaryParams rp;
  rp.dim = 32; rp.base = 10000000.0f; rp.mrope_section = {11,11,10};
  rp.style = "interleaved"; rp.attention_scaling = 1.0f;
  mx::array inv = qsa_compute_inv_freq(rp);
  mx::eval(inv);
  mx::save("/tmp/dbg_inv_cpp.npy", inv);
  return 0;
}
