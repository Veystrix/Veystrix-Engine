#include "qsa_full.hpp"
#include <mlx/mlx.h>
namespace mx = mlx::core;
using namespace veystrix::engine;
int main(int argc, char** argv) {
  mx::array x = mx::load("/tmp/dbg_rope_x.npy");
  mx::array pos = mx::load("/tmp/dbg_rope_pos.npy");
  QSARotaryParams rp;
  rp.dim = 32; rp.base = 10000000.0f; rp.mrope_section = {11,11,10};
  rp.style = "interleaved"; rp.attention_scaling = 1.0f;
  mx::array out = qsa_apply_rope(x, pos, rp);
  mx::eval(out);
  mx::save("/tmp/dbg_rope_cpp.npy", out);
  return 0;
}
