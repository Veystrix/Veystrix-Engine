// probe_embed.cpp -- OWNER-RUN diagnostic (new file, no shared edits).
//
// Reuses the EXACT embed loading path of real_checkpoint_generate.cpp by
// including that file with main() renamed (all its helpers live in anonymous
// namespaces, so no symbol collisions). Dumps the gather-then-dequantize
// embedding for the real prompt and the tiled hidden fed to layer 0, so it can
// be compared bit-for-bit with Python's layer_L00_input.f32.
//
// Usage: /tmp/probe_embed <checkpoint-dir> <flat-config> <out-dir>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define main realgen_main_disabled
#include "real_checkpoint_generate.cpp"
#undef main

static void dump_f32(const std::string& path, const mx::array& a) {
  mx::array f = mx::contiguous(mx::astype(a, mx::float32));
  mx::eval(f);
  std::ofstream o(path, std::ios::binary);
  o.write(reinterpret_cast<const char*>(f.data<float>()),
          static_cast<std::streamsize>(f.size() * sizeof(float)));
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <checkpoint-dir> <flat-config> <out-dir>\n",
                 argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  Config cfg = read_config(argv[2]);
  const int H = get_int(cfg, "hidden_size");
  const int hc = get_int(cfg, "hc_count");
  const int gs = get_int(cfg, "group_size", 32);
  const int bits = get_int(cfg, "bits", 4);
  const std::vector<int> prompt = split_ints(get_str(cfg, "prompt_ids"));
  const int S = static_cast<int>(prompt.size());

  DenseLoader dl(ckpt, cfg, gs, bits);
  DenseLoader::QTriplet embed =
      dl.quantized("language_model.model.embed_tokens");
  std::vector<int32_t> ids(prompt.begin(), prompt.end());
  mx::array f(ids.data(), mx::Shape{S}, mx::int32);

  // Exact copy of embed_all in real_checkpoint_generate.cpp:
  // gather packed rows FIRST, then mx::dequantize.
  mx::array w = mx::take(embed.w, f, 0);
  mx::array s = mx::take(embed.s, f, 0);
  mx::array b = mx::take(embed.b, f, 0);
  mx::array row = mx::dequantize(w, s, b, gs, bits, "affine");
  mx::array emb = mx::reshape(row, mx::Shape{1, S, H});
  mx::array tiled = mx::tile(emb, std::vector<int>{1, 1, hc});

  const std::string out = std::string(argv[3]) + "/";
  dump_f32(out + "cpp_embed_all.f32", emb);
  dump_f32(out + "cpp_tiled_all.f32", tiled);
  mx::array last = mx::reshape(
      mx::slice(tiled, mx::Shape{0, S - 1, 0}, mx::Shape{1, S, hc * H}),
      mx::Shape{1, hc * H});
  dump_f32(out + "cpp_tiled_last.f32", last);
  std::printf("embed dtype=%ld dequant dtype=%ld S=%d H=%d hc=%d\n",
              (long)static_cast<mx::Dtype::Val>(emb.dtype()),
              (long)static_cast<mx::Dtype::Val>(row.dtype()), S, H, hc);
  std::printf("wrote cpp_embed_all.f32 [1,%d,%d], cpp_tiled_all.f32 [1,%d,%d], "
              "cpp_tiled_last.f32 [1,%d]\n", S, H, S, hc * H, hc * H);
  return 0;
}
