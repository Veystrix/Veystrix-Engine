// probe_l0_weights.cpp -- OWNER-RUN diagnostic (new file, no shared edits).
// Reuses real_checkpoint_generate.cpp's EXACT DenseLoader path (include with
// main renamed) to load every L0 weight the end-to-end batched prefill uses,
// and dumps them in the same raw format as dump_python_moe_intermediates.py /
// dump_python_intermediates.py so they can be compared bit-for-bit.
//
// Usage: /tmp/probe_l0_weights <checkpoint-dir> <flat-config> <out-dir>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define main realgen_main_disabled
#include "real_checkpoint_generate.cpp"
#undef main

static void dump(const std::string& dir, const std::string& name,
                 const mx::array& a) {
  std::ofstream o(dir + "/" + name + ".bin", std::ios::binary);
  if (a.dtype() == mx::uint32) {
    mx::array f = mx::contiguous(a);
    mx::eval(f);
    o.write(reinterpret_cast<const char*>(f.data<uint32_t>()),
            static_cast<std::streamsize>(f.size() * sizeof(uint32_t)));
  } else {
    mx::array f = mx::contiguous(mx::astype(a, mx::float32));
    mx::eval(f);
    o.write(reinterpret_cast<const char*>(f.data<float>()),
            static_cast<std::streamsize>(f.size() * sizeof(float)));
  }
}

static void dumpq(const std::string& dir, const std::string& name,
                  const Linear& L) {
  if (L.q) {
    dump(dir, name + "_weight", L.q->w);
    dump(dir, name + "_scales", L.q->scales);
    dump(dir, name + "_biases", L.q->biases);
  } else if (L.dense) {
    dump(dir, name + "_dense", *L.dense);
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <ckpt> <config> <out>\n", argv[0]);
    return 2;
  }
  Config cfg = read_config(argv[2]);
  const int H = get_int(cfg, "hidden_size");
  const int hc = get_int(cfg, "hc_count");
  const int low = get_int(cfg, "hc_lowrank");
  const int gs = get_int(cfg, "group_size", 32);
  const int bits = get_int(cfg, "bits", 4);
  const int hcH = hc * H;
  DenseLoader dl(argv[1], cfg, gs, bits);
  const std::string out = argv[3];
  const std::string b = "language_model.model.layers.0.";

  // Hyper-connection norm gains and quantized projections.
  dump(out, "hc_attn_norm_w", dl.dense(b + "attn_hyper_connection.hc_norm.weight"));
  dump(out, "hc_mlp_norm_w", dl.dense(b + "mlp_hyper_connection.hc_norm.weight"));
  dumpq(out, "hc_attn_input_mix_weight_down",
        dl.linear(b + "attn_hyper_connection.input_mix_weight_down", hcH));
  dumpq(out, "hc_attn_input_mix_weight_up",
        dl.linear(b + "attn_hyper_connection.input_mix_weight_up", low));
  dumpq(out, "hc_attn_block_inject_weight",
        dl.linear(b + "attn_hyper_connection.block_inject_weight", hcH));
  dumpq(out, "hc_mlp_input_mix_weight_down",
        dl.linear(b + "mlp_hyper_connection.input_mix_weight_down", hcH));
  dumpq(out, "hc_mlp_input_mix_weight_up",
        dl.linear(b + "mlp_hyper_connection.input_mix_weight_up", low));
  dumpq(out, "hc_mlp_block_inject_weight",
        dl.linear(b + "mlp_hyper_connection.block_inject_weight", hcH));

  // Router + shared expert.
  dumpq(out, "gate_w", dl.linear(b + "mlp.gate.weight", 0));
  dumpq(out, "w_shared_gate", dl.linear(b + "mlp.shared_expert.gate_proj", H));
  dumpq(out, "w_shared_up", dl.linear(b + "mlp.shared_expert.up_proj", H));
  dumpq(out, "w_shared_down", dl.linear(b + "mlp.shared_expert.down_proj", 0));
  dumpq(out, "w_shared_sgate", dl.linear(b + "mlp.shared_expert_gate.weight", H));

  // GDN weights (the end-to-end path loads these via DenseLoader too).
  const std::string a = b + "linear_attn.";
  dumpq(out, "gdn_in_proj_qkv", dl.linear(a + "in_proj_qkv", H));
  dumpq(out, "gdn_in_proj_z", dl.linear(a + "in_proj_z", H));
  dumpq(out, "gdn_in_proj_b", dl.linear(a + "in_proj_b", H));
  dumpq(out, "gdn_in_proj_a", dl.linear(a + "in_proj_a", H));
  dump(out, "w_conv1d_weight", dl.dense(a + "conv1d.weight"));
  dump(out, "w_A_log", dl.dense(a + "A_log"));
  dump(out, "w_dt_bias", dl.dense(a + "dt_bias"));
  dump(out, "w_norm_weight", dl.dense(a + "norm.weight"));
  dumpq(out, "w_outproj", dl.linear(a + "out_proj", get_int(cfg, "value_dim")));

  std::printf("dumped L0 weights from C++ DenseLoader to %s\n", out.c_str());
  return 0;
}
