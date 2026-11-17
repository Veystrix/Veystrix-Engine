#!/usr/bin/env python3
"""Generate a tiny deterministic 2-shard safetensors checkpoint for the
cross-shard MoE slot-bookkeeping test, plus a flat raw dump for the test's
all-expert-resident reference.

Layout (all 9 tensors of one MoE layer):
  gate_proj.{weight,scales,biases}, up_proj.{...}: [E, INTER, HIDDEN/8|/G]
  down_proj.{weight,scales,biases}:                [E, HIDDEN, INTER/8|/G]
Shard A holds tensors 0..7; shard B holds tensor 8 (down_proj.biases).
Shard B's data section is padded so tensor 8's ABSOLUTE file offset satisfies
(offset - off0) % rb0 == 0, the trigger for the old offset-derived-miss bug.

Usage: gen_crossshard_checkpoint.py <outdir>
"""
import json, os, struct, sys

E, HIDDEN, INTER, GROUP, BITS = 16, 32, 32, 32, 4
PREFIX = "language_model.model.layers.0.mlp.switch_mlp"

# name, dtype, shape
SPECS = [
    ("gate_proj.weight",  "U32",  [E, INTER, HIDDEN // 8]),
    ("gate_proj.scales",  "BF16", [E, INTER, HIDDEN // GROUP]),
    ("gate_proj.biases",  "BF16", [E, INTER, HIDDEN // GROUP]),
    ("up_proj.weight",    "U32",  [E, INTER, HIDDEN // 8]),
    ("up_proj.scales",    "BF16", [E, INTER, HIDDEN // GROUP]),
    ("up_proj.biases",    "BF16", [E, INTER, HIDDEN // GROUP]),
    ("down_proj.weight",  "U32",  [E, HIDDEN, INTER // 8]),
    ("down_proj.scales",  "BF16", [E, HIDDEN, INTER // GROUP]),
    ("down_proj.biases",  "BF16", [E, HIDDEN, INTER // GROUP]),
]


def row_bytes(dtype, shape):
    per = 4 if dtype == "U32" else 2
    n = 1
    for d in shape[1:]:
        n *= d
    return n * per


def make_data(seed, dtype, shape):
    n = 1
    for d in shape:
        n *= d
    per = 4 if dtype == "U32" else 2
    # deterministic bytes
    raw = bytearray()
    x = (seed * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF
    for _ in range(n * per):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        raw.append(x & 0xFF)
    if dtype == "BF16":
        # Keep scales/biases modest, finite, positive: reinterpret bytes as
        # u16 then clamp the exponent byte into a small range.
        for i in range(0, len(raw), 2):
            raw[i + 1] = 0x3E + (raw[i + 1] % 4)  # ~0.015625..0.03125
    return bytes(raw)


def shard(header, data, path):
    hb = header.encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(data)


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    data = [make_data(i + 1, dt, sh) for i, (_, dt, sh) in enumerate(SPECS)]
    rb = [row_bytes(dt, sh) for _, dt, sh in SPECS]
    assert all(len(data[i]) == E * rb[i] for i in range(9))

    a_names = [f"{PREFIX}.{n}" for n, _, _ in SPECS[:8]]
    b_name = f"{PREFIX}.{SPECS[8][0]}"

    # data_start_A = 8 + len(header_A); off0 = data_start_A (tensor 0 at 0).
    def header_a(offs):
        d = {}
        for idx, name in enumerate(a_names):
            dt, sh = SPECS[idx][1], SPECS[idx][2]
            d[name] = {"dtype": dt, "shape": sh,
                       "data_offsets": [offs[idx], offs[idx] + len(data[idx])]}
        return json.dumps(d, separators=(",", ":"))

    offs_a, cur = [], 0
    for i in range(8):
        offs_a.append(cur)
        cur += len(data[i])
    ha = header_a(offs_a)
    data_start_A = 8 + len(ha.encode())
    off0 = data_start_A
    rb0 = rb[0]

    # Shard B: solve pad so (data_start_B + pad - off0) % rb0 == 0, iterating
    # because pad appears in the header and can change its length.
    pad = 0
    for _ in range(12):
        hb = json.dumps({b_name: {"dtype": SPECS[8][1], "shape": SPECS[8][2],
                                  "data_offsets": [pad, pad + len(data[8])]}},
                        separators=(",", ":"))
        data_start_B = 8 + len(hb.encode())
        new_pad = (off0 - data_start_B) % rb0
        # Keep absolute offset STRICTLY ABOVE off0: the old bug's filter skips
        # reads with offset < off0, so a tensor below it would not trigger.
        if data_start_B + new_pad <= off0:
            new_pad += rb0
        if new_pad == pad:
            break
        pad = new_pad
    assert (data_start_B + pad - off0) % rb0 == 0, "Bug2 trigger not arranged"
    data_b = b"\x00" * pad + data[8]

    shard(ha, b"".join(data[:8]), os.path.join(outdir, "shardA.safetensors"))
    shard(hb, data_b, os.path.join(outdir, "shardB.safetensors"))

    # Flat dump for the test's all-resident reference.
    with open(os.path.join(outdir, "all_tensors.bin"), "wb") as f:
        for d in data:
            f.write(d)
    meta = {
        "E": E, "HIDDEN": HIDDEN, "INTER": INTER, "GROUP": GROUP, "BITS": BITS,
        "PREFIX": PREFIX,
        "shardA": "shardA.safetensors", "shardB": "shardB.safetensors",
        "tensors": [{"name": n, "dtype": dt, "shape": sh, "row_bytes": rb[i]}
                    for i, (n, dt, sh) in enumerate(SPECS)],
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"generated cross-shard checkpoint in {outdir}")
    print(f"  off0={off0} rb0={rb0} shardB pad={pad} "
          f"abs8={data_start_B + pad} (abs8-off0)%rb0={(data_start_B + pad - off0) % rb0}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "crossshard_ckpt")
