#!/usr/bin/env python3
"""Stage 3/1 helper for the bit-exact fc2 all-zero K-group skip (Task B).

Two modes:

  repack  : read fc2.weight / fc2.scales / fc2.biases of one expert from a
            checkpoint and write the K-group-major byte file consumed by the
            C++ repack_fc2_kgroup_major() layout (chunk = hidden rows of
            [8 x uint32 LE packed weights, scale, bias], groups contiguous).
            The original dense format remains the default fallback in C++.

  probe   : given a relu2 activation tensor (.npy), report exactly-zero
            activation and fully-zero 64-neuron-group fractions (stage 1).
            This is the hook the owner runs on the real model; this script
            never runs the model itself.

Examples
--------
  python gen_fc2_skip_vectors.py repack \
      --shard model-00001-of-00022.safetensors \
      --prefix backbone.layers.0.mixer.switch_mlp \
      --expert 0 --hidden 2688 --intermediate 1856 --group 64 --bits 4 \
      --out fc2_e0.bin

  python gen_fc2_skip_vectors.py probe --acts relu2.npy --group 64

  python gen_fc2_skip_vectors.py selftest
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

LAYOUT = "fc2_kgroup_major_v1"


def chunk_layout(hidden: int, intermediate: int, group: int, bits: int,
                 s_itemsize: int):
    assert intermediate % group == 0
    n_groups = intermediate // group
    words = group * bits // 32            # 8 for G64/b4
    w_bytes = hidden * words * 4
    s_bytes = hidden * s_itemsize
    b_bytes = s_bytes
    chunk = w_bytes + s_bytes + b_bytes
    return dict(n_groups=n_groups, words_per_group=words, chunk_bytes=chunk,
                total_bytes=chunk * n_groups, w_bytes=w_bytes,
                s_bytes=s_bytes, b_bytes=b_bytes)


def _load_tensor(args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = ("fc2.weight", "fc2.scales", "fc2.biases")
    if getattr(args, "shard", None):
        from safetensors.numpy import load_file
        with open(args.shard, "rb") as f:
            blobs = load_file(f.read())
        prefix = args.prefix.rstrip(".")
        return tuple(blobs[f"{prefix}.{n}"][args.expert] for n in names)
    if getattr(args, "npy_dir", None):
        d = Path(args.npy_dir)
        return tuple(np.load(d / f"{n.replace('.', '_')}.npy") for n in names)
    raise SystemExit("provide --shard/--prefix/--expert or --npy-dir")


def repack(args) -> None:
    w, s, b = _load_tensor(args)
    hidden, intermediate = args.hidden, args.intermediate
    group, bits = args.group, args.bits
    if w.dtype != np.uint32:
        raise SystemExit(f"fc2.weight must be uint32 packed, got {w.dtype}")
    if w.shape != (hidden, intermediate // 8):
        raise SystemExit(f"unexpected weight shape {w.shape}")
    if s.shape != (hidden, intermediate // group) or b.shape != s.shape:
        raise SystemExit(f"unexpected scale/bias shapes {s.shape} / {b.shape}")

    s_raw = np.ascontiguousarray(s).tobytes()
    b_raw = np.ascontiguousarray(b).tobytes()
    w_raw = np.ascontiguousarray(w).astype("<u4", copy=False).tobytes()
    si, bi = s.dtype.itemsize, b.dtype.itemsize
    L = chunk_layout(hidden, intermediate, group, bits, si)
    if bi != si:
        raise SystemExit("scale/bias itemsize mismatch")

    words = L["words_per_group"]
    n_groups = L["n_groups"]
    w_row = intermediate // 8
    out = bytearray()
    for g in range(n_groups):
        for n in range(hidden):
            wo = (n * w_row + g * words) * 4
            out += w_raw[wo:wo + words * 4]
            so = (n * n_groups + g) * si
            out += s_raw[so:so + si]
            out += b_raw[so:so + si]
    assert len(out) == L["total_bytes"], (len(out), L["total_bytes"])

    out_path = Path(args.out)
    out_path.write_bytes(out)
    meta = dict(layout=LAYOUT, hidden=hidden, intermediate=intermediate,
                group_size=group, bits=bits, scale_dtype=str(s.dtype),
                bias_dtype=str(b.dtype), scale_itemsize=si, **L)
    out_path.with_suffix(out_path.suffix + ".json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_path} ({len(out)} bytes) + .json metadata")
    print(json.dumps(meta, indent=2))


def probe(args) -> None:
    a = np.load(args.acts).astype(np.float32, copy=False)
    if a.ndim < 2:
        raise SystemExit("activations must be [rows, intermediate]")
    group = args.group
    last = a.shape[-1]
    if last % group:
        raise SystemExit("intermediate not divisible by group")
    rows = int(np.prod(a.shape[:-1]))
    flat = a.reshape(rows, last)
    zero_acts = int(np.count_nonzero(flat == 0.0))
    n_groups = last // group
    gv = flat.reshape(rows, n_groups, group)
    group_zero_all_rows = np.all(gv == 0.0, axis=(0, 2))
    zids = [int(i) for i in np.nonzero(group_zero_all_rows)[0]]
    print(json.dumps(dict(
        total_activations=rows * last, zero_activations=zero_acts,
        activation_zero_fraction=zero_acts / (rows * last),
        rows=rows, intermediate=last, n_groups=n_groups,
        zero_groups=len(zids), group_zero_fraction=len(zids) / n_groups,
        zero_group_ids=zids), indent=2))


def selftest() -> None:
    hidden, intermediate, group, bits = 8, 128, 64, 4
    rng = np.random.default_rng(0)
    w = (rng.integers(0, 2**32, size=(hidden, intermediate // 8),
                      dtype=np.uint64) % (2**32)).astype(np.uint32)
    s = rng.random((hidden, intermediate // group)).astype(np.float32)
    b = rng.random((hidden, intermediate // group)).astype(np.float32)
    L = chunk_layout(hidden, intermediate, group, bits, s.dtype.itemsize)
    w_raw = np.ascontiguousarray(w).astype("<u4", copy=False).tobytes()
    s_raw = np.ascontiguousarray(s).tobytes()
    b_raw = np.ascontiguousarray(b).tobytes()
    words, n_groups, w_row = L["words_per_group"], L["n_groups"], intermediate // 8
    out = bytearray()
    for g in range(n_groups):
        for n in range(hidden):
            wo = (n * w_row + g * words) * 4
            out += w_raw[wo:wo + words * 4]
            so = (n * n_groups + g) * s.dtype.itemsize
            out += s_raw[so:so + s.dtype.itemsize]
            out += b_raw[so:so + b.dtype.itemsize]
    assert len(out) == L["total_bytes"]
    got = struct.unpack_from("<I", out, L["chunk_bytes"])[0]
    assert got == int(w[0, 8]), (got, w[0, 8])
    print("selftest OK: chunk_bytes=%d total=%d group1/row0 word matches"
          % (L["chunk_bytes"], L["total_bytes"]))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("repack")
    r.add_argument("--shard"); r.add_argument("--prefix", default="")
    r.add_argument("--expert", type=int, default=0)
    r.add_argument("--npy-dir")
    r.add_argument("--hidden", type=int, default=2688)
    r.add_argument("--intermediate", type=int, default=1856)
    r.add_argument("--group", type=int, default=64)
    r.add_argument("--bits", type=int, default=4)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=repack)
    pr = sub.add_parser("probe")
    pr.add_argument("--acts", required=True)
    pr.add_argument("--group", type=int, default=64)
    pr.set_defaults(fn=probe)
    st = sub.add_parser("selftest")
    st.set_defaults(fn=lambda _a: selftest())
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
