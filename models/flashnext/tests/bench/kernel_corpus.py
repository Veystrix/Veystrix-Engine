#!/usr/bin/env python3
"""Decode shape corpus for the Stage 4 kernel laboratory (pure Python).

Stage 0 records every `mx.quantized_matmul`, `mx.gather_qmm` and custom Metal
launch shape of a real run (`stage0.shape_trace`). The lab needs the decode
subset with a call count per decode step, so that a microbenchmark of each
shape can be weighted into a model of one token. This module extracts that
subset and refuses to guess when the artifact does not agree with itself.

Three things about the artifact are easy to get wrong and are checked here:

- A production artifact carries each arm twice, under `raw_arms` and under
  `conditions[].arms`. A naive walk for `operand_shapes` dicts counts every
  arm twice and doubles every call count. Arms are keyed by (condition,
  index), counted once, and the two views must be identical.
- A decode step count is inferred two independent ways and must agree. (a)
  From the artifact's own bookkeeping: arms that carry shape telemetry times
  `stage0.metrics.target_rounds` (decode forward passes per arm). (b) From a
  per-layer invariant in the call counts: the shared-expert gate projection
  (output width 1) runs once per layer per forward pass, so
  decode_calls / prefill_calls is the number of decode passes per prefill
  pass.
- The custom Metal MoE path covers layers - 1 layers per step. Layer 0 runs
  MLX `gather_qmm` (see `StreamingSwitchGLU`), so metal layers plus gather
  layers must equal the layer count.

Shape classes. Decode: dense `quantized_matmul` with `x` of `[1, 1, in]`,
`metal.*` rows whose first operand has one row, and `gather_qmm` rows whose
routing indices address a single token (layer 0 at decode; an extension of
the brief, because they are decode work). Prefill: everything else. Rows with
2 to 8 tokens, which a verification round would produce, cannot be told from
a short prefill by shape and are labelled prefill; none occur in the current
artifact.

Data types are not in the trace. `DTYPES` records what the code and the
checkpoint headers say, and how that was determined.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FIXTURE = REPO_ROOT / "models/flashnext/tests/inputs/stage4-shape-corpus.json"

DENSE = "mlx.quantized_matmul"
GATHER = "mlx.gather_qmm"
BITS = 4
GROUP_SIZE = 32
SCALE_BYTES = 2

# Cached rows are for decode steps at M=1. See module docstring for the
# honest limits of this classification.
DECODE, PREFILL = "decode", "prefill"

DTYPES = {
    "activation": {
        "dtype": "bfloat16",
        "confidence": "derived from code, not observed in the trace",
        "how": (
            "The decode input is nn.QuantizedEmbedding(...) = mx.dequantize "
            "with bf16 scales, so bfloat16. Qwen4ExpRMSNorm (and the "
            "patch_rmsnorm replacement) compute in float32 and cast back to "
            "the input dtype. Hyper-connection mixing, GDN, attention and the "
            "MoE block are all bfloat16 in and out. benchmarks bench_glue and "
            "compiled.py self-tests also build their decoder-layer arrays as "
            "bfloat16."
        ),
        "conflict": (
            "metal_runtime.py (header comment of _mlx_qmv_header) and the unit "
            "test test_production_width_float32_activation_and_bfloat16_"
            "metadata_are_exact state that Flash-Next feeds the custom "
            "kernels float32 activations. Nothing in the model code path "
            "produces float32 hidden states. The two cannot be reconciled "
            "from code alone, so the lab benchmarks the MoE executor with "
            "both and labels them."
        ),
    },
    "packed_weight": {
        "dtype": "uint32",
        "confidence": "verified",
        "how": "safetensors headers of the checkpoint: every quantized .weight is U32; "
               "the dense shapes in the trace are [out, in/8].",
    },
    "scales_biases": {
        "dtype": "bfloat16",
        "confidence": "verified",
        "how": "safetensors headers of the checkpoint (index of 22 shards): every "
               ".scales and .biases tensor, dense and switch_mlp, is BF16. "
               "ExpertPool and slab_pack records store them as 2-byte values "
               "(102,400 bytes per 640x80 part), and _get_dummy_streamed_weights "
               "builds them as bfloat16.",
    },
    "router_scores_and_shared_y": {
        "dtype": "bfloat16",
        "confidence": "derived from code, not observed",
        "how": "softmax and renormalisation of bf16 router logits, "
               "shared_y = sigmoid(shared_expert_gate(x)) * shared_expert(x) in bf16 "
               "(adaptive_topk.apply with FLASHNEXT_FUSED_SHARED=1, the default). "
               "mlx_vlm Qwen3_5MoeSparseMoeBlock.gate is not quantized (BF16 "
               "[512, 2560], no scales), so the router matmul is not in the trace.",
    },
    "gdn": {
        "dtype": "q,k,v bfloat16; a,b bfloat16; A_log,dt_bias bfloat16; state float32",
        "confidence": "derived from code and checkpoint headers",
        "how": "A_log and dt_bias are BF16 in the checkpoint; the recurrent state is "
               "created as float32 (gated_delta_update, mamba_ssm_dtype float32); "
               "g = exp(-exp(A_log.f32) * softplus(a + dt_bias)) is float32.",
    },
}

# Roles come from mlx_vlm qwen4_exp/language.py and qwen3_5; they are an
# annotation for the reader, not a measurement. Keyed by (in, out).
ROLES = {
    (10240, 320): "hyper-connection input_mix_down: 2 per layer + final mixer",
    (320, 10240): "hyper-connection input_mix_up: 2 per layer + final mixer",
    (10240, 4): "hyper-connection block_inject: 2 per layer",
    (2560, 1): "shared_expert_gate: 1 per layer",
    (2560, 640): "shared_expert gate_proj and up_proj: 2 per layer; QSA indexer index_qk_proj: 1 per full-attention layer",
    (640, 2560): "shared_expert down_proj: 1 per layer",
    (2560, 10240): "GDN in_proj_qkv: 1 per GDN layer; PLE key_proj: 1",
    (2560, 6144): "GDN in_proj_z: 1 per GDN layer",
    (2560, 48): "GDN in_proj_a and in_proj_b: 2 per GDN layer",
    (6144, 2560): "GDN out_proj (1 per GDN layer) and attention o_proj (1 per full-attention layer)",
    (2560, 12288): "attention q_proj with output gate: 1 per full-attention layer",
    (2560, 512): "attention k_proj and v_proj: 2 per full-attention layer",
    (2560, 2560): "PLE value_proj: 1",
    (2560, 248320): "lm_head (skipped on cache-commit-only rounds)",
}


class CorpusError(ValueError):
    """The artifact is inconsistent, or does not have the structure this module expects."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_arms(artifact: dict) -> tuple[dict, int]:
    """Arms that carry a shape trace, one entry per (condition, index).

    Returns ({key: arm}, duplicate_views). Views of the same arm must agree.
    """
    seen: dict = {}

    def key_for(path):
        if len(path) == 3 and path[0] == "raw_arms":
            return (str(path[1]), int(path[2]))
        if len(path) == 4 and path[0] == "conditions" and path[2] == "arms":
            name = artifact["conditions"][path[1]].get("condition", path[1])
            return (str(name), int(path[3]))
        return ("path",) + tuple(str(part) for part in path)

    def visit(node, path):
        if isinstance(node, dict):
            stage0 = node.get("stage0")
            if isinstance(stage0, dict) and stage0.get("shape_trace"):
                seen.setdefault(key_for(path), []).append((path, node))
                return
            for name, child in node.items():
                visit(child, path + (name,))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, path + (index,))

    visit(artifact, ())
    arms, duplicates = {}, 0
    for key, views in seen.items():
        first_path, first = views[0]
        for other_path, other in views[1:]:
            if first["stage0"]["shape_trace"] != other["stage0"]["shape_trace"]:
                raise CorpusError(
                    f"two views of arm {key} disagree: {first_path} and {other_path}"
                )
            duplicates += 1
        arms[key] = first
    return arms, duplicates


def _freeze(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _shapes(row) -> list[list[int]]:
    return [[int(d) for d in shape] for shape in row["operand_shapes"]]


def classify(row: dict) -> str:
    """`decode` or `prefill` for one trace row (see module docstring)."""
    name, shapes = row["operation"], _shapes(row)
    first = shapes[0]
    if name == DENSE:
        return DECODE if len(first) == 3 and first[0] == 1 and first[1] == 1 else PREFILL
    if name.startswith("metal."):
        return DECODE if first[0] == 1 else PREFILL
    if name == GATHER:
        route = shapes[-1]
        tokens = math.prod(route[:-1]) if len(route) >= 2 else route[0]
        return DECODE if tokens == 1 else PREFILL
    return PREFILL


def q4_bytes(out_width: int, in_width: int) -> dict:
    """Bytes one Q4 G32 matrix streams: packed weights plus bf16 scales and biases."""
    weight = out_width * in_width // 2
    meta = out_width * in_width // GROUP_SIZE * SCALE_BYTES
    return {"weight": weight, "scales": meta, "biases": meta, "total": weight + 2 * meta}


def describe(row: dict) -> dict:
    """Derived widths and weight bytes of one row, with the packing cross-checked."""
    name, shapes = row["operation"], _shapes(row)
    out = {}
    if name == DENSE:
        x, w = shapes[0], shapes[1]
        in_width, out_width = x[-1], w[0]
        if w[1] * 32 // BITS != in_width:
            raise CorpusError(f"{name} weight {w} does not pack {in_width} inputs at {BITS} bits")
        out.update(kind="dense", in_width=in_width, out_width=out_width, rows=math.prod(x[:-1]))
        out["weight_bytes"] = q4_bytes(out_width, in_width)["total"]
        role = ROLES.get((in_width, out_width))
        if role:
            out["role"] = role
    elif name == GATHER:
        x, w = shapes[0], shapes[1]
        experts, out_width, packed = w
        in_width = x[-1]
        if packed * 32 // BITS != in_width:
            raise CorpusError(f"{name} weight {w} does not pack {in_width} inputs")
        out.update(kind="gather", in_width=in_width, out_width=out_width, experts=experts,
                   rows=1, slots=int(shapes[-1][-1]))
        out["weight_bytes"] = experts * q4_bytes(out_width, in_width)["total"]
    elif name.startswith("metal."):
        x, w, routes = shapes[0], shapes[1], shapes[4]
        experts, out_width, packed = w
        in_width = x[-1]
        if packed * 32 // BITS != in_width:
            raise CorpusError(f"{name} weight {w} does not pack {in_width} inputs")
        out.update(kind="metal", in_width=in_width, out_width=out_width, experts=experts,
                   rows=routes[0], slots=routes[1])
        out["weight_bytes"] = experts * q4_bytes(out_width, in_width)["total"]
    else:
        raise CorpusError(f"unknown operation {name!r}")
    return out


def _aggregate(arms: dict) -> dict:
    """{(operation, shapes, options): {arm_key: calls}} over unique arms."""
    table: dict = {}
    for key, arm in arms.items():
        for row in arm["stage0"]["shape_trace"]:
            ident = (row["operation"], _freeze(row["operand_shapes"]), _freeze(row["options"]))
            counts = table.setdefault(ident, {})
            counts[key] = counts.get(key, 0) + int(row["calls"])
    return table


def _row_from(ident, counts, first_row):
    return {
        "operation": ident[0],
        "operand_shapes": first_row["operand_shapes"],
        "options": first_row["options"],
        "calls_by_arm": {"/".join(map(str, key)): n for key, n in sorted(counts.items())},
        "total_calls": sum(counts.values()),
    }


def infer_decode_steps(arms: dict, rows: list[dict], token_limit=None) -> dict:
    """Decode forward passes, inferred from the artifact and from call counts.

    Raises CorpusError when the two methods disagree.
    """
    if not arms:
        raise CorpusError("no arm carries shape telemetry")
    per_arm_rounds = []
    for key, arm in sorted(arms.items()):
        metrics = (arm["stage0"].get("metrics") or {})
        rounds = metrics.get("target_rounds", token_limit)
        if rounds is None:
            raise CorpusError(f"arm {key} has no target_rounds and the artifact has no token_limit")
        per_arm_rounds.append(int(rounds))
    if len(set(per_arm_rounds)) != 1:
        raise CorpusError(f"arms disagree on decode rounds: {per_arm_rounds}")
    method_a = {
        "arms_with_shape_telemetry": len(arms),
        "decode_rounds_per_arm": per_arm_rounds[0],
        "token_limit": token_limit,
        "total_steps": len(arms) * per_arm_rounds[0],
        "source": "stage0.metrics.target_rounds (token_limit when absent) times unique arms",
    }

    gate = [r for r in rows if r["operation"] == DENSE and r["out_width"] == 1]
    prefill_gate = [r for r in gate if r["class"] == PREFILL]
    decode_gate = [r for r in gate if r["class"] == DECODE]
    if not prefill_gate or not decode_gate:
        raise CorpusError("per-layer invariant needs the out=1 projection in both decode and prefill")
    prefill_rows = {r["rows"] for r in prefill_gate}
    if len(prefill_rows) != 1:
        raise CorpusError(f"prefill ran in more than one chunk length {sorted(prefill_rows)}; "
                          "the layer invariant assumes one prefill pass")
    per_arm_steps, layers_per_arm = set(), set()
    for key in arms:
        name = "/".join(map(str, key))
        pre = sum(r["calls_by_arm"].get(name, 0) for r in prefill_gate)
        dec = sum(r["calls_by_arm"].get(name, 0) for r in decode_gate)
        tokens = arms[key]["stage0"].get("metrics", {}).get("computed_prefill_tokens")
        if tokens is not None and int(tokens) != next(iter(prefill_rows)):
            raise CorpusError(f"arm {name} prefilled {tokens} tokens in one pass of "
                              f"{next(iter(prefill_rows))} rows; chunked prefill is not supported")
        if pre == 0 or dec % pre:
            raise CorpusError(f"arm {name}: {dec} decode and {pre} prefill gate calls are not a whole ratio")
        layers_per_arm.add(pre)
        per_arm_steps.add(dec // pre)
    if len(layers_per_arm) != 1 or len(per_arm_steps) != 1:
        raise CorpusError(f"arms disagree: layers {sorted(layers_per_arm)}, steps {sorted(per_arm_steps)}")
    layers, steps_each = layers_per_arm.pop(), per_arm_steps.pop()
    method_b = {
        "layers": layers,
        "decode_passes_per_arm": steps_each,
        "total_steps": steps_each * len(arms),
        "source": "decode calls / prefill calls of the shared-expert gate projection "
                  "(out width 1, one call per layer per forward pass)",
    }
    if method_a["total_steps"] != method_b["total_steps"]:
        raise CorpusError(
            f"decode steps disagree: artifact bookkeeping gives {method_a['total_steps']} "
            f"({len(arms)} arms x {method_a['decode_rounds_per_arm']} rounds), the per-layer "
            f"invariant gives {method_b['total_steps']} ({steps_each} per arm x {len(arms)} arms, "
            f"{layers} layers)"
        )
    return {"steps": method_a["total_steps"], "layers": layers,
            "by_artifact": method_a, "by_invariant": method_b}


def _check_metal_coverage(rows: list[dict], steps: int, layers: int) -> dict:
    """Metal layers + layer-0 gather layers per step must equal the layer count."""
    gate = [r for r in rows if r["class"] == DECODE and r["operation"] == "metal.gate_qmv"]
    if not gate:
        return {"metal_layers_per_step": 0, "gather_layers_per_step": 0, "explains_layer_count": False}
    metal = sum(r["total_calls"] for r in gate)
    gather = sum(r["total_calls"] for r in rows if r["class"] == DECODE and r["operation"] == GATHER)
    if metal % steps or gather % (3 * steps):
        raise CorpusError(f"metal ({metal}) or gather ({gather}) calls are not a whole number per step ({steps})")
    metal_layers, gather_layers = metal // steps, gather // (3 * steps)
    if metal_layers + gather_layers != layers:
        raise CorpusError(
            f"{metal_layers} metal layers + {gather_layers} gather layers per step != {layers} layers"
        )
    return {"metal_layers_per_step": metal_layers, "gather_layers_per_step": gather_layers,
            "explains_layer_count": True,
            "note": "the custom Metal MoE path skips layer 0, which runs mlx.gather_qmm "
                    "(gate, up and down) at decode; metal kernels therefore appear layers - 1 times per step"}


def _logit_rounds(arms: dict):
    """Rounds that computed logits, from target_round events; None when events are absent."""
    total = 0
    for arm in arms.values():
        events = arm["stage0"].get("events")
        if events is None:
            return None
        total += sum(1 for e in events if e.get("event") == "target_round" and not e.get("cache_commit_only"))
    return total


def extract(path, repo_root=None) -> dict:
    """Decode corpus, steps, dtype notes and prefill rows from one production artifact."""
    path = Path(path)
    root = Path(repo_root) if repo_root else REPO_ROOT
    artifact = json.loads(path.read_text())
    arms, duplicates = find_arms(artifact)
    table = _aggregate(arms)
    first = {}
    for arm in arms.values():
        for row in arm["stage0"]["shape_trace"]:
            first.setdefault((row["operation"], _freeze(row["operand_shapes"]), _freeze(row["options"])), row)
    rows = []
    for ident, counts in table.items():
        row = _row_from(ident, counts, first[ident])
        row["class"] = classify(row)
        row.update(describe(row))
        rows.append(row)
    rows.sort(key=lambda r: (r["class"], r["operation"], _freeze(r["operand_shapes"])))

    inferred = infer_decode_steps(arms, rows, artifact.get("token_limit"))
    steps, layers = inferred["steps"], inferred["layers"]
    coverage = _check_metal_coverage(rows, steps, layers)
    logit_rounds = _logit_rounds(arms)

    decode, prefill = [], []
    for row in rows:
        if row["class"] == PREFILL:
            prefill.append(row)
            continue
        total = row["total_calls"]
        row["calls_per_step"] = total // steps if total % steps == 0 else total / steps
        if total % steps:
            explained = (logit_rounds is not None and logit_rounds > 0 and total % logit_rounds == 0
                         and row.get("out_width") == max(r.get("out_width", 0) for r in rows if r["operation"] == DENSE))
            if not explained:
                raise CorpusError(
                    f"{row['operation']} {row['operand_shapes']}: {total} calls over {steps} decode "
                    "steps is fractional and no mechanism in the artifact explains it"
                )
            row["fraction_explained"] = (
                f"runs once per round that computes logits: {logit_rounds} of {steps} decode rounds "
                "(the remaining rounds are cache_commit_only and skip lm_head)"
            )
        decode.append(row)

    try:
        relative = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        relative = str(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"artifact": relative, "sha256": sha256_file(path),
                   "unique_arms": len(arms), "duplicate_views_ignored": duplicates,
                   "token_limit": artifact.get("token_limit")},
        "decode_steps": inferred,
        "layer_coverage": coverage,
        "quantization": {"bits": BITS, "group_size": GROUP_SIZE, "mode": "affine",
                         "bytes": "weight out*in/2, scales and biases out*in/32*2 each"},
        "dtypes": DTYPES,
        "decode": decode,
        "prefill": prefill,
    }


def fixture_view(extracted: dict) -> dict:
    """What is committed: the decode corpus, with prefill rows reduced to a summary."""
    view = {key: value for key, value in extracted.items() if key != "prefill"}
    view["prefill_summary"] = {
        "rows": len(extracted["prefill"]),
        "total_calls": sum(r["total_calls"] for r in extracted["prefill"]),
        "note": "prefill rows are identified by class='prefill' in extract() and excluded here",
    }
    return view


def dump(view: dict) -> str:
    return json.dumps(view, sort_keys=True, separators=(",", ":")) + "\n"


def load(path=DEFAULT_FIXTURE) -> dict:
    """The committed fixture, validated for schema version."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise CorpusError(f"{path}: schema_version {data.get('schema_version')} != {SCHEMA_VERSION}")
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact", type=Path)
    # The corpus is a test input under tests/inputs, not a result, so it is not an output option.
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args(argv)
    try:
        extracted = extract(args.artifact)
    except CorpusError as error:
        print(f"kernel_corpus: {error}", file=sys.stderr)
        return 2
    view = fixture_view(extracted)
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    args.fixture.write_text(dump(view))
    steps = view["decode_steps"]
    print(f"wrote {args.fixture}")
    print(f"decode steps {steps['steps']} ({view['source']['unique_arms']} arms, "
          f"{view['source']['duplicate_views_ignored']} duplicate views ignored), layers {steps['layers']}")
    for row in view["decode"]:
        print(f"  {row['calls_per_step']:>7} /step  {row['operation']:22s} "
              f"{row.get('in_width', '')!s:>6} -> {row.get('out_width', '')!s:<7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
