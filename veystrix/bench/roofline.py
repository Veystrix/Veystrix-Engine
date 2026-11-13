"""Versioned Stage 0 metrics, independent of MLX and model weights."""
from __future__ import annotations

import statistics

SCHEMA_VERSION = 1


def _rate(work, ns):
    return work * 1e9 / ns if ns is not None and ns > 0 else None


def _delta(before, after):
    return after - before if before is not None and after is not None else None


def arm_metrics(events, computed_tokens, read_bytes, io_wait_ns):
    marks = {row["event"]: row["ns"] for row in events}
    emitted = [row["ns"] for row in events if row["event"] == "token_emitted"]
    rounds = [row for row in events if row["event"] == "target_round"]
    verify = [row for row in rounds if row.get("verification")]
    drafts = [row["duration_ns"] for row in events if row["event"] == "draft_round"]
    engine_ns = _delta(marks.get("decode_start"), marks.get("decode_commit"))
    prefill_ns = _delta(marks.get("prefill_start"), marks.get("first_logits_ready"))
    admission_ns = _delta(marks.get("engine_admission_start"), marks.get("engine_admission_end"))
    callback_ns = _delta(marks.get("prefill_callback"), marks.get("prefill_callback_end"))
    ttft_ns = _delta(marks.get("admission"), emitted[0] if emitted else None)
    if computed_tokens is None:
        prefill_rate = None
    else:
        prefill_rate = _rate(computed_tokens, prefill_ns)
    non_io_ns = (engine_ns - io_wait_ns if engine_ns is not None and
                 io_wait_ns is not None and 0 <= io_wait_ns <= engine_ns else None)
    effective_gb_s = (read_bytes / io_wait_ns if read_bytes is not None and
                      io_wait_ns and io_wait_ns > 0 else None)
    count = len(emitted)
    read_mb = read_bytes / count / 1e6 if count and read_bytes is not None else None
    return {
        "ttft_ns": ttft_ns,
        "ttft_less_benchmark_callback_ns": ttft_ns - callback_ns if ttft_ns is not None and callback_ns is not None else None,
        "engine_admission_ns": admission_ns,
        "first_sample_ns": _delta(marks.get("first_logits_ready"), marks.get("first_sample_ready")),
        "tokenization_ns": _delta(marks.get("tokenization_start"), marks.get("tokenization_end")),
        "prefix_lookup_restore_ns": _delta(marks.get("prefix_lookup_start"), marks.get("prefix_lookup_end")),
        "prefill_ns": prefill_ns,
        "computed_prefill_tokens": computed_tokens,
        "computed_prefill_tok_s": prefill_rate,
        "stream_decode_tok_s": _rate(count - 1, emitted[-1] - emitted[0]) if count > 1 else None,
        "engine_decode_tok_s": _rate(count, engine_ns),
        "engine_decode_ns": engine_ns,
        "target_rounds": len(rounds),
        "median_draft_round_ns": statistics.median(drafts) if drafts else None,
        "target_rounds_per_token": len(rounds) / count if count else None,
        "draft_rounds": len(drafts),
        "median_target_round_ns": statistics.median(r["duration_ns"] for r in rounds) if rounds else None,
        "verification_rounds": len(verify),
        "accepted_tokens_per_verification": (
            sum(r["accepted"] for r in verify) / len(verify)
            if verify and all("accepted" in r for r in verify) else None
        ),
        "physical_read_bytes": read_bytes,
        "physical_mb_per_token": read_mb,
        "observed_io_wait_ns": io_wait_ns,
        "effective_storage_gb_s": effective_gb_s,
        "non_io_ms_per_token": non_io_ns / count / 1e6 if count and non_io_ns is not None else None,
        "upper_bound_io_only_tok_s": effective_gb_s * 1000 / read_mb if effective_gb_s and read_mb else None,
        "upper_bound_compute_only_tok_s": _rate(count, non_io_ns),
        "upper_bound_at_3gb_s_tok_s": (
            _rate(count, non_io_ns + read_bytes / 3)
            if non_io_ns is not None and read_bytes is not None else None
        ),
    }


def resolution(values):
    """Observed run-to-run resolution, available after five retained runs."""
    if len(values) < 5:
        return None
    p10, *_, p90 = statistics.quantiles(values, n=10, method="inclusive")
    return {"count": len(values), "p10": p10, "median": statistics.median(values),
            "p90": p90, "resolution_half_width": (p90 - p10) / 2}


def paired_overhead(baseline, observed):
    if len(baseline) != len(observed) or not baseline:
        raise ValueError("telemetry overhead requires complete matched pairs")
    if any(left["token_sha256"] != right["token_sha256"]
           for left, right in zip(baseline, observed)):
        raise ValueError("telemetry arms produced different token digests")
    band = resolution([
        (left["gen_rate"] / right["gen_rate"] - 1) * 100
        for left, right in zip(baseline, observed)
    ])
    return {"generation_time_overhead_percent": band,
            "under_one_percent": None if band is None else band["p90"] < 1.0}


def report(evidence, collected, drop, *, paired=False):
    conditions = {}
    for condition, rows in collected.items():
        arms = [
            {"round": index + 1, "token_sha256": row["token_sha256"],
             "sampled_available_floor_bytes": row["stage0"]["memory"]["sampled_available_floor_bytes"],
             **row["stage0"]["metrics"]}
            for index, row in enumerate(rows) if index >= drop and "stage0" in row
        ]
        if arms:
            metrics = ("ttft_ns", "computed_prefill_tok_s", "stream_decode_tok_s", "engine_decode_tok_s")
            conditions[condition] = {
                "arms": arms,
                "sampled_memory_floor_pass": (
                    all(arm["sampled_available_floor_bytes"] >= 3 * 2**30 for arm in arms)
                    if all(arm["sampled_available_floor_bytes"] is not None for arm in arms) else None
                ),
                "resolution": {
                    key: resolution([arm[key] for arm in arms if arm[key] is not None])
                    for key in metrics
                },
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {key: evidence[key] for key in (
            "stage0_provenance", "checkpoint_identity", "source_fingerprints",
            "prompt_sha256", "sampling", "token_limit", "protocol",
        )},
        "conditions": conditions,
        "paired_overhead": paired_overhead(collected["baseline"][drop:], collected["telemetry"][drop:])
        if paired else None,
    }
