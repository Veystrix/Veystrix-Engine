from veystrix.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(context, _record_path):
    return [context.python, "-m", bench_module("flashnext", "bench_server_short_paired"),
            "--runtime", context.runtime, "--checkpoint", context.checkpoint,
            "--performance-v2", "--least-stale"]


TEST = TestSpec(
    id="least-stale", title="Paired Least-Stale expert eviction", category="diagnostic",
    explanation="Compares LRU and Least-Stale in fresh guarded servers with a fixed expert pool.",
    why="Checks exact greedy tokens and first-logit identity while measuring the effect of cache eviction.",
    script=command, metrics=COMMON_METRICS + ("first prefill logit digest",), promotion=False,
    controls={"sampling": "greedy", "max_tokens": "24", "rounds": "3", "order": "AB, BA, AB"},
    source="models/flashnext/tests/bench/bench_server_short_paired.py",
)
