from veystrix.testsuite.api import COMMON_METRICS, TestSpec, bench_module


def command(context, _record_path):
    return [context.python, "-m", bench_module("flashnext", "bench_server_short_paired"),
            "--runtime", context.runtime, "--checkpoint", context.checkpoint,
            "--performance-v2", "--pool-policy", "lru", "cost-aware"]


TEST = TestSpec(
    id="cost-aware", title="Paired cost-aware expert eviction", category="diagnostic",
    explanation="Compares LRU and cost-aware expert eviction in fresh guarded servers with the same fixed expert pool.",
    why="Checks exact greedy tokens and first-logit identity, requires equal mapped pool bytes in both arms, and "
        "measures the effect of cache eviction.",
    script=command, metrics=COMMON_METRICS + ("first prefill logit digest",), promotion=False,
    controls={"sampling": "greedy", "max_tokens": "24", "rounds": "3", "order": "AB, BA, AB", "pool": "2 GB"},
    source="models/flashnext/tests/bench/bench_server_short_paired.py",
)
