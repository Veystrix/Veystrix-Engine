"""Optional full-row logit fingerprint for greedy checkpoint comparisons."""
from __future__ import annotations

import hashlib
import os

FLAG = "VEYSTRIX_BENCH_LOGIT_DIGEST"


def reference_sampler(sampler):
    """Capture the first logit row delivered to the prompt sampler."""
    if os.environ.get(FLAG) != "1" or not sampler.settings.greedy:
        return sampler

    recorded = False

    def sample(logits):
        nonlocal recorded
        if not recorded:
            import mlx.core as mx
            import numpy as np
            from models.flashnext import flight_recorder

            row = np.asarray(logits.reshape(-1).astype(mx.float32), dtype="<f4")
            flight_recorder.record("prefill_logit_reference", dtype=str(logits.dtype),
                                   vocab_size=row.size, sha256=hashlib.sha256(row.tobytes()).hexdigest())
            recorded = True
        return sampler(logits)

    return sample
