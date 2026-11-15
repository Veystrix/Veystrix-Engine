"""Opt-in block-parallel Nemotron Mamba prefill with FP32 boundary states."""
from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models.nemotron_h import NemotronHMamba2Mixer
from mlx_lm.models.ssm import compute_dt, ssm_attn

BLOCK = 64
BLOCKS_PER_GROUP = 4


def parallel_ssm(x, a_log, b, c, residual, dt, dt_bias, state, limits):
    """Run independent local scans in parallel and propagate block boundaries."""
    batch, length, heads, width = x.shape
    groups, state_width = b.shape[-2:]
    repeats = heads // groups
    outputs = []
    for start in range(0, length, BLOCK * BLOCKS_PER_GROUP):
        size = min(BLOCK * BLOCKS_PER_GROUP, length - start)
        block = min(BLOCK, size)
        # The final short block uses the stock path below. Padding dt would
        # decay the state beyond the final real token.
        count = size // block
        end = start + count * block
        local_x = x[:, start:end].reshape(batch * count, block, heads, width)
        local_b = b[:, start:end].reshape(batch * count, block, groups, state_width)
        local_c = c[:, start:end].reshape(batch * count, block, groups, state_width)
        local_dt = dt[:, start:end].reshape(batch * count, block, heads)
        y, end_states = ssm_attn(local_x, a_log, local_b, local_c, residual, local_dt, dt_bias, None, limits)
        decay = mx.exp(mx.cumsum(compute_dt(local_dt, dt_bias, limits) * -mx.exp(a_log), axis=1))
        decay = decay.reshape(batch, count, block, heads)
        end_states = end_states.reshape(batch, count, heads, width, state_width).astype(mx.float32)
        incoming = []
        if state is None:
            state = mx.zeros((batch, heads, width, state_width), dtype=mx.float32)
        for index in range(count):
            incoming.append(state)
            state = end_states[:, index] + decay[:, index, -1, :, None, None] * state
        boundary = mx.stack(incoming, axis=1).reshape(batch * count, 1, groups, repeats, width, state_width)
        projection = (boundary @ local_c.reshape(batch * count, block, groups, 1, state_width, 1)).squeeze(-1)
        correction = projection.reshape(batch * count, block, heads, width)
        correction *= decay.reshape(batch * count, block, heads, 1)
        output = (y.astype(mx.float32) + correction).astype(x.dtype).reshape(batch, count * block, heads, width)
        mx.eval(output, state)
        outputs.append(output)
        tail = start + size
        if end < tail:
            output, state = ssm_attn(x[:, end:tail], a_log, b[:, end:tail], c[:, end:tail], residual,
                                     dt[:, end:tail], dt_bias, state, limits)
            outputs.append(output)
    return (outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)), state


class ParallelMambaMixer(NemotronHMamba2Mixer):
    def _ssm(self, hidden_states, b, c, dt, cache, mask):
        batch, length, _ = hidden_states.shape
        if length < 2 * BLOCK or mask is not None or (cache is not None and cache.lengths is not None):
            return super()._ssm(hidden_states, b, c, dt, cache, mask)
        hidden = hidden_states.reshape(batch, length, self.num_heads, self.head_dim)
        b = b.reshape(batch, length, self.n_groups, self.ssm_state_size)
        c = c.reshape(batch, length, self.n_groups, self.ssm_state_size)
        y, state = parallel_ssm(hidden, self.A_log, b, c, self.D.astype(hidden.dtype), dt, self.dt_bias,
                                cache[1] if cache is not None else None, self.time_step_limit)
        if cache is not None:
            cache[1] = state
        return y.reshape(batch, length, self.intermediate_size)


def enabled():
    return os.environ.get("VEYSTRIX_NEMOTRON_PARALLEL_MAMBA") == "1"


def fingerprint(model):
    if any(isinstance(layer.mixer, ParallelMambaMixer) for layer in model.layers):
        return {"mamba_scan": f"parallel-v1-{BLOCK}x{BLOCKS_PER_GROUP}"}
    return {}
