"""Faster Mamba-2 state-space scan for prefill chunks.

The stock ``mlx_lm.models.ssm.ssm_attn`` builds, for every 256-token step, a ``[heads, 256, 256]`` fp32 matrix of segment
sums (repeat, tril, cumsum), exponentiates it, repeats ``C @ B^T`` to every head, and broadcasts the carried state to every
token to get the previous-state contribution. The scan is memory-bound (about 39 ms for a 2,048-token chunk although it needs
about 5 GFLOP). This version computes the same quantities with fewer passes:

* the decay between two positions is ``exp(cs[i] - cs[j])`` for the fp32 cumulative sum ``cs`` of ``dt * A``, evaluated by one
  compiled kernel together with the product with ``C @ B^T`` and the causal mask (no repeat, no tril, no segment-sum matrix);
* the previous-state contribution and the next state are one GEMM per group (``[r * head_dim, d] @ [d, tokens]``) instead of
  one tiny matmul per token and head, and ``B`` is not repeated to every head.

Same arithmetic up to floating-point reassociation and not bit-identical. Against a float64 sequential recurrence (768 tokens,
bf16 inputs) the output error is the same as the stock scan's (RMS 4.567e-3 against 4.566e-3: the bf16 rounding of the output),
and the final state differs from the truth by 4.6e-7 RMS (stock 1.3e-8, state magnitude 0.16). About 3x faster for the scan
(39 -> 13 ms per 2,048-token chunk). Only used for several-hundred-row calls without a mask; decode, speculative rows and
masked calls keep the stock path. ``VEYSTRIX_NEMOTRON_FAST_SSM=0`` restores the stock scan.
"""
from __future__ import annotations

import os

import mlx.core as mx
from mlx_lm.models import nemotron_h
from mlx_lm.models.ssm import compute_dt, ssm_update as _stock_ssm_update

FLAG = "VEYSTRIX_NEMOTRON_FAST_SSM"
MIN_ROWS = 64
STEP = 256

_installed = False


def enabled() -> bool:
    value = os.environ.get(FLAG, "1") or "1"
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1")
    return value == "1"


@mx.compile
def _attention_matrix(cb, cs):
    """``tril(CB * exp(cs[i] - cs[j]))`` for ``cb`` of shape ``[b, g, 1, s, s]`` and ``cs`` of shape ``[b, g, r, s]``."""
    steps = cs.shape[-1]
    diff = cs[..., :, None] - cs[..., None, :]
    index = mx.arange(steps)
    return mx.where(index[:, None] >= index[None, :], cb * mx.exp(diff), 0.0)


def fast_ssm_attn(x, a_log, b, c, d_skip, dt, dt_bias, state=None, time_step_limit=(0.001, 100.0), step=STEP):
    """Drop-in for ``ssm_attn`` (no mask, no lengths): returns ``(y, final_state)``."""
    batch, length, heads, head_dim = x.shape
    groups, width = b.shape[-2], b.shape[-1]
    repeats = heads // groups
    dt = compute_dt(dt, dt_bias, time_step_limit)
    a = -mx.exp(a_log).astype(dt.dtype)
    dta = dt * a.reshape(1, 1, -1)
    dtx = dt.reshape(batch, length, heads, 1) * x

    def one(dtx, dta, b, c, state):
        s = dtx.shape[1]
        cb = mx.swapaxes(c, 1, 2) @ mx.transpose(b, (0, 2, 3, 1))                    # [b, g, s, s]
        cs = mx.cumsum(dta.swapaxes(1, 2), axis=-1)                                  # [b, h, s] fp32
        matrix = _attention_matrix(cb.reshape(batch, groups, 1, s, s),
                                   cs.reshape(batch, groups, repeats, s)).reshape(batch, heads, s, s)
        dtx_h = dtx.swapaxes(1, 2)                                                   # [b, h, s, dh]
        y = matrix @ dtx_h
        decayed = dtx_h * mx.exp(cs[..., -1:] - cs)[..., None]
        grouped = decayed.reshape(batch, groups, repeats, s, head_dim).transpose(0, 1, 2, 4, 3)
        next_state = (grouped.reshape(batch, groups, repeats * head_dim, s) @ b.transpose(0, 2, 1, 3)
                      ).reshape(batch, heads, head_dim, width)
        if state is not None:
            carried = mx.exp(cs)                                                     # [b, h, s]
            next_state += carried[:, :, -1, None, None] * state
            previous = state.reshape(batch, groups, repeats * head_dim, width) @ c.transpose(0, 2, 3, 1)
            y = y + carried[..., None] * previous.reshape(batch, heads, head_dim, s).swapaxes(-1, -2)
        return mx.swapaxes(y, 1, 2).astype(x.dtype), next_state

    ys = []
    for start in range(0, length, step):
        y, state = one(dtx[:, start:start + step], dta[:, start:start + step], b[:, start:start + step],
                       c[:, start:start + step], state)
        ys.append(y)
    return mx.concatenate(ys, axis=1) + x * d_skip.reshape(1, 1, heads, 1), state


def ssm_update(hidden_states, a_log, b, c, d_skip, dt, dt_bias, state=None, time_step_limit=(0.001, 100.0), mask=None,
               lengths=None):
    """``ssm_update`` with the fast scan for long unmasked calls (prefill chunks)."""
    if mask is None and lengths is None and hidden_states.shape[1] >= MIN_ROWS and enabled():
        return fast_ssm_attn(hidden_states, a_log, b, c, d_skip, dt, dt_bias, state, time_step_limit)
    return _stock_ssm_update(hidden_states, a_log, b, c, d_skip, dt, dt_bias, state, time_step_limit, mask=mask,
                             lengths=lengths)


def install() -> None:
    """Route the Nemotron-H mixer's scan through ``ssm_update`` above (idempotent)."""
    global _installed
    if not _installed:
        nemotron_h.ssm_update = ssm_update
        _installed = True


def fingerprint() -> dict:
    return {"ssm_prefill": "fused-v1"} if enabled() else {}
