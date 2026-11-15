"""Small, synchronous model quanta for independently admitted conversations.

All forwards run on the server's inference thread. Each quantum exits its
runtime scopes and evaluates persistent arrays before another request can run.
"""
from __future__ import annotations

from time import perf_counter

from veystrix.backends.base import GenerationCancelled
from veystrix.sampling import Sampler
from veystrix.text import stream_decode

PREFILL_TOKENS = 128


def generate(backend, max_tokens, *, dispatch, dispatch_decode=None, out=None, on_prefilled=None,
             on_prefill_progress=None, on_decode_token=None, should_cancel=None):
    """Use the backend's singleton arithmetic and advance one cache at a time."""
    cancel = should_cancel or (lambda: False)
    prompt = list(backend.pending)
    if not prompt:
        return "", backend.batch_stats()
    cached = len(backend.tape)
    limit = backend.batch_context()
    remaining = max(0, limit - cached - len(prompt))
    horizon = remaining if max_tokens < 0 else min(remaining, max_tokens)
    if cached + len(prompt) > limit:
        raise ValueError("prompt exceeds the admitted context window")

    def checked(fn):
        if cancel():
            raise GenerationCancelled
        with backend.batch_scope():
            return fn()

    sampler = Sampler(backend.sampling)
    prefill_time = 0.0
    backend.last_prefix = None
    backend.turn_closed = False
    first = None

    def prepare():
        backend.batch_prepare(cached + len(prompt))
        return backend.batch_prefix_plan(prompt, cached, PREFILL_TOKENS)

    offset, anchor = dispatch(lambda: checked(prepare), False)
    restored = offset - cached
    # Absolute boundaries keep the plan stable across a continued conversation.
    while backend.pending:
        width = PREFILL_TOKENS - offset % PREFILL_TOKENS
        if anchor is not None and offset < anchor:
            width = min(width, anchor - offset)
        chunk = backend.pending[:width]
        final = len(chunk) == len(backend.pending)

        def prefill():
            began = perf_counter()
            token = backend.batch_prefill(chunk, offset, final, sampler)
            backend.tape.extend(chunk)
            del backend.pending[:len(chunk)]
            if anchor is not None and len(backend.tape) == anchor:
                backend.batch_save_prefix()
            return token, perf_counter() - began
        first, seconds = dispatch(lambda: checked(prefill), False)
        offset += len(chunk)
        prefill_time += seconds
        if on_prefill_progress is not None:
            on_prefill_progress(offset - cached, len(prompt))
    if on_prefilled is not None:
        on_prefilled()

    partial, pieces = [], []
    next_token = first
    decode_time = 0.0
    produced = 0
    finish = "length"
    for index in range(horizon):
        value = next_token
        if value in backend.stops:
            finish = "stop"
            break

        def step():
            began = perf_counter()
            sampler.observe(value)
            token = backend.batch_decode(value, sampler, index + 1 < horizon)
            backend.tape.append(value)
            return token, perf_counter() - began
        if dispatch_decode is not None and hasattr(backend, "batch_decode_group"):
            sampler.observe(value)
            next_token, seconds = dispatch_decode((backend, value, sampler, index + 1 < horizon), cancel)
            backend.tape.append(value)
        else:
            next_token, seconds = dispatch(lambda: checked(step), True)
        decode_time += seconds
        produced += 1
        piece = stream_decode(backend.tokenizer, partial, value)
        if on_decode_token is not None:
            on_decode_token(value, piece)
        if piece:
            pieces.append(piece)
            if out is not None:
                out(piece)
    return "".join(pieces), backend.batch_stats(
        finish=finish, tokens=produced, seconds=decode_time, prompt_tokens=len(prompt) - restored,
        prefill_seconds=prefill_time,
    )
