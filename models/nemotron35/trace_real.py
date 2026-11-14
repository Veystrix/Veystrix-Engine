"""Layer-by-layer divergence tracer: the streamed Nemotron engine against the stock mlx-lm reference.

``python -m models.nemotron35.trace_real CHECKPOINT [--tokens 48] [--decode 3] [--chunk 16] [--pool-gb 0]
[--int8-kv] [--swap-limit-mb 2000] [--out PATH]``

``verify_real`` shows that the final logits of the production engine and of a layer-at-a-time stock reference differ
slightly. This tool finds the first operation that differs. For every phase (prefill, then each forced decode step)
it walks the 52 layers one at a time and, for each layer ``i``, computes three things:

* **reference**: the stock block (loaded from the checkpoint, used, discarded) applied to the reference's own
  propagated input, advancing the reference's own cache.
* **forced**: the ENGINE's real layer ``streamed.model.backbone.layers[i]`` (real expert pool path) applied to the
  SAME input as the reference, with engine caches that have only ever seen reference inputs. If every earlier layer is
  forced, any difference in the output of layer ``i`` is caused by layer ``i`` alone (or by state that this layer
  itself accumulated in earlier phases).
* **free**: the engine's layer ``i`` applied to the engine's own propagated input with a third, independent cache set.
  This reproduces the cumulative drift ``verify`` sees at the logits.

Outputs are compared as float32 numpy arrays over all positions of all chunks of the phase. Equality is by value
(like ``verify_real.compare``): ``+0.0`` and ``-0.0`` count as equal, NaN equals NaN.

For the first forced-divergent layer of each kind in the prefill and the first decode phase, a ``suboperations`` list
walks the block's own operations on the identical input in both implementations (the pre-norm, then the router, the
routed experts given the reference's indices, the weighted sum, the shared expert and the mixer output for MoE; the
projections and the mixer output for Mamba and attention) and names the first one that differs. The sub-operation
calls run on cache copies, so they do not disturb the reference, forced or free caches.

Notes on fidelity: the engine's free-walk production decode is token-major (``model.backbone(...)``); here the free
walk is layer-major with the engine's chunk boundaries, masks and MoE layer bracket (as ``decode_many`` and
``_segment_layer_major`` do), and the last-position logits of every phase are reported against the reference for
both the forced and the free walk. The forced logits apply only the last layer to reference input.

Memory: one stock reference block at a time (deleted and the allocator cache cleared after every layer) plus the
streamed engine. Run it alone, plugged in, with other heavy apps closed. It stops if swap in use passes
``--swap-limit-mb`` (default 2000).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache

from . import streamed_experts
from .attention import AttentionCache
from .config import validate_config
from .inventory import _read_json
from .streamed_model import load_streamed
from .verify_real import Reference, compare, swap_mb

DRILL_PHASES = ('prefill', 'decode1')
CACHED = ('mamba', 'attention')


# -- numeric helpers ---------------------------------------------------------------------------------------------
def _np(array) -> np.ndarray:
    return np.asarray(array.astype(mx.float32))


def _cat(parts):
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """``a`` is the reference, ``b`` the engine side. Both float32 numpy arrays."""
    if a.shape != b.shape:
        return {'bit_identical': False, 'shape_mismatch': [list(a.shape), list(b.shape)], 'elements': int(a.size),
                'max_abs_diff': float('inf'), 'mean_abs_diff': float('inf'), 'differing_elements': int(a.size),
                'max_abs_ref': float(np.max(np.abs(a))) if a.size else 0.0}
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    diff = np.where(same, np.float32(0.0), np.abs(b - a))
    out = {'max_abs_diff': float(diff.max()) if diff.size else 0.0,
           'mean_abs_diff': float(diff.mean(dtype=np.float64)) if diff.size else 0.0,
           'differing_elements': int(a.size - np.count_nonzero(same)), 'elements': int(a.size),
           'bit_identical': bool(same.all()), 'max_abs_ref': float(np.max(np.abs(a))) if a.size else 0.0}
    if a.ndim == 3 and not out['bit_identical']:
        rows = (~same).reshape(a.shape[1], -1).any(axis=1)
        out['first_differing_position'] = int(np.argmax(rows))
        out['differing_positions'] = int(rows.sum())
    return out


def _parts_metrics(reference_parts, other_parts) -> dict:
    return _metrics(_np(_cat(reference_parts)), _np(_cat(other_parts)))


# -- cache helpers -----------------------------------------------------------------------------------------------
def _state_list(cache) -> list:
    """The cache's state arrays (``None`` entries kept); an empty KV cache has none."""
    if cache is None:
        return []
    if not isinstance(cache, ArraysCache) and getattr(cache, 'offset', 0) == 0:
        return []
    return list(cache.state)


def _snapshot(cache):
    """An independent cache object with fresh array handles (mx arrays are immutable, but ``KVCache`` assigns into its
    buffers in place, so the copy must not share the Python objects). Same recipe as ``clone_cache``."""
    if isinstance(cache, ArraysCache):
        fresh = ArraysCache(size=len(cache.state))
        fresh.state = [None if a is None else a[...] for a in cache.state]
        return fresh
    fresh = type(cache)()
    if hasattr(cache, 'cancelled'):
        fresh.cancelled = cache.cancelled
    if getattr(cache, 'offset', 0):
        fresh.state = [a[...] for a in cache.state]
    return fresh


def _state_metrics(reference_cache, other_cache) -> dict:
    """Compare cache state arrays. Skipped (and said so) when the cache classes differ."""
    if type(reference_cache) is not type(other_cache):
        return {'state_comparison': f'skipped: {type(reference_cache).__name__} vs {type(other_cache).__name__}',
                'state_bit_identical': None, 'state_max_abs_diff': None}
    ours, theirs = _state_list(reference_cache), _state_list(other_cache)
    if len(ours) != len(theirs):
        return {'state_comparison': 'compared', 'state_bit_identical': False, 'state_max_abs_diff': float('inf')}
    identical, worst = True, 0.0
    for a, b in zip(ours, theirs):
        if a is None or b is None:
            identical = identical and a is None and b is None
            continue
        m = _metrics(_np(a), _np(b))
        identical = identical and m['bit_identical']
        worst = max(worst, m['max_abs_diff'])
    return {'state_comparison': 'compared', 'state_bit_identical': identical, 'state_max_abs_diff': worst}


def _mask(kind, h, cache):
    if kind == 'attention':
        return create_attention_mask(h, cache)
    if kind == 'mamba':
        return create_ssm_mask(h, cache)
    return None


def _apply(module, kind, h, cache, last):
    """One chunk through one block, evaluated eagerly (cache state too on the last chunk, as the engine does)."""
    out = module(h, mask=_mask(kind, h, cache), cache=cache)
    arrays = [a for a in _state_list(cache) if a is not None] if last else []
    mx.eval(out, *arrays)
    return out


# -- sub-operation drill-down --------------------------------------------------------------------------------------
def _step(name, reference, engine, **extra) -> dict:
    mx.eval(reference, engine)
    out = {'name': name}
    out.update(_metrics(_np(reference), _np(engine)))
    out.update(reference_dtype=str(reference.dtype), engine_dtype=str(engine.dtype),
               reference_shape=list(reference.shape), engine_shape=list(engine.shape))
    out.update(extra)
    return out


def _attempt(steps, name, thunk):
    try:
        value = thunk()
    except Exception as error:               # the stock layout is read from the installed mlx-lm, not assumed
        steps.append({'name': name, 'error': repr(error)})
        return None
    return value


def _drill_moe(stock, engine, xn) -> list:
    steps = []
    s, e = stock.mixer, engine.mixer

    def gate():
        inds_s, scores_s = s.gate(xn)
        inds_e, scores_e = e.gate(xn)
        mx.eval(inds_s, scores_s, inds_e, scores_e)
        a, b = np.asarray(inds_s), np.asarray(inds_e)
        in_order = bool(np.array_equal(a, b))
        as_sets = a.shape == b.shape and bool(np.array_equal(np.sort(a, -1), np.sort(b, -1)))
        step = {'name': 'gate_indices', 'bit_identical': in_order, 'identical_as_sets_per_token': as_sets,
                'tokens_with_different_sets': (int((np.sort(a, -1) != np.sort(b, -1)).any(-1).sum())
                                               if a.shape == b.shape else None),
                'reference_dtype': str(inds_s.dtype), 'engine_dtype': str(inds_e.dtype),
                'reference_shape': list(a.shape), 'engine_shape': list(b.shape)}
        steps.append(step)
        sc = _step('gate_scores', scores_s, scores_e)
        if as_sets:      # weights compared per expert id as well, in case only the order differs
            oa, ob = np.argsort(a, -1, kind='stable'), np.argsort(b, -1, kind='stable')
            aligned = _metrics(np.take_along_axis(_np(scores_s), oa, -1), np.take_along_axis(_np(scores_e), ob, -1))
            sc['aligned_by_expert_max_abs_diff'] = aligned['max_abs_diff']
            sc['aligned_by_expert_bit_identical'] = aligned['bit_identical']
        steps.append(sc)
        return inds_s, scores_s

    routed = _attempt(steps, 'gate', gate)
    if routed is None:
        return steps
    inds, scores = routed          # the REFERENCE's routing feeds both sides from here on
    latent = getattr(s, 'moe_latent_size', None) is not None
    x = xn
    if latent:
        def fc1():
            nonlocal x
            ref_x, eng_x = s.fc1_latent_proj(xn), e.fc1_latent_proj(xn)
            steps.append(_step('fc1_latent_proj', ref_x, eng_x))
            x = ref_x
        _attempt(steps, 'fc1_latent_proj', fc1)

    ys = {}

    def routed_experts():
        ys['s'], ys['e'] = s.switch_mlp(x, inds), e.switch_mlp(x, inds)
        steps.append(_step('routed_experts', ys['s'], ys['e'], reference_class=type(s.switch_mlp).__name__,
                           engine_class=type(e.switch_mlp).__name__))
    _attempt(steps, 'routed_experts', routed_experts)

    def weigh(y):
        return (y * scores[..., None]).sum(axis=-2).astype(y.dtype)     # NemotronHMoE.__call__, same inline formula
    sums = {}
    if ys:
        def weighted():
            sums['s'], sums['e'] = weigh(ys['s']), weigh(ys['e'])
            steps.append(_step('weighted_sum', sums['s'], sums['e']))
        _attempt(steps, 'weighted_sum', weighted)
    if sums and latent:
        def fc2():
            steps.append(_step('fc2_latent_proj', s.fc2_latent_proj(sums['s']), e.fc2_latent_proj(sums['e'])))
        _attempt(steps, 'fc2_latent_proj', fc2)
    if hasattr(s, 'shared_experts') and hasattr(e, 'shared_experts'):
        _attempt(steps, 'shared_experts',
                 lambda: steps.append(_step('shared_experts', s.shared_experts(xn), e.shared_experts(xn))))
    _attempt(steps, 'mixer_output', lambda: steps.append(
        _step('mixer_output', s(xn), e(xn), reference_class=type(s).__name__, engine_class=type(e).__name__)))
    return steps


def _drill_cached(kind, stock, engine, xn, ref_cache, forced_cache) -> list:
    steps = []
    s, e = stock.mixer, engine.mixer
    before = _state_metrics(ref_cache, forced_cache)
    steps.append({'name': 'cache_state_before', 'precondition': True, **before,
                  'bit_identical': before['state_bit_identical'] is not False,
                  'reference_class': type(ref_cache).__name__, 'engine_class': type(forced_cache).__name__})
    projections = ('in_proj',) if kind == 'mamba' else ('q_proj', 'k_proj', 'v_proj')
    for name in projections:
        if hasattr(s, name) and hasattr(e, name):
            _attempt(steps, name, lambda name=name: steps.append(_step(name, getattr(s, name)(xn),
                                                                         getattr(e, name)(xn))))

    def mixer():
        out_s = s(xn, mask=_mask(kind, xn, ref_cache), cache=ref_cache)
        out_e = e(xn, mask=_mask(kind, xn, forced_cache), cache=forced_cache)
        steps.append(_step('mixer_output', out_s, out_e, reference_class=type(s).__name__,
                           engine_class=type(e).__name__))
        mx.eval(*[a for a in _state_list(ref_cache) + _state_list(forced_cache) if a is not None])
        after = _state_metrics(ref_cache, forced_cache)
        steps.append({'name': 'cache_state_after_mixer', **after,
                      'bit_identical': after['state_bit_identical'] is not False,
                      'reference_class': type(ref_cache).__name__, 'engine_class': type(forced_cache).__name__})
    _attempt(steps, 'mixer_output', mixer)
    return steps


def _drill(kind, stock, engine, x, ref_cache, forced_cache) -> list:
    """Sub-operations of one block on the identical input ``x`` (pre-layer hidden state of one chunk). The caches are
    private snapshots taken before the layer advanced the real ones."""
    steps = []
    xn = _attempt(steps, 'norm', lambda: stock.norm(x))
    if xn is None:
        return steps
    _attempt(steps, 'norm', lambda: steps.append(_step('norm', xn, engine.norm(x))))
    if kind == 'moe':
        steps.extend(_drill_moe(stock, engine, xn))
    elif kind in CACHED:
        steps.extend(_drill_cached(kind, stock, engine, xn, ref_cache, forced_cache))
    else:
        _attempt(steps, 'mixer_output', lambda: steps.append(
            _step('mixer_output', stock.mixer(xn), engine.mixer(xn), reference_class=type(stock.mixer).__name__,
                  engine_class=type(engine.mixer).__name__)))
    return steps


def _first_differing(steps):
    return next((s['name'] for s in steps if not s.get('precondition') and s.get('bit_identical') is False), None)


# -- the walk --------------------------------------------------------------------------------------------------------
def _walk_phase(ref, core, caches, phase, chunks):
    backbone, pool = core.model.backbone, core.streamed.pool
    types = ref.types
    drilled = set()
    count = len(chunks)
    embed = ref.load_embedding()
    ref_hidden = [embed(mx.array([c], dtype=mx.int32)) for c in chunks]
    del embed
    free_hidden = [backbone.embeddings(mx.array([c], dtype=mx.int32)) for c in chunks]
    mx.eval(*ref_hidden, *free_hidden)
    mx.clear_cache()
    report = {'phase': phase, 'tokens': sum(len(c) for c in chunks), 'chunk_sizes': [len(c) for c in chunks],
              'embedding': _parts_metrics(ref_hidden, free_hidden), 'layers': []}
    forced_last = ref_hidden
    counter = 0
    for i, kind in enumerate(types):
        block = ref.load_block(i)
        layer = backbone.layers[i]
        cached = kind in CACHED
        ref_cache = ref.caches.get(i)
        forced_cache = caches['forced'][counter] if cached else None
        free_cache = caches['free'][counter] if cached else None
        counter += cached
        ref_in = ref_hidden
        want = phase in DRILL_PHASES and kind not in drilled
        snaps_ref, snaps_forced = [], []

        ref_out = []
        for j, h in enumerate(ref_in):
            if want and cached:
                snaps_ref.append(_snapshot(ref_cache))
            ref_out.append(_apply(block, kind, h, ref_cache, j == count - 1))

        forced_out, free_out = [], []
        if kind == 'moe':
            pool.begin_layer(i)
        try:
            for j, h in enumerate(ref_in):
                if want and cached:
                    snaps_forced.append(_snapshot(forced_cache))
                forced_out.append(_apply(layer, kind, h, forced_cache, j == count - 1))
            for j, h in enumerate(free_hidden):
                free_out.append(_apply(layer, kind, h, free_cache, j == count - 1))

            entry = {'layer': i, 'kind': kind, 'reference_dtype': str(ref_out[0].dtype),
                     'forced_dtype': str(forced_out[0].dtype), 'free_dtype': str(free_out[0].dtype)}
            reference = _np(_cat(ref_out))
            entry['forced'] = _metrics(reference, _np(_cat(forced_out)))
            entry['free'] = _metrics(reference, _np(_cat(free_out)))
            del reference
            if cached:
                forced_state, free_state = _state_metrics(ref_cache, forced_cache), _state_metrics(ref_cache, free_cache)
                entry['state_comparison'] = forced_state['state_comparison']
                entry['state_bit_identical'] = forced_state['state_bit_identical']
                entry['state_max_abs_diff'] = forced_state['state_max_abs_diff']
                entry['free_state_bit_identical'] = free_state['state_bit_identical']
                entry['free_state_max_abs_diff'] = free_state['state_max_abs_diff']
            if want and not entry['forced']['bit_identical']:
                drilled.add(kind)
                first = next(j for j in range(count)
                             if not _metrics(_np(ref_out[j]), _np(forced_out[j]))['bit_identical'])
                try:
                    steps = _drill(kind, block, layer, ref_in[first],
                                   snaps_ref[first] if cached else None, snaps_forced[first] if cached else None)
                except Exception as error:
                    steps = [{'error': repr(error)}]
                entry['suboperation_chunk'] = first
                entry['suboperations'] = steps
                entry['first_differing_suboperation'] = _first_differing(steps)
        finally:
            if kind == 'moe':
                pool.end_layer()
        report['layers'].append(entry)
        ref_hidden, free_hidden, forced_last = ref_out, free_out, forced_out
        del block, snaps_ref, snaps_forced, ref_in
        mx.clear_cache()
        ref._guard()

    norm, head = ref.load_head()
    ref_logits = head(norm(ref_hidden[-1][:, -1:, :]))[:, -1, :]
    del norm, head
    mx.eval(ref_logits)

    def engine_logits(hidden):
        logits = core.model.lm_head(backbone.norm_f(hidden[-1][:, -1:, :]))[:, -1, :]
        mx.eval(logits)
        return logits
    free_logits, forced_logits = engine_logits(free_hidden), engine_logits(forced_last)
    report['logits'] = {'free_vs_reference': compare(free_logits, ref_logits),
                        'forced_vs_reference': compare(forced_logits, ref_logits)}
    ref_np = _np(ref_logits).reshape(-1)
    report['reference_greedy_token'] = int(np.argsort(-ref_np)[0])
    mx.clear_cache()
    ref._guard()
    return report


def _summarise(report):
    phases = report['phases']
    by_kind, free_max, first, first_state = {}, {}, None, None
    for p in phases:
        kinds, free_max[p['phase']] = {}, []
        for entry in p['layers']:
            slot = kinds.setdefault(entry['kind'], {'divergent': 0, 'total': 0, 'divergent_layers': []})
            slot['total'] += 1
            free_max[p['phase']].append(entry['free']['max_abs_diff'])
            if not entry['forced']['bit_identical']:
                slot['divergent'] += 1
                slot['divergent_layers'].append(entry['layer'])
                if first is None:
                    first = {'phase': p['phase'], 'layer': entry['layer'], 'kind': entry['kind']}
            if entry.get('state_bit_identical') is False and first_state is None:
                first_state = {'phase': p['phase'], 'layer': entry['layer'], 'kind': entry['kind']}
        by_kind[p['phase']] = kinds
    report['first_forced_divergence'] = first
    report['first_forced_state_divergence'] = first_state
    report['forced_divergent_layers_by_kind'] = by_kind
    report['free_max_abs_by_layer'] = free_max
    report['all_forced_bit_identical'] = first is None
    report['all_forced_state_bit_identical'] = first_state is None


def trace(root, *, tokens=48, decode=3, chunk=16, pool_gb=0.0, int8_kv=False, swap_limit_mb=2000.0,
          config_check=None) -> dict:
    root = Path(root).expanduser()
    config = _read_json(root / 'config.json', 32 * 1024 * 1024, 'config.json')
    (config_check or validate_config)(config)
    rng = np.random.default_rng(1234)               # the prompt of verify_real.verify
    low = 100 if config['vocab_size'] > 1000 else 3
    prompt = [int(t) for t in rng.integers(low, min(config['vocab_size'], 30000), size=tokens)]
    report = {'checkpoint': str(root), 'prompt_tokens': tokens, 'decode_steps': decode, 'chunk_tokens': chunk,
              'pool_gb': pool_gb, 'int8_kv': int8_kv, 'gather': streamed_experts.GATHER, 'phases': []}
    began = time.time()
    from .backend_core import EngineCore
    streamed = load_streamed(root, pool_bytes=int(pool_gb * 2 ** 30), verify=False, config_check=config_check)
    pool = streamed.pool
    speculate = pool.speculate
    pool.speculate = False
    try:
        core = EngineCore(streamed, chunk_tokens=chunk, int8_kv=int8_kv)
        ref = Reference(root, config, swap_limit_mb)
        caches = {'forced': core._make_cache(), 'free': core._make_cache()}
        for entry in caches['forced'] + caches['free']:
            if isinstance(entry, AttentionCache):
                entry.cancelled = lambda: pool.cancel_check()
        chunks = [prompt[i:i + chunk] for i in range(0, len(prompt), chunk)]
        for phase in ['prefill'] + [f'decode{k + 1}' for k in range(decode)]:
            phase_report = _walk_phase(ref, core, caches, phase, chunks)
            phase_report['swap_mb'] = swap_mb()
            report['phases'].append(phase_report)
            chunks = [[phase_report['reference_greedy_token']]]
    finally:
        pool.end_layer()
        pool.speculate = speculate
        streamed.close()
    report['seconds'] = round(time.time() - began, 1)
    _summarise(report)
    report['all_free_greedy_agree'] = all(p['logits']['free_vs_reference']['greedy_agree'] for p in report['phases'])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('checkpoint')
    parser.add_argument('--tokens', type=int, default=48)
    parser.add_argument('--decode', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=16, help='prefill chunk tokens of the engine under test')
    parser.add_argument('--pool-gb', type=float, default=0.0)
    parser.add_argument('--int8-kv', action='store_true', help='engine attention uses the INT8 block cache')
    parser.add_argument('--swap-limit-mb', type=float, default=2000.0)
    parser.add_argument('--out', default=None, help='also write the JSON report to this path')
    ns = parser.parse_args(argv)
    report = trace(ns.checkpoint, tokens=ns.tokens, decode=ns.decode, chunk=ns.chunk, pool_gb=ns.pool_gb,
                   int8_kv=ns.int8_kv, swap_limit_mb=ns.swap_limit_mb)
    text = json.dumps(report, indent=2)
    print(text)
    if ns.out:
        Path(ns.out).expanduser().write_text(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
