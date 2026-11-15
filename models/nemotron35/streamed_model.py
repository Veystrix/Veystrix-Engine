"""Build the Nemotron-H model with SSD-streamed routed experts.

The dense core (embeddings, Mamba, attention, router, shared experts, lm_head;
about 1.17 GiB for the 4-bit checkpoint) is read from the shards and held
resident. The stacked routed experts are never loaded: each ``switch_mlp`` is
replaced with ``StreamedSwitchMLP``. Parameter names are those of the installed
mlx-lm ``nemotron_h.Model``, so the same weights load unchanged into the
in-memory reference (used by the synthetic equivalence tests).

``load_streamed`` reads real weights. It is only called after the explicit
authorization/admission gates in the backend; tests exercise it on tiny
synthetic checkpoints written in the identical layout.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import nemotron_h

from .config import validate_config
from .inventory import CheckpointError, verify_checkpoint, verify_inventory, expected_tensors, _read_json
from .safetensors_index import index_checkpoint
from .streamed_experts import ExpertPool, NextGate, StackedExpertReader, StreamedSwitchMLP

_NP = {'U32': np.uint32, 'F32': np.float32, 'F16': np.float16, 'BF16': np.uint16}


def _read_tensor(directory: Path, desc: dict):
    fd = os.open(directory / desc['shard'], os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        if desc['offset'] + desc['length'] > os.fstat(fd).st_size:
            raise CheckpointError('shard truncated')
        data = os.pread(fd, desc['length'], desc['offset'])
    finally:
        os.close(fd)
    if len(data) != desc['length']:
        raise CheckpointError('short tensor read')
    array = mx.array(np.frombuffer(data, dtype=_NP[desc['dtype']]).reshape(desc['shape']))
    return array.view(mx.bfloat16) if desc['dtype'] == 'BF16' else array


@contextlib.contextmanager
def _stub_switch_mlp(factory):
    """Make mlx-lm construct streamed experts instead of 128 resident ones."""
    from .attention import StreamedAttention
    from . import mamba_scan
    original, attention = nemotron_h.SwitchMLP, nemotron_h.NemotronHAttention
    mamba = nemotron_h.NemotronHMamba2Mixer
    nemotron_h.SwitchMLP = factory
    nemotron_h.NemotronHAttention = StreamedAttention
    if mamba_scan.enabled():
        nemotron_h.NemotronHMamba2Mixer = mamba_scan.ParallelMambaMixer
    try:
        yield
    finally:
        nemotron_h.SwitchMLP = original
        nemotron_h.NemotronHAttention = attention
        nemotron_h.NemotronHMamba2Mixer = mamba


class StreamedModel:
    """The model plus the objects that own its resources."""

    def __init__(self, model, pool, reader, config, dense_bytes):
        self.model, self.pool, self.reader = model, pool, reader
        self.config, self.dense_bytes = config, dense_bytes

    def close(self) -> None:
        self.pool.clear()
        self.pool.close()
        self.reader.close()


def load_streamed(directory, *, pool_bytes: int, config_check=None, verify: bool = True,
                  prefetch: bool = True, staged_bytes: int = 384 * 2**20) -> StreamedModel:
    """Load the dense core resident and stream the routed experts from SSD.

    ``verify`` runs the full structural verification (headers, inventory,
    extents) first. ``config_check`` defaults to the strict public-config check;
    tests of tiny synthetic checkpoints pass a permissive one explicitly.
    """
    root = Path(directory).expanduser()
    check = config_check or validate_config
    if verify:
        verify_checkpoint(root, config_check=check)
    config = _read_json(root / 'config.json', 32 * 1024 * 1024, 'config.json')
    check(config)
    index = index_checkpoint(root)
    verify_inventory(index, config)
    group = config['quantization']['group_size']
    args = nemotron_h.ModelArgs.from_dict(config)
    moe_layers = [i for i, kind in enumerate(config['layers_block_type']) if kind == 'moe']
    reader = StackedExpertReader(root, index, layers=moe_layers, expert_count=config['n_routed_experts'],
                                 hidden=config['hidden_size'], moe_dim=config['moe_intermediate_size'],
                                 group_size=group)
    pool = ExpertPool(reader, pool_bytes, staged_bytes=staged_bytes)
    order = iter(moe_layers)
    with _stub_switch_mlp(lambda *a, **k: StreamedSwitchMLP(pool, next(order), group)):
        model = nemotron_h.Model(args)
    dense = {name: _read_tensor(root, desc) for name, desc in index.items() if '.switch_mlp.' not in name}
    nn.quantize(model, group_size=group, bits=4, mode='affine',
                class_predicate=lambda path, module: hasattr(module, 'to_quantized') and f'{path}.scales' in dense)
    model.load_weights(list(dense.items()), strict=True)
    mx.eval(model.parameters())
    if prefetch:   # each MoE layer predicts the next one's experts from its own input (exactness is unaffected)
        for position, here in enumerate(moe_layers[:-1]):
            there = moe_layers[position + 1]
            gate = model.backbone.layers[there].mixer.gate
            switch = model.backbone.layers[here].mixer.switch_mlp
            switch.next_gate = NextGate(there, gate.weight, gate.e_score_correction_bias,
                                        config['num_experts_per_tok'])
            # MOE-04/05: the layer after next is prefetched from the previous pass's routes when opted in.
            switch.lookahead_layer = (moe_layers[position + 2]
                                      if position + 2 < len(moe_layers) else None)
    if os.environ.get("VEYSTRIX_NEMOTRON_OVERLAP_SHARED", "1") != "0":
        _overlap_shared_experts(model, moe_layers)
    from . import fast_ssm
    fast_ssm.install()                  # prefill-sized scans use the fused kernel unless VEYSTRIX_NEMOTRON_FAST_SSM=0
    from . import cache_routing
    cache_routing.install(model)         # off unless VEYSTRIX_NEMOTRON_CACHE_ROUTING is observe or swap
    try:
        _attach_pack(reader, root, index)
    except BaseException:
        pool.close()
        reader.close()
        raise
    dense_bytes = sum(v.nbytes for v in dense.values())
    return StreamedModel(model, pool, reader, config, dense_bytes)


class OverlappedSharedMoE(nemotron_h.NemotronHMoE):
    """The stock MoE block, except that the shared expert's result may already have been started by the routed-expert
    layer (``StreamedSwitchMLP``) while the SSD was being read. The values are the same operations on the same input."""

    def __call__(self, x):
        residuals = x
        inds, scores = self.gate(x)
        if self.moe_latent_size is not None:
            x = self.fc1_latent_proj(x)
        self.switch_mlp.shared_result = None
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.moe_latent_size is not None:
            y = self.fc2_latent_proj(y)
        if self.config.n_shared_experts is not None:
            started, self.switch_mlp.shared_result = self.switch_mlp.shared_result, None
            y = y + (started[1] if started is not None and started[0] is residuals else self.shared_experts(residuals))
        return y


def _overlap_shared_experts(model, moe_layers) -> None:
    """Let each routed-expert layer start its block's shared expert before it waits for the routed experts' reads.
    Not for latent-projection models (the layer then sees a projected input, not the block's input)."""
    for layer in moe_layers:
        mixer = model.backbone.layers[layer].mixer
        if getattr(mixer, "moe_latent_size", None) is not None or not hasattr(mixer, "shared_experts"):
            continue
        mixer.__class__ = OverlappedSharedMoE
        mixer.switch_mlp.shared_fn = lambda x, mixer=mixer: mixer.shared_experts(x)


def _attach_pack(reader, root, index) -> None:
    """Open the expert pack, if any, and decide how it is used.

    ``VEYSTRIX_NEMOTRON_EXPERT_MAP``: ``0`` (default) never maps; ``auto`` maps the pack found at its default
    path for this checkpoint and otherwise streams from the shards; ``1`` requires a mapped pack.
    ``VEYSTRIX_NEMOTRON_EXPERT_PACK`` names a pack explicitly. Without ``…_MAP=1`` an explicit pack keeps its
    earlier meaning: positioned reads from the pack."""
    from models.shared.expert_pack import ExpertPack, default_path

    from .streamed_experts import MAP_FLAG

    mode = (os.environ.get(MAP_FLAG) or "0").strip().lower()
    if mode not in ("auto", "0", "1"):
        raise ValueError(f"{MAP_FLAG} must be auto, 0 or 1")
    explicit = os.environ.get("VEYSTRIX_NEMOTRON_EXPERT_PACK")
    if explicit and mode != "1":
        reader.pack = ExpertPack(Path(explicit).expanduser(), root, index, "nemotron35")
        return
    if mode == "0":
        return
    path = Path(explicit).expanduser() if explicit else default_path(root, index, "nemotron35")
    if not path.is_file():
        if mode == "1":
            raise CheckpointError(f"{MAP_FLAG}=1 but there is no expert pack at {path}; build one with "
                                  "python -m models.shared.pack_experts nemotron35 <checkpoint>")
        return
    try:
        pack = ExpertPack(path, root, index, "nemotron35", mapped=True)
    except (OSError, ValueError) as error:
        if mode == "1":
            raise CheckpointError(f"expert pack {path} cannot be used: {error}") from error
        print(f"expert pack {path} ignored ({error}); streaming experts from the checkpoint", file=sys.stderr)
        return
    reader.pack = pack
    if not reader.use_mapping():
        pack.close()
        reader.pack = None
        if mode == "1":
            raise CheckpointError(f"expert pack {path} does not cover every routed layer on aligned records")
        print(f"expert pack {path} ignored (partial or unaligned); streaming experts from the checkpoint",
              file=sys.stderr)
