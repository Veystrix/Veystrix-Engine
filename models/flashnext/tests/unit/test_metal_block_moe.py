"""Block (row-batched) MoE kernels: host plan, bit-exactness and wiring.

The block kernels must return, for every (token, slot) pair, exactly the bits
of the per-pair kernels. The GPU tests compare raw bytes and skip cleanly when
custom Metal is unavailable.
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from models.flashnext import expert_cache, expert_pool
from models.flashnext.metal_runtime import (
    BLOCK_EMPTY,
    BLOCK_ROWS,
    MetalMoEExecutor,
    block_plan,
)

HIDDEN, INTER, TOP_K, EXPERTS = 2560, 640, 8, 64
SLOT_BIT = 0x80000000
# Tolerances of the existing MLX smoke tests (bfloat16) and the exact
# mixed-dtype tests (float32).
BF16_TOLERANCE = (2e-2, 3e-3)
F32_TOLERANCE = (1e-3, 1e-4)


class BlockPlanTests(unittest.TestCase):
    def check_plan(self, routes, rows):
        group_routes, group_pairs = block_plan(routes, rows)
        self.assertEqual(group_routes.dtype, np.uint32)
        self.assertEqual(group_pairs.dtype, np.uint32)
        self.assertEqual(group_pairs.shape, (len(group_routes), rows))
        live = group_pairs[group_pairs != BLOCK_EMPTY]
        self.assertEqual(sorted(live.tolist()), list(range(len(routes))))
        for route, pairs in zip(group_routes.tolist(), group_pairs.tolist()):
            used = [pair for pair in pairs if pair != BLOCK_EMPTY]
            self.assertEqual(pairs[:len(used)], used)       # padding only at the end
            self.assertTrue(used)                           # no group is all padding
            self.assertEqual(used, sorted(used))
            self.assertTrue(all(routes[pair] == route for pair in used))
        return group_routes.tolist(), group_pairs.tolist()

    def test_groups_follow_first_appearance_and_keep_pair_order(self):
        routes, pairs = self.check_plan([7, 3, 7, 9, 3, 7], 4)
        self.assertEqual(routes, [7, 3, 9])
        self.assertEqual(pairs, [
            [0, 2, 5, BLOCK_EMPTY],
            [1, 4, BLOCK_EMPTY, BLOCK_EMPTY],
            [3, BLOCK_EMPTY, BLOCK_EMPTY, BLOCK_EMPTY],
        ])

    def test_long_groups_split_into_consecutive_chunks(self):
        routes, pairs = self.check_plan([5] * 5 + [6] + [5] * 2, 2)
        self.assertEqual(routes, [5, 5, 5, 5, 6])
        self.assertEqual(pairs, [
            [0, 1], [2, 3], [4, 6], [7, BLOCK_EMPTY], [5, BLOCK_EMPTY],
        ])

    def test_a_group_of_exactly_rows_is_not_split_or_padded(self):
        routes, pairs = self.check_plan([1, 1, 1, 1], 4)
        self.assertEqual((routes, pairs), ([1], [[0, 1, 2, 3]]))

    def test_slot_bit_routes_are_kept_raw(self):
        raw = [SLOT_BIT | 2, 2, SLOT_BIT | 2]
        routes, pairs = self.check_plan(raw, 2)
        self.assertEqual(routes, [SLOT_BIT | 2, 2])
        self.assertEqual(pairs, [[0, 2], [1, BLOCK_EMPTY]])

    def test_every_pair_appears_once_for_random_routes(self):
        rng = np.random.default_rng(3)
        for rows in BLOCK_ROWS:
            for size in (1, 2, 8, 16, 40, 80):
                with self.subTest(rows=rows, size=size):
                    routes = rng.integers(0, 6, size=size).tolist()
                    self.check_plan(routes, rows)

    def test_accepts_numpy_and_nested_routes(self):
        flat, _ = self.check_plan([4, 4, 5, 5], 2)
        nested, _ = block_plan(np.array([[4, 4], [5, 5]], dtype=np.uint32), 2)
        self.assertEqual(flat, nested.tolist())

    def test_rejects_unsupported_rows_and_empty_routes(self):
        for rows in (0, 1, 3, 5, 16):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                block_plan([1, 2], rows)
        with self.assertRaises(ValueError):
            block_plan([], 2)
        with self.assertRaises(ValueError):
            block_plan(np.zeros((0, 8), dtype=np.uint32), 4)


def _tolerance(dtype, mx):
    return BF16_TOLERANCE if dtype == mx.bfloat16 else F32_TOLERANCE


class BlockGpuCase(unittest.TestCase):
    """Skips unless custom Metal is available; builds synthetic Q4 experts."""

    @classmethod
    def setUpClass(cls):
        if sys.platform != "darwin" or os.environ.get("GITHUB_ACTIONS") == "true":
            raise unittest.SkipTest("block kernels need a local macOS GPU")
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        if mx.default_device().type != mx.DeviceType.gpu:
            raise unittest.SkipTest("MLX is not using a GPU device")
        cls.mx = mx
        cls._experts = {}

    @classmethod
    def tearDownClass(cls):
        cls._experts = {}

    @classmethod
    def experts(cls, group_size):
        """Stacked projections and the slab pack holding the same bytes."""
        if group_size not in cls._experts:
            cls._experts = {group_size: cls._build_experts(group_size)}
        return cls._experts[group_size]

    @classmethod
    def _build_experts(cls, group_size):
        from models.flashnext.slab_pack import HEADER_SIZE, get_slab_layout

        mx = cls.mx
        layout = get_slab_layout(group_size)
        rng = np.random.default_rng(100 + group_size)
        raw = np.zeros(HEADER_SIZE + EXPERTS * layout.record_stride, dtype=np.uint8)
        packs = {}
        for name, (rows, width) in (
            ("gate_proj", (INTER, HIDDEN)),
            ("up_proj", (INTER, HIDDEN)),
            ("down_proj", (HIDDEN, INTER)),
        ):
            groups = width // group_size
            weight = rng.integers(0, 2**32, size=(EXPERTS, rows, width // 8), dtype=np.uint32)
            scales = mx.array(
                rng.uniform(0.0005, 0.002, size=(EXPERTS, rows, groups)).astype(np.float32)
            ).astype(mx.bfloat16)
            biases = mx.array(
                rng.uniform(-0.01, 0.01, size=(EXPERTS, rows, groups)).astype(np.float32)
            ).astype(mx.bfloat16)
            parts = {
                "weight": weight,
                "scales": np.asarray(mx.view(scales, mx.uint16)),
                "biases": np.asarray(mx.view(biases, mx.uint16)),
            }
            for part, values in parts.items():
                for expert in range(EXPERTS):
                    data = np.ascontiguousarray(values[expert]).view(np.uint8).reshape(-1)
                    start = HEADER_SIZE + expert * layout.record_stride + layout.offset(name, part)
                    raw[start:start + data.size] = data
            packs[name] = (mx.array(weight), scales, biases)
        slab = mx.array(raw)
        # What the pool path hands the executor next to the slab pack.
        dummy = {
            name: (
                mx.zeros((1, *weight.shape[1:]), dtype=mx.uint32),
                mx.zeros((1, *scales.shape[1:]), dtype=mx.bfloat16),
                mx.zeros((1, *scales.shape[1:]), dtype=mx.bfloat16),
            )
            for name, (weight, scales, _) in packs.items()
        }
        mx.eval(*(part for parts in packs.values() for part in parts), slab)
        return SimpleNamespace(packs=packs, slab=slab, dummy=dummy)

    # -- inputs ----------------------------------------------------------

    def routing(self, pattern, rows):
        """Expert ids per token (``TOP_K`` each) for one overlap pattern."""
        rng = np.random.default_rng(len(pattern) * 31 + rows)
        if pattern == "same":
            experts = [list(range(TOP_K)) for _ in range(rows)]
        elif pattern == "disjoint":
            experts = [list(range(r * TOP_K, (r + 1) * TOP_K)) for r in range(rows)]
        elif pattern == "mixed":
            experts = [list(range(4 * r, 4 * r + TOP_K)) for r in range(rows)]
        elif pattern == "one_shared":
            experts = [[0] + list(range(1 + 7 * r, 8 + 7 * r)) for r in range(rows)]
        else:
            raise ValueError(pattern)
        return [rng.permutation(row).tolist() for row in experts]

    def tensors(self, dtype, rows, seed=0):
        mx = self.mx
        rng = np.random.default_rng(seed + rows)
        x = mx.array(rng.normal(0, 0.25, size=(rows, HIDDEN)).astype(np.float32)).astype(dtype)
        scores = mx.array(rng.uniform(0.05, 0.2, size=(rows, TOP_K)).astype(np.float32)).astype(dtype)
        shared = mx.array(rng.normal(0, 0.05, size=(rows, HIDDEN)).astype(np.float32)).astype(dtype)
        activation = mx.array(
            rng.normal(0, 0.3, size=(rows, TOP_K, INTER)).astype(np.float32)
        ).astype(dtype)
        return SimpleNamespace(x=x, scores=scores, shared=shared, activation=activation)

    def block_for(self, routes, rows):
        group_routes, group_pairs = block_plan(np.asarray(routes).reshape(-1), rows)
        return self.mx.array(group_routes), self.mx.array(group_pairs)

    def bits(self, array):
        mx = self.mx
        mx.eval(array)
        unsigned = mx.uint16 if array.dtype == mx.bfloat16 else mx.uint32
        return np.asarray(mx.view(array.reshape(-1), unsigned))

    def assert_bits_equal(self, actual, expected, label):
        self.assertEqual(actual.dtype, expected.dtype, label)
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertTrue(np.array_equal(self.bits(actual), self.bits(expected)), label)

    def executor(self, group_size, **kwargs):
        executor = MetalMoEExecutor(
            EXPERTS, HIDDEN, TOP_K, backend="metal", group_size=group_size, **kwargs
        )
        if not executor.available:
            self.skipTest(executor.fallback_reason)
        return executor


class BlockBitExactTests(BlockGpuCase):
    DTYPES = ("bfloat16", "float32")
    PATTERNS = ("same", "disjoint", "mixed", "one_shared")

    def slab_routes(self, pattern, rows):
        experts = self.routing(pattern, rows)
        return self.mx.array([[SLOT_BIT | e for e in row] for row in experts], dtype=self.mx.uint32)

    def check_projections(self, group_size):
        """Gate, fused up+SwiGLU and slot-input down, block against per-pair."""
        mx = self.mx
        pool = self.experts(group_size)
        executor = self.executor(group_size)
        from models.flashnext.metal_runtime import _as_projection

        gate, up, down = (_as_projection(pool.packs[name])
                          for name in ("gate_proj", "up_proj", "down_proj"))
        for dtype_name in self.DTYPES:
            dtype = getattr(mx, dtype_name)
            for rows in (2, 3, 4, 8):
                inputs = self.tensors(dtype, rows)
                for pattern in self.PATTERNS:
                    routes = self.slab_routes(pattern, rows)
                    gate_ref = executor._metal_projection(
                        inputs.x, routes, gate, INTER, False,
                        slab_pack=pool.slab, proj_name="gate_proj")
                    up_ref = executor._metal_fused_up_swiglu(
                        inputs.x, routes, gate_ref, up, INTER, slab_pack=pool.slab)
                    down_ref = executor._metal_projection(
                        inputs.activation, routes, down, HIDDEN, True,
                        slab_pack=pool.slab, proj_name="down_proj")
                    for block_rows in BLOCK_ROWS:
                        label = (f"G{group_size} {dtype_name} M={rows} {pattern} "
                                 f"block={block_rows}")
                        block = self.block_for(routes, block_rows)
                        gate_out = executor._metal_block(
                            inputs.x, block, gate, INTER, False, TOP_K,
                            slab_pack=pool.slab, proj_name="gate_proj")
                        self.assert_bits_equal(gate_out, gate_ref, "gate " + label)
                        up_out = executor._metal_block(
                            inputs.x, block, up, INTER, False, TOP_K,
                            slab_pack=pool.slab, proj_name="up_proj", gate_out=gate_ref)
                        self.assert_bits_equal(up_out, up_ref, "up+swiglu " + label)
                        down_out = executor._metal_block(
                            inputs.activation, block, down, HIDDEN, True, TOP_K,
                            slab_pack=pool.slab, proj_name="down_proj")
                        self.assert_bits_equal(down_out, down_ref, "down " + label)

    def test_projections_group_size_32(self):
        self.check_projections(32)

    def test_projections_group_size_64(self):
        self.check_projections(64)

    def check_execute(self, group_size, streamed):
        mx = self.mx
        pool = self.experts(group_size)
        for fused in (False, True):
            executor = self.executor(group_size, fused_up_swiglu=fused)
            for dtype_name in self.DTYPES:
                dtype = getattr(mx, dtype_name)
                for rows in (2, 3, 4, 8):
                    inputs = self.tensors(dtype, rows, seed=5)
                    for pattern in self.PATTERNS:
                        experts = self.routing(pattern, rows)
                        if streamed:
                            routes = mx.array(experts, dtype=mx.uint32)
                            options = {}
                            weights = pool.packs
                        else:
                            routes = self.slab_routes(pattern, rows)
                            options = {"slab_pack": pool.slab}
                            weights = pool.dummy
                        for shared in (None, inputs.shared):
                            expected = executor.execute(
                                inputs.x, routes, weights, scores=inputs.scores,
                                shared_y=shared, **options)
                            for block_rows in BLOCK_ROWS:
                                label = (f"G{group_size} streamed={streamed} fused={fused} "
                                         f"{dtype_name} M={rows} {pattern} block={block_rows} "
                                         f"shared={shared is not None}")
                                actual = executor.execute(
                                    inputs.x, routes, weights, scores=inputs.scores,
                                    shared_y=shared, block=self.block_for(routes, block_rows),
                                    **options)
                                self.assertEqual(executor.last_path, "custom-metal")
                                self.assert_bits_equal(actual, expected, label)

    def test_execute_slab_group_size_32(self):
        self.check_execute(32, streamed=False)

    def test_execute_slab_group_size_64(self):
        self.check_execute(64, streamed=False)

    def test_execute_streamed_arrays(self):
        self.check_execute(32, streamed=True)
        self.check_execute(64, streamed=True)

    def test_return_all_exposes_the_same_gate_and_up(self):
        mx = self.mx
        pool = self.experts(32)
        executor = self.executor(32, fused_up_swiglu=True)
        inputs = self.tensors(mx.bfloat16, 4)
        routes = self.slab_routes("mixed", 4)
        expected = executor.execute(
            inputs.x, routes, pool.dummy, scores=inputs.scores, slab_pack=pool.slab,
            return_all=True)
        actual = executor.execute(
            inputs.x, routes, pool.dummy, scores=inputs.scores, slab_pack=pool.slab,
            return_all=True, block=self.block_for(routes, 4))
        for name, got, want in zip(("gate", "up", "down"), actual, expected):
            self.assert_bits_equal(got, want, name)

    def test_general_impl_shapes_and_narrow_outputs(self):
        """Non-fast input widths (tail) and output widths that are not tiles."""
        mx = self.mx
        from models.flashnext.metal_runtime import Q4G32Projection

        rng = np.random.default_rng(9)
        experts = 10
        for group_size, in_width in ((32, 768), (32, 800), (64, 768), (32, 1024), (64, 1536)):
            for out_width in (4, 6, 8, 12, 18, 24):
                if in_width % 512 == 0 and out_width % 8:
                    continue  # the fast per-pair kernel writes past the output here
                weight = mx.array(rng.integers(
                    0, 2**32, size=(experts, out_width, in_width // 8), dtype=np.uint32))
                scales = mx.array(rng.uniform(
                    0.0005, 0.002, size=(experts, out_width, in_width // group_size)
                ).astype(np.float32)).astype(mx.bfloat16)
                biases = mx.array(rng.uniform(
                    -0.01, 0.01, size=(experts, out_width, in_width // group_size)
                ).astype(np.float32)).astype(mx.bfloat16)
                projection = Q4G32Projection(weight, scales, biases)
                executor = MetalMoEExecutor(
                    experts, in_width, TOP_K, backend="metal", group_size=group_size)
                for dtype in (mx.bfloat16, mx.float32):
                    x = mx.array(rng.normal(0, 0.25, size=(3, in_width)).astype(np.float32)
                                 ).astype(dtype)
                    routes = mx.array(rng.integers(0, 3, size=(3, TOP_K)), dtype=mx.uint32)
                    expected = executor._metal_projection(
                        x, routes, projection, out_width, False, proj_name="gate_proj")
                    for block_rows in BLOCK_ROWS:
                        label = f"G{group_size} in={in_width} out={out_width} {dtype} block={block_rows}"
                        actual = executor._metal_block(
                            x, self.block_for(routes, block_rows), projection, out_width,
                            False, TOP_K, proj_name="gate_proj")
                        self.assert_bits_equal(actual, expected, label)
                    if out_width % 8 == 0:
                        # The per-pair fused kernel is only well defined for
                        # whole tiles.
                        gate_out = expected
                        fused = executor._metal_fused_up_swiglu(
                            x, routes, gate_out, projection, out_width)
                        actual = executor._metal_block(
                            x, self.block_for(routes, 4), projection, out_width, False,
                            TOP_K, proj_name="up_proj", gate_out=gate_out)
                        self.assert_bits_equal(actual, fused, "swiglu " + label)

    def test_slot_input_with_general_impl_matches_per_pair(self):
        mx = self.mx
        from models.flashnext.metal_runtime import Q4G32Projection

        rng = np.random.default_rng(10)
        experts, in_width, out_width = 5, 768, 40
        weight = mx.array(rng.integers(0, 2**32, size=(experts, out_width, in_width // 8),
                                       dtype=np.uint32))
        scales = mx.array(rng.uniform(0.0005, 0.002, size=(experts, out_width, in_width // 64)
                                      ).astype(np.float32)).astype(mx.bfloat16)
        biases = mx.zeros_like(scales)
        projection = Q4G32Projection(weight, scales, biases)
        executor = MetalMoEExecutor(experts, 64, 4, backend="metal", group_size=64)
        x = mx.array(rng.normal(0, 0.25, size=(2, 4, in_width)).astype(np.float32)).astype(mx.bfloat16)
        routes = mx.array([[0, 1, 1, 2], [2, 2, 0, 4]], dtype=mx.uint32)
        expected = executor._metal_projection(
            x, routes, projection, out_width, True, proj_name="down_proj")
        actual = executor._metal_block(
            x, self.block_for(routes, 2), projection, out_width, True, 4,
            proj_name="down_proj")
        self.assert_bits_equal(actual, expected, "slot input, general impl")


def _first_experts(pool, count):
    """The first ``count`` experts of the stacked projections."""
    return {
        name: tuple(part[:count] for part in parts) for name, parts in pool.packs.items()
    }


class BlockReferenceTests(BlockGpuCase):
    def test_block_output_agrees_with_the_mlx_reference(self):
        mx = self.mx
        for group_size in (32, 64):
            pool = self.experts(group_size)
            packs = _first_experts(pool, 32)
            reference = MetalMoEExecutor(
                EXPERTS, HIDDEN, TOP_K, backend="reference", group_size=group_size)
            executor = self.executor(group_size, fused_up_swiglu=True)
            for dtype in (mx.bfloat16, mx.float32):
                for pattern in ("same", "mixed", "disjoint"):
                    rows = 4
                    inputs = self.tensors(dtype, rows, seed=2)
                    experts = self.routing(pattern, rows)
                    routes = mx.array(experts, dtype=mx.uint32)
                    expected = reference.execute(
                        inputs.x, routes, packs, scores=inputs.scores)
                    actual = executor.execute(
                        inputs.x, routes, packs, scores=inputs.scores,
                        block=self.block_for(routes, 4))
                    self.assertEqual(reference.last_path, "reference")
                    rtol, atol = _tolerance(dtype, mx)
                    mx.eval(actual, expected)
                    np.testing.assert_allclose(
                        np.asarray(actual.astype(mx.float32)),
                        np.asarray(expected.astype(mx.float32)),
                        rtol=rtol, atol=atol,
                        err_msg=f"G{group_size} {dtype} {pattern}")


class BlockKernelCacheTests(BlockGpuCase):
    def test_group_count_and_token_count_do_not_enter_the_kernel_key(self):
        mx = self.mx
        pool = self.experts(64)
        executor = self.executor(64)
        from models.flashnext.metal_runtime import _as_projection

        gate = _as_projection(pool.packs["gate_proj"])
        built = []
        for rows, pattern in ((3, "same"), (4, "disjoint"), (4, "mixed")):
            inputs = self.tensors(mx.bfloat16, rows)
            experts = self.routing(pattern, rows)
            routes = mx.array([[SLOT_BIT | e for e in row] for row in experts], dtype=mx.uint32)
            block = self.block_for(routes, 4)
            built.append((block[0].shape[0], rows))
            mx.eval(executor._metal_block(
                inputs.x, block, gate, INTER, False, TOP_K,
                slab_pack=pool.slab, proj_name="gate_proj"))
        self.assertEqual(len({groups for groups, _ in built}), 3, built)
        self.assertEqual(len(executor._kernels), 1)
        (key,) = executor._kernels
        self.assertEqual(key[-1], 4)

    def test_block_rows_select_distinct_named_kernels(self):
        captured = []

        class Backend(SimpleNamespace):
            def metal_kernel(self, **kwargs):
                captured.append(kwargs)
                return lambda **_: None

        pool_shape = (np.zeros((2, 16, 4), dtype=np.uint32),
                      np.zeros((2, 16, 1), dtype=np.float32),
                      np.zeros((2, 16, 1), dtype=np.float32))
        from models.flashnext.metal_runtime import Q4G32Projection

        executor = MetalMoEExecutor(
            2, 32, 2, backend=Backend(available=True, supports_custom_moe=True))
        x = np.zeros((2, 32), dtype=np.float32)
        projection = Q4G32Projection(*pool_shape)
        for rows in (2, 4):
            routes, pairs = block_plan([0, 1, 0, 1], rows)
            executor._metal_block(x, (routes, pairs), projection, 16, False, 2,
                                  proj_name="gate_proj")
        executor._metal_projection(x, np.zeros((2, 2), dtype=np.uint32), projection, 16, False,
                                   proj_name="gate_proj")
        names = [call["name"] for call in captured]
        self.assertEqual(len(set(names)), 3, names)
        self.assertTrue(names[0].endswith("_block2") and names[1].endswith("_block4"))
        self.assertNotIn("block", names[2])
        self.assertEqual(captured[0]["input_names"][4:6], ["group_routes", "group_pairs"])
        self.assertEqual(captured[0]["compile_options"], {"math_mode": "safe"})


class BlockBoundaryTests(BlockGpuCase):
    def test_a_profiled_boundary_times_the_block_launch_under_its_label(self):
        mx = self.mx
        pool = self.experts(64)
        from models.flashnext.metal_runtime import _as_projection

        gate = _as_projection(pool.packs["gate_proj"])
        inputs = self.tensors(mx.bfloat16, 2)
        routes = mx.array([[SLOT_BIT | e for e in range(TOP_K)]] * 2, dtype=mx.uint32)
        for boundary in ("gate_qmv", "up_qmv"):
            executor = self.executor(64, profile_boundaries=True, profile_boundary=boundary)
            out = executor._metal_block(
                inputs.x, self.block_for(routes, 2), gate, INTER, False, TOP_K,
                slab_pack=pool.slab, proj_name=boundary.replace("qmv", "proj"))
            mx.eval(out)
            labels = [event["label"] for event in executor.boundary_profile["events"]]
            self.assertEqual(labels, [boundary])
            self.assertTrue(any(key[7] == boundary.replace("qmv", "proj")
                                for key in executor._kernels))


class BlockApiTests(BlockGpuCase):
    def test_block_with_a_pool_activation_is_rejected(self):
        mx = self.mx
        pool = self.experts(32)
        executor = self.executor(32)
        inputs = self.tensors(mx.bfloat16, 2)
        routes = self.mx.array([[SLOT_BIT | e for e in range(TOP_K)]] * 2, dtype=mx.uint32)
        with self.assertRaises(ValueError):
            executor.execute(
                inputs.x, routes, pool.dummy, scores=inputs.scores, slab_pack=pool.slab,
                pool_activation=inputs.activation[:2], block=self.block_for(routes, 2))

    def test_malformed_blocks_are_rejected(self):
        mx = self.mx
        pool = self.experts(32)
        executor = self.executor(32)
        inputs = self.tensors(mx.bfloat16, 2)
        routes = mx.array([[SLOT_BIT | e for e in range(TOP_K)]] * 2, dtype=mx.uint32)
        good_routes, good_pairs = block_plan(np.asarray(routes).reshape(-1), 2)

        def run(block):
            return executor.execute(inputs.x, routes, pool.dummy, scores=inputs.scores,
                                    slab_pack=pool.slab, block=block)

        for label, block, error in (
            ("not a pair", (mx.array(good_routes),), ValueError),
            ("rows", (mx.array(good_routes), mx.zeros((len(good_routes), 3), dtype=mx.uint32)),
             ValueError),
            ("groups", (mx.array(good_routes[:-1]), mx.array(good_pairs)), ValueError),
            ("empty", (mx.zeros((0,), dtype=mx.uint32), mx.zeros((0, 2), dtype=mx.uint32)),
             ValueError),
            ("too many groups", (mx.zeros((17,), dtype=mx.uint32),
                                 mx.zeros((17, 2), dtype=mx.uint32)), ValueError),
            ("route rank", (mx.array(good_routes)[None], mx.array(good_pairs)), ValueError),
            ("dtype", (mx.array(good_routes).astype(mx.int32), mx.array(good_pairs)), TypeError),
        ):
            with self.subTest(label), self.assertRaises(error):
                run(block)

    def test_an_unavailable_executor_uses_the_reference_path(self):
        mx = self.mx
        pool = self.experts(32)
        inputs = self.tensors(mx.float32, 2)
        routes = mx.array([[e for e in range(TOP_K)]] * 2, dtype=mx.uint32)
        packs = _first_experts(pool, 16)
        executor = MetalMoEExecutor(EXPERTS, HIDDEN, TOP_K, backend="reference")
        expected = executor.execute(inputs.x, routes, packs, scores=inputs.scores)
        actual = executor.execute(inputs.x, routes, packs, scores=inputs.scores,
                                  block=self.block_for(routes, 2))
        self.assertEqual(executor.last_path, "reference")
        self.assert_bits_equal(actual, expected, "reference fallback")


class _Pool:
    """Just enough of ExpertPool for ``_pool_pass`` with every expert resident."""

    slots = 40
    usable_slots = 40
    buffer_mx = "pack"

    def assign(self, layer_id, wanted):
        return {expert: index for index, expert in enumerate(wanted)}, []

    def prefetch_ahead(self, layer_id):
        pass


class PoolPassWiringTests(unittest.TestCase):
    """What ``_pool_pass`` hands the executor; the executor itself is faked."""

    FLAG = "FLASHNEXT_METAL_BLOCK_MOE"
    ROWS = "FLASHNEXT_METAL_BLOCK_MOE_ROWS"

    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.mx = mx

    def run_pool_pass(self, tokens, *, available=True, env=None):
        mx = self.mx
        calls = []

        class Executor:
            def __init__(self):
                self.available = available

            def execute(self, x, routes, weights, **kwargs):
                calls.append((routes, kwargs))
                return mx.zeros((tokens, 16), dtype=mx.float32)

        gate = SimpleNamespace(cache=SimpleNamespace(store=object()), group_size=64)
        layer = SimpleNamespace(
            layer_id=3, gate_proj=gate, hits=0, misses=0,
            _executor=lambda *args: Executor(),
            _get_dummy_streamed_weights=lambda hidden: ("gate", "up", "down"),
        )
        routed = [(token * 4 + slot) % 10 for token in range(tokens) for slot in range(TOP_K)]
        indices = mx.zeros((tokens, TOP_K), dtype=mx.uint32)
        flat_input = mx.zeros((tokens, 16), dtype=mx.float32)
        scores = mx.zeros((tokens, TOP_K), dtype=mx.float32)
        env = env or {}
        with mock.patch.object(expert_pool, "for_store", return_value=_Pool()), \
             mock.patch.dict(os.environ, env):
            for key in (self.FLAG, self.ROWS):
                if key not in env:
                    os.environ.pop(key, None)
            expert_cache.StreamingSwitchGLU._pool_pass(
                layer, indices, routed, flat_input, scores, None, None, None)
        (routes, kwargs), = calls
        return routed, routes, kwargs

    def test_flag_off_passes_no_plan(self):
        for tokens in (1, 4):
            with self.subTest(tokens=tokens):
                _, _, kwargs = self.run_pool_pass(tokens)
                self.assertNotIn("block", kwargs)
        _, _, kwargs = self.run_pool_pass(4, env={self.FLAG: "0"})
        self.assertNotIn("block", kwargs)

    def test_a_single_token_never_uses_the_block_path(self):
        _, _, kwargs = self.run_pool_pass(1, env={self.FLAG: "1"})
        self.assertNotIn("block", kwargs)

    def test_flag_on_builds_a_two_row_plan_from_the_pool_routes(self):
        for tokens in (2, 3, 4, 5, 8):
            with self.subTest(tokens=tokens):
                routed, routes, kwargs = self.run_pool_pass(tokens, env={self.FLAG: "1"})
                group_routes, group_pairs = kwargs["block"]
                self.assertEqual(group_pairs.shape[1], 2)
                expected = block_plan(np.asarray(routes).reshape(-1), 2)
                self.assertTrue(np.array_equal(np.asarray(group_routes), expected[0]))
                self.assertTrue(np.array_equal(np.asarray(group_pairs), expected[1]))
                # Raw route values carry the pool slot bit.
                self.assertTrue(all(int(value) & SLOT_BIT for value in np.asarray(group_routes)))
                self.assertEqual(
                    sorted(int(p) for p in np.asarray(group_pairs).reshape(-1) if p != BLOCK_EMPTY),
                    list(range(tokens * TOP_K)))

    def test_the_rows_setting_selects_four_or_eight(self):
        for value, tokens, rows in (("2", 8, 2), ("4", 8, 4), ("8", 3, 8), ("4", 2, 4)):
            with self.subTest(rows=value, tokens=tokens):
                _, _, kwargs = self.run_pool_pass(
                    tokens, env={self.FLAG: "1", self.ROWS: value})
                self.assertEqual(kwargs["block"][1].shape[1], rows)

    def test_an_invalid_rows_setting_falls_back_to_two(self):
        for value in ("", "1", "3", "16", "four"):
            with self.subTest(rows=value):
                _, _, kwargs = self.run_pool_pass(
                    4, env={self.FLAG: "1", self.ROWS: value})
                self.assertEqual(kwargs["block"][1].shape[1], 2)

    def test_an_unavailable_executor_gets_no_plan(self):
        _, _, kwargs = self.run_pool_pass(4, available=False, env={self.FLAG: "1"})
        self.assertNotIn("block", kwargs)


if __name__ == "__main__":
    unittest.main()
