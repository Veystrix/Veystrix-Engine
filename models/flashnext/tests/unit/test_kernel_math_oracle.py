"""GPU-free oracle for the byte math that the expert kernels rely on.

NumPy computes the expected slab stride from the layout's declared tensor
shapes; the engine's own arithmetic must agree. Runs on any CI runner, with
no Metal device and no checkpoint.
"""
from __future__ import annotations

import unittest

import numpy as np

from models.flashnext import preflight as P
from models.flashnext.slab_pack import get_slab_layout


def oracle_bytes_from_shapes(layout) -> int:
    """Sum declared tensor bytes: uint32 words (4 B) or bf16 metadata (2 B)."""
    total = np.int64(0)
    for name, (rows, cols) in layout.shapes:
        element_bytes = 4 if name.endswith("weight") else 2
        total += np.int64(rows) * np.int64(cols) * element_bytes
    return int(total)


def offset_bytes(layout, name: str) -> int:
    rows, cols = layout.shape(*name.split("."))
    return int(np.int64(rows) * np.int64(cols) * (4 if name.endswith("weight") else 2))


class KernelMathOracleTests(unittest.TestCase):
    def test_slab_stride_matches_declared_shapes(self):
        for group in (32, 64):
            with self.subTest(group=group):
                layout = get_slab_layout(group)
                self.assertEqual(layout.record_stride, oracle_bytes_from_shapes(layout))

    def test_expert_record_bytes_matches_the_slab_stride(self):
        for group in (32, 64):
            with self.subTest(group=group):
                layout = get_slab_layout(group)
                self.assertEqual(P.expert_record_bytes(2560, 640, 4, group), layout.record_stride)

    def test_slab_offsets_are_contiguous_and_in_order(self):
        for group in (32, 64):
            layout = get_slab_layout(group)
            cursor = 0
            for name, offset in layout.offsets:
                self.assertEqual(offset, cursor, (group, name))
                cursor += offset_bytes(layout, name)
            self.assertEqual(cursor, layout.record_stride)

    def test_malformed_quant_dimensions_do_not_raise(self):
        for hidden, intermediate, bits, group in ((0, 640, 4, 32), (2560, 640, 4, 0),
                                                   ("bad", 640, 4, 32), (2560, None, 4, 32)):
            with self.subTest(hidden=hidden, group=group):
                self.assertEqual(P.expert_record_bytes(hidden, intermediate, bits, group), 0)

    def test_tensor_inventory_has_no_collisions(self):
        layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)]
        families = P.expected_families(layer_types)
        names = P.tensor_names(families)
        self.assertEqual(len(names), len(set(names)))
        expansions = {"WSB": 3, "W": 1, "": 1}
        self.assertEqual(len(names), sum(expansions[spec] for spec in families.values()))


if __name__ == "__main__":
    unittest.main()
