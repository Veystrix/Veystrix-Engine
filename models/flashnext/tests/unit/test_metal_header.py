"""MLX header slicing: the pinned wheel is unchanged and a newer one is tolerated."""
from __future__ import annotations

from pathlib import Path
import unittest

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - MLX is optional in CI
    mx = None

from models.flashnext.metal_runtime import _mlx_qmv_header, _slice_qmv_header

_FAST_HEAD_OLD = "template <typename T, int group_size, int bits>\nMETAL_FUNC void qmv_fast_impl("
_FAST_HEAD_NEW = (
    "template <typename T, int group_size, int bits, bool partial_rows = false>\n"
    "METAL_FUNC void qmv_fast_impl("
)


def _wheel_text() -> str:
    return (
        Path(mx.__file__).resolve().parent / "include/mlx/backend/metal/kernels/quantized.h"
    ).read_text()


# Verbatim copy of the implementation before the slicing was factored out.
def _legacy_header(fma_dequant: bool) -> str:
    """Load the exact Q4 vector helpers shipped with the active MLX wheel."""
    path = (
        Path(mx.__file__).resolve().parent
        / "include/mlx/backend/metal/kernels/quantized.h"
    )
    text = path.read_text()
    helper_end = text.index(
        "template <typename U, int values_per_thread, int bits>\ninline void\nqouter"
    )
    qmv_start = text.index(
        "template <typename T, int group_size, int bits>\n"
        "METAL_FUNC void qmv_fast_impl"
    )
    qmv_end = text.index("// Affine analog of fp_qmv_wide", qmv_start)
    header = text[:helper_end] + text[qmv_start:qmv_end]
    # The stock MLX helpers use one type for activations and affine metadata.
    # FlashNext stores float32 activations with bfloat16 scales and biases.
    # Keep the MLX reduction body, but use a separate metadata type so MLX
    # does not insert per-projection metadata conversion kernels.
    impl_start = text.index(
        "template <typename T, int group_size, int bits>\n"
        "METAL_FUNC void qmv_impl",
        qmv_start,
    )
    fast = text[qmv_start:impl_start]
    impl = text[impl_start:qmv_end]
    fast = fast.replace(
        "template <typename T, int group_size, int bits>",
        "template <typename T, int group_size, int bits, typename M>",
        1,
    ).replace("qmv_fast_impl", "qmv_fast_mixed_impl", 1)
    impl = impl.replace(
        "template <typename T, int group_size, int bits>",
        "template <typename T, int group_size, int bits, typename M>",
        1,
    ).replace("qmv_impl", "qmv_mixed_impl", 1)
    for old, new in (
        ("const device T* scales", "const device M* scales"),
        ("const device T* biases", "const device M* biases"),
        ("const device T* sl", "const device M* sl"),
        ("const device T* bl", "const device M* bl"),
    ):
        fast = fast.replace(old, new)
        impl = impl.replace(old, new)
    header += fast + impl

    def _make_accumulate(src: str, old_name: str, new_name: str) -> str:
        acc = src.replace(old_name, new_name, 1)
        acc = acc.replace("device T* y,", "thread float* combined, float score,", 1)
        acc = acc.replace("y += tid.x * out_vec_size + out_row;", "// y unused", 1)
        acc = acc.replace("y += tid.x * out_vec_size + used_out_row;", "// y unused", 1)
        acc = acc.replace(
            "y[row] = static_cast<T>(result[row]);",
            "combined[row] += float(static_cast<T>(result[row])) * score;",
        )
        return acc

    fast_acc = _make_accumulate(fast, "qmv_fast_mixed_impl", "qmv_fast_accumulate_impl")
    impl_acc = _make_accumulate(impl, "qmv_mixed_impl", "qmv_accumulate_impl")
    header += fast_acc + impl_acc
    # Custom-kernel helpers receive small local dimensions, not buffer
    # address-space constants. Pass them by value so older Metal compilers do
    # not reject references without an explicit address space.
    if fma_dequant:
        old = "return scale * accum + sum * bias;"
        if header.count(old) != 2:
            raise RuntimeError("MLX affine dot helpers changed; cannot enable explicit FMA")
        header = header.replace(old, "return metal::fma(scale, accum, sum * bias);")
    return header.replace("const constant int&", "const int")


@unittest.skipIf(mx is None, "MLX is unavailable")
class PinnedWheelHeaderTests(unittest.TestCase):
    def test_header_is_byte_identical_to_the_previous_implementation(self):
        for fma in (False, True):
            with self.subTest(fma_dequant=fma):
                self.assertEqual(_mlx_qmv_header(fma), _legacy_header(fma))

    def test_pinned_wheel_text_still_has_the_two_template_heads(self):
        self.assertIn(_FAST_HEAD_OLD, _wheel_text())


@unittest.skipIf(mx is None, "MLX is unavailable")
class SlicingLogicTests(unittest.TestCase):
    def setUp(self):
        self.text = _wheel_text()

    def newer_text(self):
        self.assertEqual(self.text.count(_FAST_HEAD_OLD), 1)
        return self.text.replace(_FAST_HEAD_OLD, _FAST_HEAD_NEW)

    def test_extra_template_parameters_keep_their_position_after_the_metadata_type(self):
        header = _slice_qmv_header(self.newer_text(), False, "0.32.4")
        mixed = (
            "template <typename T, int group_size, int bits, typename M, bool partial_rows = false>\n"
            "METAL_FUNC void "
        )
        self.assertIn(mixed + "qmv_fast_mixed_impl(", header)
        self.assertIn(mixed + "qmv_fast_accumulate_impl(", header)
        plain = "template <typename T, int group_size, int bits, typename M>\nMETAL_FUNC void "
        self.assertIn(plain + "qmv_mixed_impl(", header)
        self.assertIn(plain + "qmv_accumulate_impl(", header)

    def test_newer_text_differs_from_the_pinned_text_only_in_the_template_head(self):
        pinned = _slice_qmv_header(self.text, False, "0.32.2")
        newer = _slice_qmv_header(self.newer_text(), False, "0.32.4")
        normalised = newer.replace(", bool partial_rows = false>", ">")
        self.assertEqual(normalised, pinned)

    def test_a_missing_pattern_names_the_pattern_and_the_mlx_version(self):
        text = self.text
        cases = {
            "device T* y,": text.replace("device T* y,", "device T* out,"),
            "y += tid.x * out_vec_size + used_out_row;": text.replace(
                "y += tid.x * out_vec_size + used_out_row;", "y += 0;"),
            "y[row] = static_cast<T>(result[row]);": text.replace(
                "y[row] = static_cast<T>(result[row]);", "y[row] = result[row];"),
            "const device T* sl": text.replace("const device T* sl", "const device T *sl"),
            "const device T* scales": text.replace("const device T* scales", "const device T *scales"),
        }
        for pattern, broken in cases.items():
            with self.subTest(pattern=pattern):
                with self.assertRaises(RuntimeError) as raised:
                    _slice_qmv_header(broken, False, "9.9.9")
                self.assertIn(pattern, str(raised.exception))
                self.assertIn("9.9.9", str(raised.exception))

    def test_a_missing_anchor_raises_runtime_error(self):
        for old in (_FAST_HEAD_OLD, "METAL_FUNC void qmv_impl(", "// Affine analog of fp_qmv_wide",
                    "inline void\nqouter"):
            with self.subTest(anchor=old):
                with self.assertRaises(RuntimeError):
                    _slice_qmv_header(self.text.replace(old, "// gone", 1), False, "9.9.9")

    def test_explicit_fma_requires_both_dot_helpers(self):
        broken = self.text.replace("return scale * accum + sum * bias;", "return accum;", 1)
        with self.assertRaisesRegex(RuntimeError, "9.9.9"):
            _slice_qmv_header(broken, True, "9.9.9")
        _slice_qmv_header(broken, False, "9.9.9")


if __name__ == "__main__":
    unittest.main()
