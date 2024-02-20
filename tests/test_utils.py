"""Tests for `utils`."""

import unittest

import jax
import numpy as onp
import parameterized

from agjax import utils


class WrappedValueTest(unittest.TestCase):
    def test_flatten_unflatten(self):
        wrapped = utils.WrappedValue(value=(1, 2, 3, 4))
        leaves, treedef = jax.tree_util.tree_flatten(wrapped)
        self.assertSequenceEqual(leaves, ())
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertSequenceEqual(restored.value, (1, 2, 3, 4))

    def test_wrapped_repr(self):
        wrapped = utils.WrappedValue(value="my_string")
        self.assertSequenceEqual(str(wrapped), "_WrappedValue(my_string)")


class ValidateIdxTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        (
            ([6], 3),
            ([-4], 3),
            ([-4, -3, -2], 3),
        )
    )
    def test_invalid_idx_out_of_bounds(self, idx, sequence_length):
        with self.assertRaisesRegex(ValueError, "Found out of bounds values"):
            utils.validate_idx_for_sequence_len(idx, sequence_length)

    @parameterized.parameterized.expand(
        (
            ([1, 1], 3),
            ([1, -2], 3),
        )
    )
    def test_invalid_idx_duplicate_value(self, idx, sequence_length):
        with self.assertRaisesRegex(ValueError, "Found duplicate values"):
            utils.validate_idx_for_sequence_len(idx, sequence_length)


class SplitMergeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        (
            ([1, 2, 3, 4], [0, 1]),
            ([1, 2, 3, 4], [0, -1]),
        )
    )
    def test_merge_undoes_split(self, sequence, idx):
        a, b = utils.split(sequence, idx)
        merged = utils.merge(a, b, idx)
        for s, m in zip(sequence, merged):
            onp.testing.assert_array_equal(s, m)
