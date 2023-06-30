"""Tests for `wrapper`."""

import autograd
import autograd.numpy as npa
import jax
import jax.numpy as jnp
import numpy as onp
import parameterized
import unittest

from agjax import wrapper


TEST_FNS_AND_ARGS = (
    (  # Basic scalar-valued function.
        lambda x: x**2,
        (3.0,),
    ),
    (  # Scalar-valued function with two input arguments.
        lambda x, y: x**2 + y,
        (3.0, 4.0),
    ),
    (  # Two arguments, two outputs.
        lambda x, y: (x**2 + y, x - y),
        (3.0, 4.0),
    ),
    (  # Two arguments, two outputs, complex.
        lambda x, y: (x**2 + y, x - y),
        (3.0 + 1.0j, 4.0 + 0.5j),
    ),
    (  # Two arguments, two outputs, complex.
        lambda x, y, z: (x**2 + y + z, x - y),
        (3.0 + 1.0j, 4.0 + 0.5j, -11.0),
    ),
    (  # Returns a pytree.
        lambda x, y: {"a": x**2 + y, "b": (x - y, y - x)},
        (3.0 + 1.0j, 4.0 + 0.5j),
    ),
    (  # Arguments and outputs include pytree.
        lambda x, y: {
            "a": (x["a0"] + x["a1"]) ** 2 + y,
            "b": (x["a0"] - y, y - x["a1"]),
        },
        ({"a0": 3.0 + 1.0j, "a1": 22.0j}, 4.0 + 0.5j),
    ),
)


class WrapperTest(unittest.TestCase):
    @parameterized.parameterized.expand(TEST_FNS_AND_ARGS)
    def test_wrapped_matches_autograd(self, autograd_fn, args):
        # Tests case where all arguments can be differentiated
        # with respect to, and all outputs are differentiable.
        expected_outputs = autograd_fn(*args)

        outputnums = (
            tuple(range(len(expected_outputs)))
            if isinstance(expected_outputs, tuple)
            else 0
        )
        wrapped = wrapper.wrap_for_jax(
            autograd_fn,
            nondiff_argnums=(),
            nondiff_outputnums=(),
        )
        onp.testing.assert_array_equal(expected_outputs, wrapped(*args))

        def autograd_scalar_fn(*args):
            outputs = autograd_fn(*args)
            outputs = jax.tree_util.tree_leaves(outputs)
            return npa.sum([npa.sum(npa.abs(o) ** 2) for o in outputs])

        def jax_scalar_fn(*args):
            outputs = wrapped(*args)
            outputs = jax.tree_util.tree_leaves(outputs)
            return jnp.sum(jnp.asarray([jnp.sum(jnp.abs(o) ** 2) for o in outputs]))

        expected_grad = autograd.grad(autograd_scalar_fn)(*args)
        grad = jax.grad(jax_scalar_fn)(*args)

        for g, eg in zip(
            jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(expected_grad)
        ):
            onp.testing.assert_allclose(g, eg)

    def test_nondiff_argnums_and_outputnums(self):
        def autograd_fn(x, y, int_arg, str_arg):
            return (
                npa.sum(x**2 + y * int_arg),
                x - y + 2,
                str_arg * int_arg,
            )

        wrapped = wrapper.wrap_for_jax(
            autograd_fn, nondiff_argnums=(2, 3), nondiff_outputnums=2
        )

        args = (0.3 + 2.2j, -11.0 + 0.0j, 3, "test")
        expected_outputs = ((0.3 + 2.2j) ** 2 - 11.0 * 3, 13.3 + 2.2j, "testtesttest")

        # Check that directly calling the wrapped function gives the expected
        # output. This validates `_fn`.
        wrapped_outputs = wrapped(*args)
        onp.testing.assert_allclose(wrapped_outputs[0], expected_outputs[0])
        onp.testing.assert_allclose(wrapped_outputs[1], expected_outputs[1])
        self.assertEqual(wrapped_outputs[2], expected_outputs[2])

        def jax_scalar_fn(*args):
            a, b, c = wrapped(*args)
            return a + b, c

        expected_value = (0.3 + 2.2j) ** 2 - 11.0 * 3 + 13.3 + 2.2j
        expected_aux = "testtesttest"
        expected_grad = (2 * args[0] + 1, 2)

        # Check that directly calling the wrapped function gives the expected
        # output. This validates `_fn`.
        value, aux = jax_scalar_fn(*args)
        onp.testing.assert_allclose(value, expected_value)
        self.assertEqual(aux, expected_aux)

        # Check that `value_and_grad` returns the correct value and gradient.
        # The value is computed by `fwd_fn`, and so this tests a different
        # codepath than when the value is computed directly.
        (value, aux), grad = jax.value_and_grad(
            jax_scalar_fn,
            has_aux=True,
            holomorphic=True,
            argnums=(0, 1),
        )(*args)
        onp.testing.assert_allclose(value, expected_value)
        self.assertEqual(aux, expected_aux)
        for e, g in zip(expected_grad, grad):
            onp.testing.assert_allclose(e, g)


class SplitMergeTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        (
            ([1, 2, 3], [6]),
            ([1, 2, 3], [-4]),
            ([1, 2, 3], [-4, -3, -2]),
        )
    )
    def test_invalid_idx_out_of_bounds(self, sequence, idx):
        with self.assertRaisesRegex(ValueError, "Found out of bounds values"):
            wrapper._split(sequence, idx)

    @parameterized.parameterized.expand(
        (
            ([1, 2, 3], [1, 1]),
            ([1, 2, 3], [1, -2]),
        )
    )
    def test_invalid_idx_duplicate_value(self, sequence, idx):
        with self.assertRaisesRegex(ValueError, "Found duplicate values"):
            wrapper._split(sequence, idx)

    @parameterized.parameterized.expand(
        (
            ([1, 2, 3, 4], [0, 1]),
            ([1, 2, 3, 4], [0, -1]),
        )
    )
    def test_merge_undoes_split(self, sequence, idx):
        a, b = wrapper._split(sequence, idx)
        merged = wrapper._merge(a, b, idx)
        for s, m in zip(sequence, merged):
            onp.testing.assert_array_equal(s, m)
