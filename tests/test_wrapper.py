# type: ignore
"""Tests for `wrapper`."""

import unittest

import autograd
import autograd.numpy as npa
import jax
import jax.numpy as jnp
import numpy as onp
import parameterized

from agjax import wrapper

TEST_FNS_AND_ARGS = (
    (  # Basic scalar-valued function, real outputs.
        lambda x: x**2,
        (3.0,),
    ),
    (  # Scalar-valued function with two input arguments, real outputs.
        lambda x, y: x**2 + y,
        (3.0, 4.0),
    ),
    (  # Two arguments, scalar output wrapped in a tuple, real outputs.
        lambda x, y: (x**2 + y,),
        (3.0, 4.0),
    ),
    (  # Two arguments, two outputs, real outputs.
        lambda x, y: (x**2 + y, x - y),
        (3.0, 4.0),
    ),
    (  # Two arguments, two outputs, real arguments, complex outputs.
        lambda x, y: (x**2 + 1j * y**2, 1j * x**2 - y**2),
        (3.0, 4.0),
    ),
    (  # Two arguments, two outputs, complex arguments, real outputs.
        lambda x, y: (npa.abs(x) ** 2 + npa.abs(y), npa.abs(x + y)),
        (3.0 + 1.0j, 4.0 + 0.5j),
    ),
    (  # Two arguments, two outputs, complex outputs.
        lambda x, y: (x**2 + y, x - y),
        (3.0 + 1.0j, 4.0 + 0.5j),
    ),
    (  # Two arguments, two outputs, complex outputs.
        lambda x, y, z: (x**2 + y + z, x - y),
        (3.0 + 1.0j, 4.0 + 0.5j, -11.0),
    ),
    (  # Returns a pytree, complex outputs.
        lambda x, y: {"a": x**2 + y, "b": (x - y, y - x)},
        (3.0 + 1.0j, 4.0 + 0.5j),
    ),
    (  # Arguments and outputs include pytree, complex outputs.
        lambda x, y: {
            "a": (x["a0"] + x["a1"]) ** 2 + y,
            "b": (x["a0"] - y, y - x["a1"]),
        },
        ({"a0": 3.0 + 1.0j, "a1": 22.0j}, 4.0 + 0.5j),
    ),
)


class WrapperTest(unittest.TestCase):
    @parameterized.parameterized.expand(([2], [-3]))
    def test_out_of_bounds_nondiff_argnums(self, nondiff_argnums):
        def fn(x, y):
            return (npa.sum(x + y), y)

        wrapped = wrapper.wrap_for_jax(fn, nondiff_argnums=nondiff_argnums)
        with self.assertRaisesRegex(ValueError, "Found out of bounds"):
            wrapped(1.0, 2.0)

    @parameterized.parameterized.expand(([(1, 1)], [(1, -1)]))
    def test_duplicate_nondiff_argnums(self, nondiff_argnums):
        def fn(x, y):
            return (npa.sum(x + y), y)

        wrapped = wrapper.wrap_for_jax(fn, nondiff_argnums=nondiff_argnums)
        with self.assertRaisesRegex(ValueError, "Found duplicate"):
            wrapped(1.0, 2.0)

    @parameterized.parameterized.expand(([2], [-3]))
    def test_out_of_bounds_nondiff_outputnums(self, nondiff_outputnums):
        def fn(x, y):
            return (npa.sum(x + y), y)

        wrapped = wrapper.wrap_for_jax(fn, nondiff_outputnums=nondiff_outputnums)
        with self.assertRaisesRegex(ValueError, "Found out of bounds"):
            wrapped(1.0, 2.0)

    @parameterized.parameterized.expand(([(1, 1)], [(1, -1)]))
    def test_duplicate_nondiff_outputnums(self, nondiff_outputnums):
        def fn(x, y):
            return (npa.sum(x + y), y)

        wrapped = wrapper.wrap_for_jax(fn, nondiff_outputnums=nondiff_outputnums)
        with self.assertRaisesRegex(ValueError, "Found duplicate"):
            wrapped(1.0, 2.0)

    def test_function_has_no_differentiable_outputs(self):
        def fn(x, y):
            return (x, y)

        wrapped = wrapper.wrap_for_jax(fn, nondiff_outputnums=(0, 1))
        with self.assertRaisesRegex(ValueError, "At least one differentiable output"):
            wrapped(1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "At least one differentiable output"):
            jax.grad(wrapped)(1.0, 2.0)
        with self.assertRaisesRegex(ValueError, "At least one differentiable output"):
            jax.value_and_grad(wrapped)(1.0, 2.0)

    @parameterized.parameterized.expand(TEST_FNS_AND_ARGS)
    def test_wrapped_matches_autograd(self, autograd_fn, args):
        # Tests case where all arguments can be differentiated
        # with respect to, and all outputs are differentiable.
        expected_outputs = autograd_fn(*args)

        wrapped = wrapper.wrap_for_jax(
            autograd_fn,
            nondiff_argnums=(),
            nondiff_outputnums=(),
        )
        for v, ev in zip(
            jax.tree_util.tree_leaves(wrapped(*args)),
            jax.tree_util.tree_leaves(expected_outputs),
        ):
            onp.testing.assert_allclose(v, ev)

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
