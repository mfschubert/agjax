# type: ignore
"""Tests for `wrapper`."""

import unittest

import autograd
import autograd.numpy as npa
import jax
import jax.numpy as jnp
import jaxlib
import numpy as onp
from parameterized import parameterized

from agjax import utils
from agjax.experimental import wrapper

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
    @parameterized.expand(([2], [-3]))
    def test_out_of_bounds_nondiff_argnums(self, nondiff_argnums):
        def fn(x, y):
            return (npa.sum(x + y), y)

        wrapped = wrapper.wrap_for_jax(
            fn,
            result_shape_dtypes=(jnp.asarray(0.0), jnp.asarray(0.0)),
            nondiff_argnums=nondiff_argnums,
        )
        with self.assertRaisesRegex(ValueError, "Found out of bounds"):
            wrapped(1.0, 2.0)
        with self.assertRaisesRegex(
            jaxlib.xla_extension.XlaRuntimeError, "Found out of bounds"
        ):
            jax.grad(wrapped)(1.0, 2.0)

    @parameterized.expand(([2], [-3]))
    def test_out_of_bounds_nondiff_outputnums(self, nondiff_outputnums):
        def fn(x, y):
            return (npa.sum(x + y), y)

        wrapped = wrapper.wrap_for_jax(
            fn,
            result_shape_dtypes=(jnp.asarray(0.0), jnp.asarray(0.0)),
            nondiff_outputnums=nondiff_outputnums,
        )
        with self.assertRaisesRegex(ValueError, "Found out of bounds"):
            wrapped(1.0, 2.0)
        with self.assertRaisesRegex(
            jaxlib.xla_extension.XlaRuntimeError, "Found out of bounds"
        ):
            jax.grad(wrapped)(1.0, 2.0)

    @parameterized.expand(([(1, 1)], [(1, -1)]))
    def test_duplicate_nondiff_outputnums(self, nondiff_outputnums):
        def fn(x, y):
            return (npa.sum(x + y), y)

        wrapped = wrapper.wrap_for_jax(
            fn,
            result_shape_dtypes=(jnp.asarray(0.0), jnp.asarray(0.0)),
            nondiff_outputnums=nondiff_outputnums,
        )
        with self.assertRaisesRegex(ValueError, "Found duplicate"):
            wrapped(1.0, 2.0)

    def test_function_has_no_differentiable_outputs(self):
        def fn(x, y):
            return (x, y)

        wrapped = wrapper.wrap_for_jax(
            fn,
            result_shape_dtypes=(jnp.asarray(0.0), jnp.asarray(0.0)),
            nondiff_outputnums=(0, 1),
        )
        with self.assertRaisesRegex(ValueError, "At least one differentiable output"):
            wrapped(1.0, 2.0)
        with self.assertRaisesRegex(
            jaxlib.xla_extension.XlaRuntimeError, "At least one differentiable output"
        ):
            jax.grad(wrapped)(1.0, 2.0)
        with self.assertRaisesRegex(
            jaxlib.xla_extension.XlaRuntimeError, "At least one differentiable output"
        ):
            jax.value_and_grad(wrapped)(1.0, 2.0)

    @parameterized.expand(TEST_FNS_AND_ARGS)
    def test_wrapped_matches_autograd(self, autograd_fn, args):
        # Tests case where all arguments can be differentiated
        # with respect to, and all outputs are differentiable.
        expected_outputs = autograd_fn(*args)
        output_shape_dtypes = utils.to_jax(expected_outputs)

        wrapped = wrapper.wrap_for_jax(
            autograd_fn,
            output_shape_dtypes,
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
        def autograd_fn(x, y, int_arg1, int_arg2):
            return (
                npa.sum(x**2 + y * int_arg1),
                x - y + 2,
                int_arg1 * int_arg2,
            )

        args = (0.3 + 2.2j, -11.0 + 0.0j, 3, 5)
        expected_outputs = ((0.3 + 2.2j) ** 2 - 11.0 * 3, 13.3 + 2.2j, 15)

        output_shape_dtypes = utils.to_jax(expected_outputs)
        wrapped = wrapper.wrap_for_jax(
            autograd_fn,
            output_shape_dtypes,
            nondiff_argnums=(2, 3),
            nondiff_outputnums=2,
        )

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
        expected_aux = 15
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

    def test_vmap_wrapped_matches_autograd(self):
        def autograd_fn(x, y, int_arg1, int_arg2):
            return (
                npa.sum(x**2 + y * int_arg1),
                x - y + 2,
                int_arg1 * int_arg2,
            )

        def jax_fn(x, y, int_arg1, int_arg2):
            return (
                jnp.sum(x**2 + y * int_arg1),
                x - y + 2,
                int_arg1 * int_arg2,
            )

        args = (0.3 + 2.2j, -11.0 + 0.0j, 3, 5)
        batch_args = (
            jnp.arange(10) * args[0],
            jnp.arange(10) * args[1],
            jnp.arange(10, dtype=int) * args[2],
            jnp.arange(10, dtype=int) * args[3],
        )

        # Tests case where all arguments can be differentiated
        # with respect to, and all outputs are differentiable.
        output_shape_dtypes = utils.to_jax(autograd_fn(*args))
        wrapped = wrapper.wrap_for_jax(
            autograd_fn,
            output_shape_dtypes,
            nondiff_argnums=(2, 3),
            nondiff_outputnums=(2,),
        )

        for v, ev in zip(
            jax.tree_util.tree_leaves(jax.vmap(wrapped)(*batch_args)),
            jax.tree_util.tree_leaves(jax.vmap(jax_fn)(*batch_args)),
        ):
            onp.testing.assert_allclose(v, ev)

        def wrapped_scalar_fn(*args):
            outputs = wrapped(*args)
            outputs = jax.tree_util.tree_leaves(outputs)
            return jnp.sum(jnp.asarray([jnp.sum(jnp.abs(o) ** 2) for o in outputs]))

        def jax_scalar_fn(*args):
            outputs = jax_fn(*args)
            outputs = jax.tree_util.tree_leaves(outputs)
            return jnp.sum(jnp.asarray([jnp.sum(jnp.abs(o) ** 2) for o in outputs]))

        expected_grad = jax.vmap(jax.grad(jax_scalar_fn, argnums=(0, 1)))(*batch_args)
        grad = jax.vmap(jax.grad(wrapped_scalar_fn, argnums=(0, 1)))(*batch_args)

        for g, eg in zip(
            jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(expected_grad)
        ):
            onp.testing.assert_allclose(g, eg, rtol=1e-6)

    def test_double_vmap(self):
        def autograd_fn(x, y, int_arg1, int_arg2):
            return (
                npa.sum(x**2 + y * int_arg1),
                x - y + 2,
                int_arg1 * int_arg2,
            )

        def jax_fn(x, y, int_arg1, int_arg2):
            return (
                jnp.sum(x**2 + y * int_arg1),
                x - y + 2,
                int_arg1 * int_arg2,
            )

        args = (0.3 + 2.2j, -11.0 + 0.0j, 3, 5)
        batch_args = (
            jnp.arange(20).reshape(4, 5) * args[0],
            jnp.arange(20).reshape(4, 5) * args[1],
            jnp.arange(20, dtype=int).reshape(4, 5) * args[2],
            jnp.arange(20, dtype=int).reshape(4, 5) * args[3],
        )

        # Tests case where all arguments can be differentiated
        # with respect to, and all outputs are differentiable.
        output_shape_dtypes = utils.to_jax(autograd_fn(*args))
        wrapped = wrapper.wrap_for_jax(
            autograd_fn,
            output_shape_dtypes,
            nondiff_argnums=(2, 3),
            nondiff_outputnums=(2,),
        )

        for v, ev in zip(
            jax.tree_util.tree_leaves(jax.vmap(jax.vmap(wrapped))(*batch_args)),
            jax.tree_util.tree_leaves(jax.vmap(jax.vmap(jax_fn))(*batch_args)),
        ):
            onp.testing.assert_allclose(v, ev)

        def wrapped_scalar_fn(*args):
            outputs = wrapped(*args)
            outputs = jax.tree_util.tree_leaves(outputs)
            return jnp.sum(jnp.asarray([jnp.sum(jnp.abs(o) ** 2) for o in outputs]))

        def jax_scalar_fn(*args):
            outputs = jax_fn(*args)
            outputs = jax.tree_util.tree_leaves(outputs)
            return jnp.sum(jnp.asarray([jnp.sum(jnp.abs(o) ** 2) for o in outputs]))

        expected_grad = jax.vmap(jax.vmap(jax.grad(jax_scalar_fn, argnums=(0, 1))))(
            *batch_args
        )
        grad = jax.vmap(jax.vmap(jax.grad(wrapped_scalar_fn, argnums=(0, 1))))(
            *batch_args
        )

        for g, eg in zip(
            jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(expected_grad)
        ):
            onp.testing.assert_allclose(g, eg, rtol=1e-6)

    def test_jacrev(self):
        def autograd_fn(x, y):
            return {"a": npa.cos(x**2) / y, "b": npa.sum(x) * y}

        def jax_fn(x, y):
            return {"a": jnp.cos(x**2) / y, "b": jnp.sum(x) * y}

        result_shape_dtypes = {"a": jnp.ones((5,)), "b": jnp.ones((5,))}
        wrapped = wrapper.wrap_for_jax(autograd_fn, result_shape_dtypes)

        x_batch = jnp.arange(30, dtype=float).reshape(6, 5)
        y_batch = jnp.arange(30, 60, dtype=float).reshape(6, 5)

        expected = jax.vmap(jax.jacrev(jax_fn))(x_batch, y_batch)
        result = jax.vmap(jax.jacrev(wrapped))(x_batch, y_batch)

        for g, eg in zip(
            jax.tree_util.tree_leaves(result), jax.tree_util.tree_leaves(expected)
        ):
            onp.testing.assert_allclose(g, eg, rtol=1e-6)
