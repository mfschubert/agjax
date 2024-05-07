# Agjax -- jax wrapper for autograd-differentiable functions.
`v0.3.1`

Agjax allows existing code built with autograd to be used with the jax framework.

In particular, `agjax.wrap_for_jax` allows arbitrary autograd functions ot be differentiated using `jax.grad`. Several other function transformations (e.g. compilation via `jax.jit`) are not supported.

Meanwhile, `agjax.experimental.wrap_for_jax` supports `grad`, `jit`, `vmap`, and `jacrev`. However, it depends on certain under-the-hood behavior by jax, which is not guaranteed to remain unchanged. It also is more restrictive in terms of the valid function signatures of functions to be wrapped: all arguments and outputs must be convertible to valid jax types. (`agjax.wrap_for_jax` also supports non-jax inputs and outputs, e.g. strings.)

## Installation
```
pip install agjax
```

## Usage
Basic usage is as follows:
```python
@agjax.wrap_for_jax
def fn(x, y):
  return x * npa.cos(y)

jax.grad(fn, argnums=(0,  1))(1.0, 0.0)

# (Array(1., dtype=float32), Array(0., dtype=float32))
```

The experimental wrapper is similar, but requires that the function outputs and datatypes be specified, simiilar to `jax.pure_callback`.
```python
wrapped_fn = agjax.experimental.wrap_for_jax(
  lambda x, y: x * npa.cos(y),
  result_shape_dtypes=jnp.ones((5,)),
)

jax.jacrev(wrapped_fn, argnums=0)(jnp.arange(5, dtype=float), jnp.arange(5, 10, dtype=float))

# [[ 0.28366217  0.          0.          0.          0.        ]
#  [ 0.          0.96017027  0.          0.          0.        ]
#  [ 0.          0.          0.75390226  0.          0.        ]
#  [ 0.          0.          0.         -0.14550003  0.        ]
#  [ 0.          0.          0.          0.         -0.91113025]]
```

Agjax wrappers are intended to be quite general, and can support functions with multiple inputs and outputs as well as functions that have nondifferentiable outputs or arguments that cannot be differentiated with respect to. These should be specified using `nondiff_argnums` and `nondiff_outputnums` arguments. In the experimental wrapper, these must still be jax-convertible types, while in the standard wrapper they may have arbitrary types.

```python
@functools.partial(
  agjax.wrap_for_jax, nondiff_argnums=(2,), nondiff_outputnums=(1,)
)
def fn(x, y, string_arg):
  return x * npa.cos(y), string_arg * 2

(value, aux), grad = jax.value_and_grad(
  fn, argnums=(0, 1), has_aux=True
)(1.0, 0.0, "test")

print(f"value = {value}")
print(f"  aux = {aux}")
print(f" grad = {grad}")
```
```
value = 1.0
  aux = testtest
 grad = (Array(1., dtype=float32), Array(0., dtype=float32))
```
