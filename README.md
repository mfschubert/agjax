# Overview v0.2.0
Agjax is a jax wrapper for autograd-differentiable functions. It allows existing code built with autograd to be used with the jax framework. In particular, agjax allows an arbitrary autograd function to be differentiated using `jax.grad`. Several other function transformations (e.g. compilation via `jax.jit`) are not supported.

## Installation
Agjax is not yet a published package, but can be installed by cloning this repository and running `pip install -e agjax`.
or `make install`

## Usage
Basic usage is as follows:
```python
@agjax.wrap_for_jax
def fn(x, y):
  return x * npa.cos(y)

grad = jax.grad(fn, argnums=(0,  1))(1.0, 0.0)
print(f"grad = {grad}")
```
```
grad = (Array(1., dtype=float32), Array(0., dtype=float32))
```

Agjax is intended to be quite general, and can support functions with multiple inputs and outputs as well as functions that have nondifferentiable outputs or arguments that cannot be differentiated with respect to. These should be specified using `nondiff_argnums` and `nondiff_outputnums` arguments to `wrap_for_jax`.

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
