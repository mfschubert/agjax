"""Defines a jax wrapper for autograd-differentiable functions."""

import functools
from concurrent import futures
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import autograd
import autograd.numpy as npa
import jax
import jax.numpy as jnp
import numpy as onp
from jax import tree_util

PyTree = Any


def wrap_for_jax(
    fn: Callable[[Any], PyTree],
    nondiff_argnums: Union[int, Tuple[int, ...]] = (),
    nondiff_outputnums: Union[int, Tuple[int, ...]] = (),
    enable_jac: bool = False,
    max_workers: Optional[int] = None,
) -> Callable[[Any], PyTree]:
    """Wraps `fn` so that it can be differentiated by jax.

    Arguments should be jax types, and are converted to numpy arrays prior
    to calling the underlying autograd-differentiable `fn`. Optionally,
    nondifferentiable arguments (i.e. those which cannot be differentiated
    with respect to) may be specified; these are passed to `fn` unchanged.

    Similarly, differentiable outputs are converted to jax types; some
    outputs can be identified as non-differentiable, which are returned
    unchanged.

    Args:
        fn: The autograd-differentiable function.
        nondiff_argnums: The arguments that cannot be differentiated with
            respect to. These are passed to `fn` unchanged.
        nondiff_outputnums: The outputs that cannot be differentiated.
            These are returned exactly as returned by `fn`.
        enable_jac: Determines the implementation of the wrapped function.
            When `True`, the wrapped function can be used with `jax.jacrev`
            and `jax.jacfwd`. This may come with increased computational cost.
        max_workers: The maximum number of workers used in constructing the
            linearization of `fn` when `enable_jac` is `True`. Does nothing
            when `enable_jac` is `False`.

    Returns:
        The wrapped function.
    """
    _nondiff_argnums, _ = _ensure_tuple(nondiff_argnums)
    _nondiff_outputnums, _ = _ensure_tuple(nondiff_outputnums)
    del nondiff_argnums, nondiff_outputnums

    split_args_fn = functools.partial(_split, idx=_nondiff_argnums)
    merge_args_fn = functools.partial(_merge, idx=_nondiff_argnums)
    split_outputs_fn = functools.partial(_split, idx=_nondiff_outputnums)
    merge_outputs_fn = functools.partial(_merge, idx=_nondiff_outputnums)

    @functools.partial(jax.custom_vjp, nondiff_argnums=_nondiff_argnums)
    def _fn(*args_jax: Any) -> PyTree:
        # Arguments that can be differentiated with respect to are jax arrays, and
        # must be converged to numpy. Extract these, convert to numpy, and remerge.
        nondiff_args, diff_args = split_args_fn(args_jax)
        args = merge_args_fn(nondiff_args, _to_numpy(diff_args))
        outputs = fn(*args)
        _validate_nondiff_outputnums_for_outputs(_nondiff_outputnums, outputs)
        # Convert differentiable outputs to jax arrays.
        outputs, is_tuple_outputs = _ensure_tuple(outputs)
        nondiff_outputs, diff_outputs = split_outputs_fn(outputs)
        nondiff_outputs = tuple([_WrappedValue(o) for o in nondiff_outputs])
        outputs = merge_outputs_fn(nondiff_outputs, _to_jax(diff_outputs))
        return outputs if is_tuple_outputs else outputs[0]

    def _fwd_fn(*args_jax: Any) -> PyTree:
        # Convert to numpy the args that can be differentiated with respect to.
        nondiff_args, diff_args = split_args_fn(args_jax)
        diff_args = _to_numpy(diff_args)
        args = merge_args_fn(nondiff_args, diff_args)

        # Variables updated nonlocally where `fn` is evaluated.
        is_tuple_outputs: bool = None  # type: ignore[assignment]
        nondiff_outputs: Tuple[PyTree, ...] = None  # type: ignore[assignment]
        diff_outputs_treedef: tree_util.PyTreeDef = None

        def _tuple_fn(*args: Any) -> autograd.builtins.tuple:
            nonlocal is_tuple_outputs
            nonlocal nondiff_outputs
            nonlocal diff_outputs_treedef

            outputs = fn(*args)
            _validate_nondiff_outputnums_for_outputs(_nondiff_outputnums, outputs)

            outputs, is_tuple_outputs = _ensure_tuple(outputs)
            nondiff_outputs, diff_outputs = split_outputs_fn(outputs)
            nondiff_outputs = _arraybox_to_numpy(nondiff_outputs)
            nondiff_outputs = tuple([_WrappedValue(o) for o in nondiff_outputs or []])
            diff_outputs_leaves, diff_outputs_treedef = tree_util.tree_flatten(
                diff_outputs
            )
            return autograd.builtins.tuple(diff_outputs_leaves)

        diff_argnums = tuple(i for i in range(len(args)) if i not in _nondiff_argnums)
        tuple_vjp_fn, diff_outputs_leaves = autograd.make_vjp(
            _tuple_fn,
            argnum=diff_argnums,
        )(*args)
        diff_outputs = tree_util.tree_unflatten(
            diff_outputs_treedef, diff_outputs_leaves
        )
        outputs = merge_outputs_fn(nondiff_outputs, _to_jax(diff_outputs))
        outputs = outputs if is_tuple_outputs else outputs[0]

        def _vjp_fn(*diff_outputs: Any) -> PyTree:
            diff_outputs_leaves = tree_util.tree_leaves(diff_outputs)
            grad = tuple_vjp_fn(_to_numpy(diff_outputs_leaves))
            # Note that there is no value associated with nondifferentiable
            # arguments in the return of the vjp function.
            return _to_jax(grad)

        return outputs, tree_util.Partial(_vjp_fn)

    def _bwd_fn(*bwd_args: Any) -> PyTree:
        # The `bwd_args` consist of the nondifferentiable arguments, the
        # residual of the forward function (i.e. our `vjp_fn`), and the
        # vector for which the vector-jacobian product is sought.
        vjp_fn = bwd_args[len(_nondiff_argnums)]
        diff_outputs = bwd_args[len(_nondiff_argnums) + 1 :]
        return vjp_fn(*diff_outputs)

    _fn.defvjp(_fwd_fn, _bwd_fn)

    def _fn_with_unwrapped_outputs(*args_jax: Any) -> PyTree:
        _validate_idx_for_sequence_len(_nondiff_argnums, len(args_jax))
        # Wrapped version of our function with custom vjp, which unpacks the
        # wrapped values associated with nondifferentiable outputs.
        outputs = _fn(*args_jax)
        if not isinstance(outputs, tuple):
            return outputs
        return tuple([o.value if isinstance(o, _WrappedValue) else o for o in outputs])

    if not enable_jac:
        return _fn_with_unwrapped_outputs

    # If `enable_jac` is `True`, use the Jacobian-compatible wrapper.

    def _construct_linearization(*args_jax: Any) -> Callable[[Any], PyTree]:
        nondiff_args, diff_args = split_args_fn(args_jax)
        diff_args_constant = jax.lax.stop_gradient(diff_args)
        del diff_args, args_jax

        args_jax_constant = merge_args_fn(nondiff_args, diff_args_constant)
        outputs, _vjp_fn = _fwd_fn(*args_jax_constant)

        outputs, is_tuple_outputs = _ensure_tuple(outputs)
        nondiff_outputs, diff_outputs_constant = split_outputs_fn(outputs)

        # Create one-hot pytrees with the structure of `diff_outputs`, and
        # evaluate the vector-Jacobian product for each of these one-hot pytrees.
        one_hot_outputs = one_hot_like(diff_outputs_constant)
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            one_hot_vjps = executor.map(lambda o: _vjp_fn(*o), one_hot_outputs)  # type: ignore[no-any-return]

        def _linear_fn(*args_jax: PyTree) -> PyTree:
            # The linear function is roughly defined by,
            #    y = outputs + jacobian @ (args - args_constant)
            _, diff_args = split_args_fn(args_jax)

            def _output_delta_value(vjp: PyTree) -> jnp.ndarray:
                value = tree_util.tree_map(
                    lambda leaf_vjp, x, x0: jnp.sum(leaf_vjp * (x - x0)),
                    vjp,
                    diff_args,
                    diff_args_constant,
                )
                return jnp.sum(jnp.asarray(tree_util.tree_leaves(value)))

            diff_outputs_delta_flat = [_output_delta_value(vjp) for vjp in one_hot_vjps]

            diff_outputs_delta = unflatten(
                flat=jnp.asarray(diff_outputs_delta_flat),
                example=diff_outputs_constant,
            )

            diff_outputs = tree_util.tree_map(
                lambda a, b: a + b, diff_outputs_constant, diff_outputs_delta
            )

            outputs = merge_outputs_fn(nondiff_outputs, diff_outputs)
            outputs = tuple(
                [o.value if isinstance(o, _WrappedValue) else o for o in outputs]
            )
            return outputs if is_tuple_outputs else outputs[0]

        return _linear_fn

    def _linearized_fn(*args_jax: Any) -> PyTree:
        # A jax-differentiable function which is the linearization of `fn` about
        # the point `args_jax`.
        return _construct_linearization(*args_jax)(*args_jax)

    return _linearized_fn


class _WrappedValue:
    """Wraps a value treated as an auxilliary quantity of a pytree node."""

    def __init__(self, value: PyTree) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"_WrappedValue({self.value})"


tree_util.register_pytree_node(
    _WrappedValue,
    flatten_func=lambda w: ((), (w.value,)),
    unflatten_func=lambda v, _: _WrappedValue(*v),
)


def _validate_nondiff_outputnums_for_outputs(
    nondiff_outputnums: Sequence[int],
    maybe_tuple_outputs: PyTree,
) -> None:
    """Validates that `nondiff_outputnums` is compatible with a `outputs`."""
    outputs_length = (
        len(maybe_tuple_outputs) if isinstance(maybe_tuple_outputs, tuple) else 1
    )
    _validate_idx_for_sequence_len(nondiff_outputnums, outputs_length)
    if outputs_length <= len(nondiff_outputnums):
        raise ValueError(
            f"At least one differentiable output is required, but got "
            f"`nondiff_outputnums` of {nondiff_outputnums} when `fn` "
            f"has {outputs_length} output(s)."
        )


def _validate_idx_for_sequence_len(idx: Sequence[int], sequence_length: int) -> None:
    """Validates that `idx` is compatible with a sequence length."""
    if not all(i in range(-sequence_length, sequence_length) for i in idx):
        raise ValueError(
            f"Found out of bounds values in `idx`, got {idx} when "
            f"`sequence_length` is {sequence_length}."
        )
    positive_idx = [i % sequence_length for i in idx]
    if len(positive_idx) != len(set(positive_idx)):
        raise ValueError(
            f"Found duplicate values in `idx`, got {idx} when "
            f"`sequence_length` is {sequence_length}."
        )


def _split(
    a: Tuple[PyTree, ...],
    idx: Tuple[int, ...],
) -> Tuple[Tuple[PyTree, ...], Tuple[PyTree, ...]]:
    """Splits the sequence `a` into two sequences."""
    _validate_idx_for_sequence_len(idx, len(a))
    return (
        tuple([a[i] for i in idx]),
        tuple([a[i] for i in range(len(a)) if i not in idx]),
    )


def _merge(
    a: Sequence[PyTree],
    b: Sequence[PyTree],
    idx: Sequence[int],
) -> Tuple[PyTree, ...]:
    """Merges the sequences `a` and `b`, undoing a `_split` operation."""
    _validate_idx_for_sequence_len(idx, len(a) + len(b))
    positive_idx = [i % (len(a) + len(b)) for i in idx]
    iter_a = iter(a)
    iter_b = iter(b)
    return tuple(
        [
            next(iter_a) if i in positive_idx else next(iter_b)
            for i in range(len(a) + len(b))
        ]
    )


def _to_jax(tree: PyTree) -> PyTree:
    """Converts leaves of a pytree to jax arrays."""
    return tree_util.tree_map(jnp.asarray, tree)


def _to_numpy(tree: PyTree) -> PyTree:
    """Converts leaves of a pytree to numpy arrays."""
    return tree_util.tree_map(onp.asarray, tree)


def _arraybox_to_numpy(tree: PyTree) -> PyTree:
    """Converts `ArrayBox` leaves of a pytree to numpy arrays."""
    return tree_util.tree_map(
        lambda x: x._value if isinstance(x, npa.numpy_boxes.ArrayBox) else x,
        tree,
    )


def _ensure_tuple(xs: PyTree) -> Tuple[PyTree, bool]:
    """Returns `(xs, True)` if `xs` is a tuple, and `((xs,), False)` otherwise."""
    is_tuple = isinstance(xs, tuple)
    return (xs if is_tuple else (xs,)), is_tuple


def one_hot_like(tree: PyTree) -> List[PyTree]:
    """Returns a tuple of one-hot pytrees matching the structure of `tree`."""
    num = onp.sum([onp.size(leaf) for leaf in jax.tree_util.tree_leaves(tree)])
    return [one_hot_like_at_idx(tree, i) for i in range(num)]


def one_hot_like_at_idx(tree: PyTree, hot_idx: int) -> PyTree:
    """Returns a pytree one-hot at `hot_idx` matching the structure of `tree`."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)

    sizes = [onp.size(leaf) for leaf in leaves]
    assert 0 <= hot_idx < onp.sum(sizes)

    leaf_idxs = onp.cumsum([0] + sizes)
    leaf_start_idxs = leaf_idxs[:-1]

    if hot_idx == 0:
        leaf_idx = 0
    else:
        leaf_start_after_hot = leaf_start_idxs > hot_idx
        leaf_idx = onp.argmax(leaf_start_after_hot) - 1

    leaf_shape = onp.shape(leaves[leaf_idx])
    leaf_array_idx = hot_idx - leaf_start_idxs[leaf_idx]
    one_hot_leaves = [onp.zeros_like(leaf) for leaf in leaves]
    one_hot_leaves[leaf_idx][onp.unravel_index(leaf_array_idx, leaf_shape)] = 1

    return jax.tree_util.tree_unflatten(treedef, one_hot_leaves)


def unflatten(flat: jnp.ndarray, example: PyTree) -> PyTree:
    """Unflattens an array into a pytree matching the structure of `example`."""
    similar_leaves, treedef = tree_util.tree_flatten(example)
    sizes = [onp.size(leaf) for leaf in similar_leaves]
    shapes = [onp.shape(leaf) for leaf in similar_leaves]

    leaves_flat = jnp.split(flat, indices_or_sections=jnp.cumsum(jnp.asarray(sizes)))
    leaves = [leaf.reshape(shape) for leaf, shape in zip(leaves_flat, shapes)]

    return tree_util.tree_unflatten(treedef, leaves)
