"""Defines a jax wrapper for autograd-differentiable functions."""

from typing import Any, Callable, Sequence, Tuple, Union

import functools

import autograd
import autograd.numpy as npa
import jax
import jax.numpy as jnp
import numpy as onp

PyTree = Any


def wrap_for_jax(
    fn: Callable[[Any], Any],
    nondiff_argnums: Union[int, Tuple[int, ...]] = (),
    nondiff_outputnums: Union[int, Tuple[int, ...]] = (),
) -> Callable[[Any], Any]:
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
    def _fn(*args_jax: Any) -> Any:
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

    def _fwd_fn(*args_jax: Any) -> Any:
        # Split arguments that can be differentiated with respect to, convert to
        # numpy and flatten.
        nondiff_args, diff_args = split_args_fn(args_jax)
        diff_args_flat, unflatten_diff_args_fn = _flatten(_to_numpy(diff_args))

        # Variables updated nonlocally where `fn` is evaluated.
        is_tuple_outputs = None
        nondiff_outputs = None
        unflatten_outputs_fn = None

        def _flat_fn(diff_args_flat: onp.ndarray) -> onp.ndarray:
            nonlocal is_tuple_outputs
            nonlocal nondiff_outputs
            nonlocal unflatten_outputs_fn

            diff_args = unflatten_diff_args_fn(diff_args_flat)
            args = merge_args_fn(nondiff_args, diff_args)
            outputs = fn(*args)
            _validate_nondiff_outputnums_for_outputs(_nondiff_outputnums, outputs)

            outputs, is_tuple_outputs = _ensure_tuple(outputs)
            nondiff_outputs, diff_outputs = split_outputs_fn(outputs)
            nondiff_outputs = _arraybox_to_numpy(nondiff_outputs)
            nondiff_outputs = tuple([_WrappedValue(o) for o in nondiff_outputs or []])
            diff_outputs_flat, unflatten_outputs_fn = _flatten(diff_outputs)
            return diff_outputs_flat

        flat_vjp_fn, diff_outputs_flat = autograd.make_vjp(_flat_fn)(diff_args_flat)
        diff_outputs = unflatten_outputs_fn(diff_outputs_flat)  # type: ignore[misc]
        outputs = merge_outputs_fn(nondiff_outputs, _to_jax(diff_outputs))
        outputs = outputs if is_tuple_outputs else outputs[0]

        def _vjp_fn(*diff_outputs: Any) -> Any:
            diff_outputs_flat, _ = _flatten(_to_numpy(diff_outputs))
            grad_flat = flat_vjp_fn(onp.asarray(diff_outputs_flat))
            grad = unflatten_diff_args_fn(grad_flat)
            # Note that there is no value associated with nondifferentiable
            # arguments in the return of the vjp function.
            return _to_jax(grad)

        return outputs, jax.tree_util.Partial(_vjp_fn)

    def _bwd_fn(*bwd_args: Any) -> Any:
        # The `bwd_args` consist of the nondifferentiable arguments, the
        # residual of the forward function (i.e. our `vjp_fn`), and the
        # vector for which the vector-jacobian product is sought.
        vjp_fn = bwd_args[len(_nondiff_argnums)]
        diff_outputs = bwd_args[len(_nondiff_argnums) + 1 :]
        return vjp_fn(*diff_outputs)

    _fn.defvjp(_fwd_fn, _bwd_fn)

    def _fn_with_unwrapped_outputs(*args_jax: Any) -> Any:
        _validate_idx_for_sequence_len(_nondiff_argnums, len(args_jax))
        # Wrapped version of our function with custom vjp, which unpacks the
        # wrapped values associated with nondifferentiable outputs.
        outputs = _fn(*args_jax)
        if not isinstance(outputs, tuple):
            return outputs
        return tuple([o.value if isinstance(o, _WrappedValue) else o for o in outputs])

    return _fn_with_unwrapped_outputs


class _WrappedValue:
    """Wraps a value treated as an auxilliary quantity of a pytree node."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"_WrappedValue({self.value})"


jax.tree_util.register_pytree_node(
    _WrappedValue,
    flatten_func=lambda w: ((), (w.value,)),
    unflatten_func=lambda v, _: _WrappedValue(*v),
)


def _validate_nondiff_outputnums_for_outputs(
    nondiff_outputnums: Sequence[int],
    maybe_tuple_outputs: Any,
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
    a: Tuple[Any, ...],
    idx: Tuple[int, ...],
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Splits the sequence `a` into two sequences."""
    _validate_idx_for_sequence_len(idx, len(a))
    return (
        tuple([a[i] for i in idx]),
        tuple([a[i] for i in range(len(a)) if i not in idx]),
    )


def _merge(
    a: Sequence[Any],
    b: Sequence[Any],
    idx: Sequence[int],
) -> Tuple[Any, ...]:
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


def _flatten(
    tree: PyTree,
) -> Tuple[onp.ndarray, Callable[[onp.ndarray], PyTree]]:
    """Returns a pytree into a single numpy array, and an `unflatten_fn`."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    flattened = npa.concatenate([leaf.flatten() for leaf in leaves])

    sizes = [leaf.size for leaf in leaves]
    shapes = [leaf.shape for leaf in leaves]

    def unflatten_fn(flat: onp.ndarray) -> PyTree:
        flat_leaves = npa.split(flat, onp.cumsum(sizes))
        leaves = [leaf.reshape(s) for leaf, s in zip(flat_leaves, shapes)]
        return jax.tree_util.tree_unflatten(treedef, leaves)

    return flattened, unflatten_fn


def _to_jax(tree: PyTree) -> PyTree:
    """Converts leaves of a pytree to jax arrays."""
    return jax.tree_util.tree_map(jnp.asarray, tree)


def _to_numpy(tree: PyTree) -> PyTree:
    """Converts leaves of a pytree to numpy arrays."""
    return jax.tree_util.tree_map(onp.asarray, tree)


def _arraybox_to_numpy(tree: PyTree) -> PyTree:
    """Converts `ArrayBox` leaves of a pytree to numpy arrays."""
    return jax.tree_util.tree_map(
        lambda x: x._value if isinstance(x, npa.numpy_boxes.ArrayBox) else x,
        tree,
    )


def _ensure_tuple(xs: Any) -> Tuple[Any, bool]:
    """Returns `(xs, True)` if `xs` is a tuple, and `((xs,), False)` otherwise."""
    is_tuple = isinstance(xs, tuple)
    return (xs if is_tuple else (xs,)), is_tuple
