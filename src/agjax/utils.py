"""Defines a utility functions for jax-autograd wrappers."""

from typing import Any, Sequence, Tuple

import autograd.numpy as npa  # type: ignore[import-untyped]
import jax
import jax.numpy as jnp
import numpy as onp

PyTree = Any


class WrappedValue:
    """Wraps a value treated as an auxilliary quantity of a pytree node."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"_WrappedValue({self.value})"


jax.tree_util.register_pytree_node(
    WrappedValue,
    flatten_func=lambda w: ((), (w.value,)),
    unflatten_func=lambda v, _: WrappedValue(*v),
)


def validate_nondiff_outputnums_for_outputs(
    nondiff_outputnums: Sequence[int],
    maybe_tuple_outputs: Any,
) -> None:
    """Validates that `nondiff_outputnums` is compatible with a `outputs`."""
    outputs_length = (
        len(maybe_tuple_outputs) if isinstance(maybe_tuple_outputs, tuple) else 1
    )
    validate_idx_for_sequence_len(nondiff_outputnums, outputs_length)
    if outputs_length <= len(nondiff_outputnums):
        raise ValueError(
            f"At least one differentiable output is required, but got "
            f"`nondiff_outputnums` of {nondiff_outputnums} when `fn` "
            f"has {outputs_length} output(s)."
        )


def validate_nondiff_argnums_for_args(
    nondiff_argnums: Sequence[int],
    args: Tuple[Any, ...],
) -> None:
    """Validates that `nondiff_argnums` is compatible with a `args`."""
    validate_idx_for_sequence_len(nondiff_argnums, len(args))
    if len(args) <= len(nondiff_argnums):
        raise ValueError(
            f"At least argument must be differentiated with respect to, but got "
            f"`nondiff_argnums` of {nondiff_argnums} when `fn` has {len(args)} "
            f"arguments(s)."
        )


def validate_idx_for_sequence_len(idx: Sequence[int], sequence_length: int) -> None:
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


def split(
    a: Tuple[Any, ...],
    idx: Tuple[int, ...],
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Splits the sequence `a` into two sequences."""
    validate_idx_for_sequence_len(idx, len(a))
    return (
        tuple([a[i] for i in idx]),
        tuple([a[i] for i in range(len(a)) if i not in idx]),
    )


def merge(
    a: Sequence[Any],
    b: Sequence[Any],
    idx: Sequence[int],
) -> Tuple[Any, ...]:
    """Merges the sequences `a` and `b`, undoing a `_split` operation."""
    validate_idx_for_sequence_len(idx, len(a) + len(b))
    positive_idx = [i % (len(a) + len(b)) for i in idx]
    iter_a = iter(a)
    iter_b = iter(b)
    return tuple(
        [
            next(iter_a) if i in positive_idx else next(iter_b)
            for i in range(len(a) + len(b))
        ]
    )


def to_jax(tree: PyTree) -> PyTree:
    """Converts leaves of a pytree to jax arrays."""
    return jax.tree_util.tree_map(jnp.asarray, tree)


def to_numpy(tree: PyTree) -> PyTree:
    """Converts leaves of a pytree to numpy arrays."""
    return jax.tree_util.tree_map(onp.asarray, tree)


def arraybox_to_numpy(tree: PyTree) -> PyTree:
    """Converts `ArrayBox` leaves of a pytree to numpy arrays."""
    return jax.tree_util.tree_map(
        lambda x: x._value if isinstance(x, npa.numpy_boxes.ArrayBox) else x,
        tree,
    )


def ensure_tuple(xs: Any) -> Tuple[Any, bool]:
    """Returns `(xs, True)` if `xs` is a tuple, and `((xs,), False)` otherwise."""
    is_tuple = isinstance(xs, tuple)
    return (xs if is_tuple else (xs,)), is_tuple
