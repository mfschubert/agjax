"""Defines a jax wrapper for autograd-differentiable functions."""

import functools
from typing import Any, Callable, Tuple, Union

import autograd  # type: ignore[import-untyped]
import jax
import numpy as onp
from jax import tree_util

from agjax import utils


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
    _nondiff_argnums, _ = utils.ensure_tuple(nondiff_argnums)
    _nondiff_outputnums, _ = utils.ensure_tuple(nondiff_outputnums)
    del nondiff_argnums, nondiff_outputnums

    split_args_fn = functools.partial(utils.split, idx=_nondiff_argnums)
    merge_args_fn = functools.partial(utils.merge, idx=_nondiff_argnums)
    split_outputs_fn = functools.partial(utils.split, idx=_nondiff_outputnums)
    merge_outputs_fn = functools.partial(utils.merge, idx=_nondiff_outputnums)

    @functools.partial(jax.custom_vjp, nondiff_argnums=_nondiff_argnums)
    def _fn(*args_jax: Any) -> Any:
        # Arguments that can be differentiated with respect to are jax arrays, and
        # must be converged to numpy. Extract these, convert to numpy, and remerge.
        nondiff_args, diff_args = split_args_fn(args_jax)
        args = merge_args_fn(nondiff_args, utils.to_numpy(diff_args))
        outputs = fn(*args)
        utils.validate_nondiff_outputnums_for_outputs(_nondiff_outputnums, outputs)
        # Convert differentiable outputs to jax arrays.
        outputs, is_tuple_outputs = utils.ensure_tuple(outputs)
        nondiff_outputs, diff_outputs = split_outputs_fn(outputs)
        nondiff_outputs = tuple([utils.WrappedValue(o) for o in nondiff_outputs])
        outputs = merge_outputs_fn(nondiff_outputs, utils.to_jax(diff_outputs))
        return outputs if is_tuple_outputs else outputs[0]

    def _fwd_fn(*args_jax: Any) -> Any:
        # Convert to numpy the args that can be differentiated with respect to.
        nondiff_args, diff_args = split_args_fn(args_jax)
        args = merge_args_fn(nondiff_args, utils.to_numpy(diff_args))

        # Variables updated nonlocally where `fn` is evaluated.
        is_tuple_outputs: bool = None  # type: ignore[assignment]
        nondiff_outputs: Tuple[Any, ...] = None  # type: ignore[assignment]
        diff_outputs_treedef: tree_util.PyTreeDef = None  # type: ignore[assignment]

        def _tuple_fn(*args: Any) -> onp.ndarray:
            nonlocal is_tuple_outputs
            nonlocal nondiff_outputs
            nonlocal diff_outputs_treedef

            outputs = fn(*args)
            utils.validate_nondiff_outputnums_for_outputs(_nondiff_outputnums, outputs)

            outputs, is_tuple_outputs = utils.ensure_tuple(outputs)
            nondiff_outputs, diff_outputs = split_outputs_fn(outputs)
            nondiff_outputs = utils.arraybox_to_numpy(nondiff_outputs)
            nondiff_outputs = tuple(
                [utils.WrappedValue(o) for o in nondiff_outputs or []]
            )
            diff_outputs_leaves, diff_outputs_treedef = tree_util.tree_flatten(
                diff_outputs
            )
            return autograd.builtins.tuple(tuple(diff_outputs_leaves))

        diff_argnums = tuple(i for i in range(len(args)) if i not in _nondiff_argnums)
        tuple_vjp_fn, diff_outputs_leaves = autograd.make_vjp(
            _tuple_fn, argnum=diff_argnums
        )(*args)
        diff_outputs = tree_util.tree_unflatten(
            diff_outputs_treedef, diff_outputs_leaves
        )
        outputs = merge_outputs_fn(nondiff_outputs, utils.to_jax(diff_outputs))
        outputs = outputs if is_tuple_outputs else outputs[0]

        def _vjp_fn(*diff_outputs: Any) -> Any:
            diff_outputs_leaves = tree_util.tree_leaves(diff_outputs)
            grad = tuple_vjp_fn(utils.to_numpy(diff_outputs_leaves))
            # Note that there is no value associated with nondifferentiable
            # arguments in the return of the vjp function.
            return utils.to_jax(grad)

        return outputs, tree_util.Partial(_vjp_fn)

    def _bwd_fn(*bwd_args: Any) -> Any:
        # The `bwd_args` consist of the nondifferentiable arguments, the
        # residual of the forward function (i.e. our `vjp_fn`), and the
        # vector for which the vector-jacobian product is sought.
        vjp_fn = bwd_args[len(_nondiff_argnums)]
        diff_outputs = bwd_args[len(_nondiff_argnums) + 1 :]
        return vjp_fn(*diff_outputs)

    _fn.defvjp(_fwd_fn, _bwd_fn)

    def _fn_with_unwrapped_outputs(*args_jax: Any) -> Any:
        utils.validate_idx_for_sequence_len(_nondiff_argnums, len(args_jax))
        # Wrapped version of our function with custom vjp, which unpacks the
        # wrapped values associated with nondifferentiable outputs.
        outputs = _fn(*args_jax)
        if not isinstance(outputs, tuple):
            return outputs
        return tuple(
            [o.value if isinstance(o, utils.WrappedValue) else o for o in outputs]
        )

    return _fn_with_unwrapped_outputs
