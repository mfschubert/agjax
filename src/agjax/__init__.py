"""agjax - a jax wrapper for autograd-differentiable functions."""

__version__ = "v0.3.1"
__author__ = "Martin Schubert <mfschubert@gmail.com>"

__all__ = ["experimental", "wrap_for_jax"]

from agjax import experimental
from agjax.wrapper import wrap_for_jax
