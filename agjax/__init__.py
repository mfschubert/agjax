"""agjax - Agjax is a jax wrapper for autograd-differentiable functions. It allows existing code built with autograd to be used with the jax framework. In particular, agjax allows an arbitrary autograd function to be differentiated using jax.grad. Several other function transformations (e.g. compilation via jax.jit) are not supported."""

__version__ = "0.2.0"
__author__ = "Martin Schubert <mfschubert@gmail.com>"

__all__ = ["wrap_for_jax"]

from agjax.wrapper import wrap_for_jax
