"""
LDP Mechanisms for bounded rating domains.

Provides four Local Differential Privacy mechanisms for rating perturbation,
as evaluated in:

    Neera & Franca (2026). The Hidden Cost of Clipping: Privacy Mechanisms
    and Bounded Domain Bias in Recommender Systems. ESORICS 2026.

Usage:
    from mechanisms import (
        clipped_laplace_mechanism,
        bounded_laplace_mechanism,
        clipped_gaussian_mechanism,
        piecewise_mechanism,
    )
"""

from .clipped_laplace import clipped_laplace_mechanism, clipped_laplace_mechanism_batch
from .bounded_laplace import bounded_laplace_mechanism, bounded_laplace_mechanism_batch
from .clipped_gaussian import clipped_gaussian_mechanism, clipped_gaussian_mechanism_batch
from .piecewise import piecewise_mechanism, piecewise_mechanism_batch

__all__ = [
    "clipped_laplace_mechanism",
    "clipped_laplace_mechanism_batch",
    "bounded_laplace_mechanism",
    "bounded_laplace_mechanism_batch",
    "clipped_gaussian_mechanism",
    "clipped_gaussian_mechanism_batch",
    "piecewise_mechanism",
    "piecewise_mechanism_batch",
]
