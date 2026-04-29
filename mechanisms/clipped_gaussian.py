"""
Clipped Gaussian Mechanism for Local Differential Privacy.

Perturbs a rating by adding Gaussian noise, then clips the result to the
valid rating range [r_min, r_max]. Provides (ε, δ)-LDP rather than the
pure ε-LDP guarantee of Laplace-based mechanisms — a weaker guarantee that,
as shown empirically in the associated paper, does not translate to utility gain.

Like the Clipped Laplace mechanism, clipping introduces systematic estimator
bias at boundary ratings (see Corollary 1 in the associated paper).

Reference:
    Dwork, C. & Roth, A. (2014). The algorithmic foundations of differential
    privacy. Foundations and Trends in TCS, 9(3-4), 211-487.
"""

import math
import numpy as np


def gaussian_sigma(
    epsilon: float,
    delta: float,
    r_min: float,
    r_max: float,
) -> float:
    """Compute the Gaussian noise standard deviation σ for (ε, δ)-LDP.

    Uses the analytic formula:
        σ = Δr * sqrt(2 * ln(1.25 / δ)) / ε

    where Δr = r_max - r_min is the L2 sensitivity of the rating function.

    Args:
        epsilon: Privacy budget (ε > 0).
        delta:   Failure probability (0 < δ < 1). Typically set to 1/n where
                 n is the number of users; default in the paper is 1e-5.
        r_min:   Minimum of the rating scale.
        r_max:   Maximum of the rating scale.

    Returns:
        Standard deviation σ > 0.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if r_min >= r_max:
        raise ValueError(f"r_min ({r_min}) must be less than r_max ({r_max})")

    sensitivity = r_max - r_min
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def clipped_gaussian_mechanism(
    r: float,
    epsilon: float,
    delta: float,
    r_min: float,
    r_max: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Perturb a single rating using the Clipped Gaussian mechanism.

    Args:
        r:       True rating value. Must satisfy r_min <= r <= r_max.
        epsilon: Privacy budget (ε > 0).
        delta:   Failure probability (0 < δ < 1).
        r_min:   Minimum of the rating scale.
        r_max:   Maximum of the rating scale.
        rng:     Optional numpy random Generator for reproducibility.

    Returns:
        Perturbed rating clipped to [r_min, r_max].
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = gaussian_sigma(epsilon, delta, r_min, r_max)
    noise = rng.normal(loc=0.0, scale=sigma)
    return float(np.clip(r + noise, r_min, r_max))


def clipped_gaussian_mechanism_batch(
    ratings: np.ndarray,
    epsilon: float,
    delta: float,
    r_min: float,
    r_max: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perturb an array of ratings using the Clipped Gaussian mechanism.

    Args:
        ratings: 1-D or 2-D array of true ratings in [r_min, r_max].
        epsilon: Privacy budget (ε > 0).
        delta:   Failure probability (0 < δ < 1).
        r_min:   Minimum of the rating scale.
        r_max:   Maximum of the rating scale.
        rng:     Optional numpy random Generator for reproducibility.

    Returns:
        Array of perturbed ratings, same shape as input, clipped to
        [r_min, r_max].
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = gaussian_sigma(epsilon, delta, r_min, r_max)
    noise = rng.normal(loc=0.0, scale=sigma, size=ratings.shape)
    return np.clip(ratings + noise, r_min, r_max)
