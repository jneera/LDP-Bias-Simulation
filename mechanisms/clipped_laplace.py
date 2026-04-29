"""
Clipped Laplace Mechanism for Local Differential Privacy.

Perturbs a rating by adding Laplace noise, then clips the result to the
valid rating range [r_min, r_max]. Satisfies ε-LDP; clipping is a
post-processing step that does not affect the privacy guarantee (Dwork &
Roth, 2014). However, clipping introduces systematic estimator bias at
boundary ratings — see Proposition 1 in the associated paper.

Reference:
    Berlioz et al. (2015). Applying differential privacy to matrix
    factorization. RecSys 2015.
"""

import numpy as np


def clipped_laplace_mechanism(
    r: float,
    epsilon: float,
    r_min: float,
    r_max: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Perturb a single rating using the Clipped Laplace mechanism.

    Args:
        r:       True rating value. Must satisfy r_min <= r <= r_max.
        epsilon: Privacy budget (ε > 0). Smaller values give stronger privacy.
        r_min:   Minimum of the rating scale.
        r_max:   Maximum of the rating scale.
        rng:     Optional numpy random Generator for reproducibility.
                 If None, uses numpy's default global RNG.

    Returns:
        Perturbed rating clipped to [r_min, r_max].

    Raises:
        ValueError: If epsilon <= 0 or r_min >= r_max.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if r_min >= r_max:
        raise ValueError(f"r_min ({r_min}) must be less than r_max ({r_max})")

    if rng is None:
        rng = np.random.default_rng()

    sensitivity = r_max - r_min
    scale = sensitivity / epsilon
    noise = rng.laplace(loc=0.0, scale=scale)
    return float(np.clip(r + noise, r_min, r_max))


def clipped_laplace_mechanism_batch(
    ratings: np.ndarray,
    epsilon: float,
    r_min: float,
    r_max: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perturb an array of ratings using the Clipped Laplace mechanism.

    Args:
        ratings: 1-D or 2-D array of true ratings in [r_min, r_max].
        epsilon: Privacy budget (ε > 0).
        r_min:   Minimum of the rating scale.
        r_max:   Maximum of the rating scale.
        rng:     Optional numpy random Generator for reproducibility.

    Returns:
        Array of perturbed ratings, same shape as input, clipped to
        [r_min, r_max].
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if r_min >= r_max:
        raise ValueError(f"r_min ({r_min}) must be less than r_max ({r_max})")

    if rng is None:
        rng = np.random.default_rng()

    sensitivity = r_max - r_min
    scale = sensitivity / epsilon
    noise = rng.laplace(loc=0.0, scale=scale, size=ratings.shape)
    return np.clip(ratings + noise, r_min, r_max)
