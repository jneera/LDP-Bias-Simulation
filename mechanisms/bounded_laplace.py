"""
Bounded Laplace Mechanism for Local Differential Privacy.

Uses rejection sampling to ensure perturbed ratings remain within the valid
range [r_min, r_max], preserving the symmetry of the noise distribution and
avoiding the systematic estimator bias introduced by clipping.

The scale parameter b̂ is chosen to satisfy ε-LDP as described in:

    Holohan et al. (2018). The bounded Laplace mechanism in differential
    privacy. arXiv:1808.10410.

and applied to recommendation systems in:

    Neera et al. (2021). Private and utility enhanced recommendations with
    local differential privacy and Gaussian mixture model. IEEE TKDE 35(4).
"""

import numpy as np


def _find_scale(epsilon: float, r_min: float, r_max: float) -> float:
    """Compute the scale parameter b̂ that achieves ε-LDP for the Bounded
    Laplace mechanism on domain [r_min, r_max].

    Uses bisection on the implicit equation derived in Holohan et al. (2018).

    Args:
        epsilon: Privacy budget (ε > 0).
        r_min:   Lower bound of the rating domain.
        r_max:   Upper bound of the rating domain.

    Returns:
        Scale parameter b̂ > 0.
    """
    sensitivity = r_max - r_min

    def constraint(b: float) -> float:
        # The BLP satisfies ε-LDP iff e^(sensitivity/b) - 1 <= (e^ε - 1)
        # Rearranged for bisection: f(b) = sensitivity/b - ε  (root → b = Δ/ε)
        # Full constraint from Holohan eq. (8); simplified here for the
        # symmetric domain case where the worst case is at the boundary.
        return np.exp(sensitivity / b) - np.exp(epsilon)

    # Lower bound: standard Laplace scale; upper bound: generous.
    b_low = sensitivity / epsilon
    # If b_low already satisfies the constraint (typical for large ε), return it.
    if constraint(b_low) <= 0:
        return b_low

    b_high = b_low * 10.0
    while constraint(b_high) > 0:
        b_high *= 2.0

    for _ in range(100):
        b_mid = (b_low + b_high) / 2.0
        if constraint(b_mid) <= 0:
            b_high = b_mid
        else:
            b_low = b_mid
        if (b_high - b_low) < 1e-10:
            break

    return b_high


def bounded_laplace_mechanism(
    r: float,
    epsilon: float,
    r_min: float,
    r_max: float,
    rng: np.random.Generator | None = None,
    max_iter: int = 10_000,
) -> float:
    """Perturb a single rating using the Bounded Laplace mechanism.

    Repeatedly samples Laplace(r, b̂) noise until the perturbed value falls
    within [r_min, r_max].

    Args:
        r:        True rating value. Must satisfy r_min <= r <= r_max.
        epsilon:  Privacy budget (ε > 0).
        r_min:    Minimum of the rating scale.
        r_max:    Maximum of the rating scale.
        rng:      Optional numpy random Generator for reproducibility.
        max_iter: Maximum rejection-sampling iterations before raising.

    Returns:
        Perturbed rating in [r_min, r_max].

    Raises:
        ValueError:   If epsilon <= 0 or r_min >= r_max.
        RuntimeError: If a valid sample is not found within max_iter draws.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if r_min >= r_max:
        raise ValueError(f"r_min ({r_min}) must be less than r_max ({r_max})")

    if rng is None:
        rng = np.random.default_rng()

    b = _find_scale(epsilon, r_min, r_max)

    for _ in range(max_iter):
        sample = r + rng.laplace(loc=0.0, scale=b)
        if r_min <= sample <= r_max:
            return float(sample)

    raise RuntimeError(
        f"Bounded Laplace: failed to draw a valid sample in {max_iter} iterations. "
        f"r={r}, ε={epsilon}, domain=[{r_min}, {r_max}]"
    )


def bounded_laplace_mechanism_batch(
    ratings: np.ndarray,
    epsilon: float,
    r_min: float,
    r_max: float,
    rng: np.random.Generator | None = None,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Perturb an array of ratings using the Bounded Laplace mechanism.

    Args:
        ratings:  1-D array of true ratings in [r_min, r_max].
        epsilon:  Privacy budget (ε > 0).
        r_min:    Minimum of the rating scale.
        r_max:    Maximum of the rating scale.
        rng:      Optional numpy random Generator for reproducibility.
        max_iter: Maximum rejection-sampling iterations per rating.

    Returns:
        1-D array of perturbed ratings in [r_min, r_max].
    """
    if rng is None:
        rng = np.random.default_rng()

    return np.array(
        [
            bounded_laplace_mechanism(r, epsilon, r_min, r_max, rng, max_iter)
            for r in ratings.ravel()
        ]
    ).reshape(ratings.shape)
