"""Simulated datasets for examples and tests."""

from dataclasses import dataclass

import numpy as np

from glm.families import ExponentialFamily
from glm.types import CoefficientVector, DesignMatrix, ResponseVector


# Frozen, so a dataset's fields can't be reassigned by accident (though the
# arrays themselves stay writable); eq=False, as comparing the arrays with ==
# would raise rather than return a bool.
@dataclass(frozen=True, eq=False)
class SimulatedRegression:
    """A simulated regression dataset, with the coefficients that generated
    it."""

    X: DesignMatrix
    y: ResponseVector
    coef: CoefficientVector


def make_regression(
    family: ExponentialFamily,
    *,
    n_samples: int = 1000,
    n_uncorrelated: int = 5,
    n_correlated: int = 0,
    n_zero_coefficients: int = 0,
    coef_range: tuple[float, float] = (-1.0, 1.0),
    dispersion: float = 1.0,
    intercept: bool = True,
    rng: np.random.Generator | None = None,
) -> SimulatedRegression:
    """Simulate a regression dataset from a GLM.

    The design matrix has an intercept column (unless intercept=False),
    followed by n_uncorrelated independent standard normal predictors, and
    then n_correlated predictors with a random correlation structure, each
    standardized to unit variance. Coefficients are drawn uniformly from
    coef_range, then n_zero_coefficients of them, chosen at random, are set to
    zero. The response is drawn from the family's conditional distribution,

        y | X ~ family(μ = family.inv_link(X @ coef), dispersion)

    Parameters
    ----------

    family: ExponentialFamily
        The family of the response's conditional distribution.

    n_samples: positive integer
        The number of observations.

    n_uncorrelated, n_correlated: non-negative integers
        The number of independent, and correlated, predictors.

    n_zero_coefficients: non-negative integer
        How many coefficients to set to zero, at most the number of columns.

    coef_range: (low, high)
        The range the coefficients are drawn from.

    dispersion: positive float
        The dispersion of the response's distribution, e.g. the variance for
        a Gaussian family. Ignored by families without a dispersion parameter.

    intercept: bool
        Whether the design matrix begins with an intercept column.

    rng: numpy Generator, optional
        The random number generator. Without one, draws come from numpy's
        global random state, so np.random.seed makes them reproducible.

    Returns
    -------

    dataset: SimulatedRegression
        The design matrix X, the response y, and the coefficients coef that
        generated it.
    """
    for name, value, minimum in [
        ("n_samples", n_samples, 1),
        ("n_uncorrelated", n_uncorrelated, 0),
        ("n_correlated", n_correlated, 0),
        ("n_zero_coefficients", n_zero_coefficients, 0),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    n_columns = int(intercept) + n_uncorrelated + n_correlated
    if n_columns == 0:
        raise ValueError("The design matrix must have at least one column.")
    if n_zero_coefficients > n_columns:
        raise ValueError(
            f"n_zero_coefficients is {n_zero_coefficients}, but there are only "
            f"{n_columns} coefficients."
        )
    if dispersion <= 0:
        raise ValueError(f"dispersion must be positive, got {dispersion}.")
    if rng is None:
        # As for ExponentialFamily.sample: seed from numpy's global state.
        rng = np.random.default_rng(np.random.randint(np.iinfo(np.int64).max))

    columns = []
    if intercept:
        columns.append(np.ones((n_samples, 1)))
    columns.append(rng.normal(size=(n_samples, n_uncorrelated)))
    if n_correlated > 0:
        # Mixing independent normals with a matrix A of standard normal
        # entries gives predictors with covariance A'A, a random correlation
        # structure.
        mixing = rng.normal(size=(n_correlated, n_correlated))
        correlated = rng.normal(size=(n_samples, n_correlated)) @ mixing
        columns.append((correlated - correlated.mean(axis=0)) / correlated.std(axis=0))
    X: DesignMatrix = np.column_stack(columns)

    coef: CoefficientVector = rng.uniform(*coef_range, size=n_columns)
    zeroed = rng.choice(n_columns, size=n_zero_coefficients, replace=False)
    coef[zeroed] = 0.0
    y = family.sample(family.inv_link(X @ coef), dispersion, rng=rng)
    return SimulatedRegression(X, y, coef)
