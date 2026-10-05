"""Simulated datasets for examples and tests.

make_regression returns arrays: a design matrix and response. make_frame
returns a data frame with named predictors, ready for fitting with a formula.
Both simulate from the same model: an intercept, continuous predictors, some
of them correlated, and categorical variables, with coefficients drawn at
random, and a response drawn from an exponential family.
"""

import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from glm.families import ExponentialFamily
from glm.types import (
    CategoricalLevels,
    CoefficientVector,
    DesignMatrix,
    Formula,
    MeanVector,
    Offset,
    ResponseVector,
)


# Frozen, so a dataset's fields can't be reassigned by accident (though the
# arrays themselves stay writable); eq=False, as comparing the numpy arrays with
# == would raise rather than return a bool.
@dataclass(frozen=True, eq=False)
class SimulatedRegression:
    """A simulated regression dataset, with the coefficients that generated
    it."""

    X: DesignMatrix
    y: ResponseVector
    coef: CoefficientVector


@dataclass(frozen=True, eq=False)
class SimulatedFrame:
    """A simulated data frame, with the model that generated it.

    Attributes
    ----------

    data: pd.DataFrame
        The continuous predictors x1, x2, ..., the categorical variables, an
        exposure column (if simulated with exposures), and the response y.

    formula: str
        The formula of the model that generated the data.

    offset: array, shape (n_samples, ), or None
        The log exposures, to pass to fit as the offset, if simulated with
        exposures.

    coef: pd.Series
        The true coefficients, named as a GLM fit with the formula names its
        coefficients, so that model.coef_ - coef compares an unpenalized fit's
        estimates with the truth.

    mean: array, shape (n_samples, )
        The mean each response was drawn from, including any latent noise.
    """

    data: pd.DataFrame
    formula: Formula
    offset: Offset | None
    coef: pd.Series
    mean: MeanVector


def make_regression(
    family: ExponentialFamily,
    *,
    n_samples: int = 1000,
    n_uncorrelated: int = 5,
    n_correlated: int = 0,
    correlation: float | None = None,
    categorical: Sequence[CategoricalLevels] | None = None,
    n_zero_coefficients: int = 0,
    coef_range: tuple[float, float] = (-1.0, 1.0),
    dispersion: float = 1.0,
    intercept: bool = True,
    rng: np.random.Generator | None = None,
) -> SimulatedRegression:
    """Simulate a regression dataset from a GLM, as arrays.

    The design matrix has an intercept column (unless intercept=False),
    followed by n_uncorrelated independent standard normal predictors, then
    n_correlated predictors with a random correlation structure, each
    standardized to unit variance, and then indicator columns for each
    categorical variable's levels other than its first, reference, level.
    Coefficients are drawn uniformly from coef_range, then
    n_zero_coefficients of them, chosen at random, are set to zero. The
    response is drawn from the family's conditional distribution,

        y | X ~ family(μ = family.inv_link(X @ coef), dispersion)

    Parameters
    ----------

    family: ExponentialFamily
        The family of the response's conditional distribution.

    n_samples: positive integer
        The number of observations.

    n_uncorrelated, n_correlated: non-negative integers
        The number of independent, and correlated, continuous predictors.

    correlation: float, optional
        The correlation between each pair of correlated predictors, between
        -1 / (n_correlated - 1) and 1. By default, the correlated predictors
        have a random correlation structure, which can range from almost
        uncorrelated to almost collinear.

    categorical: sequence, optional
        For each categorical variable, either its number of levels, each
        equally likely, or each level's probability.

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
    simulation = _simulate(
        family,
        n_samples=n_samples,
        n_uncorrelated=n_uncorrelated,
        n_correlated=n_correlated,
        correlation=correlation,
        categorical=list(categorical or []),
        n_zero_coefficients=n_zero_coefficients,
        coef_range=coef_range,
        dispersion=dispersion,
        intercept=intercept,
        exposure_range=None,
        latent_noise_sd=0.0,
        rng=rng,
    )
    return SimulatedRegression(simulation.X, simulation.y, simulation.coef)


def make_frame(
    family: ExponentialFamily,
    *,
    n_samples: int = 1000,
    n_uncorrelated: int = 5,
    n_correlated: int = 0,
    correlation: float | None = None,
    categorical: Mapping[str, CategoricalLevels] | None = None,
    n_zero_coefficients: int = 0,
    coef_range: tuple[float, float] = (-1.0, 1.0),
    dispersion: float = 1.0,
    exposure_range: tuple[float, float] | None = None,
    latent_noise_sd: float = 0.0,
    rng: np.random.Generator | None = None,
) -> SimulatedFrame:
    """Simulate a regression dataset from a GLM, as a data frame.

    The model is as for make_regression, always with an intercept, and with
    the continuous predictors named x1, x2, ..., the correlated ones last. A
    categorical variable's levels are named a, b, c, ..., with a the
    reference level.

    Parameters
    ----------

    categorical: mapping, optional
        Each categorical variable's name, and either its number of levels,
        each equally likely, or each level's probability.

    exposure_range: (low, high), optional
        If given, each observation gets an exposure, drawn uniformly from this
        range, and the response's mean is proportional to it: the linear
        predictor includes the offset log(exposure).

    latent_noise_sd: non-negative float
        The standard deviation of unobserved normal noise added to the linear
        predictor. Any is enough to make the data overdispersed relative to
        the family, so that the model's variance function is wrong.

    The other parameters are as for make_regression.

    Returns
    -------

    dataset: SimulatedFrame
        The data, the formula and offset of the model that generated it, its
        true coefficients, and the true means.
    """
    categorical = dict(categorical or {})
    for name in categorical:
        if name in ("y", "exposure") or _is_continuous_name(name):
            raise ValueError(f"The categorical variable name {name!r} is taken.")
    simulation = _simulate(
        family,
        n_samples=n_samples,
        n_uncorrelated=n_uncorrelated,
        n_correlated=n_correlated,
        correlation=correlation,
        categorical=list(categorical.values()),
        n_zero_coefficients=n_zero_coefficients,
        coef_range=coef_range,
        dispersion=dispersion,
        intercept=True,
        exposure_range=exposure_range,
        latent_noise_sd=latent_noise_sd,
        rng=rng,
    )

    n_continuous = n_uncorrelated + n_correlated
    continuous_names = [f"x{j}" for j in range(1, n_continuous + 1)]
    columns: dict[str, object] = dict(
        zip(continuous_names, simulation.X[:, 1 : n_continuous + 1].T)
    )
    coef_names = ["Intercept", *continuous_names]
    for name, (codes, levels) in zip(categorical, simulation.categories):
        columns[name] = np.asarray(levels)[codes]
        coef_names += [f"C({name})[T.{level}]" for level in levels[1:]]
    if simulation.exposure is not None:
        columns["exposure"] = simulation.exposure
    columns["y"] = simulation.y

    terms = continuous_names + [f"C({name})" for name in categorical]
    return SimulatedFrame(
        data=pd.DataFrame(columns),
        formula="y ~ " + (" + ".join(terms) if terms else "1"),
        offset=None if simulation.exposure is None else np.log(simulation.exposure),
        coef=pd.Series(simulation.coef, index=coef_names),
        mean=simulation.mean,
    )


@dataclass(frozen=True, eq=False)
class _Simulation:
    X: DesignMatrix
    y: ResponseVector
    coef: CoefficientVector
    mean: MeanVector
    exposure: Offset | None
    # Each categorical variable's level codes, and level names.
    categories: list[tuple[np.ndarray, list[str]]]


def _simulate(
    family: ExponentialFamily,
    *,
    n_samples: int,
    n_uncorrelated: int,
    n_correlated: int,
    correlation: float | None,
    categorical: list[CategoricalLevels],
    n_zero_coefficients: int,
    coef_range: tuple[float, float],
    dispersion: float,
    intercept: bool,
    exposure_range: tuple[float, float] | None,
    latent_noise_sd: float,
    rng: np.random.Generator | None,
) -> _Simulation:
    """Simulate from the model shared by make_regression and make_frame."""
    for name, value, minimum in [
        ("n_samples", n_samples, 1),
        ("n_uncorrelated", n_uncorrelated, 0),
        ("n_correlated", n_correlated, 0),
        ("n_zero_coefficients", n_zero_coefficients, 0),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    probabilities = [_level_probabilities(levels) for levels in categorical]
    n_columns = (
        int(intercept)
        + n_uncorrelated
        + n_correlated
        + sum(len(p) - 1 for p in probabilities)
    )
    if n_columns == 0:
        raise ValueError("The design matrix must have at least one column.")
    if n_zero_coefficients > n_columns:
        raise ValueError(
            f"n_zero_coefficients is {n_zero_coefficients}, but there are only "
            f"{n_columns} coefficients."
        )
    if dispersion <= 0:
        raise ValueError(f"dispersion must be positive, got {dispersion}.")
    if correlation is not None and n_correlated > 0:
        # The equicorrelation matrix is positive definite only in this range.
        lower = -1 / (n_correlated - 1) if n_correlated > 1 else -1.0
        if not lower < correlation < 1:
            raise ValueError(
                f"With {n_correlated} correlated predictors, correlation must "
                f"be between {lower:.3g} and 1.0, got {correlation}."
            )
    if latent_noise_sd < 0:
        raise ValueError(
            f"latent_noise_sd must be non-negative, got {latent_noise_sd}."
        )
    if exposure_range is not None and not 0 < exposure_range[0] <= exposure_range[1]:
        raise ValueError(f"exposure_range must be positive, got {exposure_range}.")
    if rng is None:
        # As for ExponentialFamily.sample: seed from numpy's global state.
        rng = np.random.default_rng(np.random.randint(np.iinfo(np.int64).max))

    columns = []
    if intercept:
        columns.append(np.ones((n_samples, 1)))
    columns.append(rng.normal(size=(n_samples, n_uncorrelated)))
    if n_correlated > 0:
        if correlation is None:
            # Mixing independent normals with a matrix A of standard normal
            # entries gives predictors with covariance A'A, a random
            # correlation structure.
            mixing = rng.normal(size=(n_correlated, n_correlated))
        else:
            # Mixing with the Cholesky factor L of the equicorrelation matrix C,
            # with ones on the diagonal and the desired correlation elsewhere,
            # gives predictors with covariance L'L = C.
            C = np.full((n_correlated, n_correlated), correlation)
            np.fill_diagonal(C, 1.0)
            mixing = np.linalg.cholesky(C).T
        correlated = rng.normal(size=(n_samples, n_correlated)) @ mixing
        columns.append((correlated - correlated.mean(axis=0)) / correlated.std(axis=0))
    categories = []
    for p in probabilities:
        codes = rng.choice(len(p), size=n_samples, p=p)
        # Reference coding: an indicator for each level but the first.
        columns.append((codes[:, None] == np.arange(1, len(p))).astype(np.float64))
        categories.append((codes, _level_names(len(p))))
    X: DesignMatrix = np.column_stack(columns)

    coef: CoefficientVector = rng.uniform(*coef_range, size=n_columns)
    zeroed = rng.choice(n_columns, size=n_zero_coefficients, replace=False)
    coef[zeroed] = 0.0
    η = X @ coef
    exposure = None
    if exposure_range is not None:
        exposure = rng.uniform(*exposure_range, size=n_samples)
        η = η + np.log(exposure)
    if latent_noise_sd > 0:
        η = η + rng.normal(scale=latent_noise_sd, size=n_samples)
    μ = family.inv_link(η)
    y = family.sample(μ, dispersion, rng=rng)
    return _Simulation(X, y, coef, μ, exposure, categories)


def _level_probabilities(levels: CategoricalLevels) -> np.ndarray:
    if isinstance(levels, int) and not isinstance(levels, bool):
        if levels < 2:
            raise ValueError(f"A categorical needs at least 2 levels, got {levels}.")
        return np.full(levels, 1 / levels)
    p = np.asarray(levels, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError(
            f"Level probabilities must be at least 2 non-negative numbers summing "
            f"to 1, got {levels}."
        )
    return p


def _level_names(n_levels: int) -> list[str]:
    """Names that sort in level order, so the first level is the reference."""
    if n_levels <= len(string.ascii_lowercase):
        return list(string.ascii_lowercase[:n_levels])
    width = len(str(n_levels))
    return [f"level{i:0{width}d}" for i in range(1, n_levels + 1)]


def _is_continuous_name(name: str) -> bool:
    return name.startswith("x") and name[1:].isdigit()
