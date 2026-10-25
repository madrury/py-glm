from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from glm.families import Bernoulli, ExponentialFamily, Gamma, Gaussian, Poisson
from glm.glm import GLM, has_converged

N = 500


@pytest.fixture
def X() -> np.ndarray:
    rng = np.random.default_rng(154)
    return np.column_stack([np.ones(N), rng.normal(size=N)])


@pytest.mark.parametrize("mean", [5.0, 1000.0, 1e6])
def test_poisson_fit_with_large_mean_response(X: np.ndarray, mean: float) -> None:
    # Starting the intercept at the mean response itself, rather than its log,
    # overflowed exp() and produced NaN coefficients for large counts.
    rng = np.random.default_rng(154)
    y = rng.poisson(mean * np.exp(0.1 * X[:, 1])).astype(np.float64)
    model = GLM(Poisson()).fit(X, y)
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, [np.log(mean), 0.1], atol=0.05)


def test_gamma_fit_with_large_mean_response(X: np.ndarray) -> None:
    rng = np.random.default_rng(154)
    mu = 500 * np.exp(0.1 * X[:, 1])
    y = rng.gamma(shape=2.0, scale=mu / 2.0)
    model = GLM(Gamma()).fit(X, y)
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, [np.log(500), 0.1], atol=0.1)


def test_poisson_fit_with_large_offset(X: np.ndarray) -> None:
    rng = np.random.default_rng(154)
    exposure = rng.uniform(1e4, 1e5, size=N)
    y = rng.poisson(exposure * np.exp(-3 + 0.1 * X[:, 1])).astype(np.float64)
    model = GLM(Poisson()).fit(X, y, offset=np.log(exposure))
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, [-3, 0.1], atol=0.05)


@pytest.mark.parametrize("kind", ["frequency_weights", "precision_weights"])
def test_zero_weight_rows_do_not_affect_fit(X: np.ndarray, kind: str) -> None:
    rng = np.random.default_rng(154)
    y = rng.poisson(np.exp(0.5 + 0.3 * X[:, 1])).astype(np.float64)
    # Extreme responses on rows with zero weight must have no effect.
    y_contaminated, weights = y.copy(), np.ones(N)
    y_contaminated[::4], weights[::4] = 1000.0, 0.0
    model = GLM(Poisson()).fit(X, y_contaminated, **{kind: weights})
    assert model.coef_ is not None
    kept = weights > 0
    subset = GLM(Poisson()).fit(X[kept], y[kept])
    assert subset.coef_ is not None
    np.testing.assert_allclose(model.coef_, subset.coef_)
    # Rows with zero weight aren't observations.
    assert model.n == kept.sum()


def test_non_finite_estimates_raise(X: np.ndarray) -> None:
    # The MLE does not exist for an all-zero binary response.
    with pytest.raises(ValueError, match="not finite"):
        GLM(Bernoulli()).fit(X, np.zeros(N))


def test_repr() -> None:
    model = GLM(Poisson(), alpha=0.5)
    assert repr(model) == "GLM(family=Poisson(), alpha=0.5, standardize='continuous')"
    # The repr shows the constructor arguments, so it is unchanged by fitting.
    X = np.column_stack([np.ones(N), np.linspace(-1, 1, N)])
    model.fit(X, np.ones(N))
    assert repr(model) == "GLM(family=Poisson(), alpha=0.5, standardize='continuous')"


@pytest.mark.parametrize("family", [Gaussian(), Poisson()])
def test_perfect_fit_converges(family: ExponentialFamily) -> None:
    # A perfect fit has zero deviance, which must not break the relative
    # convergence check.
    X = np.column_stack([np.ones(N), np.linspace(-1, 1, N)])
    β = np.array([1.0, 0.5])
    model = GLM(family).fit(X, family.inv_link(X @ β))
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, β)


RANK_DEFICIENT_DESIGNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "duplicated column": lambda X: np.column_stack([X, X[:, 1]]),
    "linear combination": lambda X: np.column_stack([X, 2 * X[:, 1] - 3 * X[:, 0]]),
    "column of zeros": lambda X: np.column_stack([X, np.zeros(N)]),
    "constant column": lambda X: np.column_stack([X, np.full(N, 5.0)]),
    "random linear combination": lambda X: np.column_stack(
        [X, X @ np.random.uniform(-1.0, 1.0, X.shape[1])]
    ),
}


@pytest.mark.parametrize(
    "make_design", RANK_DEFICIENT_DESIGNS.values(), ids=list(RANK_DEFICIENT_DESIGNS)
)
def test_rank_deficient_design_raises(
    X: np.ndarray, make_design: Callable[[np.ndarray], np.ndarray]
) -> None:
    rng = np.random.default_rng(154)
    y = rng.poisson(np.exp(0.5 + 0.3 * X[:, 1])).astype(np.float64)
    with pytest.raises(ValueError, match="rank deficient"):
        GLM(Poisson()).fit(make_design(X), y)


def test_rank_deficient_design_names_the_dependent_columns(X: np.ndarray) -> None:
    # A different seed from the fixture's, so X2 is independent of X1.
    rng = np.random.default_rng(155)
    design = np.column_stack([X, rng.normal(size=N), X[:, 1]])
    y = rng.poisson(np.exp(0.5 + 0.3 * X[:, 1])).astype(np.float64)
    with pytest.raises(ValueError, match=r"\['X1', 'X3'\]"):
        GLM(Poisson()).fit(design, y)


def test_more_coefficients_than_observations_raises() -> None:
    rng = np.random.default_rng(154)
    X = np.column_stack([np.ones(5), rng.normal(size=(5, 6))])
    with pytest.raises(ValueError, match="more coefficients than observations"):
        GLM(Gaussian()).fit(X, rng.normal(size=5))


def test_spline_basis_duplicating_the_intercept_raises() -> None:
    # A cs() basis sums to one at every x, duplicating the intercept. This was
    # only rank deficient up to rounding error, so solving the singular system
    # never raised, and it fit silently, with nan standard errors.
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.uniform(0, 12, size=1000)})
    df["y"] = rng.poisson(0.5 + df["x"] * (np.sin(1.25 * df["x"]) + 1))
    with pytest.raises(ValueError, match="rank deficient"):
        GLM(Poisson()).fit(df, formula="y ~ cs(x, df=10)")
    # Centering the basis removes the redundancy.
    model = GLM(Poisson()).fit(df, formula="y ~ cs(x, df=10, constraints='center')")
    assert np.all(np.isfinite(model.coef_standard_error_))


@pytest.mark.parametrize(
    "make_design", RANK_DEFICIENT_DESIGNS.values(), ids=list(RANK_DEFICIENT_DESIGNS)
)
def test_penalized_fit_accepts_rank_deficient_design(
    X: np.ndarray, make_design: Callable[[np.ndarray], np.ndarray]
) -> None:
    # The penalty makes the fit well defined; every-level encoding of
    # categorical variables relies on this.
    rng = np.random.default_rng(154)
    y = rng.poisson(np.exp(0.5 + 0.3 * X[:, 1])).astype(np.float64)
    model = GLM(Poisson(), alpha=1.0).fit(make_design(X), y)
    assert model.coef_ is not None and np.all(np.isfinite(model.coef_))


def test_rank_ignores_rows_with_zero_weight(X: np.ndarray) -> None:
    # This column is non-zero only on rows with zero weight, so the rows that
    # count leave its coefficient undetermined.
    rng = np.random.default_rng(154)
    y = rng.poisson(np.exp(0.5 + 0.3 * X[:, 1])).astype(np.float64)
    only_on_zero_weight = np.zeros(N)
    only_on_zero_weight[:10] = 1.0
    weights = np.ones(N)
    weights[:10] = 0.0
    with pytest.raises(ValueError, match="rank deficient"):
        GLM(Poisson()).fit(
            np.column_stack([X, only_on_zero_weight]), y, frequency_weights=weights
        )


@pytest.mark.parametrize(
    "family, draw",
    [
        (Poisson(), lambda rng, η: rng.poisson(np.exp(η))),
        (Bernoulli(), lambda rng, η: rng.binomial(1, 1 / (1 + np.exp(-η)))),
        (Gamma(), lambda rng, η: rng.gamma(2.0, np.exp(η) / 2.0)),
    ],
)
def test_cold_start_without_intercept_handles_large_predictors(
    family: ExponentialFamily,
    draw: Callable[[np.random.Generator, np.ndarray], np.ndarray],
) -> None:
    # Without an intercept, column 0 is a predictor. Starting its coefficient
    # at the intercept-only estimate, about 0.5 here, would put the starting
    # linear predictor near 0.5 * 1500 and overflow the inverse link.
    rng = np.random.default_rng(154)
    n = 1000
    X = np.column_stack([rng.normal(1500.0, 1.0, size=n), rng.normal(size=n)])
    y = draw(rng, 0.5 + 0.3 * X[:, 1])
    model = GLM(family).fit(X, y, validate_intercept=False)
    zero_start = GLM(family).fit(X, y, validate_intercept=False, warm_start=np.zeros(2))
    assert model.coef_ is not None and zero_start.coef_ is not None
    np.testing.assert_allclose(model.coef_, zero_start.coef_)


@pytest.mark.parametrize("family", [Poisson(), Bernoulli()])
@pytest.mark.parametrize("alpha", [0.0, 5.0])
def test_stored_fit_describes_the_returned_coefficients(
    family: ExponentialFamily, alpha: float
) -> None:
    # Stop before convergence, where the last step is large enough that
    # quantities from the previous iterate would differ noticeably.
    rng = np.random.default_rng(154)
    n = 500
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(5.0, 2.0, size=n)])
    η = 0.2 + 0.5 * X[:, 1] - 0.1 * X[:, 2]
    y = family.sample(family.inv_link(η), 1.0, rng=rng)
    model = GLM(family, alpha=alpha).fit(X, y, max_iter=3)
    assert model.coef_ is not None
    μ = model.predict(X)
    np.testing.assert_allclose(model.deviance_, family.deviance(y, μ))
    dμ = family.d_inv_link(X @ model.coef_, μ)
    working_weights = dμ**2 / family.variance(μ)
    np.testing.assert_allclose(
        model.information_matrix_, X.T @ (X * working_weights[:, None])
    )


def test_has_converged_compares_predicted_decrease_to_deviance() -> None:
    gradient = np.array([1e-3, 2e-3])
    step = np.array([1e-3, 1e-3])
    # The predicted decrease is gᵗs = 3e-6, against tol times the deviance.
    assert has_converged(gradient, step, 100.0, tol=1e-7)
    assert not has_converged(gradient, step, 100.0, tol=1e-9)


def test_has_converged_at_a_perfect_fit() -> None:
    assert has_converged(np.zeros(2), np.zeros(2), 0.0, tol=1e-8)


@pytest.mark.parametrize("penalized_deviance", [np.inf, np.nan])
def test_has_not_converged_with_non_finite_deviance(penalized_deviance: float) -> None:
    assert not has_converged(np.ones(2), np.ones(2), penalized_deviance, tol=1e-8)
