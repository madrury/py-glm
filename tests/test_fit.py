import numpy as np
import pytest

from glm.families import Bernoulli, ExponentialFamily, Gamma, Gaussian, Poisson
from glm.glm import GLM

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


def test_zero_weight_rows_do_not_affect_fit(X: np.ndarray) -> None:
    rng = np.random.default_rng(154)
    y = rng.poisson(np.exp(0.5 + 0.3 * X[:, 1])).astype(np.float64)
    # Extreme responses on rows with zero weight must have no effect.
    y_contaminated, weights = y.copy(), np.ones(N)
    y_contaminated[::4], weights[::4] = 1000.0, 0.0
    model = GLM(Poisson()).fit(X, y_contaminated, sample_weights=weights)
    assert model.coef_ is not None
    kept = weights > 0
    subset = GLM(Poisson()).fit(X[kept], y[kept])
    assert subset.coef_ is not None
    np.testing.assert_allclose(model.coef_, subset.coef_)


def test_non_finite_estimates_raise(X: np.ndarray) -> None:
    # The MLE does not exist for an all-zero binary response.
    with pytest.raises(ValueError, match="not finite"):
        GLM(Bernoulli()).fit(X, np.zeros(N))


@pytest.mark.parametrize("family", [Gaussian(), Poisson()])
def test_perfect_fit_converges(family: ExponentialFamily) -> None:
    # A perfect fit has zero deviance, which must not break the relative
    # convergence check.
    X = np.column_stack([np.ones(N), np.linspace(-1, 1, N)])
    β = np.array([1.0, 0.5])
    model = GLM(family).fit(X, family.inv_link(X @ β))
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, β)
