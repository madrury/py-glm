import numpy as np
import pytest

from glm.families import Bernoulli, ExponentialFamily, Gamma, Gaussian, Poisson
from glm.glm import GLM

N = 500
ALPHAS = [0.1, 10.0, 1000.0]


@pytest.fixture
def X() -> np.ndarray:
    rng = np.random.default_rng(154)
    return np.column_stack([np.ones(N), rng.normal(size=(N, 3))])


@pytest.mark.parametrize("alpha", ALPHAS)
def test_gaussian_ridge_matches_closed_form(X: np.ndarray, alpha: float) -> None:
    rng = np.random.default_rng(154)
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(size=N)
    model = GLM(Gaussian(), alpha=alpha).fit(X, y, tol=1e-12)
    # Minimizes sum((y - X @ β)**2) + alpha * sum(β[1:]**2).
    penalty = alpha * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    expected = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, expected, rtol=1e-8)


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize(
    "family", [Gaussian(), Bernoulli(), Poisson(), Gamma()], ids=lambda f: repr(f)
)
def test_ridge_solves_penalized_score_equations(
    X: np.ndarray, family: ExponentialFamily, alpha: float
) -> None:
    rng = np.random.default_rng(154)
    μ = family.inv_link(X @ np.array([0.2, 0.5, -0.3, 0.1]))
    y = family.sample(μ, dispersion=0.5 if family.has_dispersion else 1.0)
    model = GLM(family, alpha=alpha).fit(X, y, tol=1e-14, max_iter=1000)
    assert model.coef_ is not None
    η = X @ model.coef_
    μ_hat = family.inv_link(η)
    # At the minimum of deviance + alpha * sum(β[1:]**2), half the deviance's
    # gradient, -X' (y - μ) dμ / V(μ), balances half the penalty's, alpha * β,
    # for every coefficient except the unpenalized intercept.
    score = X.T @ ((y - μ_hat) * family.d_inv_link(η, μ_hat) / family.variance(μ_hat))
    penalty_gradient = alpha * model.coef_
    penalty_gradient[0] = 0.0
    np.testing.assert_allclose(score, penalty_gradient, atol=1e-6 * max(1.0, alpha))


def test_ridge_shrinks_toward_zero(X: np.ndarray) -> None:
    rng = np.random.default_rng(154)
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(size=N)
    norms = []
    for alpha in [0.0, 1.0, 100.0, 1e4, 1e6]:
        model = GLM(Gaussian(), alpha=alpha).fit(X, y)
        assert model.coef_ is not None
        norms.append(np.sum(model.coef_[1:] ** 2))
    assert all(a > b for a, b in zip(norms, norms[1:]))
    assert norms[-1] < 1e-4 * norms[0]
