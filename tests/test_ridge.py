import numpy as np
import pandas as pd
import pytest

from glm.families import Bernoulli, ExponentialFamily, Gamma, Gaussian, Poisson
from glm.glm import GLM
from glm.standardize import Standardizer
from glm.types import Standardization

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
    model = GLM(Gaussian(), alpha=alpha, standardize="none").fit(X, y, tol=1e-12)
    # Minimizes sum((y - X @ β)**2) + alpha * sum(β[1:]**2).
    penalty = alpha * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    expected = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, expected, rtol=1e-8)


@pytest.mark.parametrize("standardize", ["none", "continuous"])
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize(
    "family", [Gaussian(), Bernoulli(), Poisson(), Gamma()], ids=lambda f: repr(f)
)
def test_ridge_solves_penalized_score_equations(
    X: np.ndarray,
    family: ExponentialFamily,
    alpha: float,
    standardize: Standardization,
) -> None:
    X = X * np.array([1.0, 1.0, 10.0, 0.1])  # Columns on different scales.
    μ = family.inv_link(X @ np.array([0.2, 0.5, -0.03, 1.0]))
    y = family.sample(μ, dispersion=0.5 if family.has_dispersion else 1.0)
    model = GLM(family, alpha=alpha, standardize=standardize).fit(
        X, y, tol=1e-14, max_iter=1000
    )
    assert model.coef_ is not None
    η = X @ model.coef_
    μ_hat = family.inv_link(η)
    # The penalty is alpha * sum((s * β)[1:]**2), with s the column scales (all
    # ones without standardizing). At its minimum, half the deviance's
    # gradient, -X' (y - μ) dμ / V(μ) (the left hand side of the score
    # equations, see Muller eq. 11), balances half the penalty's, alpha * s**2 *
    # β, for every coefficient except the unpenalized intercept.
    scales = Standardizer(standardize).fit(X).scales_
    assert scales is not None
    score = X.T @ ((y - μ_hat) * family.d_inv_link(η, μ_hat) / family.variance(μ_hat))
    penalty_gradient = alpha * scales**2 * model.coef_
    penalty_gradient[0] = 0.0
    np.testing.assert_allclose(score, penalty_gradient, atol=1e-6 * max(1.0, alpha))


def test_ridge_shrinks_toward_zero(X: np.ndarray) -> None:
    rng = np.random.default_rng(154)
    y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(size=N)
    norms = []
    for alpha in [0.0, 1.0, 100.0, 1e4, 1e6]:
        model = GLM(Gaussian(), alpha=alpha, standardize="none").fit(X, y)
        assert model.coef_ is not None
        norms.append(np.sum(model.coef_[1:] ** 2))
    assert all(a > b for a, b in zip(norms, norms[1:]))
    assert norms[-1] < 1e-4 * norms[0]


@pytest.mark.parametrize("alpha", ALPHAS)
def test_ridge_without_intercept_penalizes_every_coefficient(
    X: np.ndarray, alpha: float
) -> None:
    # Without an intercept, column 0 is an ordinary predictor, and must be
    # penalized like the others.
    X = X[:, 1:]
    rng = np.random.default_rng(154)
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(size=N)
    model = GLM(Gaussian(), alpha=alpha, standardize="none").fit(
        X, y, tol=1e-12, validate_intercept=False
    )
    expected = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)
    assert model.coef_ is not None
    np.testing.assert_allclose(model.coef_, expected, rtol=1e-8)


@pytest.mark.parametrize("standardize", ["none", "continuous"])
@pytest.mark.parametrize("family", [Gaussian(), Poisson()], ids=lambda f: repr(f))
def test_ridge_without_intercept_solves_penalized_score_equations(
    X: np.ndarray, family: ExponentialFamily, standardize: Standardization
) -> None:
    alpha = 10.0
    X = X[:, 1:] * np.array([1.0, 10.0, 0.1])  # No intercept; mixed scales.
    μ = family.inv_link(X @ np.array([0.5, -0.03, 1.0]))
    y = family.sample(μ, dispersion=0.5 if family.has_dispersion else 1.0)
    model = GLM(family, alpha=alpha, standardize=standardize).fit(
        X, y, tol=1e-14, max_iter=1000, validate_intercept=False
    )
    assert model.coef_ is not None
    η = X @ model.coef_
    μ_hat = family.inv_link(η)
    scales = Standardizer(standardize).fit(X).scales_
    assert scales is not None
    score = X.T @ ((y - μ_hat) * family.d_inv_link(η, μ_hat) / family.variance(μ_hat))
    # Every coefficient is penalized, column 0 included.
    np.testing.assert_allclose(
        score, alpha * scales**2 * model.coef_, atol=1e-6 * alpha
    )


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(154)
    df = pd.DataFrame({"x": rng.normal(size=N), "g": rng.choice(["a", "b", "c"], N)})
    g_effect = df["g"].map({"a": 0.0, "b": 0.5, "c": -0.5})
    df["y"] = rng.poisson(np.exp(0.5 + 0.3 * df["x"] + g_effect))
    return df


def test_penalized_formula_model_encodes_every_level(df: pd.DataFrame) -> None:
    model = GLM(Poisson(), alpha=1.0).fit(df, formula="y ~ x + C(g)")
    assert model.X_names == ["Intercept", "x", "C(g)[a]", "C(g)[b]", "C(g)[c]"]


def test_unpenalized_formula_model_drops_a_reference_level(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    assert model.X_names == ["Intercept", "x", "C(g)[T.b]", "C(g)[T.c]"]


def test_penalized_formula_model_does_not_depend_on_reference_level(
    df: pd.DataFrame,
) -> None:
    # Relabel the levels so they sort in a different order. With a reference
    # level dropped, the penalty would shrink toward a different baseline and
    # change the fit; with every level encoded, nothing changes.
    relabeled = df.assign(g=df["g"].map({"a": "z", "b": "b", "c": "c"}))
    model = GLM(Poisson(), alpha=10.0)
    original = model.clone().fit(df, formula="y ~ x + C(g)", tol=1e-12)
    relabeled_model = model.clone().fit(relabeled, formula="y ~ x + C(g)", tol=1e-12)
    np.testing.assert_allclose(
        original.predict(df), relabeled_model.predict(relabeled), rtol=1e-8
    )


def test_penalized_formula_model_predicts_with_every_level(df: pd.DataFrame) -> None:
    model = GLM(Poisson(), alpha=1.0).fit(df, formula="y ~ x + C(g)")
    assert model.coef_ is not None
    # New data with only one level is still encoded with every level's column.
    new = pd.DataFrame({"x": [0.0, 1.0], "g": ["c", "c"]})
    intercept, β_x, _, _, β_c = model.coef_
    expected = np.exp(intercept + β_c + β_x * new["x"].to_numpy())
    np.testing.assert_allclose(model.predict(new), expected)
