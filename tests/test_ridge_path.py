import numpy as np
import pandas as pd
import pytest

from glm.families import Gaussian, Poisson
from glm.glm import GLM
from glm.ridge import GRID_MARGIN, RidgePath, default_alphas
from glm.standardize import Standardizer

N = 1000


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(154)
    df = pd.DataFrame({"x1": rng.normal(size=N), "g": rng.choice(["a", "b", "c"], N)})
    # A second predictor, strongly correlated with the first.
    df["x2"] = 0.95 * df["x1"] + np.sqrt(1 - 0.95**2) * rng.normal(size=N)
    g_effect = df["g"].map({"a": 0.0, "b": 0.5, "c": -0.3})
    df["y"] = rng.poisson(np.exp(0.3 + 0.4 * df["x1"] + 0.2 * df["x2"] + g_effect))
    return df


@pytest.fixture
def arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(N), df["x1"], 100 * df["x2"]])
    return X, df["y"].to_numpy().astype(np.float64)


def test_default_grid(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    path = RidgePath(Poisson()).fit(*arrays)
    assert path.alphas_ is not None and path.models_ is not None
    assert len(path.alphas_) == len(path.models_) == 100
    assert np.all(np.diff(path.alphas_) < 0)


def test_supplied_alphas_are_sorted_decreasing(
    arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    path = RidgePath(Poisson(), alphas=[1.0, 100.0, 10.0]).fit(*arrays)
    assert path.alphas_ is not None
    np.testing.assert_array_equal(path.alphas_, [100.0, 10.0, 1.0])


@pytest.mark.parametrize("alphas", [[], [1.0, 0.0], [1.0, -2.0]])
def test_invalid_alphas_raise(alphas: list[float]) -> None:
    with pytest.raises(ValueError, match="positives"):
        RidgePath(Poisson(), alphas=alphas)


@pytest.mark.parametrize("n_alphas", [1, 0, 2.5, True])
def test_invalid_n_alphas_raise(n_alphas: object) -> None:
    with pytest.raises(ValueError, match="n_alphas"):
        RidgePath(Poisson(), n_alphas=n_alphas)  # type: ignore[arg-type]


def test_models_match_independent_fits(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    # Warm starts speed up fitting, but must not change the answers.
    path = RidgePath(Poisson(), n_alphas=10).fit(*arrays, tol=1e-12)
    assert path.alphas_ is not None and path.models_ is not None
    for alpha, model in zip(path.alphas_, path.models_):
        independent = GLM(Poisson(), alpha=alpha).fit(*arrays, tol=1e-12)
        assert model.coef_ is not None and independent.coef_ is not None
        np.testing.assert_allclose(
            model.coef_, independent.coef_, rtol=1e-6, atol=1e-10
        )


def test_penalty_decreases_along_path(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    # As alpha falls, the standardized coefficients can only grow.
    X, y = arrays
    path = RidgePath(Poisson()).fit(X, y)
    scales = Standardizer("continuous").fit(X).scales_
    norms = path.coefficient_table().to_numpy()[:, 1:] * scales[1:]
    norms = np.sum(norms**2, axis=1)
    assert np.all(np.diff(norms) >= -1e-10)


def test_default_grid_spans_whole_path(df: pd.DataFrame) -> None:
    # At the largest alpha the coefficients are almost entirely shrunk, and at
    # the smallest they are almost those of the unpenalized fit.
    path = RidgePath(Poisson()).fit(df, formula="y ~ x1 + x2")
    unpenalized = GLM(Poisson()).fit(df, formula="y ~ x1 + x2")
    assert path.models_ is not None and unpenalized.coef_ is not None
    most, least = path.models_[0].coef_, path.models_[-1].coef_
    assert most is not None and least is not None
    assert np.all(np.abs(most[1:]) < 0.02 * np.abs(unpenalized.coef_[1:]))
    np.testing.assert_allclose(least, unpenalized.coef_, rtol=0.02)


def test_default_alphas_span_the_eigenvalues() -> None:
    # For a Gaussian model, the information matrix of the standardized,
    # centered columns is Z'Z, so the grid spans its eigenvalues.
    rng = np.random.default_rng(154)
    x1 = rng.normal(size=N)
    x2 = 0.95 * x1 + np.sqrt(1 - 0.95**2) * rng.normal(size=N)
    X = np.column_stack([np.ones(N), x1, x2])
    y = X @ np.array([1.0, 2.0, -1.0]) + rng.normal(size=N)
    Z = Standardizer("continuous").fit(X).transform(X)[:, 1:]
    eigenvalues = np.linalg.eigvalsh(Z.T @ Z)
    alphas = default_alphas(Gaussian(), X, y)
    assert alphas[0] == pytest.approx(GRID_MARGIN * eigenvalues.max())
    assert alphas[-1] == pytest.approx(eigenvalues.min() / GRID_MARGIN)


def test_default_alphas_skip_zero_eigenvalues(df: pd.DataFrame) -> None:
    # A column for every category level sums to the intercept, giving a zero
    # eigenvalue, which must not set the bottom of the grid.
    path = RidgePath(Poisson()).fit(df, formula="y ~ x1 + C(g)")
    assert path.alphas_ is not None
    assert np.all(np.isfinite(path.alphas_)) and path.alphas_[-1] > 1e-3


def test_formula_path(df: pd.DataFrame) -> None:
    path = RidgePath(Poisson(), n_alphas=5).fit(df, formula="y ~ x1 + x2 + C(g)")
    assert path.term_columns_ == {
        "x1": ["x1"],
        "x2": ["x2"],
        "C(g)": ["C(g)[a]", "C(g)[b]", "C(g)[c]"],
    }
    table = path.coefficient_table()
    assert table.index.name == "alpha"
    assert list(table.columns) == [
        "Intercept",
        "x1",
        "x2",
        "C(g)[a]",
        "C(g)[b]",
        "C(g)[c]",
    ]
    assert path.models_ is not None
    for model in path.models_:
        # Each model on the path is a formula model in its own right.
        assert model.formula == "y ~ x1 + x2 + C(g)"
        assert model.predict(df).shape == (N,)


def test_array_path_term_columns(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    path = RidgePath(Poisson(), n_alphas=5).fit(
        *arrays, X_names=["Intercept", "a", "b"]
    )
    assert path.term_columns_ == {"a": ["a"], "b": ["b"]}


def test_unfit_path_raises() -> None:
    with pytest.raises(ValueError, match="has not been fit"):
        RidgePath(Poisson()).coefficient_table()


def test_effective_degrees_of_freedom_gaussian() -> None:
    # For a Gaussian model the weights are constant, and the effective degrees
    # of freedom are exactly the trace of the ridge hat matrix for the
    # centered, standardized penalized columns (ESL, section 3.4.1).
    rng = np.random.default_rng(154)
    X = np.column_stack([np.ones(N), rng.normal(size=(N, 3)) * [1.0, 10.0, 0.1]])
    y = X @ np.array([1.0, 0.5, -0.05, 2.0]) + rng.normal(size=N)
    path = RidgePath(Gaussian(), n_alphas=10).fit(X, y)
    assert path.alphas_ is not None and path.df_ is not None
    Z = Standardizer("continuous").fit(X).transform(X)[:, 1:]
    Z = Z - Z.mean(axis=0)
    for alpha, df in zip(path.alphas_, path.df_):
        hat = Z @ np.linalg.solve(Z.T @ Z + alpha * np.eye(3), Z.T)
        assert df == pytest.approx(np.trace(hat))


def test_effective_degrees_of_freedom_along_path(df: pd.DataFrame) -> None:
    path = RidgePath(Poisson()).fit(df, formula="y ~ x1 + x2 + C(g)")
    assert path.df_ is not None
    # They grow as the penalty relaxes, from near zero up to the number of
    # directions the penalized coefficients can vary in: x1, x2, and two
    # contrasts between the three levels, as their columns sum to the
    # intercept.
    assert np.all(np.diff(path.df_) > 0)
    assert path.df_[0] < 0.05
    assert path.df_[-1] == pytest.approx(4.0, abs=0.05)


def test_scales(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = arrays
    path = RidgePath(Poisson(), n_alphas=5).fit(X, y)
    np.testing.assert_array_equal(
        path.scales_, Standardizer("continuous").fit(X).scales_
    )
    unstandardized = RidgePath(Poisson(), n_alphas=5, standardize="none").fit(X, y)
    np.testing.assert_array_equal(unstandardized.scales_, 1.0)
