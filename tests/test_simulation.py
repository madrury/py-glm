import numpy as np
import pandas as pd
import pytest

from glm.families import Poisson
from glm.glm import GLM
from glm.simulation import Simulation

N_SIM = 5


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(154)
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df["y"] = rng.poisson(np.exp(0.5 + 0.3 * df["x"]))
    return df


def test_parametric_bootstrap_formula_model(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    models = Simulation(model).parametric_bootstrap(df, n_sim=N_SIM)
    assert len(models) == N_SIM
    for boot in models:
        assert boot.coef_ is not None and model.coef_ is not None
        assert boot.coef_.shape == model.coef_.shape
        assert boot.X_names == model.X_names
        assert boot.y_name == model.y_name
        # Bootstrap models keep the formula, so they predict from data frames.
        assert boot.predict(df).shape == (len(df),)


def test_parametric_bootstrap_array_model(df: pd.DataFrame) -> None:
    X = np.column_stack([np.ones(len(df)), df["x"]])
    y = df["y"].to_numpy()
    model = GLM(Poisson()).fit(X, y, X_names=["Intercept", "x"], y_name="y")
    models = Simulation(model).parametric_bootstrap(X, n_sim=N_SIM)
    assert len(models) == N_SIM
    for boot in models:
        assert boot.coef_ is not None
        assert boot.coef_.shape == (2,)
        assert boot.X_names == ["Intercept", "x"]
        assert boot.predict(X).shape == (len(df),)


@pytest.fixture
def exposure_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(154)
    n = 400
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    exposure = rng.uniform(1, 10, size=n)
    y = rng.poisson(exposure * np.exp(0.2 + 0.3 * X[:, 1])).astype(np.float64)
    return X, y, np.log(exposure)


def test_non_parametric_bootstrap_uses_offset(
    exposure_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, offset = exposure_data
    model = GLM(Poisson()).fit(X, y, offset=offset)
    assert model.coef_ is not None
    models = Simulation(model).non_parametric_bootstrap(
        X, y, n_sim=N_SIM, offset=offset
    )
    intercepts = np.array([boot.coef_[0] for boot in models if boot.coef_ is not None])
    # Ignoring the offset would shift each intercept by the mean log exposure
    # (about 1.6), so the bootstrap intercepts must sit far closer than that.
    assert np.all(np.abs(intercepts - model.coef_[0]) < np.mean(offset) / 4)


def test_non_parametric_bootstrap_resamples_weights_with_rows(
    exposure_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = exposure_data
    # Zero-weight rows get extreme responses: they can only influence the fit
    # if a weight is attached to the wrong row after resampling.
    sample_weights = np.ones(len(y))
    sample_weights[::4] = 0.0
    y = y.copy()
    y[::4] = 1000.0
    model = GLM(Poisson()).fit(X, y, sample_weights=sample_weights)
    assert model.coef_ is not None
    models = Simulation(model).non_parametric_bootstrap(
        X, y, n_sim=N_SIM, sample_weights=sample_weights
    )
    for boot in models:
        assert boot.coef_ is not None
        assert np.all(np.abs(boot.coef_ - model.coef_) < 0.5)
