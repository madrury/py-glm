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
