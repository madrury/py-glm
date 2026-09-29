import numpy as np
import pandas as pd
import pytest

from glm.families import Gaussian, Poisson
from glm.glm import GLM


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(154)
    n = 500
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], size=n),
        }
    )
    g_effect = df["g"].map({"a": 0.0, "b": 0.5, "c": -0.5})
    df["y"] = rng.poisson(np.exp(0.5 + 0.3 * df["x"] + g_effect))
    return df


def test_formula_fit_matches_design_matrix_fit(df: pd.DataFrame) -> None:
    formula_model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    X = np.column_stack(
        [np.ones(len(df)), df["x"], df["g"] == "b", df["g"] == "c"]
    ).astype(np.float64)
    array_model = GLM(Poisson()).fit(X, df["y"].to_numpy())
    assert formula_model.coef_ is not None and array_model.coef_ is not None
    np.testing.assert_allclose(formula_model.coef_, array_model.coef_)
    assert formula_model.X_names == ["Intercept", "x", "C(g)[T.b]", "C(g)[T.c]"]
    assert formula_model.y_name == "y"


def test_predict_uses_training_categorical_levels(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    assert model.coef_ is not None
    # New data containing only one of the training levels must still be
    # encoded with all of the training dummy columns.
    new = pd.DataFrame({"x": [0.0, 1.0], "g": ["c", "c"]})
    intercept, β_x, β_b, β_c = model.coef_
    expected = np.exp(intercept + β_c + β_x * new["x"].to_numpy())
    np.testing.assert_allclose(model.predict(new), expected)


def test_predict_uses_training_stateful_transforms(df: pd.DataFrame) -> None:
    model = GLM(Gaussian()).fit(df, formula="y ~ center(x)")
    assert model.coef_ is not None
    # center(x) must subtract the *training* mean of x, not the mean of the
    # new data.
    new = pd.DataFrame({"x": [2.0, 3.0]})
    intercept, β_x = model.coef_
    expected = intercept + β_x * (new["x"].to_numpy() - df["x"].mean())
    np.testing.assert_allclose(model.predict(new), expected)


def test_predict_with_missing_values_raises(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    # Dropping the incomplete row would return fewer predictions than rows.
    new = pd.DataFrame({"x": [1.0, np.nan, 2.0], "g": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="null values"):
        model.predict(new)


@pytest.mark.parametrize("column", ["x", "g", "y"])
def test_fit_with_missing_values_raises(df: pd.DataFrame, column: str) -> None:
    df.loc[3, column] = None
    with pytest.raises(ValueError, match="null values"):
        GLM(Poisson()).fit(df, formula="y ~ x + C(g)")


def test_formula_without_response_raises(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must specify a response"):
        GLM(Poisson()).fit(df, formula="x + C(g)")


def test_formula_model_predict_requires_data_frame(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x")
    with pytest.raises(TypeError, match="must be a DataFrame"):
        model.predict(np.ones((2, 2)))
