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


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_predict_with_infinite_values_raises(df: pd.DataFrame, value: float) -> None:
    # Formulaic's na_action doesn't treat infinite values as missing.
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    new = pd.DataFrame({"x": [1.0, value, 2.0], "g": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="X contains missing .* or infinite"):
        model.predict(new)


def test_fit_with_transform_making_infinite_values_raises(df: pd.DataFrame) -> None:
    # Transforms are applied after formulaic's check for missing values, and
    # log(0) is -inf.
    df["z"] = np.abs(df["x"])
    df.loc[3, "z"] = 0.0
    with pytest.raises(ValueError, match="X contains missing .* or infinite"):
        GLM(Poisson()).fit(df, formula="y ~ np.log(z) + C(g)")


def test_formula_without_response_raises(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must specify a response"):
        GLM(Poisson()).fit(df, formula="x + C(g)")


def test_formula_model_predict_requires_data_frame(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x")
    with pytest.raises(TypeError, match="must be a DataFrame"):
        model.predict(np.ones((2, 2)))


@pytest.mark.parametrize(
    "formula, new_g",
    [
        ("y ~ x + C(g)", "d"),
        ("y ~ x + g", "d"),  # A string column is categorical without C().
        ("y ~ x + x:C(g)", "d"),  # A categorical only in an interaction.
    ],
)
def test_unpenalized_predict_rejects_unseen_levels(
    df: pd.DataFrame, formula: str, new_g: str
) -> None:
    # Reference coding would encode the unseen level exactly as the reference
    # level, silently predicting it as level "a".
    model = GLM(Poisson()).fit(df, formula=formula)
    new = pd.DataFrame({"x": [0.0, 1.0], "g": ["b", new_g]})
    with pytest.raises(ValueError, match=r"levels the model wasn't fit with .*\['d'\]"):
        model.predict(new)


def test_unpenalized_predict_rejects_unseen_integer_levels(df: pd.DataFrame) -> None:
    codes = df.assign(k=df["g"].map({"a": 1, "b": 2, "c": 3}))
    model = GLM(Poisson()).fit(codes, formula="y ~ x + C(k)")
    with pytest.raises(ValueError, match=r"\[4\]"):
        model.predict(pd.DataFrame({"x": [0.0], "k": [4]}))


def test_unpenalized_predict_accepts_a_subset_of_levels(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + g")
    assert model.predict(pd.DataFrame({"x": [0.0, 1.0], "g": ["c", "c"]})).shape == (2,)


def test_penalized_predict_gives_unseen_levels_no_effect(df: pd.DataFrame) -> None:
    # Every level has its own indicator, shrunk toward zero, so a level with
    # no training data gets no effect: the baseline prediction.
    model = GLM(Poisson(), alpha=1.0).fit(df, formula="y ~ x + C(g)")
    assert model.coef_ is not None
    coef = pd.Series(model.coef_, index=model.X_names)
    new = pd.DataFrame({"x": [0.5], "g": ["d"]})
    expected = np.exp(coef["Intercept"] + coef["x"] * 0.5)
    np.testing.assert_allclose(model.predict(new), expected)


def test_design_matrix_encodes_with_training_specs(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    assert model.coef_ is not None
    # Rows from a single level must still get every training column.
    design = model.design_matrix(df[df["g"] == "b"])
    assert design.shape == (int(np.sum(df["g"] == "b")), 4)
    np.testing.assert_allclose(design[:, 2], 1.0)
    np.testing.assert_allclose(design[:, 3], 0.0)


def test_response_vector_takes_the_formula_response(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    np.testing.assert_allclose(model.response_vector(df), df["y"])


def test_formula_model_response_vector_rejects_y(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    with pytest.raises(ValueError, match="y must not be supplied"):
        model.response_vector(df, df["y"].to_numpy())


def test_array_model_returns_design_matrix_and_response(df: pd.DataFrame) -> None:
    X = np.column_stack([np.ones(len(df)), df["x"]])
    y = df["y"].to_numpy()
    model = GLM(Poisson()).fit(X, y)
    assert model.design_matrix(X) is X
    assert model.response_vector(X, y) is y
    with pytest.raises(TypeError, match="y must be a numpy array"):
        model.response_vector(X)
    with pytest.raises(TypeError, match="X must be a numpy array"):
        model.design_matrix(df)


def test_unfit_model_cannot_encode_data(df: pd.DataFrame) -> None:
    model = GLM(Poisson())
    with pytest.raises(ValueError, match="not fit"):
        model.design_matrix(df)
    with pytest.raises(ValueError, match="not fit"):
        model.response_vector(df)
