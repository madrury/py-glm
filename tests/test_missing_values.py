import numpy as np
import pandas as pd
import pytest

from glm.families import Poisson
from glm.glm import GLM


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(154)
    n = 100
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = rng.poisson(np.exp(0.5 + 0.3 * X[:, 1])).astype(np.float64)
    return X, y


# Infinite values are as unusable as missing ones, and must not slip through
# to a misleading error when fitting, or to silently wrong predictions.
NON_FINITE = pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])


@NON_FINITE
def test_fit_with_missing_design_matrix_values_raises(
    data: tuple[np.ndarray, np.ndarray], value: float
) -> None:
    X, y = data
    X[3, 1] = value
    with pytest.raises(ValueError, match="X contains missing .* or infinite"):
        GLM(Poisson()).fit(X, y)


@NON_FINITE
def test_fit_with_missing_response_values_raises(
    data: tuple[np.ndarray, np.ndarray], value: float
) -> None:
    X, y = data
    y[3] = value
    with pytest.raises(ValueError, match="y contains missing .* or infinite"):
        GLM(Poisson()).fit(X, y)


@NON_FINITE
def test_predict_with_missing_values_raises(
    data: tuple[np.ndarray, np.ndarray], value: float
) -> None:
    X, y = data
    model = GLM(Poisson()).fit(X, y)
    X_new = np.array([[1.0, 0.5], [1.0, value]])
    with pytest.raises(ValueError, match="X contains missing .* or infinite"):
        model.predict(X_new)


@NON_FINITE
def test_score_with_missing_response_values_raises(
    data: tuple[np.ndarray, np.ndarray], value: float
) -> None:
    X, y = data
    model = GLM(Poisson()).fit(X, y)
    y[3] = value
    with pytest.raises(ValueError, match="y contains missing .* or infinite"):
        model.score(X, y)


def test_array_model_predict_requires_array(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = data
    model = GLM(Poisson()).fit(X, y)
    with pytest.raises(TypeError, match="must be a numpy array"):
        model.predict(pd.DataFrame(X))


@NON_FINITE
@pytest.mark.parametrize(
    "argument", ["offset", "frequency_weights", "precision_weights"]
)
def test_fit_with_missing_offset_or_weights_raises(
    data: tuple[np.ndarray, np.ndarray], argument: str, value: float
) -> None:
    X, y = data
    values = {
        "offset": np.zeros(len(y)),
        "frequency_weights": np.ones(len(y)),
        "precision_weights": np.ones(len(y)),
    }
    values[argument][3] = value
    with pytest.raises(ValueError, match=f"{argument} contains missing .* or infinite"):
        GLM(Poisson()).fit(
            X,
            y,
            offset=values["offset"],
            frequency_weights=values["frequency_weights"],
            precision_weights=values["precision_weights"],
        )


@NON_FINITE
def test_predict_with_missing_offset_raises(
    data: tuple[np.ndarray, np.ndarray], value: float
) -> None:
    X, y = data
    model = GLM(Poisson()).fit(X, y)
    with pytest.raises(ValueError, match="offset contains missing .* or infinite"):
        model.predict(X[:2], offset=np.array([0.0, value]))
