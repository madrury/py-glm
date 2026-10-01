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


def test_fit_with_missing_design_matrix_values_raises(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = data
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="X contains missing"):
        GLM(Poisson()).fit(X, y)


def test_fit_with_missing_response_values_raises(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = data
    y[3] = np.nan
    with pytest.raises(ValueError, match="y contains missing"):
        GLM(Poisson()).fit(X, y)


def test_predict_with_missing_values_raises(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = data
    model = GLM(Poisson()).fit(X, y)
    X_new = np.array([[1.0, 0.5], [1.0, np.nan]])
    with pytest.raises(ValueError, match="X contains missing"):
        model.predict(X_new)


def test_array_model_predict_requires_array(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = data
    model = GLM(Poisson()).fit(X, y)
    with pytest.raises(TypeError, match="must be a numpy array"):
        model.predict(pd.DataFrame(X))


@pytest.mark.parametrize("argument", ["offset", "sample_weights"])
def test_fit_with_missing_offset_or_weights_raises(
    data: tuple[np.ndarray, np.ndarray], argument: str
) -> None:
    X, y = data
    offset, sample_weights = np.zeros(len(y)), np.ones(len(y))
    if argument == "offset":
        offset[3] = np.nan
    else:
        sample_weights[3] = np.nan
    with pytest.raises(ValueError, match=f"{argument} contains missing"):
        GLM(Poisson()).fit(X, y, offset=offset, sample_weights=sample_weights)


def test_predict_with_missing_offset_raises(
    data: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = data
    model = GLM(Poisson()).fit(X, y)
    with pytest.raises(ValueError, match="offset contains missing"):
        model.predict(X[:2], offset=np.array([0.0, np.nan]))
