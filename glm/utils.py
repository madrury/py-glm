from typing import Any

import numpy as np
import numpy.typing as npt

from glm.types import Deviance, DesignMatrix, Offset, ResponseVector, SampleWeights


def is_commensurate(X: DesignMatrix, y: ResponseVector) -> bool:
    return bool(X.shape[0] == y.shape[0])


def check_commensurate(X: DesignMatrix, y: ResponseVector) -> None:
    if not is_commensurate(X, y):
        raise ValueError("X and y are not commensurate.")


def has_intercept_column(X: DesignMatrix) -> bool:
    return bool(np.all(X[:, 0] == 1.0))


def check_intercept(X: DesignMatrix) -> None:
    if not has_intercept_column(X):
        raise ValueError("First column in matrix X is not an intercept.")


def has_missing_values(v: npt.NDArray[Any]) -> bool:
    return bool(np.isnan(v).any())


def check_no_missing_values(name: str, v: npt.NDArray[Any]) -> None:
    if has_missing_values(v):
        raise ValueError(f"{name} contains missing (NaN) values.")


def has_same_length(v: npt.NDArray[Any], w: npt.NDArray[Any]) -> bool:
    return bool(v.shape[0] == w.shape[0])


def check_offset(y: ResponseVector, offset: Offset) -> None:
    if not has_same_length(y, offset):
        raise ValueError("Offset array and y are not the same length.")


def check_sample_weights(y: ResponseVector, sample_weights: SampleWeights) -> None:
    if not has_same_length(y, sample_weights):
        raise ValueError("Sample weights array and y are not the same length.")


def has_converged(loss: Deviance, loss_prev: Deviance, tol: float) -> bool:
    if loss_prev == np.inf:
        return False
    # The relative change in loss, |loss - loss_prev| / |loss_prev| < tol, written
    # without division so that a perfect fit, with zero loss, converges.
    return bool(abs(loss - loss_prev) <= tol * abs(loss_prev))


def default_X_names(X: DesignMatrix) -> list[str]:
    n = X.shape[1]
    return ["Intercept"] + [f"X{i}" for i in range(1, n)]


def default_y_name() -> str:
    return "y"


# The helpers below are used by glm.glmnet, which is not yet annotated.
def soft_threshold(z: Any, gamma: Any) -> Any:
    abs_z = np.abs(z)
    if gamma >= abs_z:
        return 0.0
    return np.sign(z) * (abs_z - gamma)


def weighted_means(X: npt.NDArray[Any], weights: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.mean(X * weights.reshape(-1, 1), axis=0)


def weighted_dot(
    x0: npt.NDArray[Any], x1: npt.NDArray[Any], weights: npt.NDArray[Any]
) -> Any:
    return np.dot(x0, weights * x1)


def weighted_column_dots(
    X: npt.NDArray[Any], weights: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    return np.sum(X * X * weights.reshape(-1, 1), axis=0)
