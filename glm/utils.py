from collections.abc import Sequence
from typing import Any, get_args

import numpy as np
import numpy.typing as npt

from glm.types import Deviance, DesignMatrix, Offset, ResponseVector, Standardization

STANDARDIZATIONS: tuple[str, ...] = get_args(Standardization.__value__)


def is_commensurate(X: DesignMatrix, y: ResponseVector) -> bool:
    return bool(X.shape[0] == y.shape[0])


def check_commensurate(X: DesignMatrix, y: ResponseVector) -> None:
    if not is_commensurate(X, y):
        raise ValueError("X and y are not commensurate.")


def has_intercept_column(X: DesignMatrix) -> bool:
    return bool(np.all(X[:, 0] == 1.0))


def check_intercept(X: DesignMatrix) -> None:
    if not has_intercept_column(X):
        raise ValueError("Column 0 of X, X[:, 0], is not an intercept.")


def has_non_finite_values(v: npt.NDArray[Any]) -> bool:
    return bool(not np.isfinite(v).all())


def check_finite(name: str, v: npt.NDArray[Any]) -> None:
    """Raise if an array has missing or infinite values, which no fit or
    prediction can use."""
    if has_non_finite_values(v):
        raise ValueError(f"{name} contains missing (NaN) or infinite values.")


def has_same_length(v: npt.NDArray[Any], w: npt.NDArray[Any]) -> bool:
    return bool(v.shape[0] == w.shape[0])


def check_offset(y: ResponseVector, offset: Offset) -> None:
    if not has_same_length(y, offset):
        raise ValueError("Offset array and y are not the same length.")


# Used by glm.glmnet, which is not yet annotated; GLM uses check_weights.
def check_sample_weights(
    y: ResponseVector, sample_weights: npt.NDArray[np.float64]
) -> None:
    if not has_same_length(y, sample_weights):
        raise ValueError("Sample weights array and y are not the same length.")


def check_weights(
    name: str, y: ResponseVector, weights: npt.NDArray[np.float64] | None
) -> npt.NDArray[np.float64]:
    """Check one kind of weights, defaulting to ones."""
    if weights is None:
        return np.ones(y.shape[0])
    if not has_same_length(y, weights):
        raise ValueError(f"{name} and y are not the same length.")
    check_finite(name, weights)
    if np.any(weights < 0):
        raise ValueError(f"{name} must be non-negative.")
    return weights


def check_full_rank(X: DesignMatrix, X_names: list[str]) -> None:
    """Raise if the columns of X are linearly dependent, so that an
    unpenalized model's coefficients aren't uniquely determined.

    numpy's solvers don't catch this: they raise only on an exactly zero pivot,
    and rounding error leaves a rank deficient information matrix with tiny,
    non-zero ones. The fit then converges, with coefficients that are arbitrary
    along the dependent directions, and nan standard errors.

    Pass only the rows with positive weight. The information matrix XᵗWX, with
    W the diagonal working weights, has the rank of W^(1/2) X, and on rows with
    positive weight W^(1/2) is invertible, so it has the rank of X. Checking X,
    rather than XᵗWX, also keeps rank deficiency, where the coefficients aren't
    determined, apart from tiny working weights, which leave them determined
    but poorly conditioned.
    """
    n, p = X.shape
    if n < p:
        raise ValueError(
            f"The design matrix is rank deficient: it has more coefficients than "
            f"observations ({p} > {n}), so the coefficients aren't uniquely "
            f"determined. Fit with a ridge penalty (alpha > 0), or use fewer "
            f"columns."
        )
    rank = np.linalg.matrix_rank(X)
    if rank == p:
        return
    # The right singular vectors for the vanishing singular values span the
    # null space, so the columns with a weight in any of them are the dependent
    # ones. matrix_rank counts the singular values above its tolerance, which
    # come first, as svd sorts them in decreasing order.
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    null_basis = Vt[rank:]
    dependent = np.any(np.abs(null_basis) > 1e-8, axis=0)
    names = [name for name, d in zip(X_names, dependent) if d]
    raise ValueError(
        f"The design matrix is rank deficient: the columns {names} are linearly "
        f"dependent, so their coefficients aren't uniquely determined. Remove "
        f"the redundancy, e.g. by dropping a reference level of a categorical "
        f"variable, or centering a spline basis with constraints='center'; or "
        f"fit with a ridge penalty (alpha > 0)."
    )


def check_standardization(standardize: object) -> None:
    if standardize not in STANDARDIZATIONS:
        raise ValueError(
            f"standardize must be one of {STANDARDIZATIONS}, got {standardize!r}."
        )


def check_integer_at_least(name: str, value: object, minimum: int) -> None:
    # bool is a subclass of int, but True isn't a count.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer of at least {minimum}, got {value!r}."
        )


def check_alphas(alphas: Sequence[float] | None, n_alphas: int) -> None:
    """Raise unless the supplied alphas, or else the size of the default grid,
    are valid."""
    if alphas is not None:
        if len(alphas) == 0 or not all(alpha > 0 for alpha in alphas):
            raise ValueError("alphas must be a non-empty sequence of positives.")
    else:
        check_integer_at_least("n_alphas", n_alphas, 2)


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
