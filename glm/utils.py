import numpy as np
import pandas as pd


def check_types(X, y, formula):
    if not formula:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array if no formula is supplied, "
                                "got a {type(X)}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array if no formula is supplied, "
                                "got a {type(y)}")
    if formula:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame if a formula is supplied, "
                                "got a {type(X)}")

def is_commensurate(X, y):
    return X.shape[0] == y.shape[0]

def check_commensurate(X, y):
    if not is_commensurate(X, y):
        raise ValueError("X and y are not commensurate.")

def has_intercept_column(X):
    return np.all(X[:, 0] == 1.0)

def check_intercept(X):
    if not has_intercept_column(X):
        raise ValueError("First column in matrix X is not an intercept.")

def has_same_length(v, w):
    return v.shape[0] == w.shape[0]

def check_offset(y, offset):
    if not has_same_length(y, offset):
        raise ValueError("Offset array and y are not the same length.")

def check_sample_weights(y, sample_weights):
    if not has_same_length(y, sample_weights):
        raise ValueError("Sample weights array and y are not the same length.")

def has_converged(loss, loss_prev, tol):
    if loss_prev == np.inf:
        return False
    rel_change = np.abs((loss - loss_prev) / loss_prev)
    return rel_change < tol

def soft_threshold(z, gamma):
    abs_z = np.abs(z)
    if gamma >= abs_z:
        return 0.0
    return np.sign(z) * (abs_z - gamma)

def weighted_means(X, weights):
    return np.mean(X * weights.reshape(-1, 1), axis=0) 

def weighted_dot(x0, x1, weights):
    return np.dot(x0, weights * x1)

def weighted_column_dots(X, weights):
    return np.sum(X * X * weights.reshape(-1, 1), axis=0)
