import numpy as np
from itertools import cycle
from utils import (check_commensurate, check_intercept, check_offset,
                   check_sample_weights, has_converged, soft_threshold)


class ElasticNet:

    def __init__(self, lam, alpha, max_iter=10, tol=0.1**5):
        self.lam = lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y, offset=None, sample_weights=None, 
                        active_coef_list=None, j_to_active_map=None, active_coefs=None,
                        xy_dots=None, xx_dots=None):
        """Fit an elastic net with coordinate descent."""
        check_commensurate(X, y)
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])
        check_sample_weights(y, sample_weights)

        # We are choosing data structures here for efficiency.
        n_samples = X.shape[0]
        n_coef = X.shape[1]
        self.intercept_ = np.mean(y)
        if active_coefs is None:
            active_coefs = np.zeros(n_coef) 
        n_active_coefs = np.sum(active_coefs != 0)
        if active_coef_list is None:
            active_coef_list = []
            active_coef_set = set(active_coef_list)
        if j_to_active_map is None:
            j_to_active_map = {j: n_coef - 1 for j in range(n_coef)}
        if xy_dots is None:
            xy_dots = np.dot(X.T, y)
        if xx_dots is None:
            xx_dots = np.zeros((n_coef, n_coef))
        lambda_alpha = self.lam * self.alpha
        update_denom = 1 + self.lam * (1 - self.alpha)

        loss = np.inf
        prior_loss = None
        n_iter = 0
        is_converged = False
        while n_iter < self.max_iter and not is_converged:
            for j in range(n_coef):
                print()
                print("loop coef: ", j)
                print("coef: ", j)
                print("active coef idxs: ", active_coef_list)
                print("active coefs: ", active_coefs)
                print("xx_dots:\n", xx_dots)
                xj_dot_residual = (
                    xy_dots[j] 
                    - self.intercept_
                    - np.sum(xx_dots[j, :n_active_coefs] * active_coefs[:n_active_coefs]))
                partial_residual = (
                    (1 / n_samples) * xj_dot_residual 
                    + active_coefs[j_to_active_map[j]])
                new_coef = soft_threshold(partial_residual, lambda_alpha) / update_denom
                if j in active_coef_set:
                    active_coefs[j_to_active_map[j]] = new_coef
                elif new_coef != 0.0:
                    n_active_coefs += 1
                    j_to_active_map[j] = n_active_coefs - 1
                    xx_dots[j, n_active_coefs - 1] = np.dot(X[:, j], X[:, j])
                    for idx, active_coef_idx in enumerate(active_coef_list):
                        dprod = np.dot(X[:, j], X[:, active_coef_idx])
                        xx_dots[j, idx] = dprod
                        xx_dots[idx, n_active_coefs - 1] = dprod
                    active_coefs[n_active_coefs - 1] = new_coef
                    active_coef_list.append(j)
                    active_coef_set.add(j)
            n_iter += 1
