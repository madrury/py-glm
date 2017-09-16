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
                        xy_dots=None, xx_dots=None,
                        print_state=False):
        """Fit an elastic net with coordinate descent."""
        # Check inputs for validity.
        check_commensurate(X, y)
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])
        check_sample_weights(y, sample_weights)

        # Set up initial data structures if needed.
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

        # Fit the model by coordinatewise descent.
        loss = np.inf
        prior_loss = None
        n_iter = 0
        is_converged = False
        while n_iter < self.max_iter and not is_converged:
            previous_coefs = active_coefs
            for j in range(n_coef):
                if print_state:
                    self._print_state(j, active_coef_list, active_coefs, xx_dots)
                partial_residual = self._compute_partial_residual(
                    xy_dots, xx_dots, j, active_coefs, 
                    j_to_active_map, n_samples,
                    n_active_coefs)
                new_coef = (
                    soft_threshold(partial_residual, lambda_alpha) / update_denom)
                if j in active_coef_set:
                    active_coefs[j_to_active_map[j]] = new_coef
                elif new_coef != 0.0:
                    n_active_coefs += 1
                    j_to_active_map[j] = n_active_coefs - 1
                    active_coefs[n_active_coefs - 1] = new_coef
                    active_coef_list.append(j)
                    active_coef_set.add(j)
                    xx_dots = self._update_xx_dots(xx_dots, X, j, n_active_coefs)
            is_converged = self._check_converged(active_coefs, previous_coefs)
            n_iter += 1
 
        self._active_coef_list = active_coef_list
        self._j_to_active_map = j_to_active_map
        self._active_coefs = active_coefs
        self._xx_dots = xx_dots
        self._xy_dots = xy_dots
        return self

    def _compute_partial_residual(self, xy_dots, xx_dots, j, active_coefs, 
                                        j_to_active_map, n_samples,
                                        n_active_coefs,):
        partial_prediction = (
            xx_dots[j, :n_active_coefs] * active_coefs[:n_active_coefs])
        xj_dot_residual = (
            xy_dots[j] - self.intercept_ - np.sum(partial_prediction)
        partial_residual = (
            (1 / n_samples) * xj_dot_residual
            + active_coefs[j_to_active_map[j]])
        return partial_residual

    def _update_xx_dots(self, xx_dots, X, j, n_active_coefs):
        xx_dots[j, n_active_coefs - 1] = np.dot(X[:, j], X[:, j])
        for idx, active_coef_idx in enumerate(active_coef_list):
            dprod = np.dot(X[:, j], X[:, active_coef_idx])
            xx_dots[j, idx] = dprod
            xx_dots[idx, n_active_coefs - 1] = dprod
        return xx_dots

    def _check_converged(self, active_coefs, previous_coefs, n_coef):
        relative_change = np.sum(np.abs((active_coefs - previous_coefs)))
        return relative_change < n_coef * self.tol

    def _print_state(self, j, active_coef_list, active_coefs, xx_dots):
        print()
        print("loop coef: ", j)
        print("active coef idxs: ", active_coef_list)
        print("active coefs: ", active_coefs)
        print("xx_dots:\n", xx_dots)
