import numpy as np
from itertools import cycle
from utils import (check_commensurate, check_intercept, check_offset,
                   check_sample_weights, has_converged, soft_threshold)


class ElasticNet:
    """Fit an elastic net model.

    The elastic net is a regularized regularized linear regression model
    incorperating both L1 and L2 penalty terms.  It is fit by minimizing the
    penalized loss function:

        sum((y - y_hat)**2) 
            + lam * ((1 - alpha)/2 * sum(beta**2)
            + alpha * sum(abs(beta)))

    Parameters
    ----------
    lam: float
        The overall strength of the regularization.

    alpha: float in the interval [0, 1]
        The relative strengths of the L1 and L2 regularization.

    max_iter: positive integer
        The maximum number of coordinate descent cycles before brekaing out of
        the descent loop.

    tol: float
        The convergence tolerance.  The actual convergence tolerance is applied
        to the absolute multiplicitive change in the coefficient vector. When
        the coefficient vector changes by less than n_params * tol in one full
        coordinate descent cycle, the algorithm terminates.

    Attributes
    ----------
    intercept_: float
        The fit intercept in the regression.  This is stored seperately, as the
        penalty terms are *not* applied to the intercept.

    Adittionally, the following private attrutes are used by the fitting
    algorithm.  They are stored permenanty so they can be used as warm starts
    to other ElasticNet objects. This is used both when fitting an entire
    regularization path of models, and also when fitting Glmnet objects, which
    procedure by solving quadratic approximations to the Glmnet loss using the
    ElasticNet.

    Many of the arrays below use a perculiar ordering.  Instead of being
    arranged to match the order of the features in a training matrix X, they
    are instead arranged in order that the predictors enter the model.  This
    allows for efficient calculation of teh update steps in ElasticNet.fit.

    _active_coefs: array, shape (n_features,)
        The active set of coefficients. They are stored in the order that thier
        associated features enter into the model, i.e. the j'th coefficient in
        this list is associated with the j'th feature to enter the model.

    _active_coef_idx_list: array, shape (n_features,)
        The indicies of the active coefficients into the column dimension of
        the training data.  I.e. the j'th active coefficient is associated with
        the _active_coef_idx_list[j]'th column of X.

    _j_to_active_map: dictionary, positive integer => positive integer
        Maps a column index of X to an index into _active_coefs.  I.e., the
        position of the j'th column of X in the order in which coefficients
        enter into the model.

    _xx_dots: array, shape (n_features, n_features)
        The matrix of dot products of the training data X with itself, but with
        the columns permuted so that the dot products in column j are
        associated with the j'th coefficient to enter the model. This matrix is
        initialzed to zero, and filled in lazily as coefficients enter the
        model.

    _xy_dots: array, shape (n_features,)
        The dot products of the columns in the training matrix X with the
        target y. Arranged in the same order as the columns of X.
    """
    def __init__(self, lam, alpha, max_iter=10, tol=0.1**5):
        self.lam = lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.intercept_ = None
        self._active_coefs = None
        self._active_coef_idx_list = None
        self._j_to_active_map = None
        self._xx_dots = None
        self._xy_dots = None

    def fit(self, X, y, offset=None, sample_weights=None, 
                        active_coef_idx_list=None, j_to_active_map=None, active_coefs=None,
                        xy_dots=None, xx_dots=None,
                        print_state=False):
        """Fit an elastic net with coordinate descent.
        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Training data.

        y: array, shape (n_samples, )
            Target values.

        offset: array, shape (n_samples, )
            Offsets for samples.  If provided, the model fit is
                E[Y|X] = np.dot(X, coef) + offset

        sample_weights: array, shape (n_sample, )
            Sample weights used in the deviance minimized by the model. If
            provided, each term in the deviance being minimized is multiplied
            by its corrosponding weight.

        Returns
        -------
        self: ElasticNet object
            The fit model.
        """
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
        if active_coef_idx_list is None:
            active_coef_idx_list = []
            active_coef_set = set(active_coef_idx_list)
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
                    self._print_state(j, active_coef_idx_list, active_coefs, xx_dots)
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
                    active_coef_idx_list.append(j)
                    active_coef_set.add(j)
                    xx_dots = self._update_xx_dots(
                        xx_dots, X, j, n_active_coefs, active_coef_idx_list)
            is_converged = self._check_converged(
                active_coefs, previous_coefs, n_coef)
            n_iter += 1
 
        self._active_coef_idx_list = active_coef_idx_list
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
            xy_dots[j] - self.intercept_ - np.sum(partial_prediction))
        partial_residual = (
            (1 / n_samples) * xj_dot_residual
            + active_coefs[j_to_active_map[j]])
        return partial_residual

    def _update_xx_dots(self,
                        xx_dots, X, j, n_active_coefs, active_coef_idx_list):
        xx_dots[j, n_active_coefs - 1] = np.dot(X[:, j], X[:, j])
        for idx, active_coef_idx in enumerate(active_coef_idx_list):
            dprod = np.dot(X[:, j], X[:, active_coef_idx])
            xx_dots[j, idx] = dprod
            xx_dots[idx, n_active_coefs - 1] = dprod
        return xx_dots

    def _check_converged(self, active_coefs, previous_coefs, n_coef):
        relative_change = np.sum(np.abs((active_coefs - previous_coefs)))
        return relative_change < n_coef * self.tol

    def _print_state(self, j, active_coef_idx_list, active_coefs, xx_dots):
        print()
        print("loop coef: ", j)
        print("active coef idxs: ", active_coef_idx_list)
        print("active coefs: ", active_coefs)
        print("xx_dots:\n", xx_dots)
