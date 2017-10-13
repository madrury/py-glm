import numpy as np
from itertools import cycle
from .utils import (check_commensurate, check_intercept, check_offset,
                   check_sample_weights, has_converged, soft_threshold,
                   weighted_means, weighted_dot, weighted_column_dots)


class ElasticNet:
    """Fit an elastic net model.

    The elastic net is a regularized regularized linear regression model
    incorporating both L1 and L2 penalty terms.  It is fit by minimizing the
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
        The maximum number of coordinate descent cycles before breaking out of
        the descent loop.

    tol: float
        The convergence tolerance.  The actual convergence tolerance is applied
        to the absolute multiplicative change in the coefficient vector. When
        the coefficient vector changes by less than n_params * tol in one full
        coordinate descent cycle, the algorithm terminates.

    Attributes
    ----------
    intercept_: float
        The fit intercept in the regression.  This is stored separately, as the
        penalty terms are *not* applied to the intercept.

    n: integer, positive
        The number of samples used to fit the model, or the sum of the sample
        weights.

    p: integer, positive
        The number of fit parameters in the model.

    Additionally, the following private attributes are used by the fitting
    algorithm.  They are stored permanently so they can be used as warm starts
    to other ElasticNet objects. This is used both when fitting an entire
    regularization path of models, and also when fitting Glmnet objects, which
    procede by solving quadratic approximations to the Glmnet loss using the
    ElasticNet.

    The array below (and many of the other arrays used internally during
    fitting) use a peculiar ordering.  Instead of being arranged to match the
    order of the features in a training matrix X, they are instead arranged in
    order that the predictors enter the model.  This allows for efficient
    calculation of the update steps in ElasticNet.fit.

    _active_coefs: array, shape (n_features,)
        The active set of coefficients. They are stored in the order that their
        associated features enter into the model, i.e. the j'th coefficient in
        this list is associated with the j'th feature to enter the model.

    _active_coef_idx_list: array, shape (n_features,)
        The indices of the active coefficients into the column dimension of
        the training data.  I.e. the j'th active coefficient is associated with
        the _active_coef_idx_list[j]'th column of X.

    _j_to_active_map: dictionary, positive integer => positive integer
        Maps a column index of X to an index into _active_coefs.  I.e., the
        position of the j'th column of X in the order in which coefficients
        enter into the model.

    References
    ----------
    The implementation here is based on the discussion in Friedman, Hastie, and
    Tibshirani: Regularization Paths for Generalized Linear Models via
    Coordinate Descent (hereafter referenced as [FHT]).
    """
    def __init__(self, lam, alpha, max_iter=10, tol=0.1**5):
        self.lam = lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n = None
        self.p = None
        self._active_coefs = None
        self._active_coef_idx_list = None
        self._j_to_active_map = None

    def fit(self, X, y, offset=None, sample_weights=None, warm_start=None):
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
            by its corresponding weight.

        Returns
        -------
        self: ElasticNet object
            The fit model.

        Notes
        -----
        The following data structures are used internally by the fitting
        algorithm.

        x_means: array, shape (n_features,)
            The weighted column means of the training data X.

        xy_dots: array, shape (n_features,)
            The dot products of the columns in the training matrix X with the
            target y. Arranged in the same order as the columns of X.

        offset_dots: array, shape(n_features,)
            The weighted dot products of the columns of X with the offset
            vector.

        xx_dots: array, shape (n_features,)
            The weighted dot products of the columns of the training matrix X
            with themselves.  I.e., each individual column with itself.

        xtx_dots: array, shape (n_features, n_features)
            The matrix of weighted dot products of the columns of the training
            data X with themselves; this includes all such dot products, not
            just those between columns and themselves. The columns of this
            matrix are permuted so that the dot products stored in column j of
            xtx_dots are associated with the j'th coefficient to enter the
            model. The rows are in order of the columns of X. This matrix is
            initialized to zero, and filled in lazily as coefficients enter the
            model, as the cross-column dot products are only needed for the
            features currently active in the model.

        In addition, see the Attributes notes in the documentation of this
        class for information on how the coefficient estimates are managed
        internally.
        """
        # -- Check inputs for validity.
        check_commensurate(X, y)
        check_intercept(X)
        if sample_weights is None:
            sample_weights = (1 / X.shape[0]) * np.ones(X.shape[0])
        check_sample_weights(y, sample_weights)
        if offset is None:
            offset = np.zeros(X.shape[0])
        check_offset(y, offset)

        # -- Set up initial data structures.
        n_samples = X.shape[0]
        n_coef = X.shape[1]
        # Data structures used for managing the working coefficient estimates.
        if warm_start is None:
            active_coefs = np.zeros(n_coef)
            active_coef_idx_list = []
            j_to_active_map = {j: n_coef - 1 for j in range(n_coef)}
        else:
            active_coefs = warm_start._active_coefs
            active_coef_idx_list = warm_start._active_coef_idx_list
            j_to_active_map = warm_start._j_to_active_map
        active_coef_set = set(active_coef_idx_list)
        n_active_coefs = np.sum(active_coefs != 0)
        # Data structures holding weighted dot products, used in the
        # coefficient update calculations.
        weight_sum = np.sum(sample_weights)
        x_means = weighted_means(X, sample_weights)
        y_means = np.sum(sample_weights * y)
        xy_dots = weighted_dot(X.T, y, sample_weights)
        offset_dots = weighted_dot(X.T, offset, sample_weights)
        xx_dots = weighted_column_dots(X, sample_weights)
        xtx_dots = np.zeros((n_coef, n_coef))
        # Data elements involving the regularization strength, used in the
        # coefficient update calculations.
        lambda_alpha = self.lam * self.alpha
        update_denom_scale = self.lam * (1 - self.alpha)

        # -- Fit the model by coordinate-wise descent.
        loss = np.inf
        n_iter = 0
        is_converged = False
        previous_coefs = np.empty(n_coef)
        while n_iter < self.max_iter and not is_converged:
            previous_coefs[:] = active_coefs
            for j in range(n_coef):
                partial_residual = self._compute_partial_residual(
                    x_means, xy_dots, xx_dots, xtx_dots, offset_dots,
                    j, active_coefs, j_to_active_map, n_active_coefs)
                if j == 0:
                    new_coef = partial_residual / weight_sum 
                else:
                    update_denom = xx_dots[j] + update_denom_scale
                    partial_resid_threshold = soft_threshold(
                        partial_residual, lambda_alpha)
                    new_coef = partial_resid_threshold / update_denom
                if j in active_coef_set:
                    active_coefs[j_to_active_map[j]] = new_coef
                elif new_coef != 0.0:
                    n_active_coefs += 1
                    j_to_active_map[j] = n_active_coefs - 1
                    active_coefs[n_active_coefs - 1] = new_coef
                    active_coef_idx_list.append(j)
                    active_coef_set.add(j)
                    xtx_dots = self._update_xtx_dots(
                        xtx_dots, X, j, sample_weights,
                        n_active_coefs, active_coef_idx_list)
            is_converged = self._check_converged(
                active_coefs, previous_coefs, n_coef)
            n_iter += 1

        # -- Wrap up.
        self.n = n_samples
        self.p = n_coef
        self._active_coef_idx_list = active_coef_idx_list
        self._j_to_active_map = j_to_active_map
        self._active_coefs = active_coefs
        return self

    def _compute_partial_residual(self,
                                  x_means, xy_dots, xx_dots, xtx_dots, offset_dots,
                                  j, active_coefs, j_to_active_map, n_active_coefs):
        """Compute the partial residual used in the elastic net update rule.

        The partial residual is the residual from the predictions using the
        current model, excluding the coefficient that is currently being
        updated in the coordinate-wise descent (this is the partial in partial
        residual).

        Reference
        ---------
        The equations 5, 6, 8, and 9 in [FHT] describe the basic and weighted
        cases. Adding support for an offset is a simple elaboration on the
        ideas included there.
        """
        xj_dot_partial_prediction = (
            + np.sum(xtx_dots[j, :n_active_coefs] * active_coefs[:n_active_coefs]))
        xj_dot_residual = (
            xy_dots[j] - xj_dot_partial_prediction - offset_dots[j])
        partial_residual = (
            xj_dot_residual + xx_dots[j] * active_coefs[j_to_active_map[j]])
        return partial_residual

    def _update_xtx_dots(self,
                         xtx_dots, X, j, sample_weights,
                         n_active_coefs, active_coef_idx_list):
        """Update the xtx_dots matrix of weighted dot products of the columns
        in the training data with the products involving column j. This is used
        when a new predictor enters the model.
        """
        xtx_dots[j, n_active_coefs - 1] = weighted_dot(
            X[:, j], X[:, j], sample_weights)
        for idx, active_coef_idx in enumerate(active_coef_idx_list):
            dprod = weighted_dot(X[:, j], X[:, active_coef_idx], sample_weights)
            xtx_dots[j, idx] = dprod
            xtx_dots[idx, n_active_coefs - 1] = dprod
        return xtx_dots

    def _check_converged(self, active_coefs, previous_coefs, n_coef):
        """Check for convergence.

        Since prediction requires an expensive rearrangement of coefficients,
        instead of checking if the loss has stabilized, we instead check if the
        coefficients themselves have stabilized.
        """
        relative_change = np.sum(np.abs((active_coefs - previous_coefs)))
        return relative_change < n_coef * self.tol

    @property
    def coef_(self):
        """The coefficient estimates of a fit model.

        This attribute returns the coefficient estimates in the same order as
        the associated columns in X.
        """
        coef = np.zeros(self.p)
        for i, col_idx in enumerate(self._active_coef_idx_list):
            coef[col_idx] = self._active_coefs[i]
        return coef

    def predict(self, X):
        """Return predictions given an array X.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Data.

        Returns
        -------
        preds: array, shape (n_samples, )
            Predictions.
        """
        return np.dot(X, self.coef_)
