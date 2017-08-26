import numpy as np

class GLM:
    """A generalized linear model.

    GLMs are a generalization of the classical linear and logistic models to
    other conditional distributions of response y.  A GLM is specified by a
    *link function* G and a family of *conditional distributions* dist, with
    the model specification given by

        y | X ~ dist(theta = G(X * beta)

    Here beta are the parameters fit in the model, with X * beta a matrix
    multiplication just like in linear regression.  Above, theta is a
    *parameter* of the one parameter family of distributions dist.
    
    In this implementation, a specific GLM is specified with a *family* object
    of ExponentialFamily type, which contains the information about the
    conditional distribution of y, and its connection to X, needed to construct
    the model. See the documentation for ExponentialFamily for
    details.

    The model is fit to data using the well known Fisher Scoring algorithm,
    which is a version of Newton's method where the hessian is replaced with
    its expectation with respect to the assumed distribution of Y.

    Parameters
    ----------
    family: ExponentialFamily object
        The exponential family used in the model.

    alpha: float, non-negative
        The ridge regularization strength. If non-zero, the loss function
        minimized is a penalized deviance, where the penalty is alpha *
        np.sum(model.coef_**2).

    Attributes
    ----------
    family: ExponentialFamily object
        The exponential family used in the model.

    alpha: float, non-negative
        The regularization strength.

    coef_: array, shape (n_features, )
        The fit parameter estimates.  None if the model has not yet been fit.

    deviance_: float
        The final deviance of the fit model on the training data.

    Notes
    -----
    Instead of supplying a `fit_intercept` argument, we have instead assumed
    the programmer has included a column of ones as the *first* column X. The
    fit method will throw an exception if this is not the case.
    """
    def __init__(self, family, alpha=0.0):
        self.family = family
        self.alpha = alpha
        self.coef_ = None 
        self.deviance_ = None

    def fit(self, X, y, warm_start=None, offset=None, sample_weights=None,
                        max_iter=100, tol=0.1**5):
        """Fit the GLM model to training data.

        Fitting the model uses the well known Fisher scoring algorithm.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Training data.

        y: array, shape (n_samples, )
            Target values.

        warm_start: array, shape (n_features, )
            Initial values to use for the parameter estimates in the
            optimization, useful when fitting an entire regulatization path of
            models.  If not supplies, the initial intercept estimate will be
            the mean of the target array, and all other parameter estimates
            will be initialized to zero.

        offset: array, shape (n_samples, )
            Offsets for samples.  If provided, the model fit is
                E[Y|X] = family.inv_link(np.dot(X, coef) + offset)
            This is specially useful in models with exposures, as in Poisson
            regression.

        sample_weights: array, shape (n_sample, )
            Sample weights used in the deviance minimized by the model. If
            provided, each term in the deviance being minimized is multiplied
            by its corrosponding weight.

        max_iter: positive integer
            The maximum number of iterations for the fitting algorithm.

        tol: float, non-negative and less than one
            The convergence tolerance for the fitting algorithm.  The relative
            change in deviance is compared to this tolerance to check for
            convergence.

        Returns
        -------
        self: GLM object
            The fit model.
        """
        if not self._check_intercept(X):
            raise ValueError("First column in matrix X is not an intercept.")
        if warm_start is None:
            initial_intercept = np.mean(y)
            warm_start = np.zeros(X.shape[1])
            warm_start[0] = initial_intercept
        coef = warm_start
        if offset is None:
            offset = np.zeros(X.shape[0])
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])

        family = self.family
        penalized_deviance = np.inf
        is_converged = False
        n_iter = 0
        while n_iter < max_iter and not is_converged:
            nu = np.dot(X, coef) + offset
            mu = family.inv_link(nu)
            dmu = family.d_inv_link(nu, mu)
            var = family.variance(mu)
            dbeta = self._compute_dbeta(X, y, mu, dmu, var, sample_weights)           
            ddbeta = self._compute_ddbeta(X, dmu, var, sample_weights)
            if self.alpha != 0.0:
                dbeta = dbeta + self._d_penalty(coef)
                ddbeta = self._dd_penalty(ddbeta, X)
            coef = coef - np.linalg.solve(ddbeta, dbeta)
            penalized_deviance_previous = penalized_deviance
            penalized_deviance = family.penalized_deviance(
                y, mu, self.alpha, coef)
            is_converged = self._has_converged(
                penalized_deviance, penalized_deviance_previous, tol)
            n_iter += 1

        self.coef_ = coef
        self.deviance_ = family.deviance(y, mu)
        return self

    def predict(self, X):
        """Return predictions from a fit model.

        Predictions are computed using the inverse link function in the family
        used to fit the model:

            preds = family.inv_link(np.dot(X, self.coef_)

        Note that in the case of binary models, predict *does not* make class
        assignmnets, it returns a probability of class membership.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Data set.

        Returns
        -------
        preds: array, shape (n_samples, )
            Model predictions.
        """
        if not self._is_fit():
            raise ValueError(
                "Model is not fit, and cannot be used to make predictions.")
        return self.family.inv_link(np.dot(X, self.coef_))

    def score(self, X, y):
        """Return the deviance of a fit model on a given dataset.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Data set.

        y: array, shape (n_samples, )
            Labels for X.

        Returns
        -------
        deviance: array, shape (n_samples, )
            Model deviance scored using supplied data and labels.
        """
        return self.family.deviance(y, self.predict(X))

    def _is_fit(self):
        return self.coef_ is not None

    def _check_intercept(self, X):
       return np.all(X[:, 0] == 1.0)

    def _compute_dbeta(self, X, y, mu, dmu, var, sample_weights):
        working_residuals = sample_weights * (y - mu) * (dmu / var)
        return - np.sum(X * working_residuals.reshape(-1, 1), axis=0)

    def _compute_ddbeta(self, X, dmu, var, sample_weights):
        working_h_weights = (sample_weights * dmu**2 / var).reshape(-1, 1)
        return np.dot(X.T, X * working_h_weights)

    def _d_penalty(self, coef):
        dbeta_penalty = coef.copy()
        dbeta_penalty[0] = 0.0
        return dbeta_penalty

    def _dd_penalty(self, ddbeta, X):
        diag_idxs = list(range(1, X.shape[1]))
        ddbeta[diag_idxs, diag_idxs] += self.alpha
        return ddbeta

    def _has_converged(self, dev, dev_prev, tol):
        if dev_prev == np.inf:
            return False
        rel_change = np.abs((dev - dev_prev) / dev_prev)
        return rel_change < tol
