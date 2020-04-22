import numpy as np
import scipy.stats as sts
import patsy as pt

from .utils import (check_types, check_commensurate, check_intercept,
                    check_offset, check_sample_weights, has_converged,
                    default_X_names, default_y_name)


class GLM:
    """A generalized linear model.

    GLMs are a generalization of the classical linear and logistic models to
    other conditional distributions of response y.  A GLM is specified by a
    *link function* G and a family of *conditional distributions* dist, with
    the model specification given by

        y | X ~ dist(theta = G(X * beta))

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

    formula: str
        An (optional) formula specifying the model.  Used in the case that X is
        passed as a DataFrame.  For documentation on model formulas, please see
        the patsy library documentation.

    X_info: patsy.design_info.DesignInfo object.
        Contains information about the model formula used to process the
        training data frame into a design matrix.

    X_names: List[str]
        Names for the predictors.

    y_names: str
        Name for the target varaible.

    coef_: array, shape (n_features, )
        The fit parameter estimates.  None if the model has not yet been fit.

    deviance_: float
        The final deviance of the fit model on the training data.

    information_matrix_: array, shape (n_features, n_features)
        The estimated information matrix. This information matrix is evaluated
        at the fit parameters.

    n: integer, positive
        The number of samples used to fit the model, or the sum of the sample
        weights.

    p: integer, positive
        The number of fit parameters in the model.

    Notes
    -----
    Instead of supplying a `fit_intercept` argument, we have instead assumed
    the programmer has included a column of ones as the *first* column X. The
    fit method will throw an exception if this is not the case.
    """
    def __init__(self, family, alpha=0.0):
        self.family = family
        self.alpha = alpha
        self.formula = None
        self.X_info = None
        self.X_names = None
        self.y_name = None
        self.coef_ = None
        self.deviance_ = None
        self.n = None
        self.p = None
        self.information_matrix_ = None

    def fit(self, X, y=None, formula=None, *,
            X_names=None,
            y_name=None,
            **kwargs):
        """Fit the GLM model to training data.

        Fitting the model uses the well known Fisher scoring algorithm.

        Parameters
        ----------
        X: array, shape (n_samples, n_features) or pd.DataFrame
            Training data.

        y: array, shape (n_samples, )
            Target values.

        formula: str
            A formula specifying the model.  Used in the case that X is passed
            as a DataFrame.  For documentation on model formulas, please see
            the patsy library documentation.

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

        validate_intercept: bool
            Should we defensively check that the first column in the deign
            matrix is an intercept? Defaults to True.

        Returns
        -------
        self: GLM object
            The fit model.
        """
        check_types(X, y, formula)
        if formula:
            self.formula = formula
            y_array, X_array = pt.dmatrices(formula, X)
            self.X_info = X_array.design_info
            self.X_names = X_array.design_info.column_names
            self.y_name = y_array.design_info.term_names[0]
            y_array = y_array.squeeze()
            return self._fit(X_array, y_array, **kwargs)
        else:
            if X_names:
                self.X_names = X_names
            else:
                self.X_names = default_X_names(X)
            if y_name:
                self.y_name = y_names
            else:
                self.y_name = default_y_name()
            return self._fit(X, y, **kwargs)

    def _fit(self, X, y, *,
             warm_start=None,
             offset=None,
             sample_weights=None,
             max_iter=100,
             tol=0.1**5,
             validate_intercept=True):
        """Fit the GLM model to some training data.

        This method expects X and y to be numpy arrays.
        """
        check_commensurate(X, y)
        if validate_intercept:
            check_intercept(X)
        if warm_start is None:
            initial_intercept = np.mean(y)
            warm_start = np.zeros(X.shape[1])
            warm_start[0] = initial_intercept
        coef = warm_start
        if offset is None:
            offset = np.zeros(X.shape[0])
        check_offset(y, offset)
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])
        check_sample_weights(y, sample_weights)

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
            if self._is_regularized():
                dbeta = dbeta + self._d_penalty(coef)
                ddbeta = self._dd_penalty(ddbeta, X)
            coef = coef - np.linalg.solve(ddbeta, dbeta)
            penalized_deviance_previous = penalized_deviance
            penalized_deviance = family.penalized_deviance(
                y, mu, self.alpha, coef)
            is_converged = has_converged(
                penalized_deviance, penalized_deviance_previous, tol)
            n_iter += 1

        self.coef_ = coef
        self.deviance_ = family.deviance(y, mu)
        self.n = np.sum(sample_weights)
        self.p = X.shape[1]
        self.information_matrix_ = self._compute_ddbeta(X, dmu, var, sample_weights)
        return self

    def predict(self, X, offset=None):
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

        offset: array, shape (n_samples, )
            Offsets to add on the linear scale when making predictions.

        Returns
        -------
        preds: array, shape (n_samples, )
            Model predictions.
        """
        if not self._is_fit():
            raise ValueError(
                "Model is not fit, and cannot be used to make predictions.")
        if self.formula:
            X = self._make_rhs_matrix(X)
        if offset is None:
            return self.family.inv_link(np.dot(X, self.coef_))
        else:
            return self.family.inv_link(np.dot(X, self.coef_) + offset)

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

    @property
    def dispersion_(self):
        """Return an estimate of the dispersion parameter phi."""
        if not self._is_fit():
            raise ValueError("Dispersion parameter can only be estimated for a"
                             "fit model.")
        if self.family.has_dispersion:
            return self.deviance_ / (self.n - self.p)
        else:
            return np.ones(shape=self.deviance_.shape)

    @property
    def coef_covariance_matrix_(self):
        if not self._is_fit():
            raise ValueError("Parameter covariances can only be estimated for a"
                             "fit model.")
        return self.dispersion_ * np.linalg.inv(self.information_matrix_)

    @property
    def coef_standard_error_(self):
        return np.sqrt(np.diag(self.coef_covariance_matrix_))

    @property
    def p_values_(self):
        """Return an array of p-values for the fit coefficients.  These
        p-values test the hypothesis that the given parameter is zero.

        Note: We use the asymptotic normal approximation to the p-values for
        all models.
        """
        if self.alpha != 0:
            raise ValueError("P-values are not available for "
                             "regularized models.")
        p_values = []
        null_dist = sts.norm(loc=0.0, scale=1.0)
        for coef, std_err in zip(self.coef_, self.coef_standard_error_):
            z = abs(coef) / std_err
            p_value = null_dist.cdf(-z) + (1 - null_dist.cdf(z))
            p_values.append(p_value)
        return np.asarray(p_values)

    def summary(self):
        """Print a summary of the model."""
        variable_names = self.X_names
        parameter_estimates = self.coef_
        standard_errors = self.coef_standard_error_
        longest_var_name_length = max(len(name) + 2 for name in variable_names)
        format_string = "{:<" + str(longest_var_name_length) + "} {:>20} {:>15}"
        header_string = format_string.format("Name", "Parameter Estimate", "Standard Error")
        print(f"{self.family.__class__.__name__} GLM Model Summary.")
        print('=' * len(header_string))
        print(header_string)
        print('-' * len(header_string))
        format_string = "{:<" + str(longest_var_name_length) + "} {:>20.2f} {:>15.2f}"
        for name, est, se in zip(variable_names, parameter_estimates, standard_errors):
            print(format_string.format(name, est, se))

    def clone(self):
        return self.__class__(self.family, self.alpha)

    def _is_fit(self):
        return self.coef_ is not None

    def _is_regularized(self):
        return self.alpha > 0.0

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

    def _make_rhs_matrix(self, X):
        formula_parts = self.formula.split('~')
        if len(formula_parts) == 2:
            rhs_formula = formula_parts[1].strip()
        elif len(formula_parts) == 1:
            rhs_formula = formula_parts.strip()
        else:
            raise ValueError(
                f"Cannot parse model formula {self.formula} to determine right hand side!")
        X = pt.dmatrix(rhs_formula, X)
        return X
