from typing import Self

import formulaic
import numpy as np
import pandas as pd
import scipy.stats as sts

from glm.families import ExponentialFamily
from glm.standardize import Standardizer, check_standardization
from glm.types import (
    CoefficientMask,
    CoefficientVector,
    CovarianceMatrix,
    Deviance,
    DevianceGradient,
    DesignFrame,
    DesignMatrix,
    DesignSpec,
    Formula,
    InformationMatrix,
    InverseLinkDerivative,
    MeanVector,
    Offset,
    PValueVector,
    ResponseVector,
    SampleWeights,
    StandardErrorVector,
    Standardization,
    VarianceVector,
)
from glm.utils import (
    check_commensurate,
    check_intercept,
    check_no_missing_values,
    check_offset,
    check_sample_weights,
    has_converged,
    has_intercept_column,
    default_X_names,
    default_y_name,
)

NON_FINITE_ESTIMATES_MESSAGE = (
    "Fitting failed: the coefficient estimates are not finite. This can happen "
    "when the maximum likelihood estimate does not exist, e.g. a binary response "
    "that is all zeros or all ones, or one perfectly separated by the predictors."
)


class GLM:
    """A generalized linear model.

    GLMs are a generalization of the classical linear and logistic models to
    other conditional distributions of response y. A GLM is specified by a
    *link function* G and a family of *conditional distributions* dist, with
    the model specification given by

        y | X ~ dist(theta = G(X * beta))

    Here beta are the parameters fit in the model, with X * beta a matrix
    multiplication just like in linear regression. Above, theta is a
    *parameter* of the one parameter family of distributions dist.

    In this implementation, a specific GLM is specified with a *family* object
    of ExponentialFamily type, which contains the information about the
    conditional distribution of y (and its connection to X) needed to construct
    the model. See the documentation for ExponentialFamily for details.

    The model is fit to data using the well known Fisher Scoring algorithm,
    which is a version of Newton's method where the hessian is replaced with its
    expectation with respect to the assumed distribution of Y.

    Parameters
    ----------
    family: ExponentialFamily object
        The exponential family used in the model.

    alpha: float, non-negative
        The ridge regularization strength. If non-zero, the loss function
        minimized is a penalized deviance, where the penalty is alpha times
        the sum of the squared coefficients, excluding the intercept, of the
        (by default, standardized) columns of the design matrix.

    standardize: "none", "continuous", or "all"
        Which columns of the design matrix to standardize, to unit standard
        deviation, before a penalized fit, so that the penalty does not depend
        on the predictors' units. "continuous" (the default) leaves indicator
        columns, those of zeros and ones, unscaled; "all" scales them too, as
        glmnet does. Coefficients are always reported on the original scale.
        Standardizing does not change unpenalized fits, so it is skipped when
        alpha is zero.

    Attributes
    ----------
    family: ExponentialFamily object
        The exponential family used in the model.

    alpha: float, non-negative
        The regularization strength.

    formula: str
        An (optional) formula specifying the model.  Used in the case that X is
        passed as a DataFrame.  For documentation on model formulas, please see
        the formulaic library documentation.

    X_spec: formulaic.ModelSpec
        The learned mapping from a data frame to the design matrix, including
        categorical levels and stateful transforms fit on the training data.
        Used to encode new data consistently when making predictions.

    y_spec: formulaic.ModelSpec
        The learned mapping from a data frame to the response vector, from the
        left hand side of the formula.

    X_names: List[str]
        Names for the predictors.

    y_names: str
        Name for the target varaible.

    coef_: array, shape (n_features, )
        The fit parameter estimates. None if the model has not yet been fit.

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
    Instead of supplying a `fit_intercept` argument, we have instead assumed the
    programmer has included a column of ones as column 0 of X, X[:, 0]. The fit
    method will throw an exception if this is not the case.
    """

    def __init__(
        self,
        family: ExponentialFamily,
        alpha: float = 0.0,
        standardize: Standardization = "continuous",
    ) -> None:
        check_standardization(standardize)
        self.family = family
        self.alpha = alpha
        self.standardize: Standardization = standardize
        self.formula: Formula | None = None
        self.X_spec: DesignSpec | None = None
        self.y_spec: DesignSpec | None = None
        self.X_names: list[str] | None = None
        self.y_name: str | None = None
        self.coef_: CoefficientVector | None = None
        self.deviance_: Deviance | None = None
        self.n: float | None = None
        self.p: int | None = None
        self.information_matrix_: InformationMatrix | None = None

    def fit(
        self,
        X: DesignMatrix | DesignFrame,
        y: ResponseVector | None = None,
        formula: Formula | None = None,
        *,
        X_names: list[str] | None = None,
        y_name: str | None = None,
        warm_start: CoefficientVector | None = None,
        offset: Offset | None = None,
        sample_weights: SampleWeights | None = None,
        max_iter: int = 100,
        tol: float = 0.1**5,
        validate_intercept: bool = True,
    ) -> Self:
        """Fit the GLM model to training data.

        Fitting the model uses the well known Fisher scoring algorithm.

        Parameters
        ----------
        X: array, shape (n_samples, n_features) or pd.DataFrame
            Training data.

        y: array, shape (n_samples, )
            Response /Target vector.

        formula: str
            A formula specifying the model.  Used in the case that X is passed
            as a DataFrame. For documentation on model formulas, please see
            the formulaic library documentation. For penalized models (alpha
            > 0), categorical variables are encoded with a column for every
            level, rather than dropping a reference level, so that the penalty
            treats all levels alike.

        warm_start: array, shape (n_features, )
            Initial values to use for the parameter estimates in the
            optimization, useful when fitting an entire regulatization path of
            models. If not supplied, the initial intercept estimate will be the
            mean of the target array, and all other parameter estimates will be
            initialized to zero.

        offset: array, shape (n_samples, )
            Offsets for samples. If provided, the model fit is
                E[Y|X] = family.inv_link(X @ coef + offset)
            This is specially useful in models with exposures, as in Poisson
            regression.

        sample_weights: array, shape (n_sample, )
            Sample weights used in the deviance minimized by the model. If
            provided, each term in the deviance being minimized is multiplied
            by its corrosponding weight.

        max_iter: positive integer
            The maximum number of iterations for the fitting algorithm.

        tol: float, non-negative and less than one
            The convergence tolerance for the fitting algorithm. The relative
            change in deviance is compared to this tolerance to check for
            convergence.

        validate_intercept: bool
            Should we defensively check that column 0 of the design matrix,
            X[:, 0], is an intercept? Defaults to True.

        Returns
        -------
        self: GLM object
            The fit model.
        """
        design_matrix, response = self._prepare_data(X, y, formula, X_names, y_name)
        return self._fit(
            design_matrix,
            response,
            warm_start=warm_start,
            offset=offset,
            sample_weights=sample_weights,
            max_iter=max_iter,
            tol=tol,
            validate_intercept=validate_intercept,
        )

    def _prepare_data(
        self,
        X: DesignMatrix | DesignFrame,
        y: ResponseVector | None,
        formula: Formula | None,
        X_names: list[str] | None,
        y_name: str | None,
    ) -> tuple[DesignMatrix, ResponseVector]:
        """Check the training data, and turn it into a design matrix and
        response vector, learning the model's formula specs and names."""
        if formula:
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    f"X must be a DataFrame if a formula is "
                    f"supplied, got a {type(X)}"
                )
            # Raise on missing values rather than dropping rows, which would
            # silently misalign the data with any offset or sample weights.
            # Penalized models encode every level of a categorical variable,
            # rather than dropping a reference level, so that the penalty
            # treats all levels alike; the penalty keeps the fit well defined.
            matrices = formulaic.model_matrix(
                formula,
                X,
                output="numpy",
                na_action="raise",
                ensure_full_rank=not self._is_regularized(),
            )
            if not isinstance(matrices, formulaic.ModelMatrices):
                raise ValueError(
                    f"Formula {formula!r} must specify a response, e.g. 'y ~ x'."
                )
            self.formula = formula
            self.X_spec = matrices.rhs.model_spec
            self.y_spec = matrices.lhs.model_spec
            self.X_names = list(matrices.rhs.model_spec.column_names)
            self.y_name = matrices.lhs.model_spec.column_names[0]
            design_matrix: DesignMatrix = np.asarray(matrices.rhs)
            response: ResponseVector = np.asarray(matrices.lhs).squeeze(axis=1)
        else:
            if not isinstance(X, np.ndarray):
                raise TypeError(
                    f"X must be a numpy array if no formula is "
                    f"supplied, got a {type(X)}"
                )
            if not isinstance(y, np.ndarray):
                raise TypeError(
                    f"y must be a numpy array if no formula is "
                    f"supplied, got a {type(y)}"
                )
            if X_names:
                self.X_names = X_names
            else:
                self.X_names = default_X_names(X)
            if y_name:
                self.y_name = y_name
            else:
                self.y_name = default_y_name()
            design_matrix, response = X, y
        return design_matrix, response

    def _fit(
        self,
        X: DesignMatrix,
        y: ResponseVector,
        *,
        warm_start: CoefficientVector | None,
        offset: Offset | None,
        sample_weights: SampleWeights | None,
        max_iter: int,
        tol: float,
        validate_intercept: bool,
    ) -> Self:
        """Fit the GLM model to some training data.

        This method expects X and y to be numpy arrays.
        """
        check_commensurate(X, y)
        check_no_missing_values("X", X)
        check_no_missing_values("y", y)
        if validate_intercept:
            check_intercept(X)
        if offset is None:
            offset = np.zeros(X.shape[0])
        check_offset(y, offset)
        check_no_missing_values("offset", offset)
        if sample_weights is None:
            sample_weights = np.ones(X.shape[0])
        check_sample_weights(y, sample_weights)
        check_no_missing_values("sample_weights", sample_weights)
        penalized_coefficients = self._penalized_coefficients(X)

        family = self.family
        if warm_start is None:
            # Start from the intercept-only model: its estimate of the mean is
            # the weighted mean response, so its intercept is that mean on the
            # linear predictor scale, net of the average offset.
            ȳ = np.average(y, weights=sample_weights)
            # The intercept-only model fits the same mean, ȳ, for every
            # observation, so its linear predictor is constant: take the first
            # element.
            μ_null = np.full(X.shape[0], ȳ)
            with np.errstate(divide="ignore"):
                intercept = family.link(μ_null)[0]
            # E.g. an all-zero Poisson or binary response: the intercept-only
            # estimate is infinite, so no finite estimate can exist.
            if not np.isfinite(intercept):
                raise ValueError(NON_FINITE_ESTIMATES_MESSAGE)
            warm_start = np.zeros(X.shape[1])
            warm_start[0] = intercept - np.average(offset, weights=sample_weights)
        # Penalized fits work on standardized columns, so that the penalty does
        # not depend on the predictors' units. Unpenalized fits are unchanged
        # by standardizing, so skip it for them.
        standardizer = None
        design, β = X, warm_start
        if self._is_regularized() and self.standardize != "none":
            standardizer = Standardizer(self.standardize).fit(X, sample_weights)
            design = standardizer.transform(X)
            β = standardizer.coef_to_standardized(warm_start)
        η = design @ β + offset
        μ = family.inv_link(η)
        dμ = family.d_inv_link(η, μ)
        var = family.variance(μ)
        penalized_deviance = np.inf
        is_converged = False
        n_iter = 0
        while n_iter < max_iter and not is_converged:
            η = design @ β + offset
            μ = family.inv_link(η)
            dμ = family.d_inv_link(η, μ)
            var = family.variance(μ)
            dβ = self._compute_dbeta(design, y, μ, dμ, var, sample_weights)
            ddβ = self._compute_ddbeta(design, dμ, var, sample_weights)
            if self._is_regularized():
                dβ = dβ + self._d_penalty(β, penalized_coefficients)
                ddβ = self._dd_penalty(ddβ, penalized_coefficients)
            β = β - np.linalg.solve(ddβ, dβ)
            penalized_deviance_previous = penalized_deviance
            penalized_deviance = self._penalized_deviance(
                y, μ, β, penalized_coefficients
            )
            is_converged = has_converged(
                penalized_deviance, penalized_deviance_previous, tol
            )
            n_iter += 1

        if standardizer is not None:
            β = standardizer.coef_to_original(β)
        if not np.all(np.isfinite(β)):
            raise ValueError(NON_FINITE_ESTIMATES_MESSAGE)
        self.coef_ = β
        self.deviance_ = family.deviance(y, μ)
        self.n = float(np.sum(sample_weights))
        self.p = X.shape[1]
        self.information_matrix_ = self._compute_ddbeta(X, dμ, var, sample_weights)
        return self

    def predict(
        self, X: DesignMatrix | DesignFrame, offset: Offset | None = None
    ) -> MeanVector:
        """Return predictions from a fit model (estimates of E[y | X]).

        Predictions are computed using the inverse link function in the family
        used to fit the model:

            preds = family.inv_link(X @ self.coef_)

        Note: In the case of binary models, predict *does not* make class
        assignmnets, it returns a probability of class membership.

        Parameters
        ----------
        X: array, shape (n_samples, n_features) or pd.DataFrame
            Inference data.

        offset: array, shape (n_samples, )
            Offsets to add on the linear predictor when making predictions.

        Returns
        -------
        preds: array, shape (n_samples, )
            Model predictions.
        """
        if self.coef_ is None:
            raise ValueError(
                "Model is not fit, and cannot be used to make predictions."
            )
        if self.X_spec is not None:
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    f"X must be a DataFrame for a model fit with a "
                    f"formula, got a {type(X)}"
                )
            X = self._make_rhs_matrix(X)
        elif not isinstance(X, np.ndarray):
            raise TypeError(
                f"X must be a numpy array for a model fit without a "
                f"formula, got a {type(X)}"
            )
        check_no_missing_values("X", X)
        η = X @ self.coef_
        if offset is not None:
            check_no_missing_values("offset", offset)
            η = η + offset
        return self.family.inv_link(η)

    def score(self, X: DesignMatrix | DesignFrame, y: ResponseVector) -> Deviance:
        """Return the deviance of a fit model on a given dataset.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            Data set.

        y: array, shape (n_samples, )
            Response vector aligned to X.

        Returns
        -------
        deviance: float
            Model deviance scored using supplied data and labels.
        """
        return self.family.deviance(y, self.predict(X))

    @property
    def dispersion_(self) -> float:
        """Return an estimate of the dispersion parameter phi."""
        if self.deviance_ is None or self.n is None or self.p is None:
            raise ValueError(
                "Dispersion parameter can only be estimated for a fit model."
            )
        if self.family.has_dispersion:
            return self.deviance_ / (self.n - self.p)
        else:
            return 1.0

    @property
    def coef_covariance_matrix_(self) -> CovarianceMatrix:
        if self.information_matrix_ is None:
            raise ValueError(
                "Parameter covariances can only be estimated for a fit model."
            )
        return self.dispersion_ * np.linalg.inv(self.information_matrix_)

    @property
    def coef_standard_error_(self) -> StandardErrorVector:
        return np.sqrt(np.diag(self.coef_covariance_matrix_))

    @property
    def p_values_(self) -> PValueVector:
        """Return an array of p-values for the fit coefficients. These p-values
        test the hypothesis that the given parameter is zero.

        Note: We use the asymptotic normal approximation to the parameter
        estimates to compute the p-values for all models.
        """
        if self.alpha != 0:
            raise ValueError("p-values are not available for regularized models.")
        if self.coef_ is None:
            raise ValueError("p-values can only be computed for a fit model.")
        p_values = []
        null_dist = sts.norm(loc=0.0, scale=1.0)
        for β, std_err in zip(self.coef_, self.coef_standard_error_):
            z = abs(β) / std_err
            # Two tailed.
            p_value = null_dist.cdf(-z) + (1 - null_dist.cdf(z))
            p_values.append(p_value)
        return np.asarray(p_values, dtype=np.float64)

    def summary(self) -> None:
        """Print a summary of the model."""
        if self.coef_ is None or self.X_names is None:
            raise ValueError("A summary is only available for a fit model.")
        variable_names = self.X_names
        parameter_estimates = self.coef_
        standard_errors = self.coef_standard_error_
        longest_var_name_length = max(len(name) + 2 for name in variable_names)
        format_string = "{:<" + str(longest_var_name_length) + "} {:>20} {:>15}"
        header_string = format_string.format(
            "Name", "Parameter Estimate", "Standard Error"
        )
        print(f"{self.family.__class__.__name__} GLM Model Summary.")
        print("=" * len(header_string))
        print(header_string)
        print("-" * len(header_string))
        format_string = "{:<" + str(longest_var_name_length) + "} {:>20.2f} {:>15.2f}"
        for name, est, se in zip(variable_names, parameter_estimates, standard_errors):
            print(format_string.format(name, est, se))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(family={self.family!r}, alpha={self.alpha!r}, "
            f"standardize={self.standardize!r})"
        )

    def clone(self) -> Self:
        return self.__class__(self.family, self.alpha, self.standardize)

    def _is_fit(self) -> bool:
        return self.coef_ is not None

    def _is_regularized(self) -> bool:
        return self.alpha > 0.0

    def _compute_dbeta(
        self,
        X: DesignMatrix,
        y: ResponseVector,
        μ: MeanVector,
        dμ: InverseLinkDerivative,
        var: VarianceVector,
        sample_weights: SampleWeights,
    ) -> DevianceGradient:
        working_residuals = sample_weights * (y - μ) * (dμ / var)
        return -np.sum(X * working_residuals.reshape(-1, 1), axis=0)

    def _compute_ddbeta(
        self,
        X: DesignMatrix,
        dμ: InverseLinkDerivative,
        var: VarianceVector,
        sample_weights: SampleWeights,
    ) -> InformationMatrix:
        working_h_weights = (sample_weights * dμ**2 / var).reshape(-1, 1)
        information: InformationMatrix = X.T @ (X * working_h_weights)
        return information

    def _penalized_coefficients(self, X: DesignMatrix) -> CoefficientMask:
        """The penalty applies to every coefficient except the intercept, if
        the model has one."""
        penalized_coefficients = np.ones(X.shape[1], dtype=np.bool_)
        if has_intercept_column(X):
            penalized_coefficients[0] = False
        return penalized_coefficients

    def _penalized_deviance(
        self,
        y: ResponseVector,
        μ: MeanVector,
        β: CoefficientVector,
        penalized_coefficients: CoefficientMask,
    ) -> Deviance:
        penalty = self.alpha * float(np.sum(β[penalized_coefficients] ** 2))
        return self.family.deviance(y, μ) + penalty

    def _d_penalty(
        self, β: CoefficientVector, penalized_coefficients: CoefficientMask
    ) -> DevianceGradient:
        # Like dβ, the gradient of half the penalty.
        dβ_penalty: DevianceGradient = np.where(
            penalized_coefficients, self.alpha * β, 0.0
        )
        return dβ_penalty

    def _dd_penalty(
        self, ddβ: InformationMatrix, penalized_coefficients: CoefficientMask
    ) -> InformationMatrix:
        # Like ddβ, the Hessian of half the penalty, added to the diagonal.
        idxs = np.flatnonzero(penalized_coefficients)
        ddβ[idxs, idxs] += self.alpha
        return ddβ

    def _make_rhs_matrix(self, X: DesignFrame) -> DesignMatrix:
        if self.X_spec is None:
            raise ValueError("Model was not fit with a formula.")
        # Raise on missing values rather than dropping rows, which would
        # silently misalign the design matrix with the rows of X.
        design_matrix: DesignMatrix = np.asarray(
            self.X_spec.get_model_matrix(X, output="numpy", na_action="raise")
        )
        return design_matrix

    def _make_lhs_vector(self, X: DesignFrame) -> ResponseVector:
        if self.y_spec is None:
            raise ValueError("Model was not fit with a formula.")
        response: ResponseVector = np.asarray(
            self.y_spec.get_model_matrix(X, output="numpy", na_action="raise")
        ).squeeze(axis=1)
        return response

    def _copy_formula(self, other: "GLM") -> None:
        """Take another model's formula and specs, so that this model, fit to
        that model's design matrix, can make predictions from data frames."""
        self.formula = other.formula
        self.X_spec, self.y_spec = other.X_spec, other.y_spec
