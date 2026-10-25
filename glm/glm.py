import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import formulaic
from formulaic.errors import DataMismatchWarning
import numpy as np
import pandas as pd

from glm.families import ExponentialFamily, TestDistribution
from glm.standardize import Standardizer
from glm.types import (
    CoefficientMask,
    CoefficientSummary,
    CoefficientVector,
    CovarianceMatrix,
    DesignFrame,
    DesignMatrix,
    DesignSpec,
    Deviance,
    DevianceGradient,
    Formula,
    FrequencyWeights,
    InformationMatrix,
    InverseLinkDerivative,
    MeanVector,
    ObservationWeights,
    Offset,
    PValueVector,
    PrecisionWeights,
    ResponseVector,
    StandardErrorVector,
    Standardization,
    VarianceVector,
    WorkingWeights,
)
from glm.utils import (
    check_commensurate,
    check_full_rank,
    check_intercept,
    check_no_missing_values,
    check_offset,
    check_standardization,
    check_weights,
    has_intercept_column,
    default_X_names,
    default_y_name,
)

NON_FINITE_ESTIMATES_MESSAGE = (
    "Fitting failed: the coefficient estimates are not finite. This can happen "
    "when the maximum likelihood estimate does not exist, e.g. a binary response "
    "that is all zeros or all ones, or one perfectly separated by the predictors."
)


@dataclass(frozen=True)
class FitInputs:
    """Records which optional inputs a model was fit with, so that a bootstrap
    can check it is given the same ones. A bootstrap refits the model to data
    like the original's, so leaving out an offset or weights used during fitting
    with would silently change the model being bootstrapped.

    The has_binary_response flag records whether every response was 0 or 1,
    which decides whether precision weights are needed to simulate a Bernoulli
    model's responses in a parametric bootstrap.
    """

    has_offset: bool
    has_frequency_weights: bool
    has_precision_weights: bool
    has_binary_response: bool

    def check(
        self,
        offset: Offset | None,
        frequency_weights: FrequencyWeights | None,
        precision_weights: PrecisionWeights | None,
    ) -> None:
        """Raise if the given inputs don't match those the model was fit
        with."""
        mismatches = []
        for name, was_given, is_given in [
            ("offset", self.has_offset, offset is not None),
            (
                "frequency_weights",
                self.has_frequency_weights,
                frequency_weights is not None,
            ),
            (
                "precision_weights",
                self.has_precision_weights,
                precision_weights is not None,
            ),
        ]:
            if was_given and not is_given:
                mismatches.append(
                    f"the model was fit with {name}, so pass them here too"
                )
            elif is_given and not was_given:
                mismatches.append(
                    f"the model was fit without {name}, so don't pass them here"
                )
        if mismatches:
            raise ValueError(
                "The inputs must match those the model was fit with: "
                + "; ".join(mismatches)
                + "."
            )


class GLM:
    """A generalized linear model.

    GLMs are a generalization of the classical linear and logistic models to
    other conditional distributions of response y. A GLM is specified by a
    family of *conditional distributions* dist (forming an exponential family),
    and a *link function* g, which relates the conditional mean of y to a
    linear function of the predictors:

        g(E[y | X]) = X @ β,  or equivalently  E[y | X] = g⁻¹(X @ β)

    so that the model specification is

        y | X ~ dist(μ = g⁻¹(X @ β))

    Here β are the parameters fit in the model, with X @ β a matrix
    multiplication just like in linear regression, and μ is the mean of the
    distribution dist. Some families also have a dispersion parameter, such as
    the Gaussian's variance, which is shared by every observation and estimated
    from the fit.

    In this implementation, a specific GLM is specified with a *family* object
    of ExponentialFamily type, which contains the information about the
    conditional distribution of y (and its connection to X) needed to construct
    the model. See the documentation for ExponentialFamily for details.

    The model is fit to data using the well known Fisher Scoring algorithm,
    which is a Quasi Newton's method where the hessian is replaced with its
    expectation with respect to the assumed distribution of Y. This expected
    hessian is called the Fisher Information in statistical literature. When
    using a canonical link function for an exponential family, this reduces to
    exactly Newton's method and thus has quadratic convergence guarentees. When
    using a non-canonical link, such as with Gamma GLM's (which use log, not the
    reciporical function) convergence is sub-quadratic.

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
        passed to fit as a DataFrame. For model formulas, please see the
        formulaic library documentation.

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

    n: float, positive
        The number of observations the model was fit to: the sum of the
        frequency weights of the rows with a non-zero weight. Precision weights
        don't add observations: a precise or imprecise row is still one row
        either way.

    p: integer, positive
        The number of fit parameters in the model.

    fit_inputs_: FitInputs
        Which optional inputs (an offset, and each kind of weights) the model
        was fit with, so that a bootstrap can check it's given the same ones.

    Notes
    -----
    Instead of supplying a `fit_intercept` argument, we have instead assumed the
    programmer has included a column of ones as column 0 of X, X[:, 0]. The fit
    method will throw an exception if this is not the case. When using a
    formula, formulaic will add an intercept by default.
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
        self.fit_inputs_: FitInputs | None = None

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
        frequency_weights: FrequencyWeights | None = None,
        precision_weights: PrecisionWeights | None = None,
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

        frequency_weights: array, shape (n_samples, )
            How many identical observations each row stands for, e.g. for data
            compressed by storing repeated records once, with a count. Fitting
            with them is equivalent to fitting to the expanded data.

        precision_weights: array, shape (n_samples, )
            How much more precise each row's response is than a single
            observation's: the response's variance is the family's variance
            divided by the weight. For example, a proportion of successes,
            weighted by its number of trials, or an average, weighted by the
            number of values averaged. These are what R's glm calls prior
            weights.

        max_iter: positive integer
            The maximum number of iterations for the fitting algorithm.

        tol: float, non-negative and less than one
            The convergence tolerance for the fitting algorithm. Fitting stops
            when the decrease in the penalized deviance that the next step
            predicts, relative to the penalized deviance, is within this
            tolerance (see has_converged).

        validate_intercept: bool
            Should we defensively check that column 0 of the design matrix,
            X[:, 0], is an intercept? Defaults to True.


        Returns
        -------
        self: GLM object
            The fit model.

        Notes
        -----
        Frequency and precision weights give the same coefficients, but
        different degrees of freedom: only frequency weights record the number
        of observations.

        Precision weights are used when a row of data represents a mean of
        multiple units. So a row is one observation of an average, and as such,
        has smaller variance than a single observed instance. For example, a row
        may record y = k/m as the proportion of successes in m binomial trials.
        Here the variance of a single observed proportion is:

            var(y) = var(k)/m² = m μ(1 - μ) / m² = μ(1 - μ)/m

        The family's variance function alone would overestimate it as μ(1 - μ).
        Passing precision weights m fixes this issue, which leads to appropriate
        standard errors (and dispersions for families that estimate one).

        The two weights can be combined: each row's weight in the deviance is
        their product.
        """
        design_matrix, response = self._prepare_data(X, y, formula, X_names, y_name)
        return self._fit(
            design_matrix,
            response,
            warm_start=warm_start,
            offset=offset,
            frequency_weights=frequency_weights,
            precision_weights=precision_weights,
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
        response vector, remembering the model's formula specs and feature
        names."""
        if formula:
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    f"X must be a DataFrame if a formula is "
                    f"supplied, got a {type(X)}"
                )
            # Raise on missing values rather than dropping rows, which would
            # silently misalign the data with any offset or sample weights.
            matrices = formulaic.model_matrix(
                formula,
                X,
                output="numpy",
                na_action="raise",
                # Regularized models encode every level of a categorical
                # variable, rather than dropping a reference level, to ensure
                # that the penalty treats all levels alike. This does not cause
                # any rank defficiency, the penalty keeps the fit well defined.
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
        frequency_weights: FrequencyWeights | None,
        precision_weights: PrecisionWeights | None,
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
        # Record which optional inputs were given before the defaults below
        # replace the missing ones with their default values.
        fit_inputs = FitInputs(
            has_offset=offset is not None,
            has_frequency_weights=frequency_weights is not None,
            has_precision_weights=precision_weights is not None,
            has_binary_response=bool(np.all((y == 0) | (y == 1))),
        )
        if validate_intercept:
            check_intercept(X)
        if offset is None:
            offset = np.zeros(X.shape[0])
        check_offset(y, offset)
        check_no_missing_values("offset", offset)
        frequency_weights = check_weights("frequency_weights", y, frequency_weights)
        precision_weights = check_weights("precision_weights", y, precision_weights)
        # Each row's weight weight on its contribution to the deviance.
        observation_weights: ObservationWeights = frequency_weights * precision_weights
        # A ridge penalty makes the fit well defined even when the matrix is
        # rank deficient.
        if not self._is_regularized():
            # Zero weight rows do not influence the fit, so don't need to be
            # accounted for in the rank.
            # We note that we require the information matricies X^t W X (where W
            # is the diagonal matrix of working weights) to be full rank at each
            # iteration. It sufficies to ensure that X itself is full rank,
            # because rank(XᵗWX) = rank(√WX) = rank(X).
            check_full_rank(
                X[observation_weights > 0], self.X_names or default_X_names(X)
            )
        penalized_coefficients = self._penalized_coefficients(X)
        # We're gonna reference self.family a lot, so cache a local variable for
        # a bit of efficiency.
        family = self.family
        if warm_start is None:
            warm_start = self._cold_start(X, y, offset, observation_weights)
        # Penalized fits work on standardized columns, so that the penalty does
        # not depend on the predictors' units. Unpenalized fits are unchanged
        # by standardizing, so skip it for them.
        standardizer = None
        design, β = X, warm_start
        if self._is_regularized() and self.standardize != "none":
            standardizer = Standardizer(self.standardize).fit(X, frequency_weights)
            design = standardizer.transform(X)
            β = standardizer.coef_to_standardized(warm_start)
        # Begin main fitting loop:
        η = design @ β + offset
        μ = family.inv_link(η)
        dμ = family.d_inv_link(η, μ)
        var = family.variance(μ)
        penalized_deviance = self._penalized_deviance(
            y, μ, β, penalized_coefficients, observation_weights
        )
        is_converged = False
        n_iter = 0
        while n_iter < max_iter and not is_converged:
            dβ = self._dβ(design, y, μ, dμ, var, observation_weights)
            ddβ = self._ddβ(design, dμ, var, observation_weights)
            if self._is_regularized():
                dβ = dβ + self._d_penalty(β, penalized_coefficients)
                ddβ = self._dd_penalty(ddβ, penalized_coefficients)
            step = np.linalg.solve(ddβ, dβ)
            # Check before stepping, as the predicted decrease describes the
            # current β, but take the step either way: at convergence it is
            # tiny, and refines the estimate.
            is_converged = has_converged(dβ, step, penalized_deviance, tol)
            β = β - step
            # Refresh the fit at the new coefficients, so that the next
            # iteration, and the deviance and information matrix stored after
            # the loop, all describe the β we return.
            η = design @ β + offset
            μ = family.inv_link(η)
            dμ = family.d_inv_link(η, μ)
            var = family.variance(μ)
            penalized_deviance = self._penalized_deviance(
                y, μ, β, penalized_coefficients, observation_weights
            )
            n_iter += 1

        if standardizer is not None:
            β = standardizer.coef_to_original(β)
        if not np.all(np.isfinite(β)):
            raise ValueError(NON_FINITE_ESTIMATES_MESSAGE)
        self.coef_ = β
        self.deviance_ = family.deviance(y, μ, observation_weights)
        self.n = float(np.sum(frequency_weights[observation_weights > 0]))
        self.p = X.shape[1]
        # dμ and var come from the loop, which may have worked on standardized
        # columns, while the information matrix is wanted on the original ones.
        # Mixing them is fine: standardizing changes the coefficients but not
        # the linear predictor, design @ β = X @ coef_, so μ, and with it dμ
        # and var, are the same in either coordinates.
        self.information_matrix_ = self._ddβ(X, dμ, var, observation_weights)
        self.fit_inputs_ = fit_inputs
        return self

    def predict(
        self, X: DesignMatrix | DesignFrame, offset: Offset | None = None
    ) -> MeanVector:
        """Return predictions from a fit model (estimates of E[y | X]).

        Predictions are computed using the inverse link function in the family
        used to fit the model:

            preds = family.inv_link(X @ self.coef_)

        In the case of binary models, predict *does not* make class assignments,
        it returns a probability of class membership.

        For a model fit with a formula, new data may contain a level of a
        categorical variable that the training data did not. The different
        encoding for penalized and unpenalized models has an effect here:

        - An unpenalized model encodes each level relative to a reference level,
        whose indicators are all zero. So an unseen level would be
        indistinguishable from the reference level.
        - A penalized model encodes every level with its own indicator, and
        shrinks each level's coefficient toward zero. So a level with no
        training data gets no effect at all: its indicators are all zero, and
        its prediction is the baseline.

        In the unpenalized case, we raise an error instead of silently falling
        back to the reference level prediction. In the penalized case, we do not
        raise, since the predictions fall back to the baseline case.

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
        X = self.design_matrix(X)
        check_no_missing_values("X", X)
        η = X @ self.coef_
        if offset is not None:
            check_no_missing_values("offset", offset)
            η = η + offset
        return self.family.inv_link(η)

    def score(
        self,
        X: DesignMatrix | DesignFrame,
        y: ResponseVector,
        *,
        offset: Offset | None = None,
        frequency_weights: FrequencyWeights | None = None,
        precision_weights: PrecisionWeights | None = None,
    ) -> Deviance:
        """Return the deviance of a fit model on a given dataset.

        Parameters
        ----------
        X: array, shape (n_samples, n_features) or pd.DataFrame
            Data set.

        y: array, shape (n_samples, )
            Response vector aligned to X.

        offset: array, shape (n_samples, )
            Offsets to add on the linear predictor, as for predict.

        frequency_weights, precision_weights: arrays, shape (n_samples, )
            Weights on each row's contribution to the deviance, as for fit.

        Returns
        -------
        deviance: float
            The model's deviance on the data.
        """
        frequency_weights = check_weights("frequency_weights", y, frequency_weights)
        precision_weights = check_weights("precision_weights", y, precision_weights)
        observation_weights: ObservationWeights = frequency_weights * precision_weights
        return self.family.deviance(
            y, self.predict(X, offset=offset), observation_weights
        )

    def design_matrix(self, X: DesignMatrix | DesignFrame) -> DesignMatrix:
        """Return the design matrix a fit model uses for the given data.

        For a model fit with a formula, the data frame is encoded with the
        formula's specs learned when fitting, so that categorical levels and
        stateful transforms such as center(x) are encoded as they were in
        training. For a model fit without a formula, the data is already a
        design matrix, and is returned as it is.

        Parameters
        ----------
        X: array, shape (n_samples, n_features) or pd.DataFrame
            Data, a DataFrame for a model fit with a formula, and a numpy array
            for a model fit without one.

        Returns
        -------
        design_matrix: array, shape (n_samples, n_coefficients)
            The design matrix.
        """
        if not self._is_fit():
            raise ValueError("Model is not fit, and cannot encode data.")
        if self.X_spec is not None:
            if not isinstance(X, pd.DataFrame):
                raise self._X_type_error(X)
            return self._make_rhs_matrix(X)
        if not isinstance(X, np.ndarray):
            raise self._X_type_error(X)
        return X

    def response_vector(
        self, X: DesignMatrix | DesignFrame, y: ResponseVector | None = None
    ) -> ResponseVector:
        """Return the response a fit model uses for the given data.

        For a model fit with a formula, the response is taken from the data
        frame by the formula's left hand side, so y must not be supplied. For
        a model fit without a formula, y is the response, and is returned as
        it is.

        Parameters
        ----------
        X: array, shape (n_samples, n_features) or pd.DataFrame
            Data, a DataFrame for a model fit with a formula, and a numpy array
            for a model fit without one.

        y: array, shape (n_samples, )
            The response, for a model fit without a formula only.

        Returns
        -------
        response: array, shape (n_samples, )
            The response vector.
        """
        if not self._is_fit():
            raise ValueError("Model is not fit, and cannot encode data.")
        if self.X_spec is not None:
            if y is not None:
                raise ValueError(
                    "y must not be supplied for a model fit with a formula, as "
                    "the response is taken from the formula."
                )
            if not isinstance(X, pd.DataFrame):
                raise self._X_type_error(X)
            return self._make_lhs_vector(X)
        if not isinstance(y, np.ndarray):
            raise TypeError(
                f"y must be a numpy array for a model fit without a formula, "
                f"got a {type(y)}"
            )
        return y

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
        """Return an array of p-values for the fit coefficients, testing the
        hypothesis that each true coefficient is zero. The test statistic,
        estimate / standard error, is compared with the family's reference
        distribution: Student's t if it estimates a dispersion, the standard
        normal if not (see ExponentialFamily.coefficient_test). summary_frame
        gives the p-values along with the coefficients' names.
        """
        if self.alpha != 0:
            raise ValueError("p-values are not available for regularized models.")
        if self.coef_ is None:
            raise ValueError("p-values can only be computed for a fit model.")
        statistics = self.coef_ / self.coef_standard_error_
        if self.n is None or self.p is None:
            raise ValueError("p-values can only be computed for a fit model.")
        # Two tailed: the chance of a statistic at least this far from zero.
        d: TestDistribution = self.family.coefficient_test(residual_df=self.n - self.p)
        return 2 * d.distribution.sf(np.abs(statistics))

    def summary_frame(self, level: float = 0.95) -> CoefficientSummary:
        """A table of the fit coefficients, one row per coefficient.

        The columns are the estimate, its standard error, the test statistic
        for the hypothesis that the coefficient is zero (labelled "t" or "z",
        as the family's reference distribution is Student's t or the standard
        normal; see ExponentialFamily.coefficient_test), its two tailed
        p-value, and the bounds of a confidence interval. For regularized
        models, only the estimates and standard errors are given, as the
        tests and intervals aren't valid.

        Parameters
        ----------
        level: float, between 0 and 1
            The confidence level of the intervals.
        """
        if self.coef_ is None:
            raise ValueError("A summary is only available for a fit model.")
        if self.n is None or self.p is None:
            raise ValueError("p-values can only be computed for a fit model.")
        if not 0 < level < 1:
            raise ValueError(f"level must be between 0 and 1, got {level}.")
        standard_errors = self.coef_standard_error_
        frame = pd.DataFrame(
            {"estimate": self.coef_, "std error": standard_errors},
            index=self.X_names,
        )
        if self._is_regularized():
            return frame
        test: TestDistribution = self.family.coefficient_test(
            residual_df=self.n - self.p
        )
        frame[test.statistic] = self.coef_ / standard_errors
        frame["p-value"] = self.p_values_
        quantile = float(test.distribution.ppf((1 + level) / 2))
        percent = f"{100 * level:g}%"
        frame[f"lower {percent}"] = self.coef_ - quantile * standard_errors
        frame[f"upper {percent}"] = self.coef_ + quantile * standard_errors
        return frame

    def summary(self) -> None:
        """Print a summary of the model's coefficients; see summary_frame."""
        frame = self.summary_frame()
        # Annotated, as an inferred type would include the lambda's parameter
        # name, which a formatter with a differently named parameter breaks.
        formatters: dict[str, Callable[[float], str]] = {
            column: (lambda v: f"{v:.3f}") for column in frame.columns
        }
        if "p-value" in frame.columns:
            formatters["p-value"] = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{self.family.__class__.__name__} GLM Model Summary.")
        print(frame.to_string(formatters=formatters))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(family={self.family!r}, alpha={self.alpha!r}, "
            f"standardize={self.standardize!r})"
        )

    def clone(self) -> Self:
        return self.__class__(self.family, self.alpha, self.standardize)

    def _is_fit(self) -> bool:
        return self.coef_ is not None

    def _cold_start(
        self,
        X: DesignMatrix,
        y: ResponseVector,
        offset: Offset,
        observation_weights: ObservationWeights,
    ) -> CoefficientVector:
        """Starting coefficients when none are supplied.

        With an intercept, start from the intercept-only model: its estimate of
        the mean is the weighted mean response, so its intercept is that mean
        on the linear predictor scale, net of the average offset. Without one,
        start from all zero coefficients, where the linear predictor is the
        offset; putting the intercept's value on column 0 would multiply it by
        a predictor, which can overflow the inverse link.

        Parameters
        ----------
        X: array, shape (n_samples, n_features)
            The design matrix.

        y: array, shape (n_samples, )
            The response.

        offset: array, shape (n_samples, )
            The offset on the linear predictor.

        observation_weights: array, shape (n_samples, )
            Each row's weight on its contribution to the deviance.

        Returns
        -------
        β: array, shape (n_features, )
            The starting coefficients.
        """
        β: CoefficientVector = np.zeros(X.shape[1])
        if not has_intercept_column(X):
            return β
        ȳ = np.average(y, weights=observation_weights)
        # The intercept-only model fits the same mean, ȳ, for every
        # observation, so its linear predictor is constant.
        μ_null = np.full(X.shape[0], ȳ)
        with np.errstate(divide="ignore"):
            # We just need a scalar, family.link transforms arrays into arrays.
            intercept = self.family.link(μ_null)[0]
        # E.g. an all-zero Poisson or binary response: the intercept-only
        # estimate is infinite, so no finite estimate can exist.
        if not np.isfinite(intercept):
            raise ValueError(NON_FINITE_ESTIMATES_MESSAGE)
        β[0] = intercept - np.average(offset, weights=observation_weights)
        return β

    def _is_regularized(self) -> bool:
        return self.alpha > 0.0

    def _dβ(
        self,
        X: DesignMatrix,
        y: ResponseVector,
        μ: MeanVector,
        dμ: InverseLinkDerivative,
        var: VarianceVector,
        observation_weights: ObservationWeights,
    ) -> DevianceGradient:
        working_residuals = observation_weights * (y - μ) * (dμ / var)
        return -np.sum(X * working_residuals.reshape(-1, 1), axis=0)

    def _ddβ(
        self,
        X: DesignMatrix,
        dμ: InverseLinkDerivative,
        var: VarianceVector,
        observation_weights: ObservationWeights,
    ) -> InformationMatrix:
        working_weights: WorkingWeights = observation_weights * dμ**2 / var
        information: InformationMatrix = X.T @ (X * working_weights[:, None])
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
        observation_weights: ObservationWeights,
    ) -> Deviance:
        penalty = self.alpha * float(np.sum(β[penalized_coefficients] ** 2))
        return self.family.deviance(y, μ, observation_weights) + penalty

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
        # formulaic encodes a level it didn't see in training as all zero
        # indicators. With reference coding (formulaic's ensure_full_rank),
        # that's the reference level's encoding, so an unseen level would be
        # silently predicted as the reference level. Check before encoding,
        # so the error isn't preceded by formulaic's own warning.
        if self.X_spec.ensure_full_rank:
            unseen = unseen_categories(self.X_spec, X)
            if unseen:
                described = "; ".join(
                    f"{factor}: {levels}" for factor, levels in unseen.items()
                )
                raise ValueError(
                    f"X contains categorical levels the model wasn't fit with "
                    f"({described}). An unpenalized model encodes each level "
                    f"relative to a reference level, so it can't predict for "
                    f"levels it hasn't seen: refit with them, or merge them into "
                    f"levels the model has seen."
                )
        with warnings.catch_warnings():
            # With every level encoded, as for penalized models, all zero
            # indicators for an unseen level are intended (see predict), so
            # formulaic's warning that they will skew the results is not needed.
            warnings.simplefilter("ignore", DataMismatchWarning)
            # Raise on missing values rather than dropping rows, which would
            # silently misalign the design matrix with the rows of X.
            design_matrix: DesignMatrix = np.asarray(
                self.X_spec.get_model_matrix(X, output="numpy", na_action="raise")
            )
        return design_matrix

    def _X_type_error(self, X: object) -> TypeError:
        if self.X_spec is not None:
            return TypeError(
                f"X must be a DataFrame for a model fit with a formula, "
                f"got a {type(X)}"
            )
        return TypeError(
            f"X must be a numpy array for a model fit without a formula, "
            f"got a {type(X)}"
        )

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


def unseen_categories(spec: DesignSpec, X: DesignFrame) -> dict[str, list[Any]]:
    """The levels of each categorical variable in X that the model's learned
    spec has not yet seen, keyed by the variable's expression in the formula."""
    unseen = {}
    for factor, (kind, state) in spec.encoder_state.items():
        if kind.value != "categorical":
            continue
        # Let formulaic evaluate the factor's expression on X, so that a
        # categorical variable is handled however it appears in the formula.
        observed_spec = formulaic.model_matrix(f"0 + {factor}", X).model_spec
        if observed_spec is None:
            raise ValueError(f"Could not evaluate the factor {factor} on X.")
        _, observed_state = observed_spec.encoder_state[factor]
        new_levels = [
            level
            for level in observed_state["categories"]
            if level not in state["categories"]
        ]
        if new_levels:
            unseen[str(factor)] = new_levels
    return unseen


def has_converged(
    gradient: DevianceGradient,
    step: CoefficientVector,
    penalized_deviance: Deviance,
    tol: float,
) -> bool:
    """Has Fisher scoring converged, using the squared Newton decrement.

    Fisher scoring minimizes a quadratic model of the penalized deviance F
    around the current coefficients. With g the gradient of ½F, and H its
    expected Hessian (the Fisher information, up to the dispersion, plus the
    penalty's Hessian), the model is:

        ½F(β - s) ≈ ½F(β) - gᵗs + ½ sᵗHs,

    minimized by the step s = H⁻¹g. That step decreases the quadratic
    approximation of F (twice the model above) by

        λ² = gᵗH⁻¹g = sᵗHs = gᵗs,

    the squared Newton decrement [1], the decrease in F that the step
    anticipates. We stop when this predicted decrease is small relative to F
    itself, λ² ≤ tol · |F|, so that tol is unitless, as in the classic rule of
    comparing successive deviances.

    Two other views of λ²:

      - The squared length of the step, in the norm defined by H:
        λ² = sᵗHs.
      - The squared length of the gradient, in the norm defined by H⁻¹:
        λ² = gᵗH⁻¹g.

    From this point of view, the stopping criterion is a condition on the size
    of the step, or of the gradient, scaled by how precisely the data and the
    penalty determine the coefficients in each direction.

    Since H is positive definite, λ² is zero exactly when the gradient is, and
    shrinks to zero with it as the fit converges. This justifies the condition
    as a stopping criterion.

    Both norms are invariant under invertible linear changes of coordinates, so
    the rule is too. For example, in an unpenalized fit, rescaling a column of
    the design matrix, or fitting with or without standardization, doesn't
    change when fitting stops, as it would for a rule on the plain Euclidean
    norm of the gradient.

    Parameters
    ----------
    gradient: array, shape (n_features, )
        The gradient of half the penalized deviance, g.

    step: array, shape (n_features, )
        The Fisher scoring step, H⁻¹g, to be subtracted from the
        coefficients.

    penalized_deviance: float
        The penalized deviance F, at the coefficients the gradient was
        computed at.

    tol: float
        The convergence tolerance, on the predicted decrease in F relative to
        F.

    Returns
    -------
    converged: bool
        Whether the predicted decrease is within the tolerance. Always False
        if F isn't finite.

    References
    ----------
    .. [1] S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge
       University Press, 2004, Section 9.5.1, "The Newton decrement".
    """
    if not np.isfinite(penalized_deviance):
        return False
    newton_decrement_squared = float(gradient @ step)
    return newton_decrement_squared <= tol * abs(penalized_deviance)
