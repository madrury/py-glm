"""Fitting ridge regularization paths."""

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from glm.families import ExponentialFamily, _generator
from glm.glm import GLM, NON_FINITE_ESTIMATES_MESSAGE
from glm.simulation import coefficient_table
from glm.standardize import Standardizer
from glm.types import (
    AlphaGrid,
    CVDeviances,
    CoefficientTable,
    CoefficientVector,
    ColumnScales,
    DesignFrame,
    DesignMatrix,
    EffectiveDegreesOfFreedom,
    FoldDeviances,
    FoldFrequencyWeights,
    FoldWeights,
    Formula,
    FrequencyWeights,
    InformationEigenvalues,
    LinearPredictor,
    MeanVector,
    ObservationWeights,
    Offset,
    PrecisionWeights,
    ResponseVector,
    Standardization,
    TermColumns,
    WorkingWeights,
)
from glm.utils import (
    check_alphas,
    check_integer_at_least,
    check_standardization,
    check_weights,
    has_intercept_column,
)

# The default grid spans from this many times the largest eigenvalue, to this
# many times smaller than the smallest; see default_alphas.
GRID_MARGIN = 100.0


class RidgePath:
    """Fit a GLM along a grid of ridge penalty strengths.

    Models are fit from the largest alpha to the smallest, each warm starting
    from the previous model's coefficients, which makes fitting the whole path
    cheap. Every alpha must be strictly positive, so models specified with a
    formula encode each level of all categorical variables without linear
    algebra errors arising from singular design matrices.

    The default alpha grid is constructed to cover a range from (within a close
    approximation) an intercept only model through an unpenalized model. This is
    created as a log-scale spaced grid with upper and lower bounds computed
    based on the eigenvalue spectrum of an information matrix. See
    default_alphas for details on this computation.

    Parameters
    ----------

    family: ExponentialFamily
        The exponential family used in the models.

    alphas: sequence of positive floats, optional
        The penalty strengths to fit, in any order. By default, a grid is
        chosen from the data; see default_alphas.

    n_alphas: integer, at least 2
        The number of penalty strengths in the default grid.

    standardize: "none", "continuous", or "all"
        Which columns to standardize before fitting; see GLM.

    Attributes
    ----------

    alphas_: array, shape (n_alphas, )
        The penalty strengths, in decreasing order.

    models_: list of GLM
        The fit models, one per penalty strength, in the same order.

    term_columns_: dict
        The names of the penalized coefficients belonging to each term of the
        model, for plotting coefficients in groups. For a model fit without a
        formula, each coefficient is its own term.

    scales_: array, shape (n_features, )
        The scale of each column of the design matrix, by which the penalized
        fits standardized it (ones without standardizing). Multiplying the
        coefficients by these gives the standardized coefficients, which the
        penalty acts on.

    df_: array, shape (n_alphas, )
        The effective degrees of freedom of each model, sum(d / (d + alpha)),
        where d are the eigenvalues of the model's information matrix for the
        standardized penalized columns, with the intercept profiled out (see
        default_alphas). They run from near zero, when alpha is large and all
        coefficients are shrunk, up to the dimension of vector space spanned by
        the penalized features; the intercept isn't counted. The dimension is
        usually the number of penalized coefficients, less one for each
        categorical variable encoded with a column for every level, as those
        columns sum to the intercept and thus are linearly dependent.
    """

    def __init__(
        self,
        family: ExponentialFamily,
        alphas: Sequence[float] | None = None,
        n_alphas: int = 100,
        standardize: Standardization = "continuous",
    ) -> None:
        check_standardization(standardize)
        check_alphas(alphas, n_alphas)
        self.family = family
        self.alphas = alphas
        self.n_alphas = n_alphas
        self.standardize: Standardization = standardize
        self.alphas_: AlphaGrid | None = None
        self.models_: list[GLM] | None = None
        self.term_columns_: TermColumns | None = None
        self.scales_: ColumnScales | None = None
        self.df_: EffectiveDegreesOfFreedom | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(family={self.family!r}, n_alphas={self.n_alphas!r}, "
            f"standardize={self.standardize!r})"
        )

    def fit(
        self,
        X: DesignMatrix | DesignFrame,
        y: ResponseVector | None = None,
        formula: Formula | None = None,
        *,
        X_names: list[str] | None = None,
        y_name: str | None = None,
        offset: Offset | None = None,
        frequency_weights: FrequencyWeights | None = None,
        precision_weights: PrecisionWeights | None = None,
        max_iter: int = 100,
        tol: float = 0.1**5,
        validate_intercept: bool = True,
    ) -> Self:
        """Fit a model for each penalty strength.

        The arguments are as for GLM.fit.
        """
        # Encode the data once. Any positive alpha gives the same encoding.
        template = GLM(self.family, alpha=1.0, standardize=self.standardize)
        design, response = template._prepare_data(X, y, formula, X_names, y_name)
        frequency_weights = check_weights(
            "frequency_weights", response, frequency_weights
        )
        precision_weights = check_weights(
            "precision_weights", response, precision_weights
        )
        if self.alphas is None:
            alphas = default_alphas(
                self.family,
                design,
                response,
                frequency_weights=frequency_weights,
                precision_weights=precision_weights,
                standardize=self.standardize,
                n_alphas=self.n_alphas,
            )
        else:
            alphas = np.sort(np.asarray(self.alphas, dtype=np.float64))[::-1]

        models: list[GLM] = []
        warm_start: CoefficientVector | None = None
        for alpha in alphas:
            model = GLM(self.family, alpha=float(alpha), standardize=self.standardize)
            model.fit(
                design,
                response,
                X_names=template.X_names,
                y_name=template.y_name,
                warm_start=warm_start,
                offset=offset,
                frequency_weights=frequency_weights,
                precision_weights=precision_weights,
                max_iter=max_iter,
                tol=tol,
                validate_intercept=validate_intercept,
            )
            model._copy_formula(template)
            models.append(model)
            warm_start = model.coef_

        self.alphas_ = alphas
        self.models_ = models
        self.term_columns_ = _term_columns(template, design)
        # This is intentionally a little wasteful, since we've already
        # standardized the data when fitting the model. We chose the duplicated
        # effort for a clean separation of concerns; here we need the
        # standardized columns and their scales to compute the effective degrees
        # of freedom of each alpha.
        standardizer = Standardizer(self.standardize).fit(design, frequency_weights)
        self.scales_ = standardizer.scales_
        observation_weights: ObservationWeights = frequency_weights * precision_weights
        self.df_ = _effective_degrees_of_freedom(
            models,
            alphas,
            design,
            standardizer.transform(design),
            np.zeros(design.shape[0]) if offset is None else offset,
            observation_weights,
            has_intercept_column(design),
        )
        return self

    def model_at(self, alpha: float) -> GLM:
        """The fit model at a penalty strength on the path's grid.

        Parameters
        ----------

        alpha: float
            One of the path's penalty strengths, alphas_.
        """
        if self.models_ is None or self.alphas_ is None:
            raise ValueError("The path has not been fit.")
        matches = np.flatnonzero(self.alphas_ == alpha)
        if matches.size == 0:
            raise ValueError(f"alpha {alpha!r} is not on the path's grid, alphas_.")
        # A list indexed by a numpy integer is typed Any; int keeps the GLM.
        return self.models_[int(matches[0])]

    def coefficient_table(self) -> CoefficientTable:
        """The coefficients along the path: one row per penalty strength,
        indexed by alpha, and one column per coefficient."""
        if self.models_ is None or self.alphas_ is None:
            raise ValueError("The path has not been fit.")
        table = coefficient_table(self.models_)
        table.index = self.alphas_
        table.index.name = "alpha"
        return table


class RidgePathCV:
    """Choose a ridge penalty strength by cross-validation.

    A RidgePath is fit to the full data, setting the grid of penalty
    strengths giving the collection of models to choose from, one per penalty
    strength in the grid. Then the data's observations are split into folds, and
    for each fold, a path over the same grid is fit to the other folds'
    observations, and scored by its deviance on the held-out observations.

    Observations, not rows, are split. This is an important distinction when
    rows have frequency weights: a row with frequency weight f stands for f
    independent observations, which are dealt out to the folds separately,
    exactly as when cross-validating the expanded data (see
    fold_frequency_weights). A precision-weighted row is treated as one
    observation of an average or proportion, so it is held out whole, keeping
    its weight.

    Each fold's held-out loss is its deviance per unit of observation weight,

        sum(f * p * d) / sum(f * p),

    with:

        - f the held-out frequency weights
        - p the precision weights
        - d the unit deviances

    The folds are averaged weighted by their total held-out weight, so that each
    unit of weight counts equally (rather than each fold).

    Each fold learns its own formula encoding, so a categorical level may be
    missing from a fold's training data. The penalized fold models predict it at
    the baseline, with all its indicators zero (see GLM.predict). It is
    important to fully encode each categorical instead of leaving one level out,
    otherwise the baseline will be the level left out. The default formula
    encoding when using a ridge penalty is full, so this is only a risk when
    encoding manually.

    Parameters
    ----------

    family: ExponentialFamily
        The exponential family used in the models.

    alphas: sequence of positive floats, optional
        The penalty strengths to fit, in any order. By default, a grid is
        chosen from the full data; see default_alphas.

    n_alphas: integer, at least 2
        The number of penalty strengths in the default grid.

    n_folds: integer, at least 2
        The number of folds to split the observations into.

    standardize: "none", "continuous", or "all"
        Which columns to standardize before fitting; see GLM.

    Attributes
    ----------

    alphas_: array, shape (n_alphas, )
        The penalty strengths, in decreasing order.

    path_: RidgePath
        The path fit to the full data. Holds the models to choose from.

    fold_frequency_weights_: array, shape (n_folds, n_samples)
        How many of each row's observations each fold held out. Rows with zero
        precision weight carry no weight, so are never held out.

    fold_weights_: array, shape (n_folds, )
        The total observation weight each fold held out.

    fold_deviances_: array, shape (n_folds, n_alphas)
        Each fold's held-out deviance per unit of observation weight, at each
        penalty strength.

    cv_deviance_: array, shape (n_alphas, )
        The held-out deviance per unit weight, averaged over the folds and
        indexed by regularization strength.

    cv_standard_error_: array, shape (n_alphas, )
        The standard error of cv_deviance_. Calculated from the spread of the
        various folds' deviances about their mean. With equally weighted folds,
        this is the usual standard error of a mean.

    alpha_min_: float
        The penalty strength with the smallest cv_deviance_.

    alpha_1se_: float
        The largest penalty strength whose cv_deviance_ is within one standard
        error of the smallest: the simplest model that cross-validation can't
        distinguish from the model with minimal held-out deviance.
    """

    def __init__(
        self,
        family: ExponentialFamily,
        alphas: Sequence[float] | None = None,
        n_alphas: int = 100,
        n_folds: int = 10,
        standardize: Standardization = "continuous",
    ) -> None:
        check_standardization(standardize)
        check_alphas(alphas, n_alphas)
        check_integer_at_least("n_folds", n_folds, 2)
        self.family = family
        self.alphas = alphas
        self.n_alphas = n_alphas
        self.n_folds = n_folds
        self.standardize: Standardization = standardize
        self.alphas_: AlphaGrid | None = None
        self.path_: RidgePath | None = None
        self.fold_frequency_weights_: FoldFrequencyWeights | None = None
        self.fold_weights_: FoldWeights | None = None
        self.fold_deviances_: FoldDeviances | None = None
        self.cv_deviance_: CVDeviances | None = None
        self.cv_standard_error_: CVDeviances | None = None
        self.alpha_min_: float | None = None
        self.alpha_1se_: float | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(family={self.family!r}, n_alphas={self.n_alphas!r}, "
            f"n_folds={self.n_folds!r}, standardize={self.standardize!r})"
        )

    def fit(
        self,
        X: DesignMatrix | DesignFrame,
        y: ResponseVector | None = None,
        formula: Formula | None = None,
        *,
        X_names: list[str] | None = None,
        y_name: str | None = None,
        offset: Offset | None = None,
        frequency_weights: FrequencyWeights | None = None,
        precision_weights: PrecisionWeights | None = None,
        max_iter: int = 100,
        tol: float = 0.1**5,
        validate_intercept: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Fit the full path, and cross-validate it.

        The arguments are the same as in GLM.fit, except rng.

        Parameters
        ----------

        frequency_weights: array, shape (n_samples, )
            How many identical observations each row stands for, as for
            GLM.fit. Each individual observation is assigned a fold separately
            (i.e. a single row representing 3 observations may be partitioned
            into three separate folds, as if the data represented the
            observations as three separate rows), so they must be whole numbers.

        rng: numpy Generator, optional
            The generator to split the observations into folds with. Pass one
            to make the folds reproducible.
        """
        path = RidgePath(
            self.family,
            alphas=self.alphas,
            n_alphas=self.n_alphas,
            standardize=self.standardize,
        ).fit(
            X,
            y,
            formula,
            X_names=X_names,
            y_name=y_name,
            offset=offset,
            frequency_weights=frequency_weights,
            precision_weights=precision_weights,
            max_iter=max_iter,
            tol=tol,
            validate_intercept=validate_intercept,
        )
        assert path.alphas_ is not None and path.models_ is not None
        alphas = path.alphas_
        template = path.models_[0]
        response = template.response_vector(X, y)
        frequency_weights = check_weights(
            "frequency_weights", response, frequency_weights
        )
        precision_weights = check_weights(
            "precision_weights", response, precision_weights
        )
        if offset is None:
            offset = np.zeros(response.shape[0])

        # Rows with zero precision weight carry no weight, so dealing their
        # observations to folds would only unbalance them. They stay in every
        # training set, where their zero weight leaves the fits unchanged.
        folds = fold_frequency_weights(
            np.where(precision_weights > 0, frequency_weights, 0.0),
            self.n_folds,
            _generator(rng),
        )
        fold_weights = np.empty(self.n_folds)
        fold_deviances = np.empty((self.n_folds, len(alphas)))
        for k, held_out_frequency_weights in enumerate(folds):
            training_frequency_weights = frequency_weights - held_out_frequency_weights
            training_rows = np.flatnonzero(training_frequency_weights > 0)
            held_out_rows = np.flatnonzero(held_out_frequency_weights > 0)
            held_out_weights: ObservationWeights = (
                held_out_frequency_weights[held_out_rows]
                * precision_weights[held_out_rows]
            )
            fold_weights[k] = np.sum(held_out_weights)
            fold_path = RidgePath(
                self.family, alphas=list(alphas), standardize=self.standardize
            ).fit(
                _take_rows(X, training_rows),
                None if y is None else y[training_rows],
                formula,
                X_names=X_names,
                y_name=y_name,
                offset=offset[training_rows],
                frequency_weights=training_frequency_weights[training_rows],
                precision_weights=precision_weights[training_rows],
                max_iter=max_iter,
                tol=tol,
                validate_intercept=validate_intercept,
            )
            assert fold_path.models_ is not None
            held_out_X = _take_rows(X, held_out_rows)
            # For a formula model, predict re-encodes the held-out data frame
            # for every model on the path, though the models all share one
            # encoding. Encoding once, with the first model's spec, would save
            # that, but predict only takes data frames for formula models, so
            # it would mean computing the predictions by hand here. We chose
            # the straightforward version; revisit if this is ever too slow.
            for j, model in enumerate(fold_path.models_):
                μ = model.predict(held_out_X, offset=offset[held_out_rows])
                deviance = self.family.deviance(
                    response[held_out_rows], μ, held_out_weights
                )
                fold_deviances[k, j] = deviance / fold_weights[k]

        cv_deviance_mean: CVDeviances = np.average(
            fold_deviances, axis=0, weights=fold_weights
        )
        cv_deviance_standard_error = _cv_standard_error(
            fold_deviances, fold_weights, cv_deviance_mean
        )
        best = np.argmin(cv_deviance_mean)
        threshold = cv_deviance_mean[best] + cv_deviance_standard_error[best]
        # The alphas decrease, so the first within the threshold is the largest.
        within = np.flatnonzero(cv_deviance_mean <= threshold)[0]

        self.alphas_ = alphas
        self.path_ = path
        self.fold_frequency_weights_ = folds
        self.fold_weights_ = fold_weights
        self.fold_deviances_ = fold_deviances
        self.cv_deviance_ = cv_deviance_mean
        self.cv_standard_error_ = cv_deviance_standard_error
        self.alpha_min_ = float(alphas[best])
        self.alpha_1se_ = float(alphas[within])
        return self

    @property
    def model_min_(self) -> GLM:
        """The model, fit to the full data, at alpha_min_."""
        if self.path_ is None or self.alpha_min_ is None:
            raise ValueError("The cross-validation has not been fit.")
        return self.path_.model_at(self.alpha_min_)

    @property
    def model_1se_(self) -> GLM:
        """The model, fit to the full data, at alpha_1se_."""
        if self.path_ is None or self.alpha_1se_ is None:
            raise ValueError("The cross-validation has not been fit.")
        return self.path_.model_at(self.alpha_1se_)


def fold_frequency_weights(
    frequency_weights: FrequencyWeights,
    n_folds: int,
    rng: np.random.Generator,
) -> FoldFrequencyWeights:
    """Deal the observations that rows encapsulate to cross-validation
    folds.

    A row with frequency weight f stands for f independent observations, and
    each is assigned a fold separately. This splits the data exactly as
    randomly partitioning the expanded data, with one row per observation,
    into folds of as equal sizes as possible would. That is, a row's
    observations may be split across several folds. A row with frequency
    weight one, such as a row with only a precision weight, is held out whole
    by a single fold.

    Splitting a row's copies means a held-out observation often has an exact
    duplicate in training. This isn't leakage: if the copies are independent
    observations, new data will often duplicate training data too.

    Parameters
    ----------

    frequency_weights: array, shape (n_samples, )
        How many observations each row stands for: whole numbers, summing to
        at least n_folds.

    n_folds: integer, at least 2
        The number of folds.

    rng: numpy Generator
        The generator to assign the folds with.

    Returns
    -------

    folds: array, shape (n_folds, n_samples)
        How many of each row's observations each fold holds out. Each column
        sums to the row's frequency weight.
    """
    if not np.all(frequency_weights == np.round(frequency_weights)):
        raise ValueError(
            "frequency_weights must be whole numbers to split the observations "
            "they stand for into folds."
        )
    remaining = frequency_weights.astype(np.int64)
    n_observations = int(np.sum(remaining))
    if n_observations < n_folds:
        raise ValueError(
            f"There must be at least as many observations as folds, got "
            f"{n_observations} observations for {n_folds} folds."
        )
    folds = np.empty((n_folds, remaining.shape[0]))
    residue = n_observations % n_folds
    for k in range(n_folds):
        # Randomly partitioning the observations into folds draws each fold's
        # observations without replacement from those not yet assigned.
        # The residue, left over by dividing evenly, goes one observation
        # each to the first few folds.
        fold_size = n_observations // n_folds + (k < residue)
        drawn = rng.multivariate_hypergeometric(remaining, fold_size)
        folds[k] = drawn
        remaining = remaining - drawn
    return folds


def default_alphas(
    family: ExponentialFamily,
    X: DesignMatrix,
    y: ResponseVector,
    *,
    frequency_weights: FrequencyWeights | None = None,
    precision_weights: PrecisionWeights | None = None,
    standardize: Standardization = "continuous",
    n_alphas: int = 100,
) -> AlphaGrid:
    """A grid of penalty strengths spanning the whole regularization path.

    Ridge shrinks a fit along each eigen-direction of the information matrix
    by the factor d / (d + alpha), where d is that direction's eigenvalue: by
    half when alpha = d, barely when alpha is much smaller, and almost
    completely when alpha is much larger (Hastie, Tibshirani and Friedman, The
    Elements of Statistical Learning, 2nd ed., section 3.4.1, for linear
    regression; a GLM is locally a weighted linear regression, with the
    information matrix in place of XᵗX). So the path happens between the
    eigenvalues, and the grid is log-spaced from GRID_MARGIN times the largest
    eigenvalue down to the smallest nonzero eigenvalue divided by GRID_MARGIN.

    The information matrix is evaluated at the intercept-only model, for the
    (standardized) penalized columns, with the unpenalized intercept profiled
    out. Zero eigenvalues, e.g. from a categorical variable with a column for
    every level, which sum to the intercept, are skipped: the intercept
    absorbs those directions, so the penalty never needs to resolve them.
    Offsets are ignored, which is harmless for choosing the grid's range.
    """
    frequency_weights = check_weights("frequency_weights", y, frequency_weights)
    precision_weights = check_weights("precision_weights", y, precision_weights)
    observation_weights: ObservationWeights = frequency_weights * precision_weights
    # The intercept-only model fits the weighted mean response everywhere.
    μ = np.full(X.shape[0], np.average(y, weights=observation_weights))
    with np.errstate(divide="ignore"):
        η = family.link(μ)
    if not np.all(np.isfinite(η)):
        raise ValueError(NON_FINITE_ESTIMATES_MESSAGE)
    Z = Standardizer(standardize).fit(X, frequency_weights).transform(X)
    eigenvalues = _profiled_eigenvalues(
        Z,
        _working_weights(family, η, μ, observation_weights),
        has_intercept_column(X),
    )
    nonzero = eigenvalues[eigenvalues > 1e-10 * eigenvalues.max()]
    if nonzero.size == 0:
        raise ValueError("The penalized columns of X carry no information.")
    grid: AlphaGrid = np.logspace(
        np.log10(GRID_MARGIN * nonzero.max()),
        np.log10(nonzero.min() / GRID_MARGIN),
        n_alphas,
    )
    return grid


def _take_rows(
    X: DesignMatrix | DesignFrame, rows: npt.NDArray[np.intp]
) -> DesignMatrix | DesignFrame:
    """The given rows of a design matrix or data frame."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[rows]
    return X[rows]


def _cv_standard_error(
    fold_deviances: FoldDeviances,
    fold_weights: FoldWeights,
    cv_deviance: CVDeviances,
) -> CVDeviances:
    """The standard error of the weighted mean of the fold deviances: the
    square root of their weighted variance, with the usual K / (K - 1)
    correction for K folds, over K."""
    K = fold_deviances.shape[0]
    variance = (
        K
        / (K - 1)
        * np.average((fold_deviances - cv_deviance) ** 2, axis=0, weights=fold_weights)
    )
    standard_error: CVDeviances = np.sqrt(variance / K)
    return standard_error


def _term_columns(template: GLM, X: DesignMatrix) -> TermColumns:
    """Group the penalized coefficients' names by the model term they belong
    to, or one per coefficient for a model fit without a formula."""
    X_names = template.X_names or []
    penalized = template._penalized_coefficients(X)
    if template.X_spec is None:
        return {name: [name] for name, p in zip(X_names, penalized) if p}
    term_columns: TermColumns = {}
    for term, columns in template.X_spec.term_slices.items():
        names = [X_names[j] for j in range(columns.start, columns.stop) if penalized[j]]
        if names:
            term_columns[str(term)] = names
    return term_columns


def _working_weights(
    family: ExponentialFamily,
    η: LinearPredictor,
    μ: MeanVector,
    observation_weights: ObservationWeights,
) -> WorkingWeights:
    """The weights W in the information matrix XᵗWX, at the means μ."""
    # Observation weights here, since this formula arises from an expected
    # second derivative of the deviance, which is observation weighted.
    working_weights: WorkingWeights = (
        observation_weights * family.d_inv_link(η, μ) ** 2 / family.variance(μ)
    )
    return working_weights


def _profiled_eigenvalues(
    Z: DesignMatrix, working_weights: WorkingWeights, has_intercept: bool
) -> InformationEigenvalues:
    """The eigenvalues of the information matrix of the penalized columns of
    Z, with the intercept, if there is one, profiled out."""
    Z_penalized = Z[:, 1:] if has_intercept else Z
    if Z_penalized.shape[1] == 0:
        raise ValueError("The model has no penalized coefficients.")
    if has_intercept:
        # The penalty doesn't act on the intercept, so whatever the penalized
        # coefficients are, the intercept is refit freely alongside them. The
        # eigenvalues we want measure the information about the penalized
        # coefficients that remains *after* the intercept has been fit.
        # Their block of the information matrix, ZᵗWZ, overstates it: a column
        # with a large mean partly mimics the intercept, and ZᵗWZ credits it
        # with information that really pins down the response's overall level.
        #
        # Removing what the intercept can explain is called "profiling it out".
        #
        # Write the full information matrix XᵗWX, intercept first, in blocks:
        #
        #   [ 1ᵗW1  1ᵗWZ ]
        #   [ ZᵗW1  ZᵗWZ ]
        #
        # The information about Z's coefficients with the intercept profiled
        # out is the Schur complement of the intercept's block:
        #
        #   ZᵗWZ - ZᵗW1 (1ᵗW1)⁻¹ 1ᵗWZ = (Z - 1m)ᵗ W (Z - 1m),
        #
        # where m = (1ᵗW1)⁻¹ 1ᵗWZ is the vector of the columns' means, weighted
        # by the working weights W (expanding the right hand side gives the
        # left). So centering each column at its weighted mean, and then
        # forming ZᵗWZ, computes exactly the profiled information. The
        # Standardizer's centering doesn't suffice: it uses the frequency weights,
        # not the working weights, and doesn't center indicator columns.
        Z_penalized = Z_penalized - np.average(
            Z_penalized, axis=0, weights=working_weights
        )
    information = Z_penalized.T @ (Z_penalized * working_weights[:, None])
    # The matrix is positive semi-definite; clip rounding error below zero.
    eigenvalues: InformationEigenvalues = np.clip(
        np.linalg.eigvalsh(information), 0.0, None
    )
    return eigenvalues


def _effective_degrees_of_freedom(
    models: list[GLM],
    alphas: AlphaGrid,
    X: DesignMatrix,
    Z: DesignMatrix,
    offset: Offset,
    observation_weights: ObservationWeights,
    has_intercept: bool,
) -> EffectiveDegreesOfFreedom:
    """sum(d / (d + alpha)) for each model, with d the eigenvalues of its
    information matrix, at its own fitted means. The penalty acts on the
    standardized coefficients, so the eigenvalues are for the standardized
    columns Z of the design matrix X."""
    df = np.empty(len(models))
    for i, (model, alpha) in enumerate(zip(models, alphas)):
        family = model.family
        if model.coef_ is None:
            raise ValueError("The path's models must be fit.")
        η = X @ model.coef_ + offset
        μ = family.inv_link(η)
        d = _profiled_eigenvalues(
            Z, _working_weights(family, η, μ, observation_weights), has_intercept
        )
        df[i] = np.sum(d / (d + alpha))
    return df
