"""Fitting ridge regularization paths."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from glm.encoding import encode_design, learn_formula
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
    DesignSpec,
    DevianceWeights,
    EffectiveDegreesOfFreedom,
    FoldDeviances,
    FoldFrequencyWeights,
    FoldWeights,
    Formula,
    FrequencyWeights,
    InformationEigenvalues,
    Offset,
    PrecisionWeights,
    ResponseVector,
    RowIndices,
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
        model, used for plotting coefficients in groups. For a model fit without
        a formula, each coefficient is its own term.

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
        standardize: Standardization = "continuous",
    ) -> None:
        check_standardization(standardize)
        self.family = family
        self.standardize: Standardization = standardize
        self.alphas_: AlphaGrid | None = None
        self.models_: list[GLM] | None = None
        self.term_columns_: TermColumns | None = None
        self.scales_: ColumnScales | None = None
        self.df_: EffectiveDegreesOfFreedom | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(family={self.family!r}, "
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
        alphas: Sequence[float] | None = None,
        n_alphas: int = 100,
        max_iter: int = 100,
        tol: float = 0.1**5,
        validate_intercept: bool = True,
    ) -> Self:
        """Fit a model for each penalty strength.

        The arguments are as for GLM.fit, except alphas and n_alphas.

        Parameters
        ----------

        alphas: sequence of positive floats, optional
            The penalty strengths to fit, in any order. By default, a grid is
            chosen from the data; see default_alphas.

        n_alphas: integer, at least 2
            The number of penalty strengths in the default grid. Ignored when
            alphas are supplied.
        """
        check_alphas(alphas, n_alphas)
        # We construct a template GLM to help encode the data, so the value of
        # alpha for this template does not matter as long as its not zero.
        template = GLM(self.family, alpha=1.0, standardize=self.standardize)
        design, response = template._prepare_data(X, y, formula, X_names, y_name)
        frequency_weights = check_weights(
            "frequency_weights", response, frequency_weights
        )
        precision_weights = check_weights(
            "precision_weights", response, precision_weights
        )
        deviance_weights: DevianceWeights = frequency_weights * precision_weights
        has_intercept = has_intercept_column(design)

        # Learn the standardization once from this path's training data.
        # Every model on the path, the default grid calculation, and the
        # effective degrees of freedom calculation of this grid, all use this
        # one standardizer.
        standardizer = Standardizer(self.standardize).fit(design, frequency_weights)
        standardized_design = standardizer.transform(design)
        grid: AlphaGrid
        if alphas is None:
            grid = default_alphas(
                self.family,
                standardized_design,
                response,
                deviance_weights=deviance_weights,
                has_intercept=has_intercept,
                n_alphas=n_alphas,
            )
        else:
            grid = np.sort(np.asarray(alphas, dtype=np.float64))[::-1]

        # Execute: fit over the grid of alphas.
        models: list[GLM] = []
        warm_start: CoefficientVector | None = None
        for alpha in grid:
            # We fit each model in the path to the already encoded data,
            #
            # Note: we don't supply a formula herewhen constructing the object.
            # After fitting we manually inject the correct formula and specs so
            # that prediction on a new dataset can encode data correctly.
            model = GLM(self.family, alpha=float(alpha), standardize=self.standardize)
            model._prepare_data(
                design,
                response,
                formula=None,
                X_names=template.X_names,
                y_name=template.y_name,
            )
            model._fit(
                design,
                response,
                warm_start=warm_start,
                offset=offset,
                frequency_weights=frequency_weights,
                precision_weights=precision_weights,
                max_iter=max_iter,
                tol=tol,
                validate_intercept=validate_intercept,
                standardizer=standardizer,
            )
            model._copy_formula(template)
            models.append(model)
            warm_start = model.coef_

        self.alphas_ = grid
        self.models_ = models
        self.term_columns_ = _term_columns(template, design)
        self.scales_ = standardizer.scales_
        # The penalty acts on the standardized coefficients, so the effective
        # degrees of freedom need the eigenvalues of the information matrix of
        # the standardized columns.
        self.df_ = _effective_degrees_of_freedom(
            models,
            grid,
            design,
            standardized_design,
            np.zeros(design.shape[0]) if offset is None else offset,
            deviance_weights,
            has_intercept,
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

    Each fold's held-out loss is its deviance per unit of deviance weight,

        sum(f * p * d) / sum(f * p),

    with:

        - f the held-out frequency weights
        - p the precision weights
        - d the unit deviances

    The folds are averaged weighted by their total held-out weight, so that each
    unit of weight counts equally (rather than each fold).

    Each fold learns its own formula encoding, from its training data alone,
    before any model is fit (see cv_encoding_spec). This has a couple of
    consequences:

    For categorical variables, this means that a level of a categorical variable
    may be missing from a fold's training data. In this case the resulting fit
    model predicts any test observation with this category at the baseline, with
    all its indicators zero (see GLM.predict). This means it is important to
    *fully* encode each categorical instead of leaving one level out, otherwise
    the baseline will be the level left out. The default formula encoding when
    using a ridge penalty is to fully encode each categorical, so there is only
    a risk when encoding manually.

    For continuous variables, this means that the standardizing means and scales
    are computed from the fold's training data alone, so the held-out data has
    no influence on the penalty. GLM.fit standardizes internally and returns
    coefficients on the original scale.

    Parameters
    ----------

    family: ExponentialFamily
        The exponential family used in the models.

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
        The total deviance weight each fold held out.

    fold_deviances_: array, shape (n_folds, n_alphas)
        Each fold's held-out deviance per unit of deviance weight, at each
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
    """

    def __init__(
        self,
        family: ExponentialFamily,
        n_folds: int = 10,
        standardize: Standardization = "continuous",
    ) -> None:
        check_standardization(standardize)
        check_integer_at_least("n_folds", n_folds, 2)
        self.family = family
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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(family={self.family!r}, "
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
        alphas: Sequence[float] | None = None,
        n_alphas: int = 100,
        max_iter: int = 100,
        tol: float = 0.1**5,
        validate_intercept: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Self:
        """Fit the full path, and cross-validate it.

        The arguments are the same as in GLM.fit, except alphas, n_alphas and
        rng.

        Parameters
        ----------

        alphas: sequence of positive floats, optional
            The penalty strengths to fit, in any order. By default, a grid is
            chosen from the full data; see default_alphas. Every fold's path is
            fit over the same grid.

        n_alphas: integer, at least 2
            The number of penalty strengths in the default grid. Ignored when
            alphas are supplied.

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
        # We begin by fitting a ridge path on the entire dataset. The resulting
        # path of models are not part of the cross validation loop, i.e., they
        # do not enter into estimation of the test error.
        #
        # But they do serve some important purposes:
        #   - The alpha grid computed by this RidgePath is used for each
        #     subsequent RidgePath in the cross validation loop. This ensures
        #     every path uses the same grid of alphas.
        #   - The model along this path at the minimum CV error is the "best"
        #     model, we don't need to retrain after determining the optimal
        #     alpha.
        # The entire path of trained models is retained as self.path_, in case
        # the user desires some other selection criteria.
        path = RidgePath(self.family, standardize=self.standardize).fit(
            X,
            y,
            formula,
            X_names=X_names,
            y_name=y_name,
            offset=offset,
            frequency_weights=frequency_weights,
            precision_weights=precision_weights,
            alphas=alphas,
            n_alphas=n_alphas,
            max_iter=max_iter,
            tol=tol,
            validate_intercept=validate_intercept,
        )
        assert path.alphas_ is not None and path.models_ is not None
        grid = path.alphas_
        # Any model in the path would work here, we're just pulling out the
        # response vector, and they're all the same.
        response = path.models_[0].response_vector(X, y)
        frequency_weights = check_weights(
            "frequency_weights", response, frequency_weights
        )
        precision_weights = check_weights(
            "precision_weights", response, precision_weights
        )
        if offset is None:
            offset = np.zeros(response.shape[0])

        # Rows with zero precision weight do not influence the fit, so dealing
        # their observations to folds would only unbalance them. They stay in
        # every training set, where their zero weight leaves the fits unchanged.
        folds = fold_frequency_weights(
            np.where(precision_weights > 0, frequency_weights, 0.0),
            self.n_folds,
            _generator(rng),
        )

        # Plan the cross validation fits:
        # We record what each fold trains on and holds out, and learn its
        # encoding from its training rows alone. No models are fit here, we
        # return a data object encapsulating the computation to be done.
        encoding = cv_encoding_spec(X, formula, response, frequency_weights, folds)

        # Execute the cross validation fits:
        # Fit each fold's full alpha path, and score it on the observations it
        # held out. We've already constructed the model matrix specifications,
        # so each RidgePath is fit to design matrices, without a formula. Each
        # fold's design matrices only live for its own iteration.
        fold_weights = np.empty(self.n_folds)
        fold_deviances = np.empty((self.n_folds, len(grid)))
        for k, fold in enumerate(encoding.folds):
            training_rows = fold.training_rows()
            held_out_rows = fold.held_out_rows()
            held_out_weights: DevianceWeights = (
                fold.held_out_frequency_weights[held_out_rows]
                * precision_weights[held_out_rows]
            )
            fold_weights[k] = np.sum(held_out_weights)
            fold_path = RidgePath(self.family, standardize=self.standardize).fit(
                fold.training_design(X),
                encoding.response[training_rows],
                offset=offset[training_rows],
                frequency_weights=fold.training_frequency_weights[training_rows],
                precision_weights=precision_weights[training_rows],
                alphas=list(grid),
                max_iter=max_iter,
                tol=tol,
                validate_intercept=validate_intercept,
            )
            assert fold_path.models_ is not None
            held_out_design = fold.held_out_design(X)
            for j, model in enumerate(fold_path.models_):
                μ = model.predict(held_out_design, offset=offset[held_out_rows])
                deviance = self.family.deviance(
                    encoding.response[held_out_rows], μ, held_out_weights
                )
                fold_deviances[k, j] = deviance / fold_weights[k]

        cv_deviance_mean: CVDeviances = np.average(
            fold_deviances, axis=0, weights=fold_weights
        )
        cv_deviance_standard_error = _cv_standard_error(
            fold_deviances, fold_weights, cv_deviance_mean
        )
        best = np.argmin(cv_deviance_mean)

        self.alphas_ = grid
        self.path_ = path
        self.fold_frequency_weights_ = folds
        self.fold_weights_ = fold_weights
        self.fold_deviances_ = fold_deviances
        self.cv_deviance_ = cv_deviance_mean
        self.cv_standard_error_ = cv_deviance_standard_error
        self.alpha_min_ = float(grid[best])
        return self

    @property
    def model_min_(self) -> GLM:
        """The model, fit to the full data, at alpha_min_."""
        if self.path_ is None or self.alpha_min_ is None:
            raise ValueError("The cross-validation has not been fit.")
        return self.path_.model_at(self.alpha_min_)


@dataclass(frozen=True)
class FoldEncoding:
    """Encapsulates what a single cross-validation fold trains on and holds out,
    and how it encodes its data.

    A row is absent from the fold's training data, or from its held-out data,
    when its frequency weight there is zero, so the weights alone describe the
    row set of a fold.

    We note: the absent rows must still be removed from the training data before
    learning the encoding: formulaic learns categorical levels and transform
    state from every row it is given, whatever the row's weight.

    Attributes
    ----------
    training_frequency_weights: array, shape (n_samples, )
        How many of each row's observations the fold trains on.

    held_out_frequency_weights: array, shape (n_samples, )
        How many of each row's observations the fold holds out.

    X_spec: formulaic.ModelSpec or None
        The design specs learned from the fold's training rows alone, or None
        when the data is already a design matrix.
    """

    training_frequency_weights: FrequencyWeights
    held_out_frequency_weights: FrequencyWeights
    X_spec: DesignSpec | None

    def training_rows(self) -> RowIndices:
        """The rows with some observations in the fold's training data."""
        return np.flatnonzero(self.training_frequency_weights > 0)

    def held_out_rows(self) -> RowIndices:
        """The rows with some observations held out by the fold."""
        return np.flatnonzero(self.held_out_frequency_weights > 0)

    def training_design(self, X: DesignMatrix | DesignFrame) -> DesignMatrix:
        """The design matrix of the fold's training rows.

        Parameters
        ----------
        X: array, shape (n_samples, n_features) or pd.DataFrame
            The full data, as passed to RidgePathCV.fit.
        """
        return self._design(X, self.training_rows())

    def held_out_design(self, X: DesignMatrix | DesignFrame) -> DesignMatrix:
        """The design matrix of the fold's held-out rows, encoded with the
        specs learned from its training rows.

        Parameters
        ----------
        X: array, shape (n_samples, n_features) or pd.DataFrame
            The full data, as passed to RidgePathCV.fit.
        """
        return self._design(X, self.held_out_rows())

    def _design(self, X: DesignMatrix | DesignFrame, rows: RowIndices) -> DesignMatrix:
        if self.X_spec is None:
            assert isinstance(X, np.ndarray)
            design: DesignMatrix = X[rows]
            return design
        assert isinstance(X, pd.DataFrame)
        return encode_design(self.X_spec, X.iloc[rows])


@dataclass(frozen=True)
class CVEncodingSpec:
    """Everything a cross-validation needs to encode each of its folds' data.

    Attributes
    ----------
    response: array, shape (n_samples, )
        The response of every row, encoded once from the full data.

    folds: list of FoldEncoding
        What each fold trains on and holds out, and a specification of its data
        encoding.
    """

    response: ResponseVector
    folds: list[FoldEncoding]


def cv_encoding_spec(
    X: DesignMatrix | DesignFrame,
    formula: Formula | None,
    response: ResponseVector,
    frequency_weights: FrequencyWeights,
    held_out_frequency_weights: FoldFrequencyWeights,
) -> CVEncodingSpec:
    """Plan a cross-validation's encodings: each fold's training and held-out
    weights, and the design specs learned from its training rows alone.

    Parameters
    ----------
    X: array, shape (n_samples, n_features) or pd.DataFrame
        The full data.

    formula: str or None
        The model's formula, or None when X is already a design matrix.

    response: array, shape (n_samples, )
        The response of every row.

    frequency_weights: array, shape (n_samples, )
        How many observations each row stands for.

    held_out_frequency_weights: array, shape (n_folds, n_samples)
        How many of each row's observations each fold holds out; see
        fold_frequency_weights.

    Returns
    -------
    encoding: CVEncodingSpec
        The plan for every fold.
    """
    folds: list[FoldEncoding] = []
    for held_out_weights in held_out_frequency_weights:
        training_weights = frequency_weights - held_out_weights
        X_spec: DesignSpec | None = None
        if formula:
            assert isinstance(X, pd.DataFrame)
            training_frame = X.iloc[np.flatnonzero(training_weights > 0)]
            # Every model on a ridge path is penalized. The encoded matrices
            # are discarded: the plan keeps only the learned specs.
            X_spec = learn_formula(
                formula, training_frame, penalized=True
            ).rhs.model_spec
        folds.append(FoldEncoding(training_weights, held_out_weights, X_spec))
    return CVEncodingSpec(response, folds)


def fold_frequency_weights(
    frequency_weights: FrequencyWeights,
    n_folds: int,
    rng: np.random.Generator,
) -> FoldFrequencyWeights:
    """Deal the (possibly multiple) observations that rows encapsulate to
    cross-validation folds.

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
    assert np.all(remaining == 0)
    return folds


def default_alphas(
    family: ExponentialFamily,
    Z: DesignMatrix,
    y: ResponseVector,
    *,
    deviance_weights: DevianceWeights,
    has_intercept: bool,
    n_alphas: int = 100,
) -> AlphaGrid:
    """A grid of penalty strengths spanning the whole regularization path.

    Ridge shrinks a fit along each eigen-direction of the information matrix
    by the factor d / (d + alpha), where d is that direction's eigenvalue: by
    half when alpha = d, barely when alpha is much smaller, and almost
    completely when alpha is much larger (Hastie, Tibshirani and Friedman, The
    Elements of Statistical Learning, 2nd ed., section 3.4.1, for linear
    regression; a GLM is locally a weighted linear regression, with the
    information matrix in place of XᵗX). This means that if:

      alpha >> largest eigenvalue, then every direction is shrunk away.
      alpha << smallest eigenvalue, then every direction is left alone

    So the meaningful stuff happens between the eigenvalues. Consequently, we
    take our alpha grid log-spaced from GRID_MARGIN times the largest eigenvalue
    down to the smallest nonzero eigenvalue divided by GRID_MARGIN.

    The information matrix is evaluated at the intercept-only model, for the
    (standardized) penalized columns, with the unpenalized intercept profiled
    out. Zero eigenvalues, e.g. from a categorical variable with a column for
    every level, which sum to the intercept, are skipped: the intercept
    absorbs those directions, so the penalty never needs to resolve them.
    Offsets are ignored, which is harmless for choosing the grid's range.

    Parameters
    ----------
    family: ExponentialFamily
        The family of the models on the path.

    Z: array, shape (n_samples, n_coefficients)
        The standardized design matrix, i.e. the columns the path's penalty
        acts on (see Standardizer). With standardize="none", the design matrix
        itself.

    y: array, shape (n_samples, )
        The response.

    deviance_weights: array, shape (n_samples, )
        Each row's weight in the deviance: its frequency weight times its
        precision weight.

    has_intercept: bool
        Is column 0 of the design matrix an intercept? Its direction is then
        profiled out of the information matrix, as it is not penalized.

    n_alphas: integer, at least 2
        The count of penalty strengths in the grid.

    Returns
    -------
    alphas: array, shape (n_alphas, )
        The grid, in decreasing order.
    """
    # The intercept-only model fits the weighted mean response everywhere.
    μ = np.full(Z.shape[0], np.average(y, weights=deviance_weights))
    with np.errstate(divide="ignore"):
        η = family.link(μ)
    if not np.all(np.isfinite(η)):
        raise ValueError(NON_FINITE_ESTIMATES_MESSAGE)
    eigenvalues = _profiled_eigenvalues(
        Z,
        family.working_weights(η, μ, deviance_weights),
        has_intercept,
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


def _profiled_eigenvalues(
    Z: DesignMatrix, working_weights: WorkingWeights, has_intercept: bool
) -> InformationEigenvalues:
    """The eigenvalues of the information matrix of the penalized columns of
    Z, with the intercept, if there is one, profiled out."""
    Z_penalized = Z[:, 1:] if has_intercept else Z
    if Z_penalized.shape[1] == 0:
        raise ValueError("The model has no penalized coefficients.")
    if has_intercept:
        # This gets a little tricky, our goal is to "profile out the intercept"
        # from the information matrix.
        #
        # The penalty doesn't act on the intercept, so whatever the penalized
        # coefficients are, the intercept is refit freely alongside them. The
        # eigenvalues we want measure the information about the penalized
        # coefficients that remains *after* the intercept has been fit.
        # Their block of the information matrix, ZᵗWZ, overstates it:
        # intuitively, a column with a large mean partly mimics the intercept,
        # and ZᵗWZ credits it with information that really pins down the
        # response's overall level.
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
        #   ZᵗWZ - ZᵗW1 (1ᵗW1)⁻¹ 1ᵗWZ
        #     = ZᵗWZ - ZᵗW1m
        #     = ZᵗW (Z - 1m)
        #     = ZᵗW (Z - 1m) - mᵗ1ᵗW (Z - 1m)  <- See prose below.
        #     = (Z - 1m)ᵗ W (Z - 1m),
        #
        # where m = (1ᵗW1)⁻¹ 1ᵗWZ is the vector of the columns' means, weighted
        # by the working weights W. The last equality holds because the centered
        # columns have weighted sum zero, 1ᵗW (Z - 1m) = 1ᵗWZ - (1ᵗW1) m = 0, so
        # subtracting mᵗ1ᵗW (Z - 1m) changes nothing. So centering each column
        # at its weighted mean, and then forming ZᵗWZ, computes exactly the
        # profiled information.
        #
        # Note that the Standardizer's centering doesn't suffice: it uses the
        # frequency weights, not the working weights, and doesn't center
        # indicator columns.
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
    deviance_weights: DevianceWeights,
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
            Z, family.working_weights(η, μ, deviance_weights), has_intercept
        )
        df[i] = np.sum(d / (d + alpha))
    return df
