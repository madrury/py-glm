"""Fitting ridge regularization paths."""

from collections.abc import Sequence
from typing import Self

import numpy as np

from glm.families import ExponentialFamily
from glm.glm import GLM, NON_FINITE_ESTIMATES_MESSAGE
from glm.simulation import coefficient_table
from glm.standardize import Standardizer, check_standardization
from glm.types import (
    AlphaGrid,
    CoefficientTable,
    CoefficientVector,
    ColumnScales,
    DesignFrame,
    DesignMatrix,
    EffectiveDegreesOfFreedom,
    Formula,
    InformationEigenvalues,
    LinearPredictor,
    MeanVector,
    Offset,
    ResponseVector,
    SampleWeights,
    Standardization,
    TermColumns,
)
from glm.utils import has_intercept_column

# The default grid spans from this many times the largest eigenvalue, to this
# many times smaller than the smallest; see default_alphas.
GRID_MARGIN = 100.0


class RidgePath:
    """Fit a GLM along a grid of ridge penalty strengths.

    Models are fit from the largest alpha to the smallest, each starting from
    the previous model's coefficients, which makes fitting the whole path cheap.
    Every alpha must be strictly positive, so models specified with a formula
    encode each level of all categorical variables.

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
        if alphas is not None:
            if len(alphas) == 0 or not all(alpha > 0 for alpha in alphas):
                raise ValueError("alphas must be a non-empty sequence of positives.")
        elif (
            isinstance(n_alphas, bool) or not isinstance(n_alphas, int) or n_alphas < 2
        ):
            raise ValueError(
                f"n_alphas must be an integer of at least 2, got {n_alphas!r}."
            )
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
        sample_weights: SampleWeights | None = None,
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
        if self.alphas is None:
            alphas = default_alphas(
                self.family,
                design,
                response,
                sample_weights=sample_weights,
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
                sample_weights=sample_weights,
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
        weights = np.ones(design.shape[0]) if sample_weights is None else sample_weights
        standardizer = Standardizer(self.standardize).fit(design, weights)
        self.scales_ = standardizer.scales_
        self.df_ = _effective_degrees_of_freedom(
            models,
            alphas,
            design,
            standardizer.transform(design),
            np.zeros(design.shape[0]) if offset is None else offset,
            weights,
            has_intercept_column(design),
        )
        return self

    def coefficient_table(self) -> CoefficientTable:
        """The coefficients along the path: one row per penalty strength,
        indexed by alpha, and one column per coefficient."""
        if self.models_ is None or self.alphas_ is None:
            raise ValueError("The path has not been fit.")
        table = coefficient_table(self.models_)
        table.index = self.alphas_
        table.index.name = "alpha"
        return table


def default_alphas(
    family: ExponentialFamily,
    X: DesignMatrix,
    y: ResponseVector,
    *,
    sample_weights: SampleWeights | None = None,
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
    information matrix in place of X'X). So the path happens between the
    eigenvalues, and the grid is log-spaced from GRID_MARGIN times the largest
    eigenvalue down to the smallest nonzero eigenvalue divided by GRID_MARGIN.

    The information matrix is evaluated at the intercept-only model, for the
    (standardized) penalized columns, with the unpenalized intercept profiled
    out. Zero eigenvalues, e.g. from a categorical variable with a column for
    every level, which sum to the intercept, are skipped: the intercept
    absorbs those directions, so the penalty never needs to resolve them.
    Offsets are ignored, which is harmless for choosing the grid's range.
    """
    weights = np.ones(X.shape[0]) if sample_weights is None else sample_weights
    # The intercept-only model fits the weighted mean response everywhere.
    μ = np.full(X.shape[0], np.average(y, weights=weights))
    with np.errstate(divide="ignore"):
        η = family.link(μ)
    if not np.all(np.isfinite(η)):
        raise ValueError(NON_FINITE_ESTIMATES_MESSAGE)
    Z = Standardizer(standardize).fit(X, weights).transform(X)
    eigenvalues = _profiled_eigenvalues(
        Z, _working_weights(family, η, μ, weights), has_intercept_column(X)
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
    sample_weights: SampleWeights,
) -> SampleWeights:
    """The weights W in the information matrix X'WX, at the means μ."""
    working_weights: SampleWeights = (
        sample_weights * family.d_inv_link(η, μ) ** 2 / family.variance(μ)
    )
    return working_weights


def _profiled_eigenvalues(
    Z: DesignMatrix, working_weights: SampleWeights, has_intercept: bool
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
        # Their block of the information matrix, Z'WZ, overstates it: a column
        # with a large mean partly mimics the intercept, and Z'WZ credits it
        # with information that really pins down the response's overall level.
        #
        # Removing what the intercept can explain is called "profiling it out".
        #
        # Write the full information matrix X'WX, intercept first, in blocks:
        #
        #   [ 1'W1  1'WZ ]
        #   [ Z'W1  Z'WZ ]
        #
        # The information about Z's coefficients with the intercept profiled
        # out is the Schur complement of the intercept's block:
        #
        #   Z'WZ - Z'W1 (1'W1)⁻¹ 1'WZ = (Z - 1m)' W (Z - 1m),
        #
        # where m = (1'W1)⁻¹ 1'WZ is the vector of the columns' means, weighted
        # by the working weights W (expanding the right hand side gives the
        # left). So centering each column at its weighted mean, and then
        # forming Z'WZ, computes exactly the profiled information. The
        # Standardizer's centering doesn't suffice: it uses the sample weights,
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
    sample_weights: SampleWeights,
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
            Z, _working_weights(family, η, μ, sample_weights), has_intercept
        )
        df[i] = np.sum(d / (d + alpha))
    return df
