"""Standardizing design matrices for penalized fitting.

A ridge/LASSO penalty is not invariant to the units of the predictors: measuring
a predictor in kilometres rather than metres makes its coefficient 1000 times
larger, and so it is penalized far more. Fitting on standardized columns, each
with unit standard deviation, makes the penalty treat predictors evenly. The
Standardizer encapsulates the transformation, so that the fitting code works on
the standardized scale while everything user facing stays on the original scale.
"""

from typing import Self

import numpy as np

from glm.types import (
    CoefficientVector,
    ColumnCenters,
    ColumnScales,
    DesignMatrix,
    FrequencyWeights,
    Standardization,
)
from glm.utils import check_standardization, has_intercept_column


class Standardizer:
    """Standardize the columns of a design matrix, and convert coefficients
    between the original and standardized scales.

    Each standardized column is z = (x - center) / scale. If column 0 is an
    intercept, it is left unchanged, as its coefficient is not penalized, and
    the other columns are centered, as the intercept absorbs the shift.
    Without an intercept, centering would change the model, so columns are
    only scaled, column 0 included.

    Parameters
    ----------

    standardize: "none", "continuous", or "all"
        Which columns to standardize. "continuous" leaves indicator columns,
        those containing only zeros and ones, unchanged. Columns with zero
        variance are always left unchanged.

    Attributes
    ----------

    centers_: array, shape (n_features, )
        The value subtracted from each column.

    scales_: array, shape (n_features, )
        The value each centered column is divided by.
    """

    def __init__(self, standardize: Standardization) -> None:
        check_standardization(standardize)
        self.standardize = standardize
        self.centers_: ColumnCenters | None = None
        self.scales_: ColumnScales | None = None

    def fit(
        self, X: DesignMatrix, frequency_weights: FrequencyWeights | None = None
    ) -> Self:
        """Learn each column's center and scale, as its mean and standard
        deviation, weighted by the frequency weights.

        Precision weights describe how precise each row's response is, not how
        common its predictors are, so they don't play a role.
        """
        weights = (
            np.ones(X.shape[0]) if frequency_weights is None else frequency_weights
        )
        means = np.average(X, axis=0, weights=weights)
        sds = np.sqrt(np.average((X - means) ** 2, axis=0, weights=weights))

        # Construct a mask indicating which columns to standardize.
        has_intercept = has_intercept_column(X)
        transformed = np.full(X.shape[1], self.standardize != "none")
        transformed &= sds > 0
        if self.standardize == "continuous":
            is_indicator = np.all((X == 0) | (X == 1), axis=0)
            transformed &= ~is_indicator
        if has_intercept:
            transformed[0] = False

        # Only center when the model contains an intercept term, otherwise
        # centering does not leave predictions invariant.
        centered = transformed & has_intercept
        self.centers_ = np.where(centered, means, 0.0)
        self.scales_ = np.where(transformed, sds, 1.0)
        return self

    def transform(self, X: DesignMatrix) -> DesignMatrix:
        centers, scales = self._fitted()
        standardized: DesignMatrix = (X - centers) / scales
        return standardized

    def coef_to_original(self, γ: CoefficientVector) -> CoefficientVector:
        """Convert coefficients fit to the standardized columns into
        coefficients for the original columns, which give the same linear
        predictor."""
        centers, scales = self._fitted()
        # In terms of the original columns, the standardized linear predictor is
        #
        #   Σⱼγⱼ(xⱼ - mⱼ)/sⱼ = Σⱼ(γⱼ / sⱼ)xⱼ - Σⱼ(γⱼ/sⱼ)mⱼ,
        #
        # so each coefficient is βⱼ = γⱼ / sⱼ, and what remains is the constant
        # -Σⱼ βⱼ mⱼ. Column 0 is never centered (m₀ = 0), so the sum is over j
        # >= 1. Since columns are only centered when column 0 is an intercept
        # (x₀ == 1), the constant can be folded into the intercept's coefficient;
        # without one, every mⱼ is zero, and so is the constant.
        β = γ / scales
        β[0] -= np.sum(β[1:] * centers[1:])
        return β

    def coef_to_standardized(self, β: CoefficientVector) -> CoefficientVector:
        """Convert coefficients for the original columns into coefficients for
        the standardized columns, which give the same linear predictor."""
        centers, scales = self._fitted()
        # The inverse of coef_to_original:
        #
        #   Σⱼ βⱼxⱼ = Σⱼ(βⱼsⱼ) (xⱼ - mⱼ)/sⱼ +  Σⱼβⱼmⱼ,
        #
        # So each standardized coefficient is γⱼ = βⱼ sⱼ for j >= 1, and the
        # intercept's coefficient takes back the constant, Σⱼ βⱼ mⱼ that
        # centering moved into it.
        γ = β * scales
        γ[0] += np.sum(β[1:] * centers[1:])
        return γ

    def _fitted(self) -> tuple[ColumnCenters, ColumnScales]:
        if self.centers_ is None or self.scales_ is None:
            raise ValueError("The Standardizer has not been fit.")
        return self.centers_, self.scales_
