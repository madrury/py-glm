"""Domain type aliases for generalized linear models.

Shapes are given in terms of n, the number of observations, and p, the number
of coefficients in the model (including the intercept).
"""

from typing import Any, Literal

import formulaic
import numpy as np
import numpy.typing as npt
import pandas as pd

# Model inputs.
type DesignMatrix = npt.NDArray[np.float64]
"""The design matrix X, shape (n, p). Column 0, X[:, 0], is the intercept."""

type DesignFrame = pd.DataFrame
"""A data frame of raw variables, turned into a design matrix by a formula."""

type Formula = str
"""A model formula, e.g. "y ~ x1 + x2"."""

type DesignSpec = formulaic.ModelSpec
"""The learned mapping from a DesignFrame to a DesignMatrix. Captures state fit
on the training data, such as categorical levels and centering constants, so
new data is encoded consistently."""

type ResponseVector = npt.NDArray[np.floating[Any] | np.integer[Any]]
"""The observed response y, shape (n,). May be integer valued, e.g. counts or
binary outcomes."""

type Offset = npt.NDArray[np.float64]
"""A known term added to the linear predictor, shape (n,)."""

type SampleWeights = npt.NDArray[np.float64]
"""Per-observation weights on the deviance, shape (n,)."""

# Penalized fitting.
type CoefficientMask = npt.NDArray[np.bool_]
"""Which coefficients a penalty applies to, shape (p,): all of them, except
the intercept when the model has one."""

type Standardization = Literal["none", "continuous", "all"]
"""Which design matrix columns to standardize before penalized fitting: none;
continuous columns only, leaving 0/1 indicator columns unscaled; or all
columns, as glmnet does. The intercept is never standardized."""

type ColumnCenters = npt.NDArray[np.float64]
"""The value subtracted from each design matrix column when standardizing,
shape (p,)."""

type ColumnScales = npt.NDArray[np.float64]
"""The value each centered design matrix column is divided by when
standardizing, shape (p,)."""

# Model structure.
type CoefficientVector = npt.NDArray[np.float64]
"""The coefficients β, shape (p,)."""

type LinearPredictor = npt.NDArray[np.float64]
"""The linear predictor η = Xβ + offset, shape (n,)."""

type MeanVector = npt.NDArray[np.float64]
"""The conditional mean μ = E[y | X], shape (n,)."""

type InverseLinkDerivative = npt.NDArray[np.float64]
"""The derivative of the mean with respect to the linear predictor, dμ/dη,
evaluated at each observation, shape (n,)."""

type VarianceVector = npt.NDArray[np.float64]
"""The variance function evaluated at the mean, V(μ), shape (n,)."""

# Inference.
type DevianceGradient = npt.NDArray[np.float64]
"""The gradient of the (penalized) deviance with respect to β, shape (p,)."""

type InformationMatrix = npt.NDArray[np.float64]
"""The Fisher information matrix, shape (p, p)."""

type CovarianceMatrix = npt.NDArray[np.float64]
"""The estimated covariance matrix of the coefficients, shape (p, p)."""

type StandardErrorVector = npt.NDArray[np.float64]
"""The estimated standard errors of the coefficients, shape (p,)."""

type PValueVector = npt.NDArray[np.float64]
"""P-values for the hypotheses that each coefficient is zero, shape (p,)."""

type Deviance = float
"""A (possibly penalized) deviance."""

# Simulation.
type SimulatedResponses = npt.NDArray[np.float64]
"""Responses simulated from a fit model, shape (n_sim, n). Each row is one
simulated response vector, so iterating yields one replicate at a time."""

type CoefficientTable = pd.DataFrame
"""Coefficient estimates from several fit models, such as bootstrap
replicates: one row per model, and one column per coefficient, named by the
models' X_names."""
