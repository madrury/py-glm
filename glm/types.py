"""Domain type aliases for generalized linear models.

Shapes are given in terms of n, the number of observations, and p, the number
of coefficients in the model (including the intercept).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import formulaic
import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import rv_continuous_frozen

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

type FrequencyWeights = npt.NDArray[np.float64]
"""How many identical observations each row stands for, shape (n,): the data
have been compressed, and a row with weight 3 is three observations."""

type PrecisionWeights = npt.NDArray[np.float64]
"""How much more precise each row's response is than a single observation's,
shape (n,): Var[y | X] = dispersion * V(μ) / weight. Typical when a response
is an average or proportion over several units, e.g. a proportion of successes
weighted by its number of trials. R's glm calls these prior weights."""

type ObservationWeights = npt.NDArray[np.float64]
"""Each row's weight in a fit, its frequency weight times its precision weight,
shape (n,): the weight on its contribution to the deviance."""

type WorkingWeights = npt.NDArray[np.float64]
"""The weights W of Fisher scoring's weighted least squares step, shape (n,):
W = observation weight * (dμ/dη)² / V(μ), so that the information matrix is
X'WX. Unlike the observation weights, they depend on the current fit."""

# Penalized fitting.
type AlphaGrid = npt.NDArray[np.float64]
"""Ridge penalty strengths along a regularization path, in decreasing order,
shape (n_alphas,)."""

type InformationEigenvalues = npt.NDArray[np.float64]
"""The eigenvalues of an information matrix, which measure how much the data
say about each of its eigen-directions."""

type EffectiveDegreesOfFreedom = npt.NDArray[np.float64]
"""The effective degrees of freedom of each model along a regularization
path, shape (n_alphas,)."""

type PathAxis = Literal["log_alpha", "df"]
"""The horizontal axis of a regularization path plot: the penalty strength,
on a log scale, or the effective degrees of freedom."""

type CoefficientScale = Literal["standardized", "original"]
"""The scale on which to plot coefficients: standardized, as the penalty
sees them, or the original scale of the predictors."""

type TermColumns = dict[str, list[str]]
"""The names of the penalized coefficients belonging to each term of a model,
e.g. {"C(g)": ["C(g)[a]", "C(g)[b]"]}, used to plot coefficients in groups."""

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

type ReferenceDistribution = "rv_continuous_frozen"
"""The distribution of a coefficient's test statistic under the hypothesis
that the coefficient is zero: a frozen scipy.stats distribution."""

type CoefficientSummary = pd.DataFrame
"""A table of a fit model's coefficients, one row per coefficient: their
estimates, standard errors, test statistics, p-values, and confidence
intervals."""

type UnitDeviances = npt.NDArray[np.float64]
"""Each observation's contribution to the deviance, shape (n,)."""

type Deviance = float
"""A (possibly penalized) deviance."""

# Simulation.
type CategoricalLevels = int | Sequence[float]
"""The levels of a simulated categorical variable: either their number, each
equally likely, or each level's probability."""

type SimulatedResponses = npt.NDArray[np.float64]
"""Responses simulated from a fit model, shape (n_sim, n). Each row is one
simulated response vector, so iterating yields one replicate at a time."""

type CoefficientTable = pd.DataFrame
"""Coefficient estimates from several fit models, such as bootstrap
replicates: one row per model, and one column per coefficient, named by the
models' X_names."""
