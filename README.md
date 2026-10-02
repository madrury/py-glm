# py-glm: Generalized Linear Models in Python

`py-glm` is a library for fitting, inspecting, and simulating from generalized
linear models in Python. Models are fit with Fisher scoring, and support model
formulas, offsets, sample weights, ridge regularization, and bootstrap
resampling.

## Installation

`py-glm` requires Python 3.13 or later, and can be installed directly from
GitHub:

```
pip install git+https://github.com/madrury/py-glm.git
```

## Fitting Models

The examples below use this simulated data:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(154)
N = 1000
df = pd.DataFrame({"Moshi": rng.uniform(-1, 1, N), "SwimSwim": rng.uniform(-1, 1, N)})
# Logistic style data.
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-(1 - 2 * df["Moshi"] + df["SwimSwim"]))))

X = np.column_stack([np.ones(N), df["Moshi"], df["SwimSwim"]])
y = df["y"].to_numpy()
```

`py-glm` supports models from various exponential families:

```python
from glm.glm import GLM
from glm.families import Gaussian, Bernoulli, Poisson, Exponential

linear_model = GLM(family=Gaussian())
logistic_model = GLM(family=Bernoulli())
poisson_model = GLM(family=Poisson())
exponential_model = GLM(family=Exponential())
```

Models with dispersion parameters are also supported. The dispersion parameters
in these models are estimated using the deviance.

```python
from glm.families import QuasiPoisson, Gamma

quasi_poisson_model = GLM(family=QuasiPoisson())
gamma_model = GLM(family=Gamma())
```

Fitting a model given data in `numpy` arrays proceeds in `sklearn` style, and
uses the Fisher scoring algorithm:

```python
logistic_model.fit(X, y)
```

**The intercept:** Instead of a `fit_intercept` argument, `py-glm` expects the
first column of the design matrix, `X[:, 0]`, to be an intercept (a column of
ones), and `fit` raises an error if it isn't. To fit a model without an intercept,
pass `validate_intercept=False`. Model formulas, below, include an intercept
automatically.

Offsets and sample weights are supported when fitting:

```python
exposure = rng.uniform(1, 10, N)
# Poisson style data.
y_count = rng.poisson(exposure * np.exp(0.5 + 0.3 * df["Moshi"]))
poisson_model.fit(X, y_count, offset=np.log(exposure))

sample_weights = rng.uniform(0, 2, N)
weighted_model = GLM(family=Bernoulli()).fit(X, y, sample_weights=sample_weights)
```

**Missing values:** `fit` and `predict` raise an error if the data contain
missing values, rather than silently dropping rows, which would misalign the
data with any offsets, sample weights, or predictions. Dealing with missing
values is thus the user's responsibility: drop or impute missing values before
fitting models.

## Model Formulas

If your data resides in a `pandas.DataFrame`, you can pass it to `fit` along
with a model formula:

```python
formula_model = GLM(family=Bernoulli())
formula_model.fit(df, formula="y ~ Moshi + SwimSwim")
```

Formulas use the [formulaic](https://matthewwardrop.github.io/formulaic/)
syntax, including categorical variables, e.g. `C(group)`, and transforms, e.g.
`center(x)`. The model stores what it learned about the formula from the
training data, such as each categorical variable's levels and the constants
used by `center`, and uses it to encode any new data, in `predict` and in any
bootstraps. So, for example, new data containing only some of a variable's
categories is still encoded with every category's column.

## Prediction

Predictions are also made in `sklearn` style:

```python
logistic_model.predict(X)
formula_model.predict(df)
```

**Note:** There is one major place we deviate from the `sklearn` interface. The
`predict` method on a `GLM` object **always** returns an estimate of the
conditional expectation `E[y | X]`. This is in contrast to `sklearn` behavior
for classification models, where it returns a class assignment. We make this
choice so that the `py-glm` library is internally consistent with its use of
`predict`. If the user would like to infer class assignments from a model, they
will need to threshold the probability returned by `predict` manually.

## Inference

Once the model is fit, parameter estimates, parameter covariance estimates, and
p-values from a standard z-test are available:

```python
logistic_model.coef_
logistic_model.coef_covariance_matrix_
logistic_model.coef_standard_error_
logistic_model.p_values_
```

To get a quick summary, use the `summary` method:

```python
formula_model.summary()
```

```
Bernoulli GLM Model Summary.
================================================
Name          Parameter Estimate  Standard Error
------------------------------------------------
Intercept                   0.98            0.09
Moshi                      -2.06            0.16
SwimSwim                    0.99            0.14
```

## Simulation

The `simulation` module samples responses from a fit model, and supports the
parametric and non-parametric bootstraps:

```python
from glm.simulation import Simulation, coefficient_table

sim = Simulation(formula_model)
simulated_responses = sim.sample(df, n_sim=100)  # Shape (100, N): one response vector per row.

parametric_boots = sim.parametric_bootstrap(df, n_sim=500)
non_parametric_boots = sim.non_parametric_bootstrap(df, n_sim=500)
```

Each bootstrap returns a list of fit models. For a model fit without a formula,
pass the response to the non-parametric bootstrap too:

```python
Simulation(logistic_model).non_parametric_bootstrap(X, y, n_sim=500)
```

`coefficient_table` collects the bootstrap estimates into a data frame, with one
row per bootstrap replicate and one column per coefficient:

```python
table = coefficient_table(parametric_boots)
table.std()  # Bootstrap standard errors.
table.quantile([0.025, 0.975])  # 95% percentile intervals.
```

The parametric bootstrap simulates from the fit model, so its standard errors
estimate the model-based standard errors. The non-parametric bootstrap
resamples the data, so its standard errors estimate the robust (sandwich)
standard errors, which remain valid if the model's variance function is wrong.

**Rare categories:** A non-parametric bootstrap resample may contain no rows
of a rare category, so that its coefficient cannot be estimated, in which case
the bootstrap raises an error. Consider merging rare categories. Ridge models,
below, are unaffected.

## Regularization

Ridge regression is supported for each model, by passing a regularization
strength when constructing the model. (The parameter is called `alpha` instead
of `lambda`, as `lambda` is a reserved word in Python.)

```python
ridge_model = GLM(family=Bernoulli(), alpha=1.0)
ridge_model.fit(X, y)
```

The penalty is `alpha` times the sum of the squared coefficients, excluding the
intercept. P-values are not available for regularized models.

## Type Annotations

`py-glm` is fully type annotated, checked with `mypy --strict`, and ships a
`py.typed` marker, so type checkers will check your code's use of the library.
The annotations use domain specific type aliases, such as `DesignMatrix`,
`ResponseVector`, and `CoefficientVector`, defined in `glm.types`.

## Development

`py-glm` uses [uv](https://docs.astral.sh/uv/) to manage its environment:

```
uv sync                  # Install the package and development dependencies.
uv run pytest            # Run the tests.
uv run mypy              # Type check.
uv run black glm tests   # Format.
```

## References

* Marlene Müller (2004). [Generalized Linear Models](http://www.marlenemueller.de/publications/HandbookCS.pdf).

## Warning

The `glmnet` code included in `glm.glmnet` is **experimental**, and is not yet
type annotated. Please use at your own risk.
