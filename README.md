# py-glm: Generalized Linear Models in Python

`py-glm` is a library for fitting, inspecting, and simulating from generalized
linear models in Python. Models are fit with Fisher scoring, and support R-style
model formulas, offsets, frequency and precision weights, hypothesis tests on
linear constraints, ridge regularization, and bootstrap resampling.

## Installation

`py-glm` requires Python 3.13 or later, and can be installed directly from
GitHub:

```
pip install git+https://github.com/madrury/py-glm.git
```

The plotting functions need `matplotlib`, which you can install along with
`py-glm` as the `plot` extra:

```
pip install "py-glm[plot] @ git+https://github.com/madrury/py-glm.git"
```

## Tutorials

The `examples` directory has a sequence of tutorial notebooks, which cover the
library in more depth than this README:

1. [A Tour of the Exponential Families](examples/01-families.ipynb)
2. [Formulas and Categorical Data](examples/02-formulas.ipynb)
3. [Inference and Bootstraps](examples/03-inference.ipynb)
4. [Frequency and Precision Weights](examples/04-weights.ipynb)
5. [Ridge Regularization and Cross Validation](examples/05-regularization.ipynb)

## Model Specification and Fitting

### Example Data
The examples below use simulated car insurance data. Each row is a policy,
observed for some fraction of its yearly period, this is called its *exposure*.
The response variable is the policy's count of claims in that exposure time:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(154)
N = 2000
df = pd.DataFrame({
    "age": rng.uniform(18, 80, N),
    "vehicle_value": rng.gamma(shape=4.0, scale=5.0, size=N),
    "region": rng.choice(["north", "south", "west"], size=N),
    "exposure": rng.uniform(0.1, 1.0, size=N),
})
# The yearly claim rate is lowest in middle age, rises with the vehicle's
# value, and varies by region.
log_rate = (
    -2.0
    + 0.0008 * (df["age"] - 45)**2
    + 0.02 * df["vehicle_value"]
    + df["region"].map({"north": 0.0, "south": 0.3, "west": -0.2})
)
df["claims"] = rng.poisson(df["exposure"] * np.exp(log_rate))
```

For now, we manually transform our data table into a design matrix, though later
we will see this is not needed:

```python
# For fitting with arrays. The first column is an explicit intercept.
X = np.column_stack([np.ones(N), df["age"], df["vehicle_value"]])
y = df["claims"].to_numpy()
# The log exposures, an offset (see Offsets and Weights, below).
offset = np.log(df["exposure"].to_numpy())
```

### Exponential Families

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
are estimated using the fit deviance.

```python
from glm.families import QuasiPoisson, Gamma

quasi_poisson_model = GLM(family=QuasiPoisson())
gamma_model = GLM(family=Gamma())
```

### Fitting
Fitting a model given data in `numpy` arrays uses the `sklearn` style:

```python
poisson_model.fit(X, y, offset=offset)
```

**The intercept:** Instead of a `fit_intercept` argument, `py-glm` expects the
first column of the design matrix, `X[:, 0]`, to be an intercept (a column of
ones), and `fit` raises an error if it isn't. To fit a model without an intercept,
pass `validate_intercept=False`. Model formulas, below, include an intercept
automatically.

**Rank-deficient designs:** Unpenalized fits raise an error if the design
matrix's columns are linearly dependent, since in this case the coefficients
aren't uniquely identified. Common examples of this happening are when a
predictor is duplicated, or a spline basis includes the intercept. Ridge models
do not require linear independence, so rank-deficient designs do not raise for
regularized regressions.

### Model Formulas
If your data resides in a `pandas.DataFrame`, you can pass it to `fit` along
with a model formula:

```python
FORMULA = "claims ~ cs(age, df=5, constraints='center') + vehicle_value + C(region)"
formula_model = GLM(family=Poisson())
formula_model.fit(df, formula=FORMULA, offset=offset)
```

Model formulas are a small specification language for transforming dataframes
into design matrices, provided by the
[formulaic](https://matthewwardrop.github.io/formulaic/) library. Include
categorical variables with `C(group)`, cubic splines with `cs(x, df=5,
constraints='center')`, and stateful transforms such as centering with
`center(x)`. For more sophisticated use, consult the `formulaic` documentation
(and give the library a well-deserved star).

The model stores what it learned about the formula encoding from the training
data, such as each categorical variable's levels and the constants used by
`center`. It is then used to encode any future data, such as in `predict`, when
testing nested models, and in any bootstraps.

For example, a new dataframe containing only some of a variable's categories is
still encoded with every category's column (some columns will be all 0.0s in
this case). On the other hand, new data including an *unseen* category will
raise (in the case of an unpenalized model); this avoids the silent error of
assuming the unseen category is equivalent to the reference one.

### Offsets and Weights
Offsets and weights are supported when fitting.

**Offsets:** An offset is a term added to the linear predictor with a fixed
coefficient of one. In the examples above, a policy's expected count of claims
is its exposure times its yearly rate, so with a log link

```
    log E[claims | X] = log(exposure) + Xβ,
```

and `offset=np.log(exposure)` ensures that the fit coefficients `coef_` describe
the yearly claims rate.

**Weights:** There are two kinds of weights, and they mean different things.

* `frequency_weights` say how many identical observations each row stands
for, as when repeated records are compressed into one row with a count. Using
`frequency_weights` in a fit is exactly equivalent to fitting to the expanded
data. For example, the policies in each region with each count of claims,
stored once with their count, give the same fit as all the policies:

```python
rows = df[["region", "claims"]].value_counts().rename("n").reset_index()
compressed_model = GLM(family=Poisson()).fit(
    rows, formula="claims ~ C(region)", frequency_weights=rows["n"].to_numpy()
)
```

* `precision_weights` say how much more precise each row's response is than a
single observation's: its variance is divided by the weight. They're the natural
weights for a response that's an average, rate, or proportion, such as a
binomial proportion of successes weighted by its count of trials. For example,
each policy's yearly claim rate, weighted by its exposure, gives exactly the
same fit as its count of claims with the offset:

```python
rate_model = GLM(family=Poisson()).fit(
    df.assign(rate=df["claims"] / df["exposure"]),
    formula="rate ~ cs(age, df=5, constraints='center') + vehicle_value + C(region)",
    precision_weights=df["exposure"].to_numpy(),
)
```

Given the same values, both types of weights give the same coefficients, but only
frequency weights count for additional observations. For families that estimate
a dispersion, the two kinds of weights give different dispersions and standard
errors. (R's `glm` calls precision weights *prior weights*.) The two kinds can
be combined.

In particular, we never use *weights* to mean the model coeffiients/parameters,
in contrast with some machine learning literature.

**Missing values:** `fit`, `predict`, and `score` raise an error if the data
contain any missing or infinite values (`np.nan`, `np.inf`, `pd.NA`, `None`),
rather than silently dropping rows, which would misalign the data with any
offsets, weights, or predictions. Dealing with missing values is thus the
user's responsibility: drop or impute missing values before fitting models.


## Prediction
Predictions from an array are also made in `sklearn` style:

```python
# Expected claims over each policy's exposure.
poisson_model.predict(X, offset=offset)
```

If fit with a dataframe and formula, only a frame is needed at prediction time
since the model stores the transformation from dataframe to design array:

```python
# Expected claims per year, as the offset defaults to zero.
formula_model.predict(df)
```

**Conditional Expectations:**
There is one major place where we deviate from the `sklearn` convention on
predictions. The `predict` method on a `GLM` object **always** returns an
estimate of the conditional expectation `E[y | X]`. This is in contrast to
`sklearn` behavior for classification models, where it returns a class
assignment. We make this choice so that the `py-glm` library is internally
consistent with its use of `predict` across all exponential families.

If the user would like to infer class assignments from a classification model,
they will need to threshold the probability returned by `predict` manually. The
author considers this best practice anyway.

## Parametric Inference
We provide inference for fit coefficients, and for null hypotheses specified as
linear constraints.

### Coefficient Inference
Once the model is fit, parameter estimates, parameter covariance estimates, and
Wald test p-values are available:

```python
formula_model.coef_
formula_model.coef_covariance_matrix_
formula_model.coef_standard_error_
formula_model.p_values_
```

`summary_frame` collects them into a data frame, one row per coefficient, with
the estimate, its standard error, the test statistic for the null hypothesis
that the coefficient is zero, its p-value, and a confidence interval:

```python
formula_model.summary_frame(level=0.95)
```

The test statistic is compared with:

  - Student's t distribution, with `n - p` degrees of freedom for families that
  estimate a dispersion parameter (it's labelled `t`)
  - The standard normal distribution for families that don't estimate a
  dispersion (labelled `z`)

To print a quick summary, use the `summary` method:

```python
formula_model.summary()
```

```
Poisson GLM Model Summary.
                                       estimate std error       z p-value lower 95% upper 95%
Intercept                                -1.653     0.150 -11.058  <0.001    -1.946    -1.360
cs(age, df=5, constraints='center')[1]    0.097     0.200   0.483   0.629    -0.295     0.489
cs(age, df=5, constraints='center')[2]   -0.158     0.154  -1.022   0.307    -0.461     0.145
cs(age, df=5, constraints='center')[3]   -0.182     0.178  -1.026   0.305    -0.530     0.166
cs(age, df=5, constraints='center')[4]   -0.169     0.175  -0.967   0.334    -0.513     0.174
cs(age, df=5, constraints='center')[5]    0.665     0.213   3.116   0.002     0.247     1.083
vehicle_value                             0.015     0.005   2.860   0.004     0.005     0.026
C(region)[T.south]                        0.279     0.133   2.092   0.036     0.018     0.540
C(region)[T.west]                        -0.203     0.153  -1.326   0.185    -0.502     0.097
```

### Joint Tests, Constraints, and Nested Models

A null hypothesis that is a linear equation in the coefficients, such as several
coefficients being zero at once, has a Wald test, specified in one of three ways.

**Constraint equations:** For example, that the region has no effect, i.e. that
both of its coefficients are zero:

```python
formula_model.tests.wald("C(region)[T.south] = 0, C(region)[T.west] = 0").summary()
```

```
Wald Test
  Null Hypothesis: C(region)[T.south] = 0, C(region)[T.west] = 0
  Constraints:     q = 2
  Statistic:       W = 11.5
  Reference:       χ²(2)
  p-value:         0.00322
```

**Nested models:** Another way to consider the same null hypothesis is that the
model excluding the region variable is correct. A *null* model nested within the
fit model, the *full* model, can be given as a formula:

```python
formula_model.tests.wald(
    null="claims ~ cs(age, df=5, constraints='center') + vehicle_value", X=df
)
```

This gives the same test as above.

Notice that here the training data is needed again, it is necessary to work out
the constraints that the nesting implies. We check that exactly the same data is
supplied, using a stored hash.

Nested models can express hypotheses that have no simple statement as constraint
equations. For example, that age has a linear effect, since a straight line is a
special case of a cubic spline:

```python
formula_model.tests.wald(
    null="claims ~ age + vehicle_value + C(region)", X=df
).summary()
```

```
Wald Test
  Null Hypothesis: claims ~ age + vehicle_value + C(region)
  Constraints:     q = 4
  Statistic:       W = 13.7
  Reference:       χ²(4)
  p-value:         0.00820
```

The test rejects, as it should: the simulated claim rate is quadratic in age.

**Constraint arrays:** For anything else, give the equation `Cβ = c` as arrays,
with `constraint=ConstraintEquation.from_arrays(C, c)`, from `glm.hypothesis`.

With a single constraint, such as `"vehicle_value = 0"`, the test is the same as
the coefficient's test in `summary_frame`. For families that estimate a
dispersion, the statistic is referred to an F distribution, rather than χ².

## Simulation, Non-Parametric Inference

The `simulation` module samples responses from a fit model, and supports the
parametric and non-parametric bootstraps:

```python
from glm.simulation import Simulation, coefficient_table

sim = Simulation(formula_model)
# Shape (100, N): one response vector per row.
simulated_responses = sim.sample(df, n_sim=100, offset=offset)

parametric_boots = sim.parametric_bootstrap(df, n_sim=500, offset=offset)
non_parametric_boots = sim.non_parametric_bootstrap(df, n_sim=500, offset=offset)
```

Each bootstrap returns a list of fit models (a `list[GLM]`). Pass the same
offset and weights the model was fit with.

For a model fit without a formula, pass the response to the non-parametric
bootstrap as well:

```python
Simulation(poisson_model).non_parametric_bootstrap(X, y, n_sim=500, offset=offset)
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
standard errors, which remain valid even if the model's variance function is
wrong.

**Rare categories:** A non-parametric bootstrap resample may contain no rows
of a rare category, so that its coefficient cannot be estimated, in which case
the bootstrap raises an error. Ridge models, below, are unaffected, as they do
not require linear independence of predictors.

## Regularization

### Ridge Regressions
Ridge regression is supported for all models, enabled by passing a
regularization strength when constructing the model. (The parameter is called
`alpha` instead of the literature convention `lambda`, as `lambda` is a reserved
word in Python.)

```python
ridge_model = GLM(family=Poisson(), alpha=1.0)
ridge_model.fit(df, formula=FORMULA, offset=offset)
```

The ridge penalty is `alpha` times the sum of the (standardized, see below)
squared coefficients, excluding any intercept. Standard errors, covariances and
p-values are not available for regularized models, and `summary_frame` gives
only the estimates.

**Standardization:** A ridge penalty depends on the units of the predictors:
measuring a predictor in kilometres rather than metres makes its coefficient
1000 times larger, and it is penalized far more. So by default, penalized models
standardize the design matrix's continuous columns to zero mean and unit
standard deviation before fitting, and report coefficients on the original
scale. The `standardize` argument controls this behaviour:

* `"continuous"` (the default) leaves indicator columns, those of zeros and
  ones, unscaled: they are already on a natural scale, and scaling them would
  shrink the coefficients of rare categories *less* than common ones.
* `"all"` scales indicator columns too, as `glmnet` does.
* `"none"` fits the columns as given, no scaling.

```python
glmnet_style_model = GLM(family=Poisson(), alpha=1.0, standardize="all")
```

**Categorical variables:** Penalized formula models encode categorical
variables with a column for *every* level, e.g. `C(g)[a]`, `C(g)[b]`, and
`C(g)[c]`, rather than dropping a reference level. This ensures the penalty
treats every level alike, and the fit doesn't depend on which level would have
been the reference.


### Ridge Paths
A `RidgePath` fits a model at each of a grid of penalty strengths, from the
largest to the smallest, starting each fit from the previous one's coefficients:

```python
from glm.ridge import RidgePath

path = RidgePath(family=Poisson())
path.fit(df, formula=FORMULA, offset=offset)

path.alphas_  # 100 penalty strengths, in decreasing order.
path.models_  # A fit GLM for each.
path.coefficient_table()  # One row per penalty strength, indexed by alpha.
path.df_  # The effective degrees of freedom of each model.
```

Pass `alphas` to `fit` to choose the penalty strengths yourself. By default, the
grid spans the eigenvalues of the model's (standardized) information matrix:
ridge shrinks a fit along each eigen-direction by the factor `d / (d + alpha)`,
for that direction's eigenvalue `d`, so the interesting part of the path happens
between the minimum and maximum eigenvalues (see Hastie, Tibshirani, and
Friedman, *The Elements of Statistical Learning*, section 3.4.1).

`plot_coefficient_paths` plots the path of every coefficient, grouped by model
term, so that every level of a categorical variable shares a color. Select
terms with a formula, plot against the effective degrees of freedom rather
than `alpha` with `x="df"`, and show coefficients on their original scale,
rather than standardized, with `scale="original"`:

```python
import matplotlib.pyplot as plt
from glm.plotting import plot_coefficient_paths

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_coefficient_paths(axes[0], path)
plot_coefficient_paths(axes[1], path, "vehicle_value + C(region)", x="df")
```

Like all of `py-glm`'s plotting functions, `plot_coefficient_paths` draws on
the axes you pass it and returns them, so you stay in control of the figure's
layout.

### Choosing the Penalty by Cross-Validation

`RidgePathCV` chooses the penalty strength by cross-validation. It splits the
data into folds, fits a path to all but each fold in turn over a grid of penalty
strengths, then scores each model using its deviance on the held-out fold:

```python
from glm.ridge import RidgePathCV
from glm.plotting import plot_cv_deviance

cv = RidgePathCV(family=Poisson(), n_folds=10)
cv.fit(df, formula=FORMULA, offset=offset, rng=np.random.default_rng(1))

cv.alpha_min_  # The penalty strength with the smallest cross-validated deviance.
cv.model_min_  # The model fit to all the data at that strength.

fig, ax = plt.subplots(figsize=(12, 4))
plot_cv_deviance(ax, cv, x="df")
```

With weights, the folds respect what each kind of weight means:
  - a frequency-weighted row's implied observations are dealt to the folds
  separately
  - a precision-weighted row is held out whole to a single fold.

## Type Annotations

`py-glm` is fully type annotated, checked with `mypy --strict`, and ships a
`py.typed` marker, so type checkers will check your code's use of the library.
The annotations extensively use domain specific type aliases, such as
`DesignMatrix`, `ResponseVector`, `CoefficientVector`, etc. These are defined
and documented in `glm.types`.

## Development

`py-glm` uses [uv](https://docs.astral.sh/uv/) to manage its environment:

```
uv sync                      # Install the package and development dependencies.
uv run pytest                # Run the tests.
uv run mypy                  # Type check.
uv run black glm tests doc   # Format.
```

## References and Further Reading

* Marlene Müller (2004). [Generalized Linear Models](http://www.marlenemueller.de/publications/HandbookCS.pdf). A friendly overview of GLM theory.
* Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009). [*The Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/), 2nd ed. Springer.
* Peter McCullagh and John Nelder (1989). *Generalized Linear Models*, 2nd ed. Chapman & Hall. The standard reference for the theory. The first Statistics textbook I ever attempted to read (unwise).
* Peter Dunn and Gordon Smyth (2018). *Generalized Linear Models With Examples in R*. Springer. A readable treatment, includes weights and the Wald, score, and likelihood ratio tests.
* Bradley Efron and Robert Tibshirani (1993). *An Introduction to the Bootstrap*. Chapman & Hall. The standard introduction to the bootstrap. By two of the great writers in statistics.
