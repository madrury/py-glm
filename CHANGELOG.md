# Changelog

Notable changes to `py-glm`, newest first. Changes that can alter the results
of an existing analysis are marked **(changes results)**.

## 0.2.0

### Fixed

- **(changes results)** Ridge fits ignored `alpha`: every penalized model was
  fit as though `alpha` were 1. Results from penalized models fit with earlier
  versions should be recomputed.
- **(changes results)** In a ridge model without an intercept, the first
  predictor's coefficient escaped the penalty. Now every coefficient but an
  actual intercept is penalized.
- **(changes results)** Weighted fits estimated the dispersion from an
  unweighted deviance, so families that estimate a dispersion gave wrong
  standard errors with weights. See the new weights, under Changed.
- `deviance_` and `information_matrix_` described the fit one iteration before
  `coef_`. They now describe `coef_`.
- Fits without an intercept could fail with non-finite estimates when a
  predictor had large values, because the starting point put an intercept
  estimate in column 0. They now start from zero coefficients.
- Infinite values in `X`, `y`, offsets or weights gave misleading errors when
  fitting, and silently wrong predictions. They now raise, as missing values
  do.

### Changed

- **(changes results)** `sample_weights` is replaced by two kinds of weights,
  `frequency_weights` and `precision_weights`, in `fit`, `score` and the
  bootstraps. They give the same coefficients but different dispersions and
  standard errors; see the README and the weights tutorial for which to use.
- **(changes results)** Penalized fits standardize the design matrix's
  continuous columns before fitting, so the penalty doesn't depend on the
  predictors' units. The `standardize` argument controls this. Coefficients
  are still reported on the original scale.
- **(changes results)** Penalized formula models encode every level of a
  categorical variable, rather than dropping a reference level, so the penalty
  treats every level alike.
- **(changes results)** For families that estimate a dispersion, p-values
  compare the test statistic with Student's t on n - p degrees of freedom, as
  R's `summary.glm` does, rather than the standard normal.
- **(changes results)** The non-parametric bootstrap resamples observations,
  not rows: a row with frequency weight f is f observations. Without weights,
  it's the usual bootstrap of rows, but it draws from the random state
  differently, so seeded results change.
- Standard errors, covariances and p-values raise for regularized models,
  which previously reported values from the unpenalized formula that weren't
  meaningful for a ridge fit. `summary_frame` gives only the estimates.
- Unpenalized fits raise if the design matrix is rank deficient, instead of
  fitting with arbitrary coefficients and NaN standard errors.
- Unpenalized formula models raise when predicting for a categorical level
  they weren't fit with, instead of silently predicting it as the reference
  level.
- Fisher scoring stops on the Newton decrement, the decrease in the deviance
  that the next step predicts, rather than on the change in deviance between
  iterations. `tol` keeps its meaning as a relative tolerance on the deviance,
  but fits can be driven much tighter.
- The bootstraps check that they're given the same offset and weights the
  model was fit with.

### Removed

- `glm.glmnet`, the experimental elastic net module. It may return, once it's
  brought up to the rest of the library's standard.

### Added

- `RidgePath`, which fits a ridge model at each of a grid of penalty
  strengths, with effective degrees of freedom, and `RidgePathCV`, which
  chooses the penalty by cross-validation, splitting weighted data with care
  for what each kind of weight means.
- `plot_coefficient_paths` and `plot_cv_deviance`, with matplotlib as an
  optional `plot` extra.
- `GLM.summary_frame`, a data frame of each coefficient's estimate, standard
  error, test statistic, p-value and confidence interval.
- `GLM.design_matrix` and `GLM.response_vector`, which encode data as a fit
  model does.
- `glm.datasets`, with `make_regression` and `make_frame` for simulating
  datasets from a GLM.
- An `rng` argument to `ExponentialFamily.sample`.
- `Bernoulli` models of proportions of successes, weighted by their counts of
  trials as precision weights.
- Tutorial notebooks in `examples`, replacing the old example notebooks.

## 0.1.0

The first versioned release.
