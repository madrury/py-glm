from typing import Any

import numpy as np
import pytest
import scipy.stats as sts
import statsmodels.api as sm
from glm.datasets import SimulatedRegression, make_regression
from glm.families import Bernoulli, ExponentialFamily, Gamma, Gaussian, Poisson
from glm.glm import GLM

N_SAMPLES = 100000
# Each family is tested on this many random regressions, one per seed.
N_REGRESSION_TESTS = 5
SEEDS = range(N_REGRESSION_TESTS)
# Each random regression draws between 1 and this many features of each kind.
MAX_UNCORR_FEATURES = 9
MAX_CORR_FEATURES = 9
# Tolerances for comparing against statsmodels. Both libraries fit the same
# model to the same data, so any difference is numerical (from convergence
# criteria), not statistical. Coefficients also need an absolute tolerance, as
# some true coefficients are zero; standard errors are strictly positive, so a
# relative tolerance suffices.
RTOL = 1e-3
COEF_ATOL = 1e-5
# Fit to a tight convergence tolerance, so that differences from statsmodels
# reflect errors rather than where each library's iterations happened to stop.
FIT_TOL = 1e-10
# Maximum allowed distance, in standard errors, between an estimate and the
# true parameter that generated the data. Each family's tests compare every
# coefficient of every regression to its true value, so we set the threshold
# with a Bonferroni bound, allowing a FALSE_FAILURE_RATE chance of a spurious
# failure per family. The bound is conservative, as the z-scores of collinear
# coefficients are correlated.
FALSE_FAILURE_RATE = 0.001
MAX_COEFFICIENTS = 1 + MAX_UNCORR_FEATURES + MAX_CORR_FEATURES  # Intercept included.
MAX_COMPARISONS = N_REGRESSION_TESTS * MAX_COEFFICIENTS
MAX_Z = sts.norm.isf(FALSE_FAILURE_RATE / (2 * MAX_COMPARISONS))


@pytest.mark.parametrize("seed", SEEDS)
def test_linear_regression(seed):
    data = simulate(Gaussian(), seed, dispersion=0.01**2)
    X, y, parameters = data.X, data.y, data.coef
    lr = GLM(family=Gaussian())
    lr.fit(X, y, tol=FIT_TOL)
    assert_near_truth(lr, parameters)
    mod = sm.OLS(y, X)
    res = mod.fit()
    assert_matches_statsmodels(lr, res)


@pytest.mark.parametrize("seed", SEEDS)
def test_logistic_regression(seed):
    data = simulate(Bernoulli(), seed)
    X, y, parameters = data.X, data.y, data.coef
    lr = GLM(family=Bernoulli())
    lr.fit(X, y, tol=FIT_TOL)
    assert_near_truth(lr, parameters)
    mod = sm.Logit(y, X)
    res = mod.fit(disp=0)
    assert_matches_statsmodels(lr, res)


@pytest.mark.parametrize("seed", SEEDS)
def test_poisson_regression(seed):
    data = simulate(Poisson(), seed, coef_range=(-0.1, 0.1))
    X, y, parameters = data.X, data.y, data.coef
    pr = GLM(family=Poisson())
    pr.fit(X, y, tol=FIT_TOL)
    assert_near_truth(pr, parameters)
    mod = sm.Poisson(y, X)
    res = mod.fit(disp=0)
    assert_matches_statsmodels(pr, res)


@pytest.mark.parametrize("seed", SEEDS)
def test_gamma_regression(seed):
    # Exponentially distributed responses: gamma with a dispersion of one.
    data = simulate(Gamma(), seed, coef_range=(-0.1, 0.1))
    X, y, parameters = data.X, data.y, data.coef
    gr = GLM(family=Gamma())
    gr.fit(X, y, tol=FIT_TOL)
    assert_near_truth(gr, parameters)
    gamma_model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.Log()))
    # We estimate the dispersion from the deviance; statsmodels defaults to
    # Pearson's chi-squared, so ask it for the deviance-based estimate.
    res = gamma_model.fit(scale="dev")
    assert_matches_statsmodels(gr, res)


def assert_near_truth(model, parameters):
    z = (model.coef_ - parameters) / model.coef_standard_error_
    assert np.all(np.abs(z) < MAX_Z)


def assert_matches_statsmodels(model, res):
    np.testing.assert_allclose(model.coef_, res.params, rtol=RTOL, atol=COEF_ATOL)
    np.testing.assert_allclose(model.coef_standard_error_, res.bse, rtol=RTOL)


def simulate(
    family: ExponentialFamily, seed: int, **kwargs: Any
) -> SimulatedRegression:
    """A random regression for the family: random numbers of correlated and
    uncorrelated features, some with zero coefficients."""
    rng = np.random.default_rng(seed)
    n_uncorrelated = int(rng.integers(1, MAX_UNCORR_FEATURES + 1))
    n_correlated = int(rng.integers(1, MAX_CORR_FEATURES + 1))
    n_zero_coefficients = int(rng.integers(n_uncorrelated + n_correlated))
    return make_regression(
        family,
        n_samples=N_SAMPLES,
        n_uncorrelated=n_uncorrelated,
        n_correlated=n_correlated,
        n_zero_coefficients=n_zero_coefficients,
        rng=rng,
        **kwargs,
    )
