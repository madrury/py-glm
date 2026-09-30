import numpy as np
import pytest
import scipy.stats as sts
import statsmodels.api as sm
import statsmodels
from glm.glm import GLM
from glm.families import Gaussian, Bernoulli, Poisson, Gamma
from generate_data import (
    make_linear_regression,
    make_logistic_regression,
    make_poisson_regression,
    make_gamma_regression,
)

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
    np.random.seed(seed)
    n_uncorr_features, n_corr_features, n_drop_features = (
        generate_regression_hyperparamters()
    )
    X, y, parameters = make_linear_regression(
        n_samples=N_SAMPLES,
        n_uncorr_features=n_uncorr_features,
        n_corr_features=n_corr_features,
        n_drop_features=n_drop_features,
        resid_sd=0.01,
    )
    lr = GLM(family=Gaussian())
    lr.fit(X, y, tol=FIT_TOL)
    assert_near_truth(lr, parameters)
    mod = sm.OLS(y, X)
    res = mod.fit()
    assert_matches_statsmodels(lr, res)


@pytest.mark.parametrize("seed", SEEDS)
def test_logistic_regression(seed):
    np.random.seed(seed)
    n_uncorr_features, n_corr_features, n_drop_features = (
        generate_regression_hyperparamters()
    )
    X, y, parameters = make_logistic_regression(
        n_samples=N_SAMPLES,
        n_uncorr_features=n_uncorr_features,
        n_corr_features=n_corr_features,
        n_drop_features=n_drop_features,
    )
    lr = GLM(family=Bernoulli())
    lr.fit(X, y, tol=FIT_TOL)
    assert_near_truth(lr, parameters)
    mod = sm.Logit(y, X)
    res = mod.fit(disp=0)
    assert_matches_statsmodels(lr, res)


@pytest.mark.parametrize("seed", SEEDS)
def test_poisson_regression(seed):
    np.random.seed(seed)
    n_uncorr_features, n_corr_features, n_drop_features = (
        generate_regression_hyperparamters()
    )
    X, y, parameters = make_poisson_regression(
        n_samples=N_SAMPLES,
        n_uncorr_features=n_uncorr_features,
        n_corr_features=n_corr_features,
        n_drop_features=n_drop_features,
        coef_range=(-0.1, 0.1),
    )
    pr = GLM(family=Poisson())
    pr.fit(X, y, tol=FIT_TOL)
    assert_near_truth(pr, parameters)
    mod = statsmodels.discrete.discrete_model.Poisson(y, X)
    res = mod.fit(disp=0)
    assert_matches_statsmodels(pr, res)


@pytest.mark.parametrize("seed", SEEDS)
def test_gamma_regression(seed):
    np.random.seed(seed)
    n_uncorr_features, n_corr_features, n_drop_features = (
        generate_regression_hyperparamters()
    )
    X, y, parameters = make_gamma_regression(
        n_samples=N_SAMPLES,
        n_uncorr_features=n_uncorr_features,
        n_corr_features=n_corr_features,
        n_drop_features=n_drop_features,
        coef_range=(-0.1, 0.1),
    )
    gr = GLM(family=Gamma())
    gr.fit(X, y, tol=FIT_TOL)
    assert_near_truth(gr, parameters)
    gamma_model = sm.GLM(
        y, X, family=sm.families.Gamma(link=statsmodels.genmod.families.links.Log())
    )
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


def generate_regression_hyperparamters():
    n_uncorr_features = np.random.choice(list(range(1, MAX_UNCORR_FEATURES + 1)))
    n_corr_features = np.random.choice(list(range(1, MAX_CORR_FEATURES + 1)))
    n_drop_features = np.random.choice(n_uncorr_features + n_corr_features)
    return n_uncorr_features, n_corr_features, n_drop_features
