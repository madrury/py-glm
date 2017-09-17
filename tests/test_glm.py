import numpy as np
import statsmodels.api as sm
import statsmodels
from glm.glm import GLM
from glm.families import Gaussian, Bernoulli, Poisson
from generate_data import (make_linear_regression, make_logistic_regression,
                           make_poisson_regression)

N_SAMPLES = 10000
TOL = 10**(-2)
N_REGRESSION_TESTS=5


def test_linear_regressions():

    def _test_random_linear_regression():
        n_uncorr_features, n_corr_features, n_drop_features = (
            generate_regression_hyperparamters())
        X, y, parameters = make_linear_regression(
            n_samples=N_SAMPLES,
            n_uncorr_features=n_uncorr_features,
            n_corr_features=n_corr_features,
            n_drop_features=n_drop_features,
            resid_sd=0.01)
        lr = GLM(family=Gaussian())
        lr.fit(X, y, tol=10**(-8))
        assert approx_equal(lr.coef_, parameters)
        mod = sm.OLS(y, X)
        res = mod.fit()
        assert approx_equal(lr.coef_, res.params)
        assert approx_equal(lr.coef_standard_error_, res.bse)

    for _ in range(N_REGRESSION_TESTS):
        _test_random_linear_regression()

def test_logistic_regressions():

    def _test_random_logistic_regression():
        n_uncorr_features, n_corr_features, n_drop_features = (
            generate_regression_hyperparamters())
        X, y, parameters = make_logistic_regression(
            n_samples=N_SAMPLES,
            n_uncorr_features=n_uncorr_features,
            n_corr_features=n_corr_features,
            n_drop_features=n_drop_features)
        lr = GLM(family=Bernoulli())
        lr.fit(X, y)
        #assert approx_equal(lr.coef_, parameters)
        mod = sm.Logit(y, X)
        res = mod.fit()
        assert approx_equal(lr.coef_, res.params)
        assert approx_equal(lr.coef_standard_error_, res.bse)

    for _ in range(N_REGRESSION_TESTS):
        _test_random_logistic_regression()

def test_poisson_regressions():

    def _test_random_poisson_regression():
        n_uncorr_features, n_corr_features, n_drop_features = (
            generate_regression_hyperparamters())
        X, y, parameters = make_poisson_regression(
            n_samples=N_SAMPLES,
            n_uncorr_features=n_uncorr_features,
            n_corr_features=n_corr_features,
            n_drop_features=n_drop_features,
            coef_range=(-0.1, 0.1))
        pr = GLM(family=Poisson())
        pr.fit(X, y)
        #assert approx_equal(pr.coef_, parameters)
        mod = statsmodels.discrete.discrete_model.Poisson(y, X)
        res = mod.fit()
        assert approx_equal(pr.coef_, res.params)
        assert approx_equal(pr.coef_standard_error_, res.bse)

    for _ in range(N_REGRESSION_TESTS):
        _test_random_poisson_regression()


def approx_equal(x0, x1, tol=TOL):
    all_within_tol = np.abs(x0 - x1) < tol
    return np.all(all_within_tol)

def generate_regression_hyperparamters():
    n_uncorr_features = np.random.choice(list(range(1, 10)))
    n_corr_features = np.random.choice(list(range(1, 10)))
    n_drop_features = np.random.choice(n_uncorr_features + n_corr_features)
    return n_uncorr_features, n_corr_features, n_drop_features
