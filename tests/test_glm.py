import numpy as np
from glm.glm import GLM
from glm.families import Gaussian, Bernoulli
from generate_data import make_linear_regression

N_SAMPLES = 100000
TOL = 10**(-1)
N_REGRESSION_TESTS=100


def test_linear_regressions():

    def _test_random_linear_regression():
        n_uncorr_features, n_corr_features, n_drop_features = (
            generate_regression_hyperparamters())
        X, y, parameters = make_linear_regression(
            n_samples=N_SAMPLES,
            n_uncorr_features=n_uncorr_features,
            n_corr_features=n_corr_features,
            n_drop_features=n_drop_features,
            resid_sd=0.05)
        lr = GLM(family=Gaussian())
        lr.fit(X, y, tol=10**(-8))
        assert approx_equal(lr.coef_, parameters)

    for _ in range(N_REGRESSION_TESTS):
        _test_random_linear_regression()


def approx_equal(x0, x1, tol=TOL):
    all_within_tol = np.abs(x0 - x1) < tol
    return np.all(all_within_tol)

def generate_regression_hyperparamters():
    n_uncorr_features = np.random.choice(list(range(1, 10)))
    n_corr_features = np.random.choice(list(range(1, 10)))
    n_drop_features = np.random.choice(n_uncorr_features + n_corr_features)
    return n_uncorr_features, n_corr_features, n_drop_features

