import numpy as np
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler

def make_linear_regression(n_samples=10000,
                           n_uncorr_features=10, n_corr_features=10,
                           n_drop_features=4,
                           include_intercept=True,
                           coef_range=(-1, 1),
                           resid_sd=0.25):
    X = make_correlated_data(
        n_samples, n_uncorr_features, n_corr_features, include_intercept)
    parameters = make_regression_coeffs(
        X, n_drop_features=n_drop_features, coef_range=coef_range)
    y = make_linear_regression_y(X, parameters, resid_sd)
    return (X, y, parameters)

def make_logistic_regression(n_samples=10000,
                             n_uncorr_features=10, n_corr_features=10,
                             n_drop_features=4,
                             include_intercept=True,
                             coef_range=(-1, 1)):
    X = make_correlated_data(
        n_samples, n_uncorr_features, n_corr_features, include_intercept)
    parameters = make_regression_coeffs(
        X, n_drop_features=n_drop_features, coef_range=coef_range)
    y = make_logistic_regression_y(X, parameters)
    return (X, y, parameters)

def make_poisson_regression(n_samples=10000,
                            n_uncorr_features=10, n_corr_features=10,
                            n_drop_features=4,
                            include_intercept=True,
                            coef_range=(-1, 1)):
    X = make_correlated_data(
        n_samples, n_uncorr_features, n_corr_features, include_intercept)
    parameters = make_regression_coeffs(
        X, n_drop_features=n_drop_features, coef_range=coef_range)
    y = make_poisson_regression_y(X, parameters)
    return (X, y, parameters)

def make_gamma_regression(n_samples=10000,
                            n_uncorr_features=10, n_corr_features=10,
                            n_drop_features=4,
                            include_intercept=True,
                            coef_range=(-1, 1)):
    X = make_correlated_data(
        n_samples, n_uncorr_features, n_corr_features, include_intercept)
    parameters = make_regression_coeffs(
        X, n_drop_features=n_drop_features, coef_range=coef_range)
    y = make_gamma_regression_y(X, parameters)
    return (X, y, parameters)


def make_uncorrelated_data(n_samples=10000, n_features=25):
    X = np.random.normal(size=(n_samples, n_features))
    return X

def make_correlated_data(n_samples=10000,
                         n_uncorr_features=10, n_corr_features=15,
                         include_intercept=True):
    X_uncorr = make_uncorrelated_data(n_samples, n_uncorr_features)
    X_corr_base = make_uncorrelated_data(n_samples, n_corr_features)
    cov_matrix = make_covariance_matrix(n_corr_features)
    X_corr = StandardScaler().fit_transform(np.dot(X_corr_base, cov_matrix))
    X = np.column_stack((X_uncorr, X_corr))
    if include_intercept:
        intercept = np.ones(n_samples).reshape(-1, 1)
        return np.column_stack((intercept, X))
    return X

def make_covariance_matrix(n_features=15):
    A = np.random.normal(size=(n_features, n_features))
    A_sq = np.dot(A.T, A)
    return sqrtm(A_sq)

def make_regression_coeffs(X, n_drop_features=None, coef_range=(-1, 1)):
    n_features = X.shape[1]
    parameters = np.random.uniform(coef_range[0], coef_range[1], size=n_features)
    if n_drop_features is not None:
        drop_idxs = np.random.choice(
            list(range(len(parameters))), size=n_drop_features, replace=False)
        parameters[drop_idxs] = 0.0
    return parameters

def make_linear_regression_y(X, parameters, resid_sd=0.25):
    y_systematic = np.dot(X, parameters)
    y = y_systematic + np.random.normal(scale=resid_sd, size=X.shape[0])
    return y

def make_logistic_regression_y(X, parameters):
    y_systematic = np.dot(X, parameters)
    p = 1 / (1 + np.exp(-y_systematic))
    return np.random.binomial(1, p=p, size=X.shape[0])

def make_poisson_regression_y(X, parameters):
    y_systematic = np.dot(X, parameters)
    mu = np.exp(y_systematic)
    return np.random.poisson(lam=mu, size=X.shape[0])

def make_gamma_regression_y(X, parameters):
    y_systematic = np.dot(X, parameters)
    mu = np.exp(y_systematic)
    return np.random.exponential(scale=mu, size=X.shape[0])
