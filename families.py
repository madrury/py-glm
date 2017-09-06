"""Exponential families for GLMs"""
from abc import ABCMeta, abstractmethod
import numpy as np


class ExponentialFamily(metaclass=ABCMeta):
    """An ExponentialFamily must implement at least four methods and define one
    attribute.

    Methods
    -------
    inv_link:
        The inverse link function.

    d_inv_link:
        The derivative of the inverse link function.

    variance:
        The variance funtion linking the mean to the variance of the
        distribution.

    deviance:
        The deviance of the family. Used as a measure of model fit.

    sample:
        A sampler from the conditional distribution of the reponse.
    """
    @abstractmethod
    def inv_link(self, nu):
        pass

    @abstractmethod
    def d_inv_link(self, nu, mu):
        pass

    @abstractmethod
    def variance(self, mu):
        pass

    @abstractmethod
    def deviance(self, y, mu):
        pass

    @abstractmethod
    def sample(self, mus, dispersion):
        pass


class ExponentialFamilyMixin:
    """Implementations of methods common to all ExponentialFamilies."""
    def penalized_deviance(self, y, mu, alpha, coef):
        return self.deviance(y, mu) + alpha*np.sum(coef[1:]**2)


class Gaussian(ExponentialFamily, ExponentialFamilyMixin):
    """A Gaussian exponential family, used to fit a classical linear model.

    The GLM fit with this family has the following structure equation:

        y | X ~ Gaussian(mu = X beta, sigma = sigma)

    Here, sigma is a nuisance parameter.
    """
    has_dispersion = True

    def inv_link(self, nu):
        return nu

    def d_inv_link(self, nu, mu):
        return np.ones(shape=nu.shape)

    def variance(self, mu):
        return np.ones(shape=mu.shape)

    def deviance(self, y, mu):
        return np.sum((y - mu)**2)

    def sample(self, mus, dispersion):
        return np.random.normal(mus, np.sqrt(dispersion))


class Bernoulli(ExponentialFamily, ExponentialFamilyMixin):
    """A Bernoulli exponential family, used to fit a classical logistic model.

    The GLM fit with this family has the following structure equation:

        y | X ~ Bernoulli(p = X beta)
    """
    has_dispersion = False

    def inv_link(self, nu):
        return 1 / (1 + np.exp(-nu))

    def d_inv_link(self, nu, mu):
        return mu * (1 - mu)

    def variance(self, mu):
        return mu * (1 - mu)

    def deviance(self, y, mu):
        return -2 * np.sum(y*np.log(mu) + (1 - y)*np.log(1 - mu))

    def sample(self, mus, dispersion):
        return np.random.binomial(1, mus)


class Poisson(ExponentialFamily, ExponentialFamilyMixin):
    """A Poisson exponential family, used to fit a Poisson regression.

    The GLM fit with this family has the following structure equation:

        y | X ~ Poisson(mu = exp(X beta))
    """
    has_dispersion = False

    def inv_link(self, nu):
        return np.exp(nu)

    def d_inv_link(self, nu, mu):
        return mu

    def variance(self, mu):
        return mu

    def deviance(self, y, mu):
        # Need to avoid explicitly calculating y*log(y) when y == 0. 
        y_log_y = np.empty(shape=y.shape)
        y_log_y[y == 0] = 0
        y_non_zero = y[y != 0]
        y_log_y[y != 0] = y_non_zero*np.log(y_non_zero)
        return 2*np.sum(mu - y - y*np.log(mu) + y_log_y)

    def sample(self, mus, dispersion):
        return np.random.poisson(mus)


class Gamma(ExponentialFamily, ExponentialFamilyMixin):

    has_dispersion = True

    def inv_link(self, nu):
        return np.exp(nu)

    def d_inv_link(self, nu, mu):
        return mu

    def variance(self, mu):
        return mu * mu

    def deviance(self, y, mu):
        return 2 * np.sum((y - mu) / mu - np.log(y / mu))

    def sample(self, mu, dispersion):
        shape, scale = dispersion, mu / dispersion
        return np.random.gamma(shape=shape, scale=scale)
