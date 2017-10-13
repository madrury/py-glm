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

        y | X ~ Gaussian(mu = X beta, sigma = dispersion)

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

    def initial_working_response(self, y):
        return y

    def initial_working_weights(self, y):
        return (1 / len(y)) * np.ones(len(y))


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

    def initial_working_response(self, y):
        return 0.5 + (y - 0.5) / 0.25

    def initial_working_weights(self, y):
        return 0.25 * np.ones(len(y))


class QuasiPoisson(ExponentialFamily, ExponentialFamilyMixin):
    """A QuasiPoisson exponential family, used to fit a possibly overdispersed
    Poisson regression.

    The GLM fit with this family has the following structure equation:

        y | X ~ Poisson(mu = exp(X beta))

    The parameter esimtates of this model are the same as a Poisson model, but
    a dispersion parameter is estimated, allowing for possibly larger standards
    errors when overdispersion is present.
    """
    has_dispersion = True

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


class Poisson(QuasiPoisson):
    """A QuasiPoisson exponential family, used to fit a possibly overdispersed
    Poisson regression.

    The GLM fit with this family has the following structure equation:

        y | X ~ Poisson(mu = exp(X beta))

    The Poisson model does not estimate a dispersion parameter; if
    overdispersion is present, the standard errors estimated in this model may
    be too small.  If this is the case, consider fitting a QuasiPoisson model.
    """
    has_dispersion = False


class Gamma(ExponentialFamily, ExponentialFamilyMixin):
    """A Gamma exponential family.

    The GLM fit with this family has the following structure equation:

        y | X ~ Gamma(shape = dispersion, scale = exp(X beta) / dispersion)

    Here, sigma is a nuisance parameter.

    Note: In this family we use the logarithmic link function, instead of the
    reciporical link function.  Although the reciporical is the canonical link,
    the logarithmic link is more commonly used in Gamma regression.
    """
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


class Exponential(Gamma):
    """An Exponential distribution exponential family.

    The GLM fit with this family has the following structure equation:

        y | X = Exponentiatl(scale = exp(X beta))

    The only difference between this family and the Gamma family is that the
    dispersion is fixed at 1.
    """
    has_dispersion = False
