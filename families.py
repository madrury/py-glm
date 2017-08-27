"""Exponential families for GLMs"""
import numpy as np


class ExponentialFamily:

    def penalized_deviance(self, y, mu, alpha, coef):
        return self.deviance(y, mu) + alpha*np.sum(coef[1:]**2)


class Gaussian(ExponentialFamily):

    has_dispersion = True

    def inv_link(self, nu):
        return nu

    def d_inv_link(self, nu, mu):
        return np.ones(shape=nu.shape)

    def variance(self, mu):
        return np.ones(shape=mu.shape)

    def deviance(self, y, mu):
        return np.sum((y - mu)**2)


class Bernoulli(ExponentialFamily):

    has_dispersion = False

    def inv_link(self, nu):
        return 1 / (1 + np.exp(-nu))

    def d_inv_link(self, nu, mu):
        return mu * (1 - mu)

    def variance(self, mu):
        return mu * (1 - mu)

    def deviance(self, y, mu):
        return -2 * np.sum(y*np.log(mu) + (1 - y)*np.log(1 - mu))


class Poisson(ExponentialFamily):

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
