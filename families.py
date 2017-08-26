"""Exponential families for GLMs"""
import numpy as np


class ExponentialFamily:

    def penalized_deviance(self, y, mu, alpha, coef):
        return self.deviance(y, mu) + alpha*np.sum(coef[1:]**2)


class Gaussian(ExponentialFamily):

    def inv_link(self, nu):
        return nu

    def d_inv_link(self, nu, mu):
        return np.ones(shape=nu.shape)

    def variance(self, mu):
        return np.ones(shape=mu.shape)

    def deviance(self, y, mu):
        return 2*np.sum((y - mu)**2)


class Bernoulli(ExponentialFamily):

    def inv_link(self, nu):
        return 1 / (1 + np.exp(-nu))

    def d_inv_link(self, nu, mu):
        return mu * (1 - mu)

    def variance(self, mu):
        return mu * (1 - mu)

    def deviance(self, y, mu):
        return 2*np.sum(y*np.log(mu) + (1 - y)*np.log(1 - mu))


class Poisson(ExponentialFamily):

    def inv_link(self, nu):
        return np.exp(nu)

    def d_inv_link(self, nu, mu):
        return mu

    def variance(self, mu):
        return mu

    def deviance(self, y, mu):
        return np.sum(mu - y - y*np.log(mu))
