"""Exponential families for GLMs"""


class Gaussian:

    def inv_link(nu):
        return nu

    def d_inv_link(nu, mu):
        return 1

    def variance(mu):
        return 1

    def deviance(y, mu):
        return 2*np.sum((y - mu)**2)


class Bernoulli:

    def inv_link(nu):
        return 1 / (1 + np.exp(nu))

    def d_inv_link(nu, mu):
        return mu * (1 - mu)

    def variance(mu):
        return mu * (1 - mu)

    def deviance(y, mu):
        return 2*np.sum(y*np.log(mu) + (1 - y)*np.log(1 - mu))


class Poission:

    def inv_link(nu):
        return np.exp(nu)

    def d_inv_link(nu, mu):
        return mu

    def variance(mu):
        return mu

    def deviance(y, mu):
        return np.sum(y*np.log(y/mu) - (y - mu))
