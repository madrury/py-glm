"""Exponential families for GLMs"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import scipy.stats as sts
from scipy.special import xlogy

from glm.types import (
    Deviance,
    InverseLinkDerivative,
    LinearPredictor,
    MeanVector,
    ReferenceDistribution,
    ResponseVector,
    DevianceWeights,
    PrecisionWeights,
    WorkingWeights,
    UnitDeviances,
    VarianceVector,
)


@dataclass(frozen=True)
class TestDistribution:
    """The reference distribution for a family's coefficient tests: the
    statistic's name, "t" or "z", and its distribution under the hypothesis
    that a coefficient is zero."""

    statistic: str
    distribution: ReferenceDistribution


class ExponentialFamily(ABC):
    """An exponential family is a one or two parameter family of probability
    distributions with a flexible parametric form that covers many common named
    families of distributions. For example, the Gaussian, Bernoulli, and Gamma
    families of distributions are exponential families.

    A Generalized Linear Model assumes the conditional distribution y | X of the
    response given the features is an exponential family, with the parameters of
    said family modeled as a function of the features X.

    An ExponentialFamily object must implement the methods below and define one
    attribute.

    Methods
    -------
    link:
        The link function, mapping the conditional mean μ = E[y | X] to the
        linear predictor η = X @ β.

    inv_link:
        The inverse link function.

    d_inv_link:
        The derivative of the inverse link function: dμ/dη.

    variance:
        The variance function linking the mean to the variance of the
        distribution.

    unit_deviance:
        Each observation's contribution to the deviance: twice the difference
        between the log-likelihoods of a saturated model, fitting every
        observation exactly, and the model, for that observation.

    deviance:
        The (optionally weighted) sum of the unit deviances, a measure of the
        model's fit. Implemented once, here, from unit_deviance.

    coefficient_test:
        The reference distribution for testing that a coefficient is zero:
        Student's t if the dispersion is estimated, the standard normal if
        not.

    sample:
        A sampler from the conditional distribution of the response. With
        precision weights w, each response is the average of w draws, with
        variance: dispersion * V(μ) / w. E.g. a proportion of successes in w
        trials. Discrete families need whole-number weights for this to make
        sense. Pass a numpy Generator as rng to make draws reproducible; without
        one, draws come from numpy's global random state, so np.random.seed
        controls them.

    Attributes
    ----------
    has_dispersion:
        Whether the family has a dispersion parameter that must be estimated.
    """

    has_dispersion: ClassVar[bool]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def coefficient_test(self, residual_df: float) -> TestDistribution:
        """The reference distribution of a coefficient's test statistic,
        estimate / standard error, under the hypothesis that the true
        coefficient is zero.

        A family with an estimated dispersion uses Student's t distribution
        with the residual degrees of freedom, n - p, accounting for the
        uncertainty in the dispersion estimate. For the Gaussian family this is
        exact; for others it's the standard approximation. A family with a fixed
        dispersion uses the standard normal: its statistics are asymptotically
        normal, and nothing extra was estimated. This is the convention of R's
        summary.glm.
        """
        if self.has_dispersion:
            return TestDistribution("t", sts.t(df=residual_df))
        return TestDistribution("z", sts.norm())

    @abstractmethod
    def link(self, mu: MeanVector) -> LinearPredictor:
        pass

    @abstractmethod
    def inv_link(self, eta: LinearPredictor) -> MeanVector:
        pass

    @abstractmethod
    def d_inv_link(self, eta: LinearPredictor, mu: MeanVector) -> InverseLinkDerivative:
        pass

    @abstractmethod
    def variance(self, mu: MeanVector) -> VarianceVector:
        pass

    @abstractmethod
    def unit_deviance(self, y: ResponseVector, mu: MeanVector) -> UnitDeviances:
        pass

    def deviance(
        self,
        y: ResponseVector,
        mu: MeanVector,
        weights: DevianceWeights | None = None,
    ) -> Deviance:
        """The sum of the unit deviances, each multiplied by its weight, if
        weights are given: each row's frequency weight times its precision
        weight."""
        unit_deviances = self.unit_deviance(y, mu)
        if weights is not None:
            unit_deviances = weights * unit_deviances
        return float(np.sum(unit_deviances))

    @abstractmethod
    def sample(
        self,
        mu: MeanVector,
        dispersion: float,
        precision_weights: PrecisionWeights | None = None,
        rng: np.random.Generator | None = None,
    ) -> ResponseVector:
        pass


class Gaussian(ExponentialFamily):
    """A Gaussian exponential family, used to fit a classical linear model.

    The GLM fit with this family has the following structure equation:

        y | X ~ Gaussian(μ = X @ β, σ² = dispersion)

    Here, the variance σ² is a nuisance parameter.
    """

    has_dispersion = True

    def link(self, mu: MeanVector) -> LinearPredictor:
        return mu

    def inv_link(self, eta: LinearPredictor) -> MeanVector:
        return eta

    def d_inv_link(self, eta: LinearPredictor, mu: MeanVector) -> InverseLinkDerivative:
        return np.ones(shape=eta.shape)

    def variance(self, mu: MeanVector) -> VarianceVector:
        return np.ones(shape=mu.shape)

    def unit_deviance(self, y: ResponseVector, mu: MeanVector) -> UnitDeviances:
        unit_deviances: UnitDeviances = (y - mu) ** 2
        return unit_deviances

    def sample(
        self,
        mu: MeanVector,
        dispersion: float,
        precision_weights: PrecisionWeights | None = None,
        rng: np.random.Generator | None = None,
    ) -> ResponseVector:
        weights = 1.0 if precision_weights is None else precision_weights
        return _generator(rng).normal(mu, np.sqrt(dispersion / weights))

    def initial_working_response(self, y: ResponseVector) -> ResponseVector:
        return y

    def initial_working_weights(self, y: ResponseVector) -> WorkingWeights:
        return (1 / len(y)) * np.ones(len(y))


class Bernoulli(ExponentialFamily):
    """A Bernoulli exponential family, used to fit a classical logistic model.

    The GLM fit with this family has the following structure equation:

        y | X ~ Bernoulli(p = 1 / (1 + exp(-X @ β)))
    """

    has_dispersion = False

    def link(self, mu: MeanVector) -> LinearPredictor:
        return np.log(mu / (1 - mu))

    def inv_link(self, eta: LinearPredictor) -> MeanVector:
        return 1 / (1 + np.exp(-eta))

    def d_inv_link(self, eta: LinearPredictor, mu: MeanVector) -> InverseLinkDerivative:
        return mu * (1 - mu)

    def variance(self, mu: MeanVector) -> VarianceVector:
        return mu * (1 - mu)

    def unit_deviance(self, y: ResponseVector, mu: MeanVector) -> UnitDeviances:
        # The binomial unit deviance, so that a response may also be a
        # proportion of successes, weighted by its number of trials (i.e.
        # binomial data). xlogy takes 0 log(0) as 0, the saturated model's
        # log-likelihood for an outcome it fits exactly. For a 0/1 response, the
        # y log(y) terms are always 0, leaving:
        #     -2 (y log(mu) + (1 - y) log(1 - mu))
        # which is maybe more familiar in ML tradition as twice the log loss
        # (binary cross-entropy).
        unit_deviances: UnitDeviances = np.asarray(
            2 * (xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu))),
            dtype=np.float64,
        )
        return unit_deviances

    def sample(
        self,
        mu: MeanVector,
        dispersion: float,
        precision_weights: PrecisionWeights | None = None,
        rng: np.random.Generator | None = None,
    ) -> ResponseVector:
        if precision_weights is None:
            return _generator(rng).binomial(1, mu)
        # The proportion of successes in w trials.
        trials = _are_whole_numbers(precision_weights, self)
        return _generator(rng).binomial(trials, mu) / trials

    def initial_working_response(self, y: ResponseVector) -> ResponseVector:
        return (y - 0.5) / 0.25

    def initial_working_weights(self, y: ResponseVector) -> WorkingWeights:
        return (1 / len(y)) * 0.25 * np.ones(len(y))


class QuasiPoisson(ExponentialFamily):
    """A QuasiPoisson exponential family, used to fit a possibly overdispersed
    Poisson regression.

    The GLM fit with this family has the following structure equation:

        y | X ~ Poisson(μ = exp(X @ β))

    The parameter esimtates of this model are the same as a Poisson model, but
    a dispersion parameter is estimated, allowing for possibly larger standards
    errors when overdispersion is present.
    """

    has_dispersion = True

    def link(self, mu: MeanVector) -> LinearPredictor:
        return np.log(mu)

    def inv_link(self, eta: LinearPredictor) -> MeanVector:
        return np.exp(eta)

    def d_inv_link(self, eta: LinearPredictor, mu: MeanVector) -> InverseLinkDerivative:
        return mu

    def variance(self, mu: MeanVector) -> VarianceVector:
        return mu

    def unit_deviance(self, y: ResponseVector, mu: MeanVector) -> UnitDeviances:
        # xlogy takes 0 log(0) as 0, the saturated model's log-likelihood for
        # a count of zero.
        unit_deviances: UnitDeviances = 2 * (xlogy(y, y / mu) - (y - mu))
        return unit_deviances

    def sample(
        self,
        mu: MeanVector,
        dispersion: float,
        precision_weights: PrecisionWeights | None = None,
        rng: np.random.Generator | None = None,
    ) -> ResponseVector:
        if precision_weights is None:
            return _generator(rng).poisson(mu)
        # The average of w counts: their total is Poisson, with mean w * mu.
        counts = _are_whole_numbers(precision_weights, self)
        return _generator(rng).poisson(counts * mu) / counts


class Poisson(QuasiPoisson):
    """A Poisson exponential family, used to fit a Poisson regression.

    The GLM fit with this family has the following structure equation:

        y | X ~ Poisson(μ = exp(X @ β))

    The Poisson model does not estimate a dispersion parameter; if
    overdispersion is present, the standard errors estimated in this model may
    be too small. If this is the case, consider fitting a QuasiPoisson model.
    """

    has_dispersion = False


class Gamma(ExponentialFamily):
    """A Gamma exponential family.

    The GLM fit with this family has the following structure equation:

        y | X ~ Gamma(shape = 1 / dispersion, scale = exp(X @ β) * dispersion)

    so that E[y | X] = exp(X @ β) and Var[y | X] = dispersion * E[y | X]**2.

    Note: In this family we use the logarithmic link function, instead of the
    reciprocal link function. Although the reciprocal is the canonical link,
    the logarithmic link is more commonly used in Gamma regression.
    """

    has_dispersion = True

    def link(self, mu: MeanVector) -> LinearPredictor:
        return np.log(mu)

    def inv_link(self, eta: LinearPredictor) -> MeanVector:
        return np.exp(eta)

    def d_inv_link(self, eta: LinearPredictor, mu: MeanVector) -> InverseLinkDerivative:
        return mu

    def variance(self, mu: MeanVector) -> VarianceVector:
        return mu * mu

    def unit_deviance(self, y: ResponseVector, mu: MeanVector) -> UnitDeviances:
        unit_deviances: UnitDeviances = 2 * ((y - mu) / mu - np.log(y / mu))
        return unit_deviances

    def sample(
        self,
        mu: MeanVector,
        dispersion: float,
        precision_weights: PrecisionWeights | None = None,
        rng: np.random.Generator | None = None,
    ) -> ResponseVector:
        # The shape is dimensionless and sets the coefficient of variation,
        # CV**2 = 1 / shape = dispersion. The scale carries the units of y.
        shape: float | PrecisionWeights = 1 / dispersion
        scale = mu * dispersion
        # The average of w draws: its shape is w times as large, and its scale
        # w times smaller, keeping the mean, with a w times smaller variance.
        if precision_weights is not None:
            shape, scale = shape * precision_weights, scale / precision_weights
        return _generator(rng).gamma(shape=shape, scale=scale)


class Exponential(Gamma):
    """An Exponential distribution exponential family.

    The GLM fit with this family has the following structure equation:

        y | X ~ Exponential(scale = exp(X @ β))

    The only difference between this family and the Gamma family is that the
    dispersion is fixed at 1.0.
    """

    has_dispersion = False


def _are_whole_numbers(
    precision_weights: PrecisionWeights, family: ExponentialFamily
) -> npt.NDArray[np.int64]:
    """Precision weights as counts of draws to be averaged, which a discrete
    family needs to be whole numbers."""
    if not np.all(precision_weights == np.round(precision_weights)):
        raise ValueError(
            f"precision_weights must be whole numbers to sample from {family!r}, "
            "as each response is the average of that many draws."
        )
    return precision_weights.astype(np.int64)


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    """The generator to draw samples with. Without one, seed a new generator
    from numpy's global random state, so that np.random.seed still makes the
    draws reproducible."""
    if rng is not None:
        return rng
    return np.random.default_rng(np.random.randint(np.iinfo(np.int64).max))
