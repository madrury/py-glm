from typing import Any, Self
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy.stats as sts
from scipy.special import wrightomega as ω


from glm.glm import GLM
from glm.families import Gaussian, Bernoulli

type Floats = NDArray[np.floating[Any]] | float


def σ(xs: Floats) -> Floats:
    return 1 / (1 + np.exp(-xs))


@dataclass
class ParameterSampleRanges:
    a: tuple[float, float]
    b: tuple[float, float]
    α: tuple[float, float]
    β: tuple[float, float]


@dataclass
class ParticipationIntensityFunction:
    """Encapsulates a linear times logistic function.

    This models the expected revenue from offering a product to a customer at a
    given price X = x. There are two random variables at play in such a scenario:

      - The Participation, D: A binary variable, does the customer purchase the
        product at the given price?
      - The Intensity, Y: A continuous variable, given the customer purchases,
        what is the received revenue.

    It is assumed that Y is deterministically zero whenever D = 0, since D and Y
    model purchase events and revenue.

    Given those variables, the expected revenue is:

        E[Y | X] = P(D = 1 | X) E[Y | X, D = 1]

    This class encapsulates a logistic function for the first factor and a
    linear function for the second:

                      a + bx
        f(x) = ----------------------
                 1 + exp(-(α + βx))
    """

    a: float
    b: float
    α: float
    β: float
    _argmax: float | None = None

    def __call__(self, xs: Floats) -> Floats:
        a, b, α, β = self.a, self.b, self.α, self.β
        return (a + b * xs) * σ(α + β * xs)

    def intensity_mean(self, xs: Floats) -> Floats:
        a, b = self.a, self.b
        return a + b * xs

    def participation_probability(self, xs: Floats) -> Floats:
        α, β = self.α, self.β
        return σ(α + β * xs)

    def λ(self) -> float:
        """The argument of the Wright omega function in the closed-form argmax."""
        a, b, α, β = self.a, self.b, self.α, self.β
        return -(b + a * β - b * α) / b

    def argmax(self) -> float:
        """Computes the argmaximum of the participation-intensity function.

        The participation-intensity function always has a unique critical point
        when bβ ≠ 0, and it is required that bβ < 0 for that critical point to
        be a maximum, bβ > 0 for a minimum.

        The closed form uses the Wright omega function ω:
            https://en.wikipedia.org/wiki/Wright_omega_function
        """
        if self._argmax is not None:
            return self._argmax
        a, b, α, β = self.a, self.b, self.α, self.β
        if b * β >= 0:
            raise ValueError(
                "The critical point is a maximum only when bβ < 0, "
                f"got b={b} and β={β}"
            )
        λ = self.λ()
        # ω(λ) = W(exp(λ)), but stays finite where exp(λ) overflows.
        self._argmax = float(-(b + a * β + b * ω(λ)) / (b * β))
        return self._argmax

    def max(self) -> float:
        return float(self(self.argmax()))

    def __str__(self) -> str:
        return (
            "ParticipationIntensityFunction("
            f"a={self.a:.2f}, "
            f"b={self.b:.2f}, "
            f"α={self.α:.2f}, "
            f"β={self.β:.2f}"
            ")"
        )


# The partial derivatives of the argmax with respect to (a, b, α, β), for the
# delta method. tests/test_participation_intensity.py checks them against finite
# differences.
@dataclass
class ArgmaxGradient:
    """Encapsulates the gradient of the critical point of the
    participation-intensity function, as it varies with a, b, α, β.

    Used to construct a delta method confidence interval for the critical point
    when the a, b, α, β parameters and their covariances are estimated from
    data.
    """

    mean: ParticipationIntensityFunction

    def __post_init__(self) -> None:
        self.a = self.mean.a
        self.b = self.mean.b
        self.α = self.mean.α
        self.β = self.mean.β
        # Raises unless the critical point is a maximum.
        self.mean.argmax()
        self.λ = self.mean.λ()
        self.w = float(ω(self.λ))

    def da(self) -> float:
        return -1 / (self.b * self.w + self.b)

    def db(self) -> float:
        return self.a / (self.b**2 * self.w + self.b**2)

    def dα(self) -> float:
        return -self.w / (self.β * self.w + self.β)

    def dβ(self) -> float:
        return (
            self.b * self.w**2 + (self.a * self.β + 2 * self.b) * self.w + self.b
        ) / (self.b * self.β**2 * (self.w + 1))

    def gradient(self) -> NDArray[np.float64]:
        return np.array([self.da(), self.db(), self.dα(), self.dβ()])


@dataclass
class ParticipationIntensityModel:
    """Estimate a participation-intensity function using generalized linear
    models for the conditional distributions:

        D | X        ~ Bernoulli(p=logistic(α + β * x))
        Y | D = 1, X ~ Gaussian(μ=a + b * x, σ²)

    The component models estimate both the parameters and their covariances,
    which are used along with the delta method to estimate the (approximate)
    standard error of the participation-intensity function's critical point.
    """

    participation: GLM = field(default_factory=lambda: GLM(family=Bernoulli()))
    intensity: GLM = field(default_factory=lambda: GLM(family=Gaussian()))
    mean_: ParticipationIntensityFunction | None = None

    def fit(self, X: pd.DataFrame) -> Self:
        self.participation.fit(X, formula="d ~ x")
        assert self.participation.coef_ is not None
        self.intensity.fit(X.query("d == 1"), formula="y ~ x")
        assert self.intensity.coef_ is not None
        self.mean_ = ParticipationIntensityFunction(
            a=self.intensity.coef_[0],
            b=self.intensity.coef_[1],
            α=self.participation.coef_[0],
            β=self.participation.coef_[1],
        )
        return self

    def summary(self) -> None:
        self.intensity.summary()
        self.participation.summary()

    def argmax_(self) -> float:
        if self.mean_ is None:
            raise ValueError("The model must be fit.")
        return self.mean_.argmax()

    def argmax_se_(self) -> float:
        if self.mean_ is None:
            raise ValueError("The model must be fit.")
        gradient = ArgmaxGradient(self.mean_).gradient()
        # The log-likelihood function splits into terms depending only on a, b,
        # and terms only depending on α, β. Thus the off diagonal terms are
        # zero.
        cov = np.block(
            [
                [self.intensity.coef_covariance_matrix_, np.zeros((2, 2))],
                [np.zeros((2, 2)), self.participation.coef_covariance_matrix_],
            ]
        )
        return float(np.sqrt(gradient @ cov @ gradient))


def sample_data(
    mean: ParticipationIntensityFunction, xs: Floats, sd: float = 0.1
) -> pd.DataFrame:
    ps = mean.participation_probability(xs)
    ds = np.random.binomial(n=1, p=ps)
    ys = np.random.normal(mean.intensity_mean(xs), scale=sd)
    return pd.DataFrame({"x": xs, "d": ds, "y": ys})


@dataclass
class CoverageSimulation:
    """Simulate data and fits to empirically evaluate the coverage properties of
    the critical point interval.
    """

    mean: ParticipationIntensityFunction

    def sample(self, n: int, nxs: int = 100) -> Self:
        models: list[ParticipationIntensityModel] = []
        for _ in range(n):
            xs = np.linspace(0.0, 3.0, num=nxs)
            X = sample_data(self.mean, xs)
            model = ParticipationIntensityModel().fit(X)
            models.append(model)
        self.models = models
        return self

    def argmaxs_(self) -> Floats:
        return np.fromiter((model.argmax_() for model in self.models), float)

    def argmax_ses_(self) -> Floats:
        return np.fromiter((model.argmax_se_() for model in self.models), float)

    def argmax_cis_(self, level: float = 0.95) -> NDArray[np.float64]:
        """Wald intervals for the true critical point, one for each sampled fit,
        centered at its estimated critical point, with standard errors from the
        delta method."""
        amaxs = self.argmaxs_()
        amax_ses = self.argmax_ses_()
        quantile = float(sts.norm.ppf((1 + level) / 2))
        return np.column_stack(
            [amaxs - quantile * amax_ses, amaxs + quantile * amax_ses]
        )
