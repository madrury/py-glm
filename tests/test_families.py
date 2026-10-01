import numpy as np
import pytest
import scipy.stats as sts

from glm.families import (
    Bernoulli,
    ExponentialFamily,
    Exponential,
    Gamma,
    Gaussian,
    Poisson,
    QuasiPoisson,
)

N_DRAWS = 200_000
# Each test checks two moments of the standardized draws. Set the threshold with
# a Bonferroni bound, allowing a FALSE_FAILURE_RATE chance of a spurious
# failure per test.
FALSE_FAILURE_RATE = 0.001
N_COMPARISONS = 2
MAX_Z = sts.norm.isf(FALSE_FAILURE_RATE / (2 * N_COMPARISONS))


@pytest.mark.parametrize(
    "family, mu_range, dispersion",
    [
        (Gaussian(), (-2.0, 2.0), 2.0),
        (Bernoulli(), (0.1, 0.9), 1.0),
        (Poisson(), (0.5, 5.0), 1.0),
        pytest.param(
            QuasiPoisson(),
            (0.5, 5.0),
            2.0,
            marks=pytest.mark.xfail(
                strict=True,
                reason="QuasiPoisson samples from a Poisson, ignoring dispersion.",
            ),
        ),
        (Gamma(), (0.5, 5.0), 0.25),
        (Exponential(), (0.5, 5.0), 1.0),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, ExponentialFamily) else None,
)
def test_sample_moments(
    family: ExponentialFamily, mu_range: tuple[float, float], dispersion: float
) -> None:
    # A GLM's samples must satisfy E[y] = mu and Var[y] = dispersion * V(mu).
    # Standardize the draws so both checks reduce to mean 0 and variance 1.
    mu = np.linspace(*mu_range, N_DRAWS)
    y = family.sample(mu, dispersion)
    r = (y - mu) / np.sqrt(dispersion * family.variance(mu))

    z_mean = r.mean() / (r.std() / np.sqrt(N_DRAWS))
    assert abs(z_mean) < MAX_Z

    # The standard error of the mean of r**2 depends on the family's tails, so
    # estimate it from the draws.
    r_squared = r**2
    z_variance = (r_squared.mean() - 1) / (r_squared.std() / np.sqrt(N_DRAWS))
    assert abs(z_variance) < MAX_Z


@pytest.mark.parametrize(
    "family, mu",
    [
        (Gaussian(), np.linspace(-5.0, 5.0, 11)),
        (Bernoulli(), np.linspace(0.01, 0.99, 11)),
        (Poisson(), np.linspace(0.1, 1000.0, 11)),
        (QuasiPoisson(), np.linspace(0.1, 1000.0, 11)),
        (Gamma(), np.linspace(0.1, 1000.0, 11)),
        (Exponential(), np.linspace(0.1, 1000.0, 11)),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, ExponentialFamily) else "",
)
def test_link_inverts_inv_link(family: ExponentialFamily, mu: np.ndarray) -> None:
    np.testing.assert_allclose(family.inv_link(family.link(mu)), mu)
