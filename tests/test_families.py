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


FAMILIES = [Gaussian(), Bernoulli(), Poisson(), QuasiPoisson(), Gamma(), Exponential()]


@pytest.mark.parametrize("family", FAMILIES, ids=repr)
def test_sample_with_rng_is_reproducible(family: ExponentialFamily) -> None:
    mu = np.linspace(0.1, 0.9, 100)
    first = family.sample(mu, 0.5, rng=np.random.default_rng(154))
    second = family.sample(mu, 0.5, rng=np.random.default_rng(154))
    other = family.sample(mu, 0.5, rng=np.random.default_rng(155))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("family", FAMILIES, ids=repr)
def test_sample_without_rng_follows_global_seed(family: ExponentialFamily) -> None:
    mu = np.linspace(0.1, 0.9, 100)
    np.random.seed(154)
    first = family.sample(mu, 0.5)
    np.random.seed(154)
    second = family.sample(mu, 0.5)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("family", FAMILIES, ids=repr)
def test_deviance_sums_unit_deviances(family: ExponentialFamily) -> None:
    rng = np.random.default_rng(154)
    mu = rng.uniform(0.2, 0.8, size=50)
    y = family.sample(mu, 0.5, rng=rng).astype(np.float64)
    if isinstance(family, Gamma):
        y = np.maximum(y, 1e-3)
    weights = rng.uniform(0.5, 2.0, size=50)
    unit = family.unit_deviance(y, mu)
    assert unit.shape == (50,)
    assert np.all(unit >= -1e-12)
    assert family.deviance(y, mu) == pytest.approx(np.sum(unit))
    assert family.deviance(y, mu, weights) == pytest.approx(np.sum(weights * unit))


@pytest.mark.parametrize(
    "family, y",
    [
        (Gaussian(), np.array([-1.0, 0.5, 3.0])),
        (Bernoulli(), np.array([0.2, 0.5, 0.9])),  # Proportions.
        (Poisson(), np.array([0.5, 2.0, 7.0])),
        (Gamma(), np.array([0.5, 2.0, 7.0])),
    ],
    ids=lambda v: repr(v) if isinstance(v, ExponentialFamily) else "",
)
def test_unit_deviance_is_zero_for_a_perfect_fit(
    family: ExponentialFamily, y: np.ndarray
) -> None:
    np.testing.assert_allclose(family.unit_deviance(y, y), 0.0, atol=1e-12)


def test_bernoulli_unit_deviance_on_binary_data() -> None:
    # For 0/1 responses, the saturated model's y log(y) terms are all zero.
    y = np.array([0.0, 1.0, 1.0, 0.0])
    mu = np.array([0.2, 0.7, 0.4, 0.9])
    expected = -2 * (y * np.log(mu) + (1 - y) * np.log(1 - mu))
    np.testing.assert_allclose(Bernoulli().unit_deviance(y, mu), expected)


def test_bernoulli_unit_deviance_on_proportions() -> None:
    y = np.array([0.25, 0.5, 0.9])
    mu = np.array([0.4, 0.5, 0.6])
    expected = 2 * (y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu)))
    np.testing.assert_allclose(Bernoulli().unit_deviance(y, mu), expected)


def test_grouped_binomial_deviance_differs_from_expanded_by_a_constant() -> None:
    # Proportions k/m weighted by their trials m, against the same trials
    # expanded into 0/1 rows. The saturated models differ, so the deviances
    # differ, but by a constant that doesn't depend on the model's means.
    trials = np.array([4, 10, 6])
    successes = np.array([1, 7, 6])
    family = Bernoulli()

    def deviances(mu: np.ndarray) -> tuple[float, float]:
        grouped = family.deviance(successes / trials, mu, weights=trials)
        expanded_y = np.concatenate(
            [np.r_[np.ones(k), np.zeros(m - k)] for k, m in zip(successes, trials)]
        )
        expanded_mu = np.repeat(mu, trials)
        return grouped, family.deviance(expanded_y, expanded_mu)

    first = deviances(np.array([0.3, 0.6, 0.8]))
    second = deviances(np.array([0.5, 0.5, 0.95]))
    assert first[0] - first[1] == pytest.approx(second[0] - second[1])
    assert first[0] != pytest.approx(first[1])


def test_poisson_unit_deviance_with_zero_counts() -> None:
    # For a count of zero, the unit deviance is 2 mu: 0 log(0) is taken as 0.
    mu = np.array([0.5, 2.0])
    np.testing.assert_allclose(Poisson().unit_deviance(np.zeros(2), mu), 2 * mu)
