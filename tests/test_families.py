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
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_sample_moments(
    family: ExponentialFamily,
    mu_range: tuple[float, float],
    dispersion: float,
    weighted: bool,
) -> None:
    # A GLM's samples must satisfy E[y] = mu and Var[y] = dispersion * V(mu) / w,
    # with precision weights w, each sample being the average of w draws.
    # Standardize the draws so both checks reduce to mean 0 and variance 1.
    mu = np.linspace(*mu_range, N_DRAWS)
    weights = None
    if weighted:
        weights = np.random.default_rng(154).integers(1, 10, N_DRAWS).astype(float)
    y = family.sample(mu, dispersion, weights)
    variance = dispersion * family.variance(mu) / (1.0 if weights is None else weights)
    r = (y - mu) / np.sqrt(variance)

    # Each check asks whether an average over the draws is as close to its
    # target as sampling noise allows. If the sampler is right, the r are
    # independent, each with mean 0 and variance 1. They aren't identically
    # distributed: each draw has its own mu and w, so their shapes differ (a
    # Poisson's skewness is 1 / √mu, say). But the Lindeberg-Feller central
    # limit theorem still applies, as no single draw dominates the average:
    # the ranges of mu and w above keep each r's third moment bounded. So their
    # average is approximately normal, with mean 0 and standard error
    # sd(r) / √N, where sd(r)² is the average of the r's variances. The pooled
    # r.std() estimates that average, plus the spread of the r's means, which is
    # zero when the sampler is right, as standardizing gave them all mean 0.
    #
    # Dividing the average by its standard error gives a z-score:
    # approximately standard normal when the sampler is right, and growing like
    # √N when it's wrong. With N = 200,000 draws, even a small error in the mean
    # gives a large z-score. Each z-score must stay within ±MAX_Z, the
    # Bonferroni bound set above, so that a correct sampler fails either check
    # with probability at most FALSE_FAILURE_RATE.
    z_mean = r.mean() / (r.std() / np.sqrt(N_DRAWS))
    assert abs(z_mean) < MAX_Z

    # The same logic applies to the variance, through the average of r**2, whose
    # target is 1. Each r**2 has mean 1 when the sampler is right, but its own
    # variance, through the underlying draw's fourth moments, which depend on mu
    # and w.  Again, the pooled standard deviation estimates the average of
    # those variances, as the r**2 share a mean.
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


@pytest.mark.parametrize("family", [Bernoulli(), Poisson()], ids=repr)
def test_discrete_samples_need_whole_number_weights(family: ExponentialFamily) -> None:
    with pytest.raises(ValueError, match="precision_weights must be whole numbers"):
        family.sample(np.full(3, 0.5), 1.0, np.array([1.0, 2.5, 3.0]))


@pytest.mark.parametrize("family", [Gaussian(), Gamma()], ids=repr)
def test_continuous_samples_allow_fractional_weights(
    family: ExponentialFamily,
) -> None:
    y = family.sample(np.full(3, 2.0), 0.5, np.array([0.5, 2.5, 3.0]))
    assert y.shape == (3,)
