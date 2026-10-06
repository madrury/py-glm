import numpy as np
import pytest
import statsmodels.api as sm

from glm.datasets import make_regression
from glm.families import Bernoulli, ExponentialFamily, Gamma, Gaussian, Poisson
from glm.glm import GLM

N = 300
FIT_TOL = 1e-12

STATSMODELS_FAMILIES = {
    "Gaussian": sm.families.Gaussian(),
    "Poisson": sm.families.Poisson(),
    "Gamma": sm.families.Gamma(sm.families.links.Log()),
}


@pytest.fixture
def weights() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(154)
    frequency = rng.integers(1, 5, size=N).astype(np.float64)
    precision = rng.uniform(0.5, 3.0, size=N)
    return frequency, precision


def simulate(family: ExponentialFamily) -> tuple[np.ndarray, np.ndarray]:
    data = make_regression(
        family,
        n_samples=N,
        n_uncorrelated=3,
        coef_range=(-0.3, 0.3),
        dispersion=0.5,
        rng=np.random.default_rng(154),
    )
    return data.X, data.y.astype(np.float64)


@pytest.mark.parametrize("kinds", ["frequency", "precision", "both"])
@pytest.mark.parametrize("family", [Gaussian(), Poisson(), Gamma()], ids=repr)
def test_weighted_fit_matches_statsmodels(
    family: ExponentialFamily,
    kinds: str,
    weights: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = simulate(family)
    frequency, precision = weights
    ours: dict[str, np.ndarray] = {}
    theirs: dict[str, np.ndarray] = {}
    if kinds in ("frequency", "both"):
        ours["frequency_weights"], theirs["freq_weights"] = frequency, frequency
    if kinds in ("precision", "both"):
        ours["precision_weights"], theirs["var_weights"] = precision, precision
    model = GLM(family).fit(X, y, tol=FIT_TOL, **ours)
    # Our dispersion is deviance-based, and fixed at 1 for the Poisson.
    result = sm.GLM(y, X, family=STATSMODELS_FAMILIES[repr(family)[:-2]], **theirs).fit(
        scale="dev" if family.has_dispersion else None
    )
    np.testing.assert_allclose(model.coef_, result.params, rtol=1e-6)
    np.testing.assert_allclose(model.coef_standard_error_, result.bse, rtol=1e-6)
    if family.has_dispersion:
        assert model.dispersion_ == pytest.approx(result.scale, rel=1e-6)
    # Only frequency weights add observations.
    assert model.n is not None and model.p is not None
    assert model.n - model.p == pytest.approx(result.df_resid)


@pytest.mark.parametrize("family", [Gaussian(), Poisson(), Gamma()], ids=repr)
def test_frequency_weights_match_expanded_data(
    family: ExponentialFamily, weights: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = simulate(family)
    frequency, _ = weights
    model = GLM(family).fit(X, y, frequency_weights=frequency, tol=FIT_TOL)
    repeats = frequency.astype(int)
    expanded = GLM(family).fit(
        np.repeat(X, repeats, axis=0), np.repeat(y, repeats), tol=FIT_TOL
    )
    np.testing.assert_allclose(model.coef_, expanded.coef_, rtol=1e-8)
    np.testing.assert_allclose(
        model.coef_standard_error_, expanded.coef_standard_error_, rtol=1e-8
    )
    assert model.deviance_ == pytest.approx(expanded.deviance_)
    assert model.dispersion_ == pytest.approx(expanded.dispersion_)


def test_precision_and_frequency_weights_differ_in_dispersion(
    weights: tuple[np.ndarray, np.ndarray],
) -> None:
    # The same weights, read as frequencies or as precisions: identical
    # estimates, but different degrees of freedom, so different dispersions.
    X, y = simulate(Gaussian())
    frequency, _ = weights
    as_frequency = GLM(Gaussian()).fit(X, y, frequency_weights=frequency)
    as_precision = GLM(Gaussian()).fit(X, y, precision_weights=frequency)
    np.testing.assert_allclose(as_frequency.coef_, as_precision.coef_)
    assert as_frequency.deviance_ == pytest.approx(as_precision.deviance_)
    assert as_frequency.n == pytest.approx(frequency.sum())
    assert as_precision.n == N
    assert as_frequency.dispersion_ < as_precision.dispersion_


def test_binomial_proportions_with_trials() -> None:
    # Proportions weighted by their trials fit the same coefficients as the
    # trials expanded into 0/1 rows, and match statsmodels' binomial.
    rng = np.random.default_rng(154)
    X = np.column_stack([np.ones(N), rng.normal(size=N)])
    trials = rng.integers(1, 20, size=N)
    successes = rng.binomial(trials, 1 / (1 + np.exp(-(X @ [0.3, -0.8]))))
    model = GLM(Bernoulli()).fit(
        X, successes / trials, precision_weights=trials.astype(np.float64), tol=FIT_TOL
    )
    result = sm.GLM(
        successes / trials, X, family=sm.families.Binomial(), var_weights=trials
    ).fit()
    np.testing.assert_allclose(model.coef_, result.params, rtol=1e-6)
    np.testing.assert_allclose(model.coef_standard_error_, result.bse, rtol=1e-6)
    assert model.deviance_ == pytest.approx(result.deviance, rel=1e-6)
    expanded_y = np.concatenate(
        [np.r_[np.ones(k), np.zeros(m - k)] for k, m in zip(successes, trials)]
    )
    expanded = GLM(Bernoulli()).fit(
        np.repeat(X, trials, axis=0), expanded_y, tol=FIT_TOL
    )
    np.testing.assert_allclose(model.coef_, expanded.coef_, rtol=1e-8)


def test_score_uses_offset_and_weights(weights: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = simulate(Poisson())
    frequency, precision = weights
    offset = np.full(N, 0.2)
    model = GLM(Poisson()).fit(X, y)
    μ = model.predict(X, offset=offset)
    expected = Poisson().deviance(y, μ, frequency * precision)
    score = model.score(
        X,
        y,
        offset=offset,
        frequency_weights=frequency,
        precision_weights=precision,
    )
    assert score == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["frequency_weights", "precision_weights"])
def test_invalid_weights_raise(kind: str) -> None:
    X, y = simulate(Poisson())
    negative = np.ones(N)
    negative[0] = -1.0
    with pytest.raises(ValueError, match=f"{kind} must be non-negative"):
        GLM(Poisson()).fit(X, y, **{kind: negative})
    with pytest.raises(ValueError, match=f"{kind} and y are not the same length"):
        GLM(Poisson()).fit(X, y, **{kind: np.ones(N - 1)})


def test_school_averages_need_precision_weights() -> None:
    # Students' scores vary independently around their school's mean, and we
    # only see each school's average score. The average of m scores has
    # variance σ² / m: one observation, more precise for a bigger school.
    # Precision weights reproduce the fit to the students' own scores;
    # frequency weights treat each school as m students who all scored its
    # average, and so understate the standard errors.
    rng = np.random.default_rng(154)
    n_schools = 50
    x = rng.normal(size=n_schools)
    sizes = rng.integers(50, 150, size=n_schools)
    school = np.repeat(np.arange(n_schools), sizes)
    scores = 60 + 5 * x[school] + rng.normal(scale=10.0, size=sizes.sum())
    averages = np.bincount(school, weights=scores) / sizes

    X = np.column_stack([np.ones(n_schools), x])
    students = GLM(Gaussian()).fit(X[school], scores, tol=FIT_TOL)
    weights = sizes.astype(np.float64)
    precision = GLM(Gaussian()).fit(X, averages, precision_weights=weights)
    frequency = GLM(Gaussian()).fit(X, averages, frequency_weights=weights)

    # The school averages are sufficient for the coefficients.
    np.testing.assert_allclose(precision.coef_, students.coef_)
    np.testing.assert_allclose(frequency.coef_, students.coef_)
    # The students' deviance splits into the variation within schools, which
    # doesn't depend on the coefficients, and the weighted school deviance.
    within_schools = np.sum((scores - averages[school]) ** 2)
    assert students.deviance_ == pytest.approx(precision.deviance_ + within_schools)
    # Only frequency weights count students as observations.
    assert precision.n == n_schools
    assert frequency.n == sizes.sum()
    # So the precision fit's standard errors are about the students' fit's.
    # It estimates σ² from only 48 residual degrees of freedom, so its
    # standard errors vary by about 10% from sample to sample: allow 30%. The
    # frequency fit's are smaller by the root of the ratio of residual degrees
    # of freedom: about ten times too small.
    np.testing.assert_allclose(
        precision.coef_standard_error_, students.coef_standard_error_, rtol=0.3
    )
    np.testing.assert_allclose(
        frequency.coef_standard_error_,
        precision.coef_standard_error_ * np.sqrt((n_schools - 2) / (sizes.sum() - 2)),
    )
