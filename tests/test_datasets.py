import dataclasses

import numpy as np
import pytest

from glm.datasets import make_regression
from glm.families import Bernoulli, Gamma, Gaussian, Poisson
from glm.glm import GLM


def test_shapes_and_intercept() -> None:
    data = make_regression(
        Gaussian(),
        n_samples=200,
        n_uncorrelated=3,
        n_correlated=2,
        rng=np.random.default_rng(154),
    )
    X, y, coef = data.X, data.y, data.coef
    assert X.shape == (200, 6) and y.shape == (200,) and coef.shape == (6,)
    np.testing.assert_array_equal(X[:, 0], 1.0)


def test_without_intercept() -> None:
    data = make_regression(
        Gaussian(), n_uncorrelated=3, intercept=False, rng=np.random.default_rng(154)
    )
    X, coef = data.X, data.coef
    assert X.shape[1] == coef.shape[0] == 3
    assert not np.all(X[:, 0] == 1.0)


def test_correlated_predictors_are_standardized_and_correlated() -> None:
    data = make_regression(
        Gaussian(),
        n_samples=5000,
        n_uncorrelated=0,
        n_correlated=4,
        rng=np.random.default_rng(154),
    )
    X = data.X
    correlated = X[:, 1:]
    np.testing.assert_allclose(correlated.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(correlated.std(axis=0), 1.0)
    off_diagonal = np.corrcoef(correlated.T)[~np.eye(4, dtype=bool)]
    assert np.max(np.abs(off_diagonal)) > 0.2


def test_coefficients() -> None:
    data = make_regression(
        Gaussian(),
        n_uncorrelated=8,
        n_zero_coefficients=3,
        coef_range=(0.5, 2.0),
        rng=np.random.default_rng(154),
    )
    coef = data.coef
    assert np.sum(coef == 0.0) == 3
    nonzero = coef[coef != 0.0]
    assert np.all((nonzero >= 0.5) & (nonzero <= 2.0))


def test_reproducible_with_rng() -> None:
    first = make_regression(Poisson(), rng=np.random.default_rng(154))
    second = make_regression(Poisson(), rng=np.random.default_rng(154))
    for field in ("X", "y", "coef"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_reproducible_with_global_seed() -> None:
    np.random.seed(154)
    first = make_regression(Poisson())
    np.random.seed(154)
    second = make_regression(Poisson())
    for field in ("X", "y", "coef"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_responses_follow_the_family() -> None:
    rng = np.random.default_rng(154)
    data = make_regression(Bernoulli(), rng=rng)
    y_binary = data.y
    assert set(np.unique(y_binary)) <= {0, 1}
    data = make_regression(Poisson(), coef_range=(-0.3, 0.3), rng=rng)
    y_count = data.y
    assert np.all(y_count >= 0) and np.all(y_count == np.round(y_count))
    data = make_regression(Gamma(), coef_range=(-0.3, 0.3), rng=rng)
    y_positive = data.y
    assert np.all(y_positive > 0)


def test_gaussian_dispersion_is_the_residual_variance() -> None:
    data = make_regression(
        Gaussian(), n_samples=20000, dispersion=4.0, rng=np.random.default_rng(154)
    )
    X, y, coef = data.X, data.y, data.coef
    assert np.var(y - X @ coef) == pytest.approx(4.0, rel=0.05)


def test_fit_recovers_coefficients() -> None:
    data = make_regression(
        Poisson(),
        n_samples=20000,
        n_uncorrelated=3,
        n_correlated=3,
        coef_range=(-0.3, 0.3),
        rng=np.random.default_rng(154),
    )
    X, y, coef = data.X, data.y, data.coef
    model = GLM(Poisson()).fit(X, y)
    z = (model.coef_ - coef) / model.coef_standard_error_
    # Seven coefficients, each with a 6e-5 chance of |z| >= 4 by chance alone,
    # so a false failure chance of about 4e-4.
    assert np.all(np.abs(z) < 4)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_uncorrelated": -1}, "n_uncorrelated"),
        ({"n_correlated": 1.5}, "n_correlated"),
        ({"n_zero_coefficients": 7}, "only 6 coefficients"),
        ({"dispersion": 0.0}, "dispersion must be positive"),
        ({"n_uncorrelated": 0, "intercept": False}, "at least one column"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_regression(Gaussian(), **kwargs)


def test_dataset_is_frozen() -> None:
    data = make_regression(Gaussian(), rng=np.random.default_rng(154))
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.coef = np.zeros(6)  # type: ignore[misc]
