import dataclasses

import numpy as np
import pandas as pd
import pytest

from glm.datasets import SimulatedFrame, make_frame, make_regression
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


def test_make_regression_with_categorical_variables() -> None:
    data = make_regression(
        Gaussian(),
        n_uncorrelated=2,
        categorical=[3, [0.5, 0.5]],
        rng=np.random.default_rng(154),
    )
    # Intercept, two continuous columns, then indicators for each level but
    # the reference: two for the first variable, one for the second.
    assert data.X.shape[1] == data.coef.shape[0] == 6
    indicators = data.X[:, 3:5]
    assert set(np.unique(indicators)) == {0.0, 1.0}
    assert np.all(indicators.sum(axis=1) <= 1)


@pytest.fixture
def frame() -> SimulatedFrame:
    return make_frame(
        Poisson(),
        n_samples=5000,
        n_uncorrelated=2,
        n_correlated=2,
        categorical={"region": 3, "plan": [0.6, 0.35, 0.05]},
        coef_range=(-0.5, 0.5),
        exposure_range=(1.0, 10.0),
        rng=np.random.default_rng(154),
    )


def test_make_frame_columns_and_formula(frame: SimulatedFrame) -> None:
    assert list(frame.data.columns) == [
        "x1",
        "x2",
        "x3",
        "x4",
        "region",
        "plan",
        "exposure",
        "y",
    ]
    assert frame.formula == "y ~ x1 + x2 + x3 + x4 + C(region) + C(plan)"
    assert sorted(frame.data["region"].unique()) == ["a", "b", "c"]
    assert frame.offset is not None
    np.testing.assert_allclose(frame.offset, np.log(frame.data["exposure"]))
    assert frame.mean.shape == (5000,)


def test_make_frame_level_probabilities(frame: SimulatedFrame) -> None:
    shares = frame.data["plan"].value_counts(normalize=True).sort_index()
    np.testing.assert_allclose(shares, [0.6, 0.35, 0.05], atol=0.02)


def test_make_frame_coefficients_match_a_fit(frame: SimulatedFrame) -> None:
    model = GLM(Poisson()).fit(frame.data, formula=frame.formula, offset=frame.offset)
    # The true coefficients are named as the fit names its coefficients.
    assert model.X_names == list(frame.coef.index)
    assert model.coef_ is not None
    z = (model.coef_ - frame.coef.to_numpy()) / model.coef_standard_error_
    # Eleven coefficients, each with a 6e-5 chance of |z| >= 4 by chance
    # alone, so a false failure chance of about 7e-4.
    assert np.all(np.abs(z) < 4)


def test_make_frame_latent_noise_overdisperses() -> None:
    def dispersion(latent_noise_sd: float) -> float:
        sim = make_frame(
            Poisson(),
            n_samples=5000,
            n_uncorrelated=2,
            coef_range=(-0.3, 0.3),
            latent_noise_sd=latent_noise_sd,
            rng=np.random.default_rng(154),
        )
        model = GLM(Poisson()).fit(sim.data, formula=sim.formula)
        μ = model.predict(sim.data)
        y = sim.data["y"].to_numpy()
        # Pearson's estimate of the dispersion. The deviance-based estimate,
        # model.dispersion_ for a QuasiPoisson, is biased upward for counts
        # this small, around 1.1 even without overdispersion.
        assert model.p is not None
        return float(np.sum((y - μ) ** 2 / μ) / (len(y) - model.p))

    # Pearson's estimate has a standard error of about sqrt(2 / n) = 0.02.
    assert dispersion(0.0) == pytest.approx(1.0, abs=0.06)
    assert dispersion(0.5) > 1.2


def test_make_frame_reproducible_with_rng() -> None:
    first = make_frame(Poisson(), categorical={"g": 3}, rng=np.random.default_rng(154))
    second = make_frame(Poisson(), categorical={"g": 3}, rng=np.random.default_rng(154))
    pd.testing.assert_frame_equal(first.data, second.data)


def test_level_names_sort_in_level_order() -> None:
    sim = make_frame(
        Gaussian(), n_samples=500, categorical={"g": 30}, rng=np.random.default_rng(154)
    )
    levels = sorted(sim.data["g"].unique())
    assert levels[0] == "level01" and len(levels) == 30
    model = GLM(Gaussian()).fit(sim.data, formula=sim.formula)
    assert model.X_names == list(sim.coef.index)


@pytest.mark.parametrize(
    "categorical, message",
    [
        ({"g": 1}, "at least 2 levels"),
        ({"g": [0.5, 0.6]}, "summing to 1"),
        ({"g": [1.0]}, "at least 2"),
        ({"x1": 3}, "is taken"),
        ({"y": 3}, "is taken"),
        ({"g": ["only"]}, "at least 2 level names"),
        ({"g": ["low", "low"]}, "must be distinct"),
        ({"g": {"low": 0.5, "high": 0.6}}, "summing to 1"),
        ({"g": True}, "can't be a bool"),
    ],
)
def test_make_frame_invalid_categorical_raises(categorical: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_frame(Gaussian(), categorical=categorical)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"latent_noise_sd": -1.0}, "latent_noise_sd"),
        ({"exposure_range": (0.0, 1.0)}, "exposure_range"),
    ],
)
def test_make_frame_invalid_arguments_raise(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_frame(Poisson(), **kwargs)


@pytest.mark.parametrize("correlation", [0.8, 0.0, -0.3])
def test_correlation(correlation: float) -> None:
    data = make_regression(
        Gaussian(),
        n_samples=20000,
        n_uncorrelated=0,
        n_correlated=3,
        correlation=correlation,
        rng=np.random.default_rng(154),
    )
    off_diagonal = np.corrcoef(data.X[:, 1:].T)[~np.eye(3, dtype=bool)]
    # A sample correlation's standard error is about (1 - ρ²) / sqrt(n) < 0.01.
    np.testing.assert_allclose(off_diagonal, correlation, atol=0.03)


@pytest.mark.parametrize("correlation", [1.0, -0.5, -1.0])
def test_invalid_correlation_raises(correlation: float) -> None:
    # With three correlated predictors, the correlation must exceed -1/2.
    with pytest.raises(ValueError, match="correlation must be between"):
        make_frame(Gaussian(), n_correlated=3, correlation=correlation)


def test_correlation_none_keeps_random_structure() -> None:
    # The default is unchanged, draw for draw.
    a = make_regression(Gaussian(), n_correlated=3, rng=np.random.default_rng(154))
    b = make_regression(
        Gaussian(), n_correlated=3, correlation=None, rng=np.random.default_rng(154)
    )
    np.testing.assert_array_equal(a.X, b.X)


@pytest.mark.parametrize(
    "levels",
    [
        ["Plains", "Mountain", "Swamp", "Island", "Forest"],
        {"Plains": 0.4, "Mountain": 0.3, "Swamp": 0.1, "Island": 0.1, "Forest": 0.1},
    ],
    ids=["names", "named probabilities"],
)
def test_make_frame_with_named_levels(levels: list[str] | dict[str, float]) -> None:
    names = list(levels)
    sim = make_frame(
        Poisson(),
        n_samples=2000,
        n_uncorrelated=1,
        categorical={"region": levels, "plan": ["Beginner", "Master"]},
        rng=np.random.default_rng(154),
    )
    region = sim.data["region"]
    assert list(region.cat.categories) == names
    assert set(region) == set(names)
    # The first level given is the reference, though it doesn't sort first.
    expected = (
        ["Intercept", "x1"]
        + [f"C(region)[T.{name}]" for name in names[1:]]
        + ["C(plan)[T.Master]"]
    )
    assert list(sim.coef.index) == expected
    model = GLM(Poisson()).fit(sim.data, formula=sim.formula)
    assert model.X_names == expected


def test_make_frame_named_level_probabilities() -> None:
    sim = make_frame(
        Gaussian(),
        n_samples=20_000,
        n_uncorrelated=0,
        categorical={"g": {"rare": 0.1, "common": 0.9}},
        rng=np.random.default_rng(154),
    )
    assert (sim.data["g"] == "rare").mean() == pytest.approx(0.1, abs=0.01)


def test_make_regression_accepts_named_levels() -> None:
    data = make_regression(
        Gaussian(),
        n_samples=100,
        n_uncorrelated=1,
        categorical=[["low", "medium", "high"]],
        rng=np.random.default_rng(154),
    )
    # An intercept, x1, and indicators for medium and high.
    assert data.X.shape == (100, 4)
