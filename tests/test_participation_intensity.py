import numpy as np
import pytest

from glm.extras.participation_intensity import (
    ArgmaxGradient,
    ParticipationIntensityFunction,
    ParticipationIntensityModel,
    sample_data,
)

# Means with an interior maximum, which needs bβ < 0.
MEANS = [
    ParticipationIntensityFunction(a=-1.0, b=5.0, α=-0.5, β=-2.0),
    ParticipationIntensityFunction(a=-0.05, b=0.55, α=-0.6, β=-1.0),
    ParticipationIntensityFunction(a=2.0, b=-1.0, α=-1.0, β=3.0),
]


@pytest.mark.parametrize("mean", MEANS)
def test_argmax_matches_grid_search(mean: ParticipationIntensityFunction) -> None:
    argmax = mean.argmax()
    xs = np.linspace(argmax - 5.0, argmax + 5.0, num=1_000_001)
    grid_argmax = xs[np.argmax(mean(xs))]
    assert argmax == pytest.approx(grid_argmax, abs=1e-5)


def test_argmax_is_finite_where_exp_overflows() -> None:
    # λ = α - 1 - aβ/b = 799, so exp(λ) overflows a float.
    mean = ParticipationIntensityFunction(a=0.0, b=1.0, α=800.0, β=-1.0)
    assert np.isfinite(mean.argmax())


@pytest.mark.parametrize("mean", MEANS)
def test_argmax_partials_match_finite_differences(
    mean: ParticipationIntensityFunction,
) -> None:
    θ = np.array([mean.a, mean.b, mean.α, mean.β])
    h = 1e-6
    finite_differences = np.empty(4)
    for i in range(4):
        e = np.zeros(4)
        e[i] = h
        upper = ParticipationIntensityFunction(*(θ + e)).argmax()
        lower = ParticipationIntensityFunction(*(θ - e)).argmax()
        finite_differences[i] = (upper - lower) / (2 * h)
    np.testing.assert_allclose(
        ArgmaxGradient(mean).gradient(), finite_differences, rtol=1e-6, atol=1e-8
    )


def test_models_do_not_share_glms() -> None:
    first = ParticipationIntensityModel()
    second = ParticipationIntensityModel()
    assert first.participation is not second.participation
    assert first.intensity is not second.intensity


def test_argmax_standard_error_uses_its_own_fit() -> None:
    np.random.seed(0)
    mean = MEANS[0]
    small = ParticipationIntensityModel().fit(
        sample_data(mean, np.linspace(0.0, 3.0, num=500))
    )
    small_se = small.argmax_se_()
    # Fitting another, larger, model must leave the first one's standard error
    # alone.
    ParticipationIntensityModel().fit(
        sample_data(mean, np.linspace(0.0, 3.0, num=50_000))
    )
    assert small.argmax_se_() == small_se
