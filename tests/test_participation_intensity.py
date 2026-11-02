import numpy as np
import pandas as pd
import pytest

from glm.extras.participation_intensity import (
    ArgmaxGradient,
    CoverageSimulation,
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

# Means without an interior maximum: bβ > 0 has a minimum, and bβ = 0 has no
# critical point.
NON_MAXIMUM_MEANS = [
    ParticipationIntensityFunction(a=-1.0, b=5.0, α=-0.5, β=2.0),
    ParticipationIntensityFunction(a=-1.0, b=0.0, α=-0.5, β=-2.0),
    ParticipationIntensityFunction(a=-1.0, b=5.0, α=-0.5, β=0.0),
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


@pytest.mark.parametrize("mean", NON_MAXIMUM_MEANS)
def test_argmax_raises_without_a_maximum(mean: ParticipationIntensityFunction) -> None:
    with pytest.raises(ValueError, match="maximum only when"):
        mean.argmax()


@pytest.mark.parametrize("mean", NON_MAXIMUM_MEANS)
def test_argmax_gradient_raises_without_a_maximum(
    mean: ParticipationIntensityFunction,
) -> None:
    with pytest.raises(ValueError, match="maximum only when"):
        ArgmaxGradient(mean)


def _frame(**columns: list[float]) -> pd.DataFrame:
    defaults = {
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "d": [1.0, 0.0, 1.0, 1.0, 0.0],
        "y": [1.0, 0.0, 4.0, 5.0, 0.0],
    }
    return pd.DataFrame(defaults | columns)


def test_fit_accepts_a_minimal_frame() -> None:
    model = ParticipationIntensityModel().fit(_frame())
    assert model.mean_ is not None


@pytest.mark.parametrize("column", ["x", "d", "y"])
def test_fit_raises_on_a_missing_column(column: str) -> None:
    with pytest.raises(ValueError, match=f"column {column} is not present"):
        ParticipationIntensityModel().fit(_frame().drop(columns=column))


def test_fit_raises_on_a_non_binary_participation_column() -> None:
    with pytest.raises(ValueError, match="not a binary"):
        ParticipationIntensityModel().fit(_frame(d=[1.0, 0.0, 1.0, 1.0, 0.5]))


def test_fit_raises_on_too_few_participating_observations() -> None:
    with pytest.raises(ValueError, match="at least 3 are needed"):
        ParticipationIntensityModel().fit(_frame(d=[1.0, 0.0, 1.0, 0.0, 0.0]))


def test_fit_raises_when_every_observation_participates() -> None:
    with pytest.raises(ValueError, match="Every observation is participating"):
        ParticipationIntensityModel().fit(_frame(d=[1.0, 1.0, 1.0, 1.0, 1.0]))


@pytest.mark.parametrize("mean", NON_MAXIMUM_MEANS)
def test_coverage_simulation_raises_without_a_true_maximum(
    mean: ParticipationIntensityFunction,
) -> None:
    with pytest.raises(ValueError, match="maximum only when"):
        CoverageSimulation(mean)


def test_coverage_simulation_is_empty_before_sampling() -> None:
    simulation = CoverageSimulation(MEANS[0])
    assert simulation.models == []
    assert simulation.dropped_count == 0
    assert simulation.argmaxs_().shape == (0,)


def test_coverage_simulation_drops_unfittable_samples() -> None:
    np.random.seed(0)
    # With three points, a sample has either at most two participating
    # observations or no non-participating ones, so every sample is dropped.
    simulation = CoverageSimulation(MEANS[0])
    with pytest.warns(UserWarning, match="Dropping a sample"):
        simulation.sample(10, nxs=3)
    assert simulation.models == []
    assert simulation.dropped_count == 10


def test_coverage_simulation_sample_resets_its_state() -> None:
    np.random.seed(0)
    simulation = CoverageSimulation(MEANS[0])
    with pytest.warns(UserWarning):
        simulation.sample(10, nxs=3)
    simulation.sample(10, nxs=500)
    # A second call replaces the first: its kept and dropped samples account
    # for exactly the 10 it drew.
    assert len(simulation.models) + simulation.dropped_count == 10
    assert len(simulation.models) > 0
