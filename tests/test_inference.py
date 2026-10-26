import numpy as np
import pandas as pd
import pytest
import scipy.stats as sts
import statsmodels.api as sm

from glm.datasets import make_frame, make_regression
from glm.families import (
    Bernoulli,
    ExponentialFamily,
    Exponential,
    Gamma,
    Gaussian,
    Poisson,
    QuasiPoisson,
)
from glm.glm import GLM

N = 200
FIT_TOL = 1e-12


@pytest.mark.parametrize(
    "family, statistic",
    [
        (Gaussian(), "t"),
        (Gamma(), "t"),
        (QuasiPoisson(), "t"),
        (Bernoulli(), "z"),
        (Poisson(), "z"),
        (Exponential(), "z"),
    ],
    ids=repr,
)
def test_coefficient_test_follows_the_dispersion(
    family: ExponentialFamily, statistic: str
) -> None:
    test = family.coefficient_test(residual_df=50)
    assert test.statistic == statistic
    expected = sts.t(df=50) if statistic == "t" else sts.norm()
    assert test.distribution.ppf(0.975) == pytest.approx(expected.ppf(0.975))


def fit_both(family: ExponentialFamily, seed: int):
    """Fit py-glm and the matching statsmodels model, with the reference
    distribution py-glm uses for the family."""
    data = make_regression(
        family,
        n_samples=N,
        n_uncorrelated=3,
        coef_range=(-0.3, 0.3),
        dispersion=0.5,
        rng=np.random.default_rng(seed),
    )
    X, y = data.X, data.y
    model = GLM(family).fit(X, y, tol=FIT_TOL)
    if isinstance(family, Gaussian):
        result = sm.OLS(y, X).fit()
    elif isinstance(family, Bernoulli):
        result = sm.Logit(y, X).fit(disp=0)
    elif isinstance(family, Poisson):
        result = sm.Poisson(y, X).fit(disp=0)
    else:
        # Our dispersion is deviance-based, and a t reference distribution.
        result = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log())).fit(
            scale="dev", use_t=True
        )
    return model, result


@pytest.mark.parametrize(
    "family", [Gaussian(), Bernoulli(), Poisson(), Gamma()], ids=repr
)
def test_summary_frame_matches_statsmodels(family: ExponentialFamily) -> None:
    model, result = fit_both(family, seed=154)
    frame = model.summary_frame(level=0.9)
    statistic = model.family.coefficient_test(1).statistic
    np.testing.assert_allclose(frame["estimate"], result.params, rtol=1e-6)
    np.testing.assert_allclose(frame["std error"], result.bse, rtol=1e-6)
    np.testing.assert_allclose(frame[statistic], result.tvalues, rtol=1e-6)
    np.testing.assert_allclose(frame["p-value"], result.pvalues, rtol=1e-5, atol=1e-12)
    intervals = np.asarray(result.conf_int(alpha=0.1))
    np.testing.assert_allclose(frame["lower 90%"], intervals[:, 0], rtol=1e-6)
    np.testing.assert_allclose(frame["upper 90%"], intervals[:, 1], rtol=1e-6)


def test_p_values_match_summary_frame() -> None:
    model, _ = fit_both(Gamma(), seed=154)
    np.testing.assert_array_equal(model.p_values_, model.summary_frame()["p-value"])


@pytest.fixture
def formula_model() -> GLM:
    sim = make_frame(
        Poisson(),
        n_samples=N,
        n_uncorrelated=2,
        categorical={"region": 3},
        coef_range=(-0.3, 0.3),
        rng=np.random.default_rng(154),
    )
    return GLM(Poisson()).fit(sim.data, formula=sim.formula)


def test_summary_frame_layout(formula_model: GLM) -> None:
    frame = formula_model.summary_frame()
    assert list(frame.index) == formula_model.X_names
    assert list(frame.columns) == [
        "estimate",
        "std error",
        "z",
        "p-value",
        "lower 95%",
        "upper 95%",
    ]


def test_regularized_summary_frame_has_only_estimates() -> None:
    sim = make_frame(Poisson(), n_samples=N, rng=np.random.default_rng(154))
    model = GLM(Poisson(), alpha=1.0).fit(sim.data, formula=sim.formula)
    assert list(model.summary_frame().columns) == ["estimate"]


@pytest.mark.parametrize(
    "attribute", ["coef_covariance_matrix_", "coef_standard_error_", "p_values_"]
)
def test_regularized_inference_raises(attribute: str) -> None:
    sim = make_frame(Poisson(), n_samples=N, rng=np.random.default_rng(154))
    model = GLM(Poisson(), alpha=1.0).fit(sim.data, formula=sim.formula)
    with pytest.raises(ValueError, match="not available for regularized models"):
        getattr(model, attribute)


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
def test_invalid_level_raises(formula_model: GLM, level: float) -> None:
    with pytest.raises(ValueError, match="level must be between 0 and 1"):
        formula_model.summary_frame(level=level)


def test_summary_prints_the_frame_without_stars(
    formula_model: GLM, capsys: pytest.CaptureFixture[str]
) -> None:
    formula_model.summary()
    printed = capsys.readouterr().out
    assert printed.startswith("Poisson GLM Model Summary.")
    assert "*" not in printed
    for name in formula_model.X_names or []:
        assert name in printed


def test_unfit_model_raises() -> None:
    with pytest.raises(ValueError, match="fit model"):
        GLM(Poisson()).summary_frame()


def test_summary_frame_is_a_data_frame(formula_model: GLM) -> None:
    assert isinstance(formula_model.summary_frame(), pd.DataFrame)
