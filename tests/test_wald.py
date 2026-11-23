import numpy as np
import pytest
import scipy.stats as sts

from glm.datasets import make_frame
from glm.families import ExponentialFamily, Poisson, QuasiPoisson
from glm.glm import GLM
from glm.hypothesis import ConstraintEquation, WaldTestResult, _wald

SPLINE_FORMULA = (
    'y ~ cs(x1, df=5, constraints="center") + cs(x2, df=5, constraints="center")'
)
LINEAR_FORMULA = "y ~ x1 + x2"


def _linearity_test(family: ExponentialFamily) -> tuple[WaldTestResult, float]:
    """Test that two spline effects are linear, when the truth is linear, and
    return the result with W, computed here from its definition."""
    sim = make_frame(
        Poisson(),
        n_samples=1000,
        n_uncorrelated=2,
        coef_range=(-0.4, 0.4),
        rng=np.random.default_rng(1),
    )
    model = GLM(family).fit(sim.data, formula=SPLINE_FORMULA)
    constraint = ConstraintEquation.from_null_model_formula(
        model, LINEAR_FORMULA, sim.data
    )
    result = _wald(model, constraint)
    assert model.coef_ is not None
    C, c = constraint.C, constraint.c
    r = C @ model.coef_ - c
    W = float(r @ np.linalg.solve(C @ model.coef_covariance_matrix_ @ C.T, r))
    return result, W


def test_known_dispersion_refers_W_to_chi2() -> None:
    result, W = _linearity_test(Poisson())
    assert result.q == 8
    assert result.p_value == pytest.approx(sts.chi2(df=result.q).sf(W))


def test_estimated_dispersion_scales_W_for_the_F_reference() -> None:
    # With large residual degrees of freedom, q·F(q, n - p) is close to χ²_q,
    # so the F test's p-value must be close to that of W against χ²_q. Referring
    # W itself to F, instead of W/q, gives a p-value orders of magnitude
    # smaller.
    result, W = _linearity_test(QuasiPoisson())
    assert result.q == 8
    assert result.p_value == pytest.approx(sts.chi2(df=result.q).sf(W), abs=0.01)
