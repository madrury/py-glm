import numpy as np
import pandas as pd
import pytest
import scipy.stats as sts
import statsmodels.api as sm

from glm.datasets import make_frame
from glm.families import ExponentialFamily, Gamma, Gaussian, Poisson, QuasiPoisson
from glm.glm import GLM
from glm.hypothesis import ConstraintEquation, WaldTestResult, _wald


def _frame(family: ExponentialFamily, seed: int = 1) -> pd.DataFrame:
    """Three continuous predictors and a three level categorical."""
    sim = make_frame(
        family,
        n_samples=500,
        n_uncorrelated=3,
        categorical={"g": 3},
        coef_range=(-0.4, 0.4),
        rng=np.random.default_rng(seed),
    )
    return sim.data


def _W(model: GLM, constraint: ConstraintEquation) -> float:
    """The Wald statistic, computed here from its definition."""
    assert model.coef_ is not None
    C, c = constraint.C, constraint.c
    r = C @ model.coef_ - c
    return float(r @ np.linalg.solve(C @ model.coef_covariance_matrix_ @ C.T, r))


def _classical_F(full: GLM, null: GLM) -> tuple[float, float]:
    """The F statistic of nested Gaussian models, from their residual sums of
    squares, and its p-value."""
    assert full.deviance_ is not None and null.deviance_ is not None
    assert full.n is not None and full.p is not None and null.p is not None
    q = full.p - null.p
    rdof = full.n - full.p
    F = ((null.deviance_ - full.deviance_) / q) / (full.deviance_ / rdof)
    return F, float(sts.f(dfn=q, dfd=rdof).sf(F))


# 7.1: a single coefficient's test is the coefficient table's test.
@pytest.mark.parametrize("family", [Poisson(), Gaussian(), Gamma()], ids=repr)
def test_single_coefficient_matches_coefficient_table(
    family: ExponentialFamily,
) -> None:
    df = _frame(family)
    model = GLM(family).fit(df, formula="y ~ x1 + x2 + C(g)")
    assert model.X_names is not None
    j = model.X_names.index("x1")
    result = model.tests.wald("x1 = 0")
    assert result.q == 1
    assert result.p_value == pytest.approx(model.p_values_[j], rel=1e-12)
    # z² or t², which is also W/1 in the F case.
    assert model.coef_ is not None
    z = model.coef_[j] / model.coef_standard_error_[j]
    assert result.statistic == pytest.approx(z**2, rel=1e-12)


# 7.2: for a Gaussian model, W/q is the classical F statistic.
def test_gaussian_matches_classical_F() -> None:
    df = _frame(Gaussian())
    full = GLM(Gaussian()).fit(df, formula="y ~ x1 + x2 + x3 + C(g)")
    null = GLM(Gaussian()).fit(df, formula="y ~ x1 + C(g)")
    F, p_value = _classical_F(full, null)
    result = full.tests.wald("x2 = 0, x3 = 0")
    assert result.statistic == pytest.approx(F, rel=1e-10)
    assert result.p_value == pytest.approx(p_value, rel=1e-8)


# 7.3: agreement with statsmodels' wald_test.
@pytest.mark.parametrize(
    "family, sm_family, use_f",
    [
        (Poisson(), sm.families.Poisson(), False),
        (Gamma(), sm.families.Gamma(sm.families.links.Log()), True),
    ],
    ids=["Poisson", "Gamma"],
)
def test_matches_statsmodels(
    family: ExponentialFamily, sm_family: sm.families.Family, use_f: bool
) -> None:
    df = _frame(family)
    model = GLM(family).fit(df, formula="y ~ x1 + x2 + x3 + C(g)", tol=1e-12)
    hypothesis = "x1 = x2, x3 = 0.1, `C(g)[T.b]` = `C(g)[T.c]`"
    result = model.tests.wald(hypothesis)
    # statsmodels estimates the dispersion with Pearson's statistic by
    # default; ours is deviance based. Poisson's dispersion is fixed at 1.
    scale = "dev" if family.has_dispersion else 1.0
    sm_result = sm.GLM(
        model.response_vector(df), model.design_matrix(df), family=sm_family
    ).fit(scale=scale, tol=1e-12)
    expected = sm_result.wald_test(
        (result.constraint.C, result.constraint.c), use_f=use_f, scalar=True
    )
    assert result.statistic == pytest.approx(float(expected.statistic), rel=1e-6)
    assert result.p_value == pytest.approx(float(expected.pvalue), rel=1e-5)


# 7.4: a null model that drops terms, with A and C built by hand.
def test_dropping_terms_matches_hand_built_constraints() -> None:
    df = _frame(Poisson())
    model = GLM(Poisson()).fit(df, formula="y ~ x1 + x2 + C(g)")
    assert model.X_names == ["Intercept", "x1", "x2", "C(g)[T.b]", "C(g)[T.c]"]
    I = np.eye(5)
    # The null, y ~ x1, keeps the intercept and x1: X₀ = XA.
    A = I[:, [0, 1]]
    # C's rows span the null space of Aᵗ: here, the dropped columns.
    C = I[[2, 3, 4], :]
    assert np.allclose(C @ A, 0.0)
    by_null = model.tests.wald(null="y ~ x1", X=df)
    by_matrix = model.tests.wald(constraint=ConstraintEquation.from_arrays(C, None))
    by_string = model.tests.wald("x2 = 0, `C(g)[T.b]` = 0, `C(g)[T.c]` = 0")
    for result in [by_matrix, by_string]:
        assert result.q == by_null.q == 3
        assert result.statistic == pytest.approx(by_null.statistic, rel=1e-10)
        assert result.p_value == pytest.approx(by_null.p_value, rel=1e-8)


# 7.5: W depends only on the hypothesis, not on how its equations are written.
def test_invariant_to_recombining_the_equations() -> None:
    df = _frame(Poisson())
    model = GLM(Poisson()).fit(df, formula="y ~ x1 + x2 + x3")
    constraint = ConstraintEquation.from_constraint_formula(
        "x1 = x2, x3 = 0.1", names=["Intercept", "x1", "x2", "x3"]
    )
    M = np.array([[2.0, 1.0], [-1.0, 3.0]])
    recombined = ConstraintEquation.from_arrays(M @ constraint.C, M @ constraint.c)
    original = model.tests.wald(constraint=constraint)
    result = model.tests.wald(constraint=recombined)
    assert result.statistic == pytest.approx(original.statistic, rel=1e-10)
    assert result.statistic == pytest.approx(_W(model, constraint), rel=1e-10)


# 7.6: a nesting that isn't a column selection: a spline against a line.
def test_spline_against_linear_matches_classical_F() -> None:
    df = _frame(Gaussian())
    full = GLM(Gaussian()).fit(df, formula="y ~ bs(x1, df=5) + x2")
    null = GLM(Gaussian()).fit(df, formula="y ~ x1 + x2")
    F, p_value = _classical_F(full, null)
    result = full.tests.wald(null="y ~ x1 + x2", X=df)
    assert result.q == 4
    assert result.statistic == pytest.approx(F, rel=1e-8)
    assert result.p_value == pytest.approx(p_value, rel=1e-6)


# The F reference needs W/q, not W.
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
    model = GLM(family).fit(
        sim.data,
        formula=(
            'y ~ cs(x1, df=5, constraints="center")'
            ' + cs(x2, df=5, constraints="center")'
        ),
    )
    constraint = ConstraintEquation.from_null_model_formula(
        model, "y ~ x1 + x2", sim.data
    )
    return _wald(model, constraint), _W(model, constraint)


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


# 7.7: errors.
@pytest.fixture
def poisson_frame() -> pd.DataFrame:
    return _frame(Poisson())


@pytest.fixture
def poisson_model(poisson_frame: pd.DataFrame) -> GLM:
    return GLM(Poisson()).fit(poisson_frame, formula="y ~ x1 + x2 + C(g)")


def test_chained_equals_raises(poisson_model: GLM) -> None:
    with pytest.raises(ValueError, match="chains"):
        poisson_model.tests.wald("x1 = x2 = 0")


def test_unknown_name_raises_listing_the_coefficients(poisson_model: GLM) -> None:
    with pytest.raises(ValueError, match=r"'x9'.*C\(g\)\[T\.b\]"):
        poisson_model.tests.wald("x9 = 0")


def test_backticked_name_with_equals_and_comma(poisson_frame: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(poisson_frame, formula="y ~ bs(x1, df=3) + x2")
    assert model.X_names is not None
    name = model.X_names[1]
    assert "=" in name and "," in name
    result = model.tests.wald(f"`{name}` = 0")
    assert result.q == 1


def test_rank_deficient_constraint_raises() -> None:
    C = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    with pytest.raises(ValueError, match="not full rank"):
        ConstraintEquation.from_arrays(C, None)


def test_misaligned_constraint_arrays_raise() -> None:
    with pytest.raises(ValueError, match="unaligned"):
        ConstraintEquation.from_arrays(np.eye(2, 3), np.zeros(3))


def test_wrong_column_count_raises(poisson_model: GLM) -> None:
    constraint = ConstraintEquation.from_arrays(np.eye(1, 3), None)
    with pytest.raises(ValueError, match="3 columns.*5 coefficients"):
        poisson_model.tests.wald(constraint=constraint)


def test_non_nested_null_raises(
    poisson_model: GLM, poisson_frame: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="not nested"):
        poisson_model.tests.wald(null="y ~ x1 + x3", X=poisson_frame)


def test_equivalent_null_raises(
    poisson_model: GLM, poisson_frame: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="equivalent"):
        poisson_model.tests.wald(null="y ~ C(g) + x2 + x1", X=poisson_frame)


def test_different_data_raises(poisson_model: GLM, poisson_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="differs from the training design"):
        poisson_model.tests.wald(null="y ~ x1", X=poisson_frame.iloc[::-1])


def test_regularized_model_raises(poisson_frame: pd.DataFrame) -> None:
    model = GLM(Poisson(), alpha=1.0).fit(poisson_frame, formula="y ~ x1 + x2")
    with pytest.raises(ValueError, match="Penalized"):
        model.tests.wald("x1 = 0")


def test_unfit_model_raises() -> None:
    with pytest.raises(ValueError, match="fit model"):
        GLM(Poisson()).tests.wald("x1 = 0")


def test_null_on_array_model_raises(
    poisson_model: GLM, poisson_frame: pd.DataFrame
) -> None:
    model = GLM(Poisson()).fit(
        poisson_model.design_matrix(poisson_frame),
        poisson_model.response_vector(poisson_frame),
    )
    with pytest.raises(ValueError, match="model formula"):
        model.tests.wald(null="y ~ x1", X=poisson_frame)


def test_null_without_X_raises(poisson_model: GLM) -> None:
    with pytest.raises(ValueError, match="requires X"):
        poisson_model.tests.wald(null="y ~ x1")


def test_unneeded_X_warns(poisson_model: GLM, poisson_frame: pd.DataFrame) -> None:
    with pytest.warns(UserWarning, match="`hypothesis`"):
        poisson_model.tests.wald("x1 = 0", X=poisson_frame)
    constraint = ConstraintEquation.from_arrays(np.eye(1, 5, 1), None)
    with pytest.warns(UserWarning, match="`constraint`"):
        poisson_model.tests.wald(constraint=constraint, X=poisson_frame)


@pytest.mark.filterwarnings("error")
def test_null_with_X_does_not_warn(
    poisson_model: GLM, poisson_frame: pd.DataFrame
) -> None:
    poisson_model.tests.wald(null="y ~ x1", X=poisson_frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"hypothesis": "x1 = 0", "null": "y ~ x1"},
        {"hypothesis": "x1 = 0", "constraint": "placeholder"},
    ],
    ids=["none", "hypothesis and null", "hypothesis and constraint"],
)
def test_exactly_one_hypothesis_required(
    poisson_model: GLM, poisson_frame: pd.DataFrame, kwargs: dict[str, object]
) -> None:
    if "constraint" in kwargs:
        kwargs["constraint"] = ConstraintEquation.from_arrays(np.eye(1, 5, 1), None)
    with pytest.raises(ValueError, match="Exactly one"):
        poisson_model.tests.wald(X=poisson_frame, **kwargs)  # type: ignore[arg-type]


# 7.8: calibration. Under a true null, p-values are roughly uniform.
def test_p_values_are_uniform_under_the_null() -> None:
    rng = np.random.default_rng(154)
    n, n_sim = 200, 300
    p_values = np.empty(n_sim)
    for i in range(n_sim):
        df = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x1", "x2", "x3"])
        df["y"] = rng.poisson(np.exp(0.5 + 0.3 * df["x1"]))
        model = GLM(Poisson()).fit(df, formula="y ~ x1 + x2 + x3")
        p_values[i] = model.tests.wald("x2 = 0, x3 = 0").p_value
    assert sts.kstest(p_values, "uniform").pvalue > 0.01
