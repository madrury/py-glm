import numpy as np
import pytest

from glm.families import Bernoulli, Poisson
from glm.glm import GLM
from glm.standardize import Standardizer
from glm.types import Standardization

N = 500
STANDARDIZATIONS: list[Standardization] = ["none", "continuous", "all"]


@pytest.fixture
def X() -> np.ndarray:
    """An intercept, two continuous columns on different scales, an indicator
    column, and a constant column."""
    rng = np.random.default_rng(154)
    return np.column_stack(
        [
            np.ones(N),
            rng.normal(loc=5.0, scale=2.0, size=N),
            rng.normal(loc=-1.0, scale=100.0, size=N),
            rng.binomial(1, 0.3, size=N),
            np.full(N, 3.0),
        ]
    ).astype(np.float64)


@pytest.fixture
def weights() -> np.ndarray:
    return np.random.default_rng(154).uniform(0.5, 2.0, size=N)


def test_fit_uses_weighted_means_and_standard_deviations(
    X: np.ndarray, weights: np.ndarray
) -> None:
    standardizer = Standardizer("all").fit(X, weights)
    means = np.average(X, axis=0, weights=weights)
    sds = np.sqrt(np.average((X - means) ** 2, axis=0, weights=weights))
    np.testing.assert_allclose(standardizer.centers_, [0, *means[1:4], 0])
    np.testing.assert_allclose(standardizer.scales_, [1, *sds[1:4], 1])


@pytest.mark.parametrize(
    "standardize, transformed",
    [
        ("none", [False, False, False, False, False]),
        ("continuous", [False, True, True, False, False]),
        ("all", [False, True, True, True, False]),
    ],
)
def test_which_columns_are_standardized(
    X: np.ndarray, standardize: Standardization, transformed: list[bool]
) -> None:
    # The intercept and the constant column are never transformed, and
    # "continuous" leaves the indicator column alone.
    Z = Standardizer(standardize).fit(X).transform(X)
    for j, is_transformed in enumerate(transformed):
        if is_transformed:
            assert Z[:, j].mean() == pytest.approx(0.0, abs=1e-10)
            assert Z[:, j].std() == pytest.approx(1.0)
        else:
            np.testing.assert_array_equal(Z[:, j], X[:, j])


def test_no_centering_without_intercept(X: np.ndarray) -> None:
    # Without an intercept to absorb it, centering would change the model.
    X_no_intercept = X[:, 1:]
    standardizer = Standardizer("all").fit(X_no_intercept)
    assert standardizer.centers_ is not None and standardizer.scales_ is not None
    np.testing.assert_array_equal(standardizer.centers_, 0.0)
    # Column 0 is an ordinary predictor, so it is scaled like the others.
    np.testing.assert_allclose(
        standardizer.scales_[:3], X_no_intercept[:, :3].std(axis=0)
    )


@pytest.mark.parametrize("standardize", STANDARDIZATIONS)
def test_coefficient_conversions_without_intercept(
    X: np.ndarray, standardize: Standardization
) -> None:
    X_no_intercept = X[:, 1:]
    standardizer = Standardizer(standardize).fit(X_no_intercept)
    β = np.array([-1.0, 0.02, 2.0, 0.3])
    γ = standardizer.coef_to_standardized(β)
    np.testing.assert_allclose(
        standardizer.transform(X_no_intercept) @ γ, X_no_intercept @ β
    )
    np.testing.assert_allclose(standardizer.coef_to_original(γ), β)


@pytest.mark.parametrize("standardize", STANDARDIZATIONS)
def test_coefficient_conversions(X: np.ndarray, standardize: Standardization) -> None:
    standardizer = Standardizer(standardize).fit(X)
    β = np.array([0.5, -1.0, 0.02, 2.0, 0.3])
    γ = standardizer.coef_to_standardized(β)
    # Both give the same linear predictor, and the conversions are inverses.
    np.testing.assert_allclose(standardizer.transform(X) @ γ, X @ β)
    np.testing.assert_allclose(standardizer.coef_to_original(γ), β)


def test_invalid_standardization_raises() -> None:
    with pytest.raises(ValueError, match="standardize must be one of"):
        Standardizer("continous")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="standardize must be one of"):
        GLM(Poisson(), alpha=1.0, standardize="continous")  # type: ignore[arg-type]


def test_unfit_standardizer_raises(X: np.ndarray) -> None:
    with pytest.raises(ValueError, match="has not been fit"):
        Standardizer("all").transform(X)


@pytest.fixture
def response(X: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(154)
    η = 0.5 + 0.3 * (X[:, 1] - 5) / 2 - 0.005 * X[:, 2] + 0.4 * X[:, 3]
    return rng.binomial(1, 1 / (1 + np.exp(-η))).astype(np.float64)


@pytest.fixture
def X_without_constant(X: np.ndarray) -> np.ndarray:
    # A constant column is collinear with the intercept, so drop it to fit.
    return X[:, :4]


def test_unpenalized_fit_ignores_standardization(
    X_without_constant: np.ndarray, response: np.ndarray
) -> None:
    fits = [
        GLM(Bernoulli(), standardize=standardize).fit(X_without_constant, response)
        for standardize in STANDARDIZATIONS
    ]
    for model in fits[1:]:
        np.testing.assert_array_equal(model.coef_, fits[0].coef_)


@pytest.mark.parametrize("standardize", ["continuous", "all"])
def test_penalized_fit_does_not_depend_on_units(
    X_without_constant: np.ndarray,
    response: np.ndarray,
    standardize: Standardization,
) -> None:
    # Measuring a predictor in different units, e.g. kilometres rather than
    # metres, rescales its coefficient, and otherwise changes nothing.
    X_rescaled = X_without_constant.copy()
    X_rescaled[:, 1] /= 1000
    model = GLM(Bernoulli(), alpha=50.0, standardize=standardize)
    original = model.clone().fit(X_without_constant, response, tol=1e-12)
    rescaled = model.clone().fit(X_rescaled, response, tol=1e-12)
    assert original.coef_ is not None and rescaled.coef_ is not None
    expected = original.coef_.copy()
    expected[1] *= 1000
    np.testing.assert_allclose(rescaled.coef_, expected, rtol=1e-6)


def test_unstandardized_penalized_fit_depends_on_units(
    X_without_constant: np.ndarray, response: np.ndarray
) -> None:
    # Without standardizing, rescaling a predictor changes how hard it is
    # penalized, so the fit genuinely changes.
    X_rescaled = X_without_constant.copy()
    X_rescaled[:, 1] /= 1000
    model = GLM(Bernoulli(), alpha=50.0, standardize="none")
    original = model.clone().fit(X_without_constant, response)
    rescaled = model.clone().fit(X_rescaled, response)
    assert original.coef_ is not None and rescaled.coef_ is not None
    assert not np.allclose(rescaled.coef_[1], original.coef_[1] * 1000, rtol=0.01)


def test_warm_start_is_on_the_original_scale(
    X_without_constant: np.ndarray, response: np.ndarray
) -> None:
    model = GLM(Bernoulli(), alpha=50.0).fit(X_without_constant, response, tol=1e-12)
    assert model.coef_ is not None
    # Starting from the solution, on the original scale, stays there.
    refit = GLM(Bernoulli(), alpha=50.0).fit(
        X_without_constant, response, warm_start=model.coef_, max_iter=1
    )
    np.testing.assert_allclose(refit.coef_, model.coef_, rtol=1e-8)
