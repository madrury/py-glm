import numpy as np
import pandas as pd
import pytest
import scipy.stats as sts

from glm.families import Poisson
from glm.glm import GLM
from glm.simulation import Simulation, coefficient_table

N_SIM = 5


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(154)
    n = 200
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], size=n),
        }
    )
    df["y"] = rng.poisson(np.exp(0.5 + 0.3 * df["x"]))
    return df


def test_parametric_bootstrap_formula_model(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    models = Simulation(model).parametric_bootstrap(df, n_sim=N_SIM)
    assert len(models) == N_SIM
    for boot in models:
        assert boot.coef_ is not None and model.coef_ is not None
        assert boot.coef_.shape == model.coef_.shape
        assert boot.X_names == model.X_names
        assert boot.y_name == model.y_name
        # Bootstrap models keep the formula, so they predict from data frames.
        assert boot.predict(df).shape == (len(df),)


def test_parametric_bootstrap_array_model(df: pd.DataFrame) -> None:
    X = np.column_stack([np.ones(len(df)), df["x"]])
    y = df["y"].to_numpy()
    model = GLM(Poisson()).fit(X, y, X_names=["Intercept", "x"], y_name="y")
    models = Simulation(model).parametric_bootstrap(X, n_sim=N_SIM)
    assert len(models) == N_SIM
    for boot in models:
        assert boot.coef_ is not None
        assert boot.coef_.shape == (2,)
        assert boot.X_names == ["Intercept", "x"]
        assert boot.predict(X).shape == (len(df),)


@pytest.fixture
def exposure_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(154)
    n = 400
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    exposure = rng.uniform(1, 10, size=n)
    y = rng.poisson(exposure * np.exp(0.2 + 0.3 * X[:, 1])).astype(np.float64)
    return X, y, np.log(exposure)


def test_non_parametric_bootstrap_uses_offset(
    exposure_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, offset = exposure_data
    model = GLM(Poisson()).fit(X, y, offset=offset)
    assert model.coef_ is not None
    models = Simulation(model).non_parametric_bootstrap(
        X, y, n_sim=N_SIM, offset=offset
    )
    intercepts = np.array([boot.coef_[0] for boot in models if boot.coef_ is not None])
    # Ignoring the offset would shift each intercept by the mean log exposure
    # (about 1.6), so the bootstrap intercepts must sit far closer than that.
    assert np.all(np.abs(intercepts - model.coef_[0]) < np.mean(offset) / 4)


def test_non_parametric_bootstrap_resamples_weights_with_rows(
    exposure_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, _ = exposure_data
    # Zero-weight rows get extreme responses: they can only influence the fit
    # if a weight is attached to the wrong row after resampling.
    frequency_weights = np.ones(len(y))
    frequency_weights[::4] = 0.0
    y = y.copy()
    y[::4] = 1000.0
    model = GLM(Poisson()).fit(X, y, frequency_weights=frequency_weights)
    assert model.coef_ is not None
    models = Simulation(model).non_parametric_bootstrap(
        X, y, n_sim=N_SIM, frequency_weights=frequency_weights
    )
    for boot in models:
        assert boot.coef_ is not None
        assert np.all(np.abs(boot.coef_ - model.coef_) < 0.5)


@pytest.fixture
def array_model(df: pd.DataFrame) -> tuple[GLM, np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(df)), df["x"]])
    y = df["y"].to_numpy()
    return GLM(Poisson()).fit(X, y), X, y


# Each public method, called with valid data for an array model, plus overrides.
SIMULATION_METHODS = {
    "sample": lambda sim, X, y, **kw: sim.sample(X, **kw),
    "parametric_bootstrap": lambda sim, X, y, **kw: sim.parametric_bootstrap(X, **kw),
    "non_parametric_bootstrap": lambda sim, X, y, **kw: sim.non_parametric_bootstrap(
        X, y, **kw
    ),
}


def test_unfit_model_raises() -> None:
    with pytest.raises(ValueError, match="require a fit model"):
        Simulation(GLM(Poisson()))


@pytest.mark.parametrize("method", SIMULATION_METHODS)
@pytest.mark.parametrize("n_sim", [0, -1, 2.5, True])
def test_invalid_n_sim_raises(
    array_model: tuple[GLM, np.ndarray, np.ndarray], method: str, n_sim: object
) -> None:
    model, X, y = array_model
    with pytest.raises(ValueError, match="n_sim must be a positive integer"):
        SIMULATION_METHODS[method](Simulation(model), X, y, n_sim=n_sim)


@pytest.mark.parametrize("method", SIMULATION_METHODS)
def test_array_model_requires_array(
    array_model: tuple[GLM, np.ndarray, np.ndarray], method: str
) -> None:
    model, X, y = array_model
    with pytest.raises(TypeError, match="X must be a numpy array"):
        SIMULATION_METHODS[method](Simulation(model), pd.DataFrame(X), y, n_sim=2)


@pytest.mark.parametrize("method", SIMULATION_METHODS)
def test_formula_model_requires_data_frame(df: pd.DataFrame, method: str) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x")
    X = np.column_stack([np.ones(len(df)), df["x"]])
    with pytest.raises(TypeError, match="X must be a DataFrame"):
        SIMULATION_METHODS[method](Simulation(model), X, None, n_sim=2)


def test_non_parametric_bootstrap_array_model_requires_y(
    array_model: tuple[GLM, np.ndarray, np.ndarray],
) -> None:
    model, X, _ = array_model
    with pytest.raises(TypeError, match="y must be a numpy array"):
        Simulation(model).non_parametric_bootstrap(X, n_sim=2)


def test_non_parametric_bootstrap_formula_model_rejects_y(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x")
    with pytest.raises(ValueError, match="y must not be supplied"):
        Simulation(model).non_parametric_bootstrap(df, df["y"].to_numpy(), n_sim=2)


def test_non_parametric_bootstrap_formula_model_uses_training_specs(
    df: pd.DataFrame,
) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ center(x) + C(g)")
    assert model.coef_ is not None
    models = Simulation(model).non_parametric_bootstrap(df, n_sim=N_SIM)
    for boot in models:
        # Replicates share the original's specs, rather than re-learning
        # category levels and centering constants from their resample.
        assert boot.X_spec is model.X_spec and boot.y_spec is model.y_spec
        assert boot.X_names == model.X_names
        assert boot.coef_ is not None
        assert boot.coef_.shape == model.coef_.shape
        assert boot.predict(df).shape == (len(df),)


@pytest.fixture
def rare_category_df(df: pd.DataFrame) -> pd.DataFrame:
    # Category "c" appears once, so many resamples contain no "c" rows.
    df = df.copy()
    df["g"] = "a"
    df.loc[df.index[: len(df) // 2], "g"] = "b"
    df.loc[df.index[-1], "g"] = "c"
    return df


def test_non_parametric_bootstrap_rare_category_raises(
    rare_category_df: pd.DataFrame,
) -> None:
    model = GLM(Poisson()).fit(rare_category_df, formula="y ~ x + C(g)")
    with pytest.raises(ValueError, match=r"no observations .* \['C\(g\)\[T.c\]'\]"):
        Simulation(model).non_parametric_bootstrap(rare_category_df, n_sim=50)


def test_non_parametric_bootstrap_rare_category_ridge(
    rare_category_df: pd.DataFrame,
) -> None:
    # The ridge penalty keeps every coefficient estimable.
    model = GLM(Poisson(), alpha=1.0).fit(rare_category_df, formula="y ~ x + C(g)")
    assert model.coef_ is not None
    models = Simulation(model).non_parametric_bootstrap(rare_category_df, n_sim=50)
    for boot in models:
        assert boot.coef_ is not None
        assert boot.coef_.shape == model.coef_.shape


def test_non_parametric_bootstrap_rank_deficient_resample_raises(
    df: pd.DataFrame,
) -> None:
    # This column equals the intercept, except in one row. Resamples without
    # that row are rank deficient, though no column is all zero.
    almost_intercept = np.ones(len(df))
    almost_intercept[-1] = 0.0
    X = np.column_stack([np.ones(len(df)), df["x"], almost_intercept])
    model = GLM(Poisson()).fit(X, df["y"].to_numpy())
    with pytest.raises(ValueError, match="rank deficient"):
        Simulation(model).non_parametric_bootstrap(X, df["y"].to_numpy(), n_sim=50)


BOOTSTRAPS = {
    "parametric_bootstrap": lambda sim, df, **kw: sim.parametric_bootstrap(df, **kw),
    "non_parametric_bootstrap": lambda sim, df, **kw: sim.non_parametric_bootstrap(
        df, **kw
    ),
}


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS)
def test_bootstrap_validates_intercept_by_default(
    df: pd.DataFrame, bootstrap: str
) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x - 1", validate_intercept=False)
    with pytest.raises(ValueError, match="not an intercept"):
        BOOTSTRAPS[bootstrap](Simulation(model), df, n_sim=2)


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS)
def test_bootstrap_without_intercept(df: pd.DataFrame, bootstrap: str) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x - 1", validate_intercept=False)
    models = BOOTSTRAPS[bootstrap](
        Simulation(model), df, n_sim=N_SIM, validate_intercept=False
    )
    for boot in models:
        assert boot.coef_ is not None
        assert boot.coef_.shape == (1,)


def test_coefficient_table(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x + C(g)")
    models = Simulation(model).parametric_bootstrap(df, n_sim=N_SIM)
    table = coefficient_table(models)
    assert table.shape == (N_SIM, len(model.X_names or []))
    assert list(table.columns) == model.X_names
    for i, boot in enumerate(models):
        assert boot.coef_ is not None
        np.testing.assert_array_equal(table.iloc[i].to_numpy(), boot.coef_)


def test_coefficient_table_requires_models() -> None:
    with pytest.raises(ValueError, match="At least one model"):
        coefficient_table([])


def test_coefficient_table_requires_fit_models(df: pd.DataFrame) -> None:
    model = GLM(Poisson()).fit(df, formula="y ~ x")
    with pytest.raises(ValueError, match="must be fit"):
        coefficient_table([model, GLM(Poisson())])


def test_coefficient_table_requires_matching_coefficients(df: pd.DataFrame) -> None:
    models = [
        GLM(Poisson()).fit(df, formula="y ~ x"),
        GLM(Poisson()).fit(df, formula="y ~ x + C(g)"),
    ]
    with pytest.raises(ValueError, match="same coefficients"):
        coefficient_table(models)


# A standard deviation estimated from n_sim bootstrap replicates has a relative
# standard error of about 1 / sqrt(2 * n_sim). We compare each of the model's
# coefficients, so set the threshold with a Bonferroni bound, as in test_glm.py.
N_SIM_SE = 400
N_COEFFICIENTS = 2
SE_FALSE_FAILURE_RATE = 0.001
SE_MAX_Z = sts.norm.isf(SE_FALSE_FAILURE_RATE / (2 * N_COEFFICIENTS))


def sandwich_standard_errors(
    model: GLM, X: np.ndarray, y: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    """The robust (sandwich) standard errors, which do not rely on the
    family's variance function being correct."""
    assert model.information_matrix_ is not None
    η = X @ model.coef_ + offset
    μ = model.family.inv_link(η)
    # Each observation's contribution to the score, the gradient of the
    # log-likelihood (up to the dispersion, which cancels below).
    scores = (
        X
        * ((y - μ) * model.family.d_inv_link(η, μ) / model.family.variance(μ))[:, None]
    )
    bread = np.linalg.inv(model.information_matrix_)
    meat = scores.T @ scores
    return np.sqrt(np.diag(bread @ meat @ bread))


@pytest.mark.parametrize("bootstrap", BOOTSTRAPS)
def test_bootstrap_standard_errors(
    exposure_data: tuple[np.ndarray, np.ndarray, np.ndarray], bootstrap: str
) -> None:
    X, y, offset = exposure_data
    model = GLM(Poisson()).fit(X, y, offset=offset)
    sim = Simulation(model)
    # The parametric bootstrap simulates from the fit model, so it estimates
    # the model-based standard errors. The non-parametric bootstrap resamples
    # the data, so it estimates the sandwich standard errors, which differ from
    # the model-based ones by sampling noise in any one dataset.
    if bootstrap == "parametric_bootstrap":
        models = sim.parametric_bootstrap(X, n_sim=N_SIM_SE, offset=offset)
        expected = model.coef_standard_error_
    else:
        models = sim.non_parametric_bootstrap(X, y, n_sim=N_SIM_SE, offset=offset)
        expected = sandwich_standard_errors(model, X, y, offset)
    ratio = coefficient_table(models).std().to_numpy() / expected
    z = (ratio - 1) * np.sqrt(2 * N_SIM_SE)
    assert np.all(np.abs(z) < SE_MAX_Z)
