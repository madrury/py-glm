import numpy as np
import pandas as pd
import pytest

from glm.encoding import encode_design, learn_formula
from glm.families import Poisson
from glm.glm import GLM
from glm.ridge import (
    RidgePath,
    RidgePathCV,
    cv_encoding_spec,
    fold_frequency_weights,
)

N = 300
N_FOLDS = 5
ALPHAS = [100.0, 10.0, 1.0, 0.1]
# Stateful transforms and a level only one row has: each fold's encoding learns
# different state from its training rows than the full data would give.
STATEFUL_FORMULA = "y ~ center(x1) + bs(x2, df=4, extrapolation='extend') + C(g)"


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(154)
    df = pd.DataFrame({"x1": rng.normal(size=N), "g": rng.choice(["a", "b", "c"], N)})
    df["x2"] = 0.95 * df["x1"] + np.sqrt(1 - 0.95**2) * rng.normal(size=N)
    g_effect = df["g"].map({"a": 0.0, "b": 0.5, "c": -0.3})
    df["y"] = rng.poisson(np.exp(0.3 + 0.4 * df["x1"] + 0.2 * df["x2"] + g_effect))
    return df


@pytest.fixture
def arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(N), df["x1"], df["x2"]])
    return X, df["y"].to_numpy().astype(np.float64)


@pytest.fixture
def weights() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(154)
    frequency = rng.integers(1, 5, size=N).astype(np.float64)
    precision = rng.uniform(0.5, 3.0, size=N)
    return frequency, precision


def test_folds_split_each_rows_observations() -> None:
    rng = np.random.default_rng(154)
    frequency = rng.integers(0, 6, size=N).astype(np.float64)
    folds = fold_frequency_weights(frequency, N_FOLDS, rng)
    assert folds.shape == (N_FOLDS, N)
    np.testing.assert_array_equal(folds, np.round(folds))
    np.testing.assert_array_equal(folds.sum(axis=0), frequency)
    # The observations are dealt out evenly, as when cross-validating the
    # expanded data.
    totals = folds.sum(axis=1)
    assert totals.max() - totals.min() <= 1


def test_folds_keep_single_observation_rows_whole() -> None:
    # Without frequency weights, as for precision-weighted rows, each row is
    # one observation, and is held out by exactly one fold.
    folds = fold_frequency_weights(np.ones(N), N_FOLDS, np.random.default_rng(154))
    np.testing.assert_array_equal(np.sort(folds, axis=0)[-1], 1.0)
    np.testing.assert_array_equal(folds.sum(axis=0), 1.0)


def test_folds_are_reproducible() -> None:
    frequency = np.full(N, 3.0)
    first = fold_frequency_weights(frequency, N_FOLDS, np.random.default_rng(1))
    second = fold_frequency_weights(frequency, N_FOLDS, np.random.default_rng(1))
    np.testing.assert_array_equal(first, second)


def test_folds_reject_fractional_frequency_weights() -> None:
    with pytest.raises(ValueError, match="whole numbers"):
        fold_frequency_weights(np.full(N, 1.5), N_FOLDS, np.random.default_rng(1))


def test_folds_reject_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least as many observations"):
        fold_frequency_weights(np.ones(3), N_FOLDS, np.random.default_rng(1))


@pytest.mark.parametrize("n_folds", [1, 0, 2.5, True])
def test_invalid_n_folds_raise(n_folds: object) -> None:
    with pytest.raises(ValueError, match="n_folds"):
        RidgePathCV(Poisson(), n_folds=n_folds)  # type: ignore[arg-type]


def test_fold_deviances_match_independent_fits(
    arrays: tuple[np.ndarray, np.ndarray], weights: tuple[np.ndarray, np.ndarray]
) -> None:
    X, y = arrays
    frequency, precision = weights
    offset = np.full(N, 0.1)
    cv = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        X,
        y,
        offset=offset,
        frequency_weights=frequency,
        precision_weights=precision,
        tol=1e-12,
        rng=np.random.default_rng(154),
        alphas=ALPHAS,
    )
    folds = fold_frequency_weights(frequency, N_FOLDS, np.random.default_rng(154))
    np.testing.assert_array_equal(cv.fold_frequency_weights_, folds)
    assert cv.fold_deviances_ is not None and cv.fold_weights_ is not None
    for k, held_out in enumerate(folds):
        training = frequency - held_out
        # The fold holds out some rows whole, and some only in part.
        assert np.any((held_out > 0) & (training == 0))
        assert np.any((held_out > 0) & (training > 0))
        held_out_weights = held_out * precision
        np.testing.assert_allclose(cv.fold_weights_[k], held_out_weights.sum())
        for j, alpha in enumerate(sorted(ALPHAS, reverse=True)):
            # Fit to every row, those held out whole getting zero weight,
            # rather than dropping them, as the fold fits do.
            model = GLM(Poisson(), alpha=alpha).fit(
                X,
                y,
                offset=offset,
                frequency_weights=training,
                precision_weights=precision,
                tol=1e-12,
            )
            μ = model.predict(X, offset=offset)
            deviance = Poisson().deviance(y, μ, held_out_weights)
            assert cv.fold_deviances_[k, j] == pytest.approx(
                deviance / held_out_weights.sum(), rel=1e-6
            )


def test_rows_without_weight_are_never_held_out(
    arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    # Most rows have zero precision weight, so if they were dealt to folds,
    # some fold would likely hold out no weight at all.
    precision = np.zeros(N)
    precision[:20] = 1.0
    cv = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        *arrays,
        precision_weights=precision,
        rng=np.random.default_rng(154),
        alphas=ALPHAS,
    )
    assert cv.fold_frequency_weights_ is not None and cv.fold_weights_ is not None
    np.testing.assert_array_equal(cv.fold_frequency_weights_[:, 20:], 0.0)
    np.testing.assert_array_equal(cv.fold_weights_, 4.0)


def test_cv_deviance_averages_folds_by_weight(
    arrays: tuple[np.ndarray, np.ndarray], weights: tuple[np.ndarray, np.ndarray]
) -> None:
    frequency, _ = weights
    cv = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        *arrays,
        frequency_weights=frequency,
        rng=np.random.default_rng(154),
        alphas=ALPHAS,
    )
    assert cv.fold_deviances_ is not None and cv.fold_weights_ is not None
    np.testing.assert_allclose(
        cv.cv_deviance_,
        np.average(cv.fold_deviances_, axis=0, weights=cv.fold_weights_),
    )
    # Equal fold weights give the usual standard error of the mean.
    equal = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        *arrays, rng=np.random.default_rng(154), alphas=ALPHAS
    )
    assert equal.fold_deviances_ is not None
    np.testing.assert_allclose(
        equal.cv_standard_error_,
        equal.fold_deviances_.std(axis=0, ddof=1) / np.sqrt(N_FOLDS),
    )


def test_full_path_matches_ridge_path(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    # The grid, and the models to choose from, come from the full data.
    cv = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        *arrays, rng=np.random.default_rng(154), n_alphas=10
    )
    path = RidgePath(Poisson()).fit(*arrays, n_alphas=10)
    assert cv.path_ is not None
    np.testing.assert_array_equal(cv.alphas_, path.alphas_)
    np.testing.assert_allclose(cv.path_.coefficient_table(), path.coefficient_table())


def test_alpha_min_has_the_smallest_cv_deviance(df: pd.DataFrame) -> None:
    cv = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        df, formula="y ~ x1 + x2 + C(g)", rng=np.random.default_rng(154), n_alphas=30
    )
    assert cv.alphas_ is not None and cv.cv_deviance_ is not None
    best = int(np.argmin(cv.cv_deviance_))
    assert cv.alpha_min_ == cv.alphas_[best]
    assert cv.model_min_.alpha == cv.alpha_min_


def test_cv_deviance_rises_when_overpenalized(df: pd.DataFrame) -> None:
    cv = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        df, formula="y ~ x1 + x2 + C(g)", rng=np.random.default_rng(154), n_alphas=30
    )
    assert cv.cv_deviance_ is not None and cv.alphas_ is not None
    # Shrinking every coefficient to zero discards real effects.
    assert cv.cv_deviance_[0] > cv.cv_deviance_.min()
    assert cv.alpha_min_ < cv.alphas_[0]


def test_level_unseen_in_training_fold(df: pd.DataFrame) -> None:
    # A level with a single row is unseen by the fold that holds it out; the
    # penalized fold models predict it at the baseline, without raising.
    df = df.copy()
    df.loc[0, "g"] = "rare"
    cv = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        df, formula="y ~ x1 + C(g)", rng=np.random.default_rng(154), n_alphas=10
    )
    assert cv.fold_deviances_ is not None
    assert np.all(np.isfinite(cv.fold_deviances_))


@pytest.fixture
def rare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[0, "g"] = "rare"
    return df


def test_folds_learn_nothing_from_held_out_data(
    rare_df: pd.DataFrame, weights: tuple[np.ndarray, np.ndarray]
) -> None:
    # Each fold's score must be what a model fit to its training rows alone
    # scores on its held-out rows, as if the held-out rows were new data.
    frequency, _ = weights
    cv = RidgePathCV(Poisson(), n_folds=N_FOLDS).fit(
        rare_df,
        formula=STATEFUL_FORMULA,
        frequency_weights=frequency,
        alphas=ALPHAS,
        tol=1e-12,
        rng=np.random.default_rng(154),
    )
    assert cv.fold_frequency_weights_ is not None and cv.fold_deviances_ is not None
    full_spec = learn_formula(STATEFUL_FORMULA, rare_df, penalized=True).rhs.model_spec
    response = rare_df["y"].to_numpy().astype(np.float64)
    for k, held_out in enumerate(cv.fold_frequency_weights_):
        training = frequency - held_out
        training_rows = np.flatnonzero(training > 0)
        held_out_rows = np.flatnonzero(held_out > 0)
        held_out_frame = rare_df.iloc[held_out_rows]
        held_out_y = response[held_out_rows]
        held_out_weights = held_out[held_out_rows]
        path = RidgePath(Poisson()).fit(
            rare_df.iloc[training_rows],
            formula=STATEFUL_FORMULA,
            frequency_weights=training[training_rows],
            alphas=ALPHAS,
            tol=1e-12,
        )
        # The same fold, with its encoding learned from all of the data: a
        # leak the comparison above must be able to detect.
        leaky_path = RidgePath(Poisson()).fit(
            encode_design(full_spec, rare_df.iloc[training_rows]),
            response[training_rows],
            frequency_weights=training[training_rows],
            alphas=ALPHAS,
            tol=1e-12,
        )
        assert path.models_ is not None and leaky_path.models_ is not None
        leaky_design = encode_design(full_spec, held_out_frame)
        for j in range(len(ALPHAS)):
            deviance = Poisson().deviance(
                held_out_y, path.models_[j].predict(held_out_frame), held_out_weights
            )
            assert cv.fold_deviances_[k, j] == pytest.approx(
                deviance / held_out_weights.sum(), rel=1e-8
            )
            leaky_deviance = Poisson().deviance(
                held_out_y,
                leaky_path.models_[j].predict(leaky_design),
                held_out_weights,
            )
            assert cv.fold_deviances_[k, j] != pytest.approx(
                leaky_deviance / held_out_weights.sum(), rel=1e-8
            )


def test_fold_encodings_ignore_held_out_rows(
    rare_df: pd.DataFrame, weights: tuple[np.ndarray, np.ndarray]
) -> None:
    # Changing the rows a fold holds out whole, even to extreme values and
    # unseen levels, leaves the encoding it learns unchanged.
    frequency, _ = weights
    folds = fold_frequency_weights(frequency, N_FOLDS, np.random.default_rng(154))
    response = rare_df["y"].to_numpy()
    held_out_whole = (folds[0] > 0) & (frequency - folds[0] == 0)
    assert np.any(held_out_whole)
    changed = rare_df.copy()
    changed.loc[held_out_whole, "x1"] = 1000.0
    changed.loc[held_out_whole, "x2"] = -1000.0
    changed.loc[held_out_whole, "g"] = "new"
    original_fold = cv_encoding_spec(
        rare_df, STATEFUL_FORMULA, response, frequency, folds
    ).folds[0]
    changed_fold = cv_encoding_spec(
        changed, STATEFUL_FORMULA, response, frequency, folds
    ).folds[0]
    assert original_fold.X_spec is not None and changed_fold.X_spec is not None
    assert original_fold.X_spec.column_names == changed_fold.X_spec.column_names
    np.testing.assert_array_equal(
        original_fold.training_design(rare_df), changed_fold.training_design(rare_df)
    )


def test_fold_encoding_weights_partition_the_observations(
    rare_df: pd.DataFrame, weights: tuple[np.ndarray, np.ndarray]
) -> None:
    frequency, _ = weights
    folds = fold_frequency_weights(frequency, N_FOLDS, np.random.default_rng(154))
    encoding = cv_encoding_spec(
        rare_df, STATEFUL_FORMULA, rare_df["y"].to_numpy(), frequency, folds
    )
    for fold in encoding.folds:
        np.testing.assert_array_equal(
            fold.training_frequency_weights + fold.held_out_frequency_weights,
            frequency,
        )


def test_unfit_cv_raises() -> None:
    with pytest.raises(ValueError, match="has not been fit"):
        RidgePathCV(Poisson()).model_min_
