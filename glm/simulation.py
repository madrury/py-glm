from collections.abc import Sequence

import numpy as np
import pandas as pd

from glm.glm import GLM
from glm.types import (
    CoefficientTable,
    DesignFrame,
    DesignMatrix,
    Offset,
    ResponseVector,
    SampleWeights,
    SimulatedResponses,
)
from glm.utils import default_X_names


def coefficient_table(models: Sequence[GLM]) -> CoefficientTable:
    """Collect the coefficient estimates of several fit models into a table.

    This is most useful for summarizing bootstrap replicates, for example their
    standard errors with table.std(), or percentile confidence intervals with
    table.quantile([0.025, 0.975]).

    Parameters
    ----------

    models: sequence of GLM
        Fit models with the same coefficients, such as those returned by
        Simulation.parametric_bootstrap or Simulation.non_parametric_bootstrap.

    Returns
    -------

    table: pd.DataFrame, shape (n_models, n_features)
        One row per model, and one column per coefficient, named by the
        models' X_names.
    """
    if not models:
        raise ValueError("At least one model is required.")
    X_names = models[0].X_names
    rows = []
    for model in models:
        if model.coef_ is None:
            raise ValueError("All models must be fit.")
        # Stacking models with different coefficients would misalign columns.
        if model.X_names != X_names:
            raise ValueError(
                f"All models must have the same coefficients, but found both "
                f"{X_names} and {model.X_names}."
            )
        rows.append(model.coef_)
    return pd.DataFrame(np.vstack(rows), columns=X_names)


class Simulation:
    """Run resampling simulations of a glm object.

    This object implements resampling strategies, for example the parametric
    and non-parametric bootstrap.

    Parameters
    ----------

    glm:
        A fit GLM object.
    """

    def __init__(self, glm: GLM) -> None:
        if glm.coef_ is None:
            raise ValueError("Simulations require a fit model.")
        self.glm = glm

    def sample(
        self,
        X: DesignMatrix | DesignFrame,
        n_sim: int = 100,
        offset: Offset | None = None,
    ) -> SimulatedResponses:
        """Sample points from the conditional distribution y | X.

        A fitted GLM model determines conditional distributions y | X.  Given a
        matrix of predictors, this method samples y from each conditional
        distribution.

        Parameters
        ----------

        X: array, shape (n_samples, n_features) or pd.DataFrame
            Data.

        n_sim: positive integer
            The number of times to sample from the conditional distributions
            y | X.

        offset: array, shape (n_samples, )
            Offsets to use in the predictions feeding into the conditional
            distributions.

        Returns
        -------

        simulations: array, shape (n_sim, n_samples)
            The sampled y values. Each row is one simulated response vector,
            so iterating over simulations yields one replicate at a time.
        """
        self._check_n_sim(n_sim)
        # predict checks that X has the right type for the model.
        μ = self.glm.predict(X, offset=offset)
        simulations = np.empty(shape=(n_sim, X.shape[0]))
        for i in range(n_sim):
            simulations[i, :] = self.glm.family.sample(μ, self.glm.dispersion_)
        return simulations

    def parametric_bootstrap(
        self,
        X: DesignMatrix | DesignFrame,
        n_sim: int = 100,
        offset: Offset | None = None,
        validate_intercept: bool = True,
    ) -> list[GLM]:
        """Fit models to parametric bootstrap samples.

        The parametric bootstrap operates by sampling data from the conditional
        distributions y | X for a fixed matrix of predictors X, and then
        fitting glm models to each pair (X, y).

        Parameters
        ----------

        X: array, shape (n_samples, n_features) or pd.DataFrame
            Data.

        n_sim: positive integer
            The number of times to sample from the conditional distributions
            y | X.

        offset: array, shape (n_samples, )
            Offsets to use in predictions feeding into the conditional
            distributions.

        validate_intercept: bool
            Check that the first column of each replicate's design matrix is an
            intercept. Set to False for models without one, e.g. a formula
            such as "y ~ x - 1".

        Returns
        -------

        models:
            The list of fit models.
        """
        self._check_n_sim(n_sim)
        design = self._design_matrix(X)
        simulations = self.sample(X, n_sim, offset=offset)
        return [
            self._fit_replicate(
                design,
                simulations[i, :],
                offset=offset,
                validate_intercept=validate_intercept,
            )
            for i in range(n_sim)
        ]

    def non_parametric_bootstrap(
        self,
        X: DesignMatrix | DesignFrame,
        y: ResponseVector | None = None,
        n_sim: int = 100,
        offset: Offset | None = None,
        sample_weights: SampleWeights | None = None,
        validate_intercept: bool = True,
    ) -> list[GLM]:
        """Fit models to non-parametric bootstrap samples.

        The non-parametric bootstrap operates by sampling data with replacement
        from the pair (X, y), and then fitting glm models to the resampled
        pairs.

        Parameters
        ----------

        X: array, shape (n_samples, n_features) or pd.DataFrame
            Data.

        y: array, shape (n_samples, )
            The response. Required for a model fit without a formula, and must
            not be supplied for a model fit with one, as the formula determines
            the response.

        n_sim: positive integer
            The number of times to resample the data.

        offset: array, shape (n_samples, )
            Offsets to use in predictions feeding into the conditional
            distributions.

        sample_weights: array, shape (n_samples, )
            Sample weights used when fitting each model. Each resampled row
            keeps its weight.

        validate_intercept: bool
            Check that the first column of each replicate's design matrix is an
            intercept. Set to False for models without one, e.g. a formula
            such as "y ~ x - 1".

        Returns
        -------

        models:
            The list of fit models.
        """
        self._check_n_sim(n_sim)
        # Encode a formula model's data once, with the specs learned when
        # fitting the original model. Re-learning them on each resample could
        # drop a rare category's column, or change the reference level or a
        # stateful transform such as center(x), so that replicates would
        # estimate different quantities.
        if self.glm.X_spec is not None:
            if not isinstance(X, pd.DataFrame):
                raise self._X_type_error(X)
            if y is not None:
                raise ValueError(
                    "y must not be supplied for a model fit with a formula, as "
                    "the response is taken from the formula."
                )
            design = self.glm._make_rhs_matrix(X)
            response = self.glm._make_lhs_vector(X)
        else:
            if not isinstance(X, np.ndarray):
                raise self._X_type_error(X)
            if not isinstance(y, np.ndarray):
                raise TypeError(
                    f"y must be a numpy array for a model fit without a formula, "
                    f"got a {type(y)}"
                )
            design, response = X, y
        n = design.shape[0]
        models = []
        for _ in range(n_sim):
            idxs = np.random.choice(n, n)
            self._check_estimable(design[idxs])
            # Per-row arrays must be resampled with the same indices as X.
            models.append(
                self._fit_replicate(
                    design[idxs],
                    response[idxs],
                    offset=None if offset is None else offset[idxs],
                    sample_weights=(
                        None if sample_weights is None else sample_weights[idxs]
                    ),
                    validate_intercept=validate_intercept,
                )
            )
        return models

    def _check_estimable(self, design: DesignMatrix) -> None:
        """Raise if an unpenalized model cannot be fit to a resampled design.

        A resample may, for example, contain no observations of a rare category,
        leaving its column all zero, so its coefficient cannot be estimated.
        """
        # A non-zero ridge penalty renders all models estimable.
        if self.glm.alpha > 0:
            return
        empty = np.all(design == 0, axis=0)
        if np.any(empty):
            X_names = self.glm.X_names or default_X_names(design)
            names = [name for name, e in zip(X_names, empty) if e]
            raise ValueError(
                f"A bootstrap resample has no observations for design columns "
                f"{names}, so their coefficients cannot be estimated. This "
                f"happens with rare categories; consider merging them."
            )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError(
                "A bootstrap resample's design matrix is rank deficient, so the "
                "coefficients cannot be estimated. This happens with rare "
                "categories or near-collinear predictors."
            )

    def _fit_replicate(
        self,
        design: DesignMatrix,
        response: ResponseVector,
        *,
        offset: Offset | None,
        sample_weights: SampleWeights | None = None,
        validate_intercept: bool,
    ) -> GLM:
        """Fit a copy of the model to one bootstrap replicate.

        Replicates are always fit on a design matrix, but keep the original
        model's names, formula and specs, so that they can be summarized, and
        make predictions from data frames, like the original model.
        """
        model = self.glm.clone()
        model.fit(
            design,
            response,
            X_names=self.glm.X_names,
            y_name=self.glm.y_name,
            offset=offset,
            sample_weights=sample_weights,
            validate_intercept=validate_intercept,
        )
        model.formula = self.glm.formula
        model.X_spec, model.y_spec = self.glm.X_spec, self.glm.y_spec
        return model

    def _design_matrix(self, X: DesignMatrix | DesignFrame) -> DesignMatrix:
        """Check the type of X, and encode it as the model's design matrix."""
        if self.glm.X_spec is not None:
            if not isinstance(X, pd.DataFrame):
                raise self._X_type_error(X)
            return self.glm._make_rhs_matrix(X)
        if not isinstance(X, np.ndarray):
            raise self._X_type_error(X)
        return X

    def _X_type_error(self, X: object) -> TypeError:
        if self.glm.X_spec is not None:
            return TypeError(
                f"X must be a DataFrame for a model fit with a formula, "
                f"got a {type(X)}"
            )
        return TypeError(
            f"X must be a numpy array for a model fit without a formula, "
            f"got a {type(X)}"
        )

    def _check_n_sim(self, n_sim: object) -> None:
        # bool is a subclass of int, but True is not a meaningful n_sim.
        if (
            isinstance(n_sim, bool)
            or not isinstance(n_sim, (int, np.integer))
            or n_sim < 1
        ):
            raise ValueError(f"n_sim must be a positive integer, got {n_sim!r}.")
