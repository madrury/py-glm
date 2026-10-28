from collections.abc import Sequence

import numpy as np
import pandas as pd

from glm.families import Bernoulli
from glm.glm import GLM, FitInputs
from glm.types import (
    CoefficientTable,
    DesignFrame,
    DesignMatrix,
    Offset,
    ResponseVector,
    FrequencyWeights,
    PrecisionWeights,
    SimulatedResponses,
)
from glm.utils import check_weights, default_X_names


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
        precision_weights: PrecisionWeights | None = None,
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

        precision_weights: array, shape (n_samples, )
            How much more precise each response is than a single observation's,
            as for GLM.fit. Each sampled response is the average of that many
            draws, e.g. a proportion of successes in that many trials. Discrete
            families like binomial and poisson need whole numbers.

        Returns
        -------

        simulations: array, shape (n_sim, n_samples)
            The sampled y values. Each row is one simulated response vector,
            so iterating over simulations yields one replicate at a time.
        """
        self._check_n_sim(n_sim)
        # predict checks that X has the right type for the model.
        μ = self.glm.predict(X, offset=offset)
        if precision_weights is not None:
            precision_weights = check_weights("precision_weights", μ, precision_weights)
        simulations = np.empty(shape=(n_sim, X.shape[0]))
        for i in range(n_sim):
            simulations[i, :] = self.glm.family.sample(
                μ, self.glm.dispersion_, precision_weights
            )
        return simulations

    def parametric_bootstrap(
        self,
        X: DesignMatrix | DesignFrame,
        n_sim: int = 100,
        offset: Offset | None = None,
        precision_weights: PrecisionWeights | None = None,
        validate_intercept: bool = True,
    ) -> list[GLM]:
        """Fit models to parametric bootstrap samples.

        The parametric bootstrap operates by sampling data from the conditional
        distributions y | X for a fixed matrix of predictors X, and then
        fitting glm models to each pair (X, y).

        The offset and weights must be those the model was fit with. Models fit
        with frequency weights aren't supported: simulating each of the
        observations a row stands for would need the expanded data, and
        simulating only their average would understate each replicate's
        dispersion. Use the non-parametric bootstrap, which resamples them
        multinomially.  Bernoulli models of proportions are not supported
        without their numbers of trials as precision weights: without them,
        nothing says how to simulate a proportion.

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

        precision_weights: array, shape (n_samples, )
            How much more precise each response is, as for GLM.fit. Each
            simulated response is the average of that many draws, e.g. a
            proportion of successes in that many trials, and each replicate is
            fit with the same weights.

        validate_intercept: bool
            Check that column 0 of each replicate's design matrix is an
            intercept. Set to False for models without one, e.g. a formula
            such as "y ~ x - 1".

        Returns
        -------

        models:
            The list of fit models.
        """
        self._check_n_sim(n_sim)
        fit_inputs = self._fit_inputs()
        if fit_inputs.has_frequency_weights:
            raise ValueError(
                "The parametric bootstrap doesn't support models fit with "
                "frequency weights. Use the non-parametric bootstrap, which "
                "resamples the observations they stand for."
            )
        fit_inputs.check(offset, None, precision_weights)
        if (
            isinstance(self.glm.family, Bernoulli)
            and not fit_inputs.has_precision_weights
            and not fit_inputs.has_binary_response
        ):
            raise ValueError(
                "The model was fit to proportions, not 0/1 responses. To "
                "simulate them, pass their numbers of trials as precision_weights, "
                "both to GLM.fit and here."
            )
        design = self.glm.design_matrix(X)
        simulations = self.sample(
            X, n_sim, offset=offset, precision_weights=precision_weights
        )
        return [
            self._fit_replicate(
                design,
                simulations[i, :],
                offset=offset,
                precision_weights=precision_weights,
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
        frequency_weights: FrequencyWeights | None = None,
        precision_weights: PrecisionWeights | None = None,
        validate_intercept: bool = True,
    ) -> list[GLM]:
        """Fit models to non-parametric bootstrap samples.

        The non-parametric bootstrap operates by sampling observations with
        replacement from the pairs (X, y), and then fitting glm models to the
        resampled pairs.

        Observations, not rows, are resampled: a row with frequency weight f
        stands for f independent observations, and each is drawn separately,
        exactly as when bootstrapping the expanded data. A replicate draws N
        observations, where N is the sum of the frequency weights, and so
        draws row i a multinomial number of times, with probability f_i / N,
        which becomes its frequency weight in the replicate. Without
        frequency weights, this is the usual bootstrap of rows.

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

        frequency_weights: array, shape (n_samples, )
            How many identical observations each row stands for, as for
            GLM.fit. Each observation is resampled separately, so they must be
            whole numbers.

        precision_weights: array, shape (n_samples, )
            How much more precise each row's response is, as for GLM.fit. A
            precision-weighted row is one observation, of an average or
            proportion, so it is resampled whole, keeping its weight.

        validate_intercept: bool
            Check that column 0 of each replicate's design matrix is an
            intercept. Set to False for models without one, e.g. a formula
            such as "y ~ x - 1".

        Returns
        -------

        models:
            The list of fit models.
        """
        self._check_n_sim(n_sim)
        self._fit_inputs().check(offset, frequency_weights, precision_weights)
        # Encode a formula model's data once, with the specs learned when
        # fitting the original model. Re-learning them on each resample could
        # drop a rare category's column, or change the reference level or a
        # stateful transform such as center(x), so that replicates would
        # estimate different quantities.
        design = self.glm.design_matrix(X)
        response = self.glm.response_vector(X, y)
        frequency_weights = check_weights(
            "frequency_weights", response, frequency_weights
        )
        if not np.all(frequency_weights == np.round(frequency_weights)):
            raise ValueError(
                "frequency_weights must be whole numbers to resample the "
                "observations they stand for."
            )
        n_observations = int(np.sum(frequency_weights))
        probabilities = frequency_weights / n_observations
        models = []
        for _ in range(n_sim):
            # How many times each row's observations are drawn.
            counts = np.random.multinomial(n_observations, probabilities)
            idxs = np.flatnonzero(counts)
            self._check_estimable(design[idxs])
            # Per-row arrays must be resampled with the same indices as X.
            models.append(
                self._fit_replicate(
                    design[idxs],
                    response[idxs],
                    offset=None if offset is None else offset[idxs],
                    frequency_weights=counts[idxs].astype(np.float64),
                    precision_weights=(
                        None if precision_weights is None else precision_weights[idxs]
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

    def _fit_inputs(self) -> FitInputs:
        if self.glm.fit_inputs_ is None:
            raise ValueError("Simulations require a fit model.")
        return self.glm.fit_inputs_

    def _fit_replicate(
        self,
        design: DesignMatrix,
        response: ResponseVector,
        *,
        offset: Offset | None,
        frequency_weights: FrequencyWeights | None = None,
        precision_weights: PrecisionWeights | None = None,
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
            frequency_weights=frequency_weights,
            precision_weights=precision_weights,
            validate_intercept=validate_intercept,
        )
        model._copy_formula(self.glm)
        return model

    def _check_n_sim(self, n_sim: object) -> None:
        # bool is a subclass of int, but True is not a meaningful n_sim.
        if (
            isinstance(n_sim, bool)
            or not isinstance(n_sim, (int, np.integer))
            or n_sim < 1
        ):
            raise ValueError(f"n_sim must be a positive integer, got {n_sim!r}.")
