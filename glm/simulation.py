import numpy as np
import pandas as pd

from glm.types import SimulatedResponses


class Simulation:
    """Run resampling simulations of a glm object.

    This object implements respampling stratgies, for example the parametric
    end non-parametric bootstrap.

    Parameters
    ----------

    glm:
        A GLM type object.
    """

    def __init__(self, glm):
        if glm.coef_ is None:
            raise ValueError("Simulations require a fit model.")
        self.glm = glm

    def sample(self, X, n_sim=100, offset=None) -> SimulatedResponses:
        """Sample points from the conditional distribution y | X.

        A fitted GLM model determines conditional distributions y | X.  Given a
        matrix of predictors, this method samples y from each conditional
        distribution.

        Parameters
        ----------

        X: array, shape (n_samples, n_features)
            Data.

        n_sim: positive integer
            The number of times to sample from the coditional distributions
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
        self._check_X(X)
        simulations = np.empty(shape=(n_sim, X.shape[0]))
        preds = self.glm.predict(X, offset=offset)
        for i in range(n_sim):
            simulations[i, :] = self.glm.family.sample(preds, self.glm.dispersion_)
        return simulations

    def parametric_bootstrap(self, X, n_sim=100, offset=None, validate_intercept=True):
        """Fit models to parameteric bootstrap samples.

        The parametric operates by sampling data from the conditional
        distributions y | X for a fixed matrix of predictors X, and then
        fitting glm models to each pair (X, y).

        Parameters
        ----------

        X: array, shape (n_sample, n_features)
            Data.

        n_sim: positive integer
            The number of times to sample from the coditional distributions
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
        self._check_X(X)
        simulations = self.sample(X, n_sim, offset=offset)
        if self.glm.X_spec is not None:
            X = self.glm._make_rhs_matrix(X)
        return [
            self._fit_replicate(
                X,
                simulations[i, :],
                offset=offset,
                validate_intercept=validate_intercept,
            )
            for i in range(n_sim)
        ]

    def non_parametric_bootstrap(
        self,
        X,
        y=None,
        n_sim=100,
        offset=None,
        sample_weights=None,
        validate_intercept=True,
    ):
        """Fit models to non-parameteric bootstrap samples.

        The non-parametric operates by sampling data with replacement from the
        pair (X, y), and then fitting glm models to the resampled pairs.

        Parameters
        ----------

        X: array, shape (n_samples, n_features)
            Data.

        y: array, shape (n_samples, )
            Targets.

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
        self._check_X(X)
        self._check_y(y)
        # Encode a formula model's data once, with the specs learned when
        # fitting the original model. Re-learning them on each resample could
        # drop a rare category's column, or change the reference level or a
        # stateful transform such as center(x), so that replicates would
        # estimate different quantities.
        if self.glm.X_spec is not None:
            design = self.glm._make_rhs_matrix(X)
            response = self.glm._make_lhs_vector(X)
        else:
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

    def _check_estimable(self, design):
        """Raise if an unpenalized model cannot be fit to a resampled design.

        A resample may, for example, contain no observations of a rare category,
        leaving its column all zero, so its coefficient cannot be estimated.
        """
        # A non-zero ridge penalty renders all models estimable.
        if self.glm.alpha > 0:
            return
        empty = np.all(design == 0, axis=0)
        if np.any(empty):
            names = [name for name, e in zip(self.glm.X_names, empty) if e]
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

    def _fit_replicate(self, design, response, **fit_kwargs):
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
            **fit_kwargs,
        )
        model.formula = self.glm.formula
        model.X_spec, model.y_spec = self.glm.X_spec, self.glm.y_spec
        return model

    def _check_n_sim(self, n_sim):
        # bool is a subclass of int, but True is not a meaningful n_sim.
        is_int = isinstance(n_sim, (int, np.integer)) and not isinstance(n_sim, bool)
        if not is_int or n_sim < 1:
            raise ValueError(f"n_sim must be a positive integer, got {n_sim!r}.")

    def _check_X(self, X):
        if self.glm.X_spec is not None:
            if not isinstance(X, pd.DataFrame):
                raise TypeError(
                    f"X must be a DataFrame for a model fit with a formula, "
                    f"got a {type(X)}"
                )
        elif not isinstance(X, np.ndarray):
            raise TypeError(
                f"X must be a numpy array for a model fit without a formula, "
                f"got a {type(X)}"
            )

    def _check_y(self, y):
        if self.glm.X_spec is not None:
            if y is not None:
                raise ValueError(
                    "y must not be supplied for a model fit with a formula, as "
                    "the response is taken from the formula."
                )
        elif not isinstance(y, np.ndarray):
            raise TypeError(
                f"y must be a numpy array for a model fit without a formula, "
                f"got a {type(y)}"
            )
