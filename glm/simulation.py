import numpy as np

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
        # TODO: Should probably check if fit first.
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

        Returns
        -------

        models:
            The list of fit models.
        """
        simulations = self.sample(X, n_sim, offset=offset)
        if self.glm.X_spec is not None:
            X = self.glm._make_rhs_matrix(X)
        models = []
        for i in range(n_sim):
            model = self.glm.clone()
            model.fit(
                X,
                simulations[i, :],
                offset=offset,
                X_names=self.glm.X_names,
                y_name=self.glm.y_name,
                validate_intercept=validate_intercept,
            )
            # Fit on the design matrix, as the simulated responses have no
            # column in X, but keep the formula so the bootstrap models can
            # make predictions from data frames.
            model.formula, model.X_spec = self.glm.formula, self.glm.X_spec
            models.append(model)
        return models

    def non_parametric_bootstrap(self, X, y=None, n_sim=100, offset=None):
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

        Returns
        -------

        models:
            The list of fit models.
        """
        models = []
        for _ in range(n_sim):
            idxs = np.random.choice(X.shape[0], X.shape[0])
            if offset:
                offset_boot = offset_boot[idxs]
            else:
                offset_boot = None
            model = self.glm.clone()
            if self.glm.formula:
                X_boot = X.iloc[idxs]
                model.fit(
                    X_boot,
                    formula=self.glm.formula,
                    offset=offset_boot,
                    validate_intercept=False,
                )
            else:
                X_boot, y_boot = X[idxs], y[idxs]
                model.fit(X_boot, y_boot, offset=offset_boot)
            models.append(model)
        return models
