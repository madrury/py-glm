import numpy as np


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
        self.glm = glm

    def sample(self, X, n_sim=100, offset=None):
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
            The sampled y values.
        """
        simulations = np.empty(shape=(n_sim, X.shape[0]))
        preds = self.glm.predict(X, offset=offset)
        for i in range(n_sim):
            simulations[i, :] = self.glm.family.sample(preds, self.glm.dispersion_)
        return simulations

    def parametric_bootstrap(self, X, n_sim=100, offset=None):
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
        models = []
        for i in range(n_sim):
            model = self.glm.clone()
            model.fit(X, simulations[i, :], offset=offset)
            models.append(model)
        return models

    def non_parametric_bootstrap(self, X, y, n_sim=100, offset=None):
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
            X_boot, y_boot = X[idxs], y[idxs]
            if offset:
                offset_boot = offset_boot[idxs]
            else:
                offset_boot = None
            model = self.glm.clone()
            model.fit(X_boot, y_boot, offset=offset_boot)
            models.append(model)
        return models
