import numpy as np


class Simulation:

    def __init__(self, glm):
        self.glm = glm

    def sample(self, X, n_sim=100, offset=None):
        simulations = np.empty(shape=(n_sim, X.shape[0]))
        preds = self.glm.predict(X, offset=offset)
        for i in range(n_sim):
            simulations[i, :] = self.glm.family.sample(preds, self.glm.dispersion_)
        return simulations

    def parameteric_bootstrap(self, X, n_sim=100, offset=None):
        simulations = self.sample(X, n_sim, offset=offset)
        models = []
        for i in range(n_sim):
            model = self.glm.clone()
            model.fit(X, simulations[i, :], offset=offset)
            models.append(model)
        return models
