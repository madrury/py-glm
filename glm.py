import numpy as np

class GLM:

    def __init__(self, family, alpha=0.0):
        """A generalized linear model.

        Parameters
        ----------

        family:
          An exponential family object.

        alpha:
          The ridge regularization strength. If non-zero, the loss function
          minimized is a penalized deviance, where the penalty is alpha *
          np.sum(model.coef_**2).
        """
        self.family = family
        self.alpha = alpha
        self.coef_ = None 
        self.deviance = None

    def fit(self, X, y, warm_start=None, max_iter=100, tol=0.1**5):
        # Check inputs.
        if not self._check_intercept(X):
            raise ValueError("First column in matrix X is not an intercept.")
        if not warm_start:
            initial_intercept = np.mean(y)
            warm_start = np.zeros(X.shape[1])
            warm_start[0] = initial_intercept
        coef = warm_start

        # Fit model.
        family = self.family
        n_iter = 0
        is_converged = False
        penalized_deviance = np.inf
        while n_iter < max_iter and not is_converged:
            nu = np.dot(X, coef)
            mu = family.inv_link(nu)
            dmu = family.d_inv_link(nu, mu)
            var = family.variance(mu)
            dbeta = - np.sum(
                X * ((y - mu) * (dmu / var)).reshape(-1, 1), axis=0)
            ddbeta = np.dot(
                X.T, X * (dmu**2 / var).reshape(-1, 1))
            if self.alpha != 0.0:
                # Add penalty term to gradients and hessians.
                dbeta_penalty = coef.copy()
                dbeta_penalty[0] = 0.0
                dbeta = dbeta + self.alpha * dbeta_penalty
                diag_idxs = list(range(1, X.shape[1]))
                ddbeta[diag_idxs, diag_idxs] += self.alpha
            coef = coef - np.linalg.solve(ddbeta, dbeta)
            penalized_deviance_previous = penalized_deviance
            penalized_deviance = family.penalized_deviance(
                y, mu, self.alpha, coef)
            is_converged = np.abs(
                penalized_deviance - penalized_deviance_previous) < tol
            print(coef, penalized_deviance)
            n_iter += 1

        self.coef_ = coef
        self.deviance = family.deviance(y, mu)
        return self

    def _check_intercept(self, X):
       return np.all(X[:, 0] == 1.0)

    def predict(self, X):
        return self.family.inv_link(np.dot(X, self.coef_))
