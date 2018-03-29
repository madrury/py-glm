# py-glm: Generalized Linear Models in Python

`py-glm` is a library for fitting, inspecting, and evaluating Generalized Linear Models in python.

# Features

## Model Fitting

`py-glm` supports models from various exponential families:

```python
from glm.glm import GLM
from glm.families import Gaussian, Bernoulli, Poisson, Exponential

linear_model = GLM(family=Gaussian())
logistic_model = GLM(family=Bernoulli())
poisson_model = GLM(family=Poisson())
exponential_model = GLM(family=Exponential())
```

Models with dispersion parameters are also supported.  The dispersion parameters
in these models is estimated using the deviance.

```python
from glm.families import QuasiPoisson, Gamma

quasi_poisson_model = GLM(family=QuasiPoisson())
gamma_model = GLM(family=Gamma())
```

Fitting a model proceeds in `sklearn` style, and uses the Fisher scoring algorithm:

```python
logistic_model.fit(X, y_logistic)
```

Offsets and sample weights are supported when fitting:

```python
linear_model.fit(X, y_linear, sample_weights=sample_weights)
poisson_nmodel.fit(X, y_poisson, offset=np.log(expos))
```

Predictions are also made in sklearn style:

```python
logistic_model.predict(X)
```


## Inference

Once the model is fit, parameter estimates and parameter covariance estimates are available:

```python
logistic_model.coef_
logistic_model.coef_covariance_matrix_
logistic_model.coef_standard_error_
```

Resampling methods are also supported: the parametric and non-parametric bootstraps:

```python
from glm.simulation import Simulation

sim = Simulation(logistic_model)
sim.parametric_bootstrap(X, n_sim=1000)
sim.non_parametric_bootstrap(X, n_sim=1000)
```

## Regularization

Ridge regression is supported for each model (note, the regularization parameter is called `alpha` instead of `lambda` due to `lambda` being a reserved word in python):

```python
logistic_model.fit(X, y_logistic, alpha=1.0)
```

## References

* Marlene Müller (2004). [Generalized Linear Models](http://www.marlenemueller.de/publications/HandbookCS.pdf).
