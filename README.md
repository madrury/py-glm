# py-glm: Generalized Linear Models in Python

`py-glm` is a library for fitting, inspecting, and evaluating Generalized Linear Models in python.

# Features

## Model Fitting

`py-glm` supports models from various exponential families:

```python
from glm import GLM
from families import Gaussian, Bernoulli, Poisson

linear_model = GLM(family=Gaussian())
logistic_model = GLM(family=Bernoulli())
model = GLM(family=Poisson())
```

Fitting a model is achieved in `sklearn` style, and uses the Fisher scoring algorithm:

```python
logistic_model.fit(X, y_logistic)
```

Offsets and sample weights are supported:

```python
linear_model.fit(X, y_linear, sample_weights=sample_weights)
poisson_nmodel.fit(X, y_poisson, offset=np.log(expos))
```

## Inference

Once the model is fit, parameter estimates and parameter covariance estimates are available:

```python
logistic_model.coef_
logistic_model.coef_covariance_matrix_
model.coef_standard_error_
```

Resampling methods are also supported: the parametric and non-parametric bootstraps:

```
from simulation import Simulation

sim = Simulation(logistic_model)
sim.parametric_bootstrap(X, n_sim=1000)
sim.non_parametric_bootstrap(X, n_sim=1000)
```

## Regularization

Ridge regression is supported for each model (note, the regularization parameter is called `alpha` instead of `lambda` due to `lambda` being a reserved word in python):

```python
logistic_model.fit(X, y_logistic, alpha=1.0)
```
