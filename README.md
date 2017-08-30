# py-glm: Generalized Linear Models in Python

`py-glm` is a library for fitting, inspecting, and evaluating Generalized Linear Models in python.

# Features

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
poissonmodel.fit(X, y_poisson, offset=np.log(expos))
```

Once the model is fit, parameter estimates and parameter covariance estimates are available:

```python
logistic_model.coef_
logistic_model.coef_covariance_matrix_
model.coef_standard_error_
```
