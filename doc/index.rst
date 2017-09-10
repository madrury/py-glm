.. glm documentation master file, created by
   sphinx-quickstart on Sat Sep  9 20:29:16 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to glm's documentation!
===============================

`py-glm` fits generalized linear models in Python.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: glm

.. module:: glm.glm

The GLM Class
-------------

The `GLM` class fits generalized linear models.

    When creating a `GLM` object, you must supply an `ExponentialFamily` object::

    logistic_model = GLM(family=Bernoulli())

    `GLM` objects can be fit and subsequently used to generate predictions.::

    logisitic_model.fit(X, y)
    logistic_model.predict(X)

.. autoclass:: GLM
   :members:

.. module:: glm.families

.. autoclass:: Gaussian
.. autoclass:: Bernoulli 
.. autoclass:: QuasiPoisson
.. autoclass:: Poisson
.. autoclass:: Gamma
.. autoclass:: Exponential



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
