"""This module contains top level functions for encoding data frames into design
matrices and responses using formulas.

Encoding happens in two steps.

1) Learning a formula on training data creates its specs (specifications):
opeartions to preform on tabular data to make a model design matrix. For
example: the categorical levels, and the state of stateful transforms such as
center(x) or spline knots.

2) Applying the specs to tabular data, training or otherwise, executes the
operations and so encodes it exactly in the same manner as the training data.

We deliberately keep these two steps seperated throughout the library, with the
exception that it is safe to learn and endode *training* data as one operation.
It is important that specs are only ever learned from *training* data, and
applied, never learned from, test data. Keeping the two concerns seperate
ensures we meet this standard.
"""

import warnings
from typing import Any

import formulaic
from formulaic.errors import DataMismatchWarning
import numpy as np

from glm.types import DesignFrame, DesignMatrix, DesignSpec, Formula, ResponseVector


def learn_formula(
    formula: Formula, X: DesignFrame, *, penalized: bool
) -> formulaic.ModelMatrices:
    """Learn a formula's specs from training data, and encode that data.

    Parameters
    ----------
    formula: str
        A formula with a response, e.g. 'y ~ x'.

    X: pd.DataFrame
        The training data. Every row is used to learn the specs, whatever its
        weight, so leave out any rows the specs must not learn from.

    penalized: bool
        Will the model fit to the encoding be penalized? Penalized models encode
        every level of a categorical variable, rather than dropping a reference
        level, so that the penalty treats all levels alike. This does not cause
        any rank deficiency, the penalty keeps the fit well defined.

    Returns
    -------
    matrices: formulaic.ModelMatrices
        The encoded design matrix (rhs) and response (lhs), each carrying the
        learned model_spec.
    """
    # Raise on missing values rather than dropping rows, which would silently
    # misalign the data with any offset or sample weights.
    matrices = formulaic.model_matrix(
        formula,
        X,
        output="numpy",
        na_action="raise",
        ensure_full_rank=not penalized,
    )
    if not isinstance(matrices, formulaic.ModelMatrices):
        raise ValueError(f"Formula {formula!r} must specify a response, e.g. 'y ~ x'.")
    return matrices


def encode_design(spec: DesignSpec, X: DesignFrame) -> DesignMatrix:
    """Encode a data frame into a design matrix with learned specs.

    Parameters
    ----------
    spec: formulaic.ModelSpec
        The learned specs of a formula's right hand side.

    X: pd.DataFrame
        The data to encode. Nothing is learned from it.

    Returns
    -------
    design_matrix: array, shape (n_samples, n_coefficients)
        The design matrix.
    """
    # When a model is unpenalized, we check for a new level, one not seen during
    # learning the spec.
    #
    # Formulaic encodes a level it didn't see in training as all zero
    # indicators. With reference coding (formulaic's ensure_full_rank), that's
    # exactly the reference level's encoding, so an unseen level would be
    # silently predicted as the reference level. We check explicitly before
    # encoding, and raise an informative error.
    #
    # We run this check before calling get_model_matrix from formulaic, so as to
    # supply a more detailed contextual error message to the caller.
    if spec.ensure_full_rank:
        unseen = unseen_categories(spec, X)
        if unseen:
            described = "; ".join(
                f"{factor}: {levels}" for factor, levels in unseen.items()
            )
            raise ValueError(
                f"X contains categorical levels the model wasn't fit with "
                f"({described}). An unpenalized model encodes each level "
                f"relative to a reference level, so it can't predict for "
                f"levels it hasn't seen: refit with them, or merge them into "
                f"levels the model has seen."
            )
    with warnings.catch_warnings():
        # With every level encoded, as for penalized models, all zero
        # indicators for an unseen level are intended (see GLM.predict), so
        # formulaic's warning that they will skew the results is not needed.
        warnings.simplefilter("ignore", DataMismatchWarning)
        # Raise on missing values rather than dropping rows, which would
        # silently misalign the design matrix with the rows of X.
        design_matrix: DesignMatrix = np.asarray(
            spec.get_model_matrix(X, output="numpy", na_action="raise")
        )
    return design_matrix


def encode_response(spec: DesignSpec, X: DesignFrame) -> ResponseVector:
    """Take the response from a data frame with learned specs.

    Parameters
    ----------
    spec: formulaic.ModelSpec
        The learned specs of a formula's left hand side.

    X: pd.DataFrame
        The data holding the response.

    Returns
    -------
    response: array, shape (n_samples, )
        The response vector.
    """
    response: ResponseVector = np.asarray(
        spec.get_model_matrix(X, output="numpy", na_action="raise")
    ).squeeze(axis=1)
    return response


def unseen_categories(spec: DesignSpec, X: DesignFrame) -> dict[str, list[Any]]:
    """The levels of each categorical variable in X that the model's learned
    spec has not yet seen, keyed by the variable's expression in the formula."""
    unseen = {}
    for factor, (kind, state) in spec.encoder_state.items():
        if kind.value != "categorical":
            continue
        # Let formulaic evaluate the factor's expression on X, so that a
        # categorical variable is handled however it appears in the formula.
        observed_spec = formulaic.model_matrix(f"0 + {factor}", X).model_spec
        if observed_spec is None:
            raise ValueError(f"Could not evaluate the factor {factor} on X.")
        _, observed_state = observed_spec.encoder_state[factor]
        new_levels = [
            level
            for level in observed_state["categories"]
            if level not in state["categories"]
        ]
        if new_levels:
            unseen[str(factor)] = new_levels
    return unseen
