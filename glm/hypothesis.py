"""This module encapsulates various hypothesis tests associated with generalized
linear models.
"""
from __future__ import annotations
from typing import cast, TYPE_CHECKING
import numpy as np
import scipy.linalg as linalg

import warnings
from typing import Self
from dataclasses import dataclass
from formulaic.utils.constraints import LinearConstraints

from glm.families import TestDistribution
from glm.types import (
    DesignFrame,
    ConstraintMatrix,
    ConstraintValues,
    ConstraintFormula,
    ModelFormula,
    DesignMatrix,
    CoefficientVector,
)
from glm.utils import check_constraint_formula, hash_array
from glm.encoding import encode_design, learn_formula

if TYPE_CHECKING:
    from glm.glm import GLM


@dataclass(frozen=True, eq=False)
class ConstraintEquation:
    """Encapsulates a linear constraint equation Cβ = c, and provides various
    methods to construct one.
    """

    C: ConstraintMatrix
    c: ConstraintValues
    description: str

    def __post_init__(self) -> None:
        C, c = self.C, self.c
        if C.ndim != 2:
            raise ValueError(f"C is not a matrix, its shape is {C.shape}.")
        if c.ndim != 1:
            raise ValueError(f"c is not a vector, its shape is {c.shape}.")
        if C.shape[0] != c.shape[0]:
            raise ValueError(
                f"Arrays are unaligned with C shape {C.shape} and c shape {c.shape}."
            )
        if not np.isfinite(C).all():
            raise ValueError(f"Constraint matrix C is not finite.")
        if not np.isfinite(c).all():
            raise ValueError(f"Constraint values c is not finite.")
        rC = np.linalg.matrix_rank(C)
        if rC != C.shape[0]:
            raise ValueError(
                f"Constraint matrix C is not full rank, rank = {rC}."
                " A constraint may be repeated."
            )

    def __repr__(self) -> str:
        return f"ConstraintEquation({self.description})"

    @classmethod
    def from_arrays(cls, C: ConstraintMatrix, c: ConstraintValues | None) -> Self:
        """Construct a ConstraintEquation from explicit left and right hand
        sides.

        Parameters
        ----------
        C: ConstraintMatrix
            Left hand side, matrix of shape (q, p).
        
        c: ConstraintValues
            Right hand side, vector of shape (q,). Defaults to zero.
        """
        if c is None:
            c = np.zeros(C.shape[0])
        return cls(C, c, f"Cβ = c, C of shape {C.shape}")

    @classmethod
    def from_constraint_formula(cls, formula: ConstraintFormula, names: list[str]) -> Self:
        """Construct a ConstraintEquation from a formula listing explicit
        constraint equations. 

        Valid examples:
            from_constraint_formula('x1 = x2, x2 = x3', names=model.X_names)
            from_constraint_formula('x1 = 0, x2 = 0', names=model.X_names)
            from_constraint_formula('x1, x2', names=model.X_names)
            from_constraint_formula('C(x)[T.a] = 0', names=model.X_names)
            from_constraint_formula('cs(x, df=3)[1] = 0', names=model.X_names)
        
        Invalid Examples:
            from_constraint_formula('x1 = x2 = x3', names=model.X_names)
            from_constraint_formula('x1 = x2 = 0', names=model.X_names)
        
        Parameters
        ----------
        formula: ConstraintFormula
            Formula listing the linear constraints.
        
        names: list[str]
            Full list of design column names for model, in order the model
            expects them. Enumerates all terms that may appear in the constraint
            formulas.
        """
        check_constraint_formula(formula)
        lc = LinearConstraints.from_spec(formula, variable_names=names)
        return cls(lc.constraint_matrix, lc.constraint_values, formula)

    @classmethod
    def from_null_model_formula(
        cls, 
        model: GLM, 
        formula: ModelFormula, 
        X: DesignFrame, 
        tol: float = 1e-6
    ) -> Self:
        """Construct a ConstraintEquation from a formula specifying a null model
        nested within a full model. 

        For example:
            model = GLM(family=Gaussian()).fit(X, formula='y ~ x1 + x2 + x3')
            ConstraintEquation.from_null_model_formula(model, 'y ~ x1', X)
        
        In the example, the null hypothesis being asserted is equivalent to:
            ConstraintEquation.from_constraint_formula(
                'x2 = 0, x3 = 0', 
                names=model.X_names
            )
        
        but more sophisticated uses are possible, for example:
            model = GLM(family=Gaussian()).fit(
                X, formula='y ~ cs(x, df=3, constraints="center")'
            )
            ConstraintEquation.from_null_model_formula(model, 'y ~ x', X)
        
        asserts the null that the effect of x on y is linear. This still works
        because the models are *nested*: there is a linear mapping A from the
        parameter space of the null model into the parameter space of the
        full model that preserves model predictions. For example: linear
        functions are within the span of cubic spline basis functions, there is
        a choice of spline parameters that give every possible linear function.
        
        This method constructs the constraint matrix C by first constructing the
        mapping A by least squares, c is always zero. The constraint equations
        then span the null space of Aᵗ.

        It is important to note that constructing C from the null model
        specification requires learning the design matrix of the null model, so
        this method requires the training frame for the full model. This
        training frame must result in the exact same design matrix for the full
        model, else this method raises.

        We require the model formula to be complete with a left hand and right
        hand side, so `~ x1` is not allowed.

        Parameters
        ----------
        model: GLM
            The full model, within which the null model is nested.

        formula: ModelFormula
            Formula specification for the null model.
        
        X: DesignFrame
            The design frame used to train the full model. This will raise if
            the exact same training data is not recovered, checked with a hash.
        
        tol: float
            Tolerance for the relative residuals used when testing if the
            designs are nested.
        """
        if model.X_spec is None:
            raise ValueError(
                f"The full model is required to be fit using a model formula."
            )
        X_full = encode_design(model.X_spec, X)
        if (
            model.fit_inputs_ is not None 
            and hash_array(X_full) != model.fit_inputs_.X_hash
        ):
            raise ValueError(
                "Design matrix for full model differs from the training design."
                " The X passed in here should be *exactly* that on which the"
                " model was trained.\n"
                f"Training hash: {model.fit_inputs_.X_hash}\n"
                f"X hash: {hash_array(X_full)}"
            )
        X_null = cast(
            DesignMatrix, 
            np.asarray(learn_formula(formula, X, penalized=False).rhs)
        )
        A, _, rank, _ = np.linalg.lstsq(X_full, X_null)
        residuals = X_full @ A - X_null
        is_nested = all(
            np.linalg.norm(residuals, axis=0) <= 
            tol * np.linalg.norm(X_null, axis=0)
        )
        # TODO: Is it worth providing an example vector that is not
        # representable?
        if not is_nested:
            raise ValueError("Full and null models are not nested.")
        if np.linalg.matrix_rank(A) == rank:
            raise ValueError(
                "The null model and the full model are equivalent, so the null"
                " hypothesis is vacuously true."
            )
        C = linalg.null_space(A.T).T
        return cls(C, np.zeros(C.shape[0]), formula)

    def meets(self, β: CoefficientVector) -> bool:
        return np.allclose(self.C @ β, self.c)


@dataclass(frozen=True)
class WaldTestResult:
    """Encapsulates the result of a Wald test.

    Parameters
    ----------
    constraint: ConstraintEquation
        The left and right hand side of the constraint equation defining the
        null hypothesis.

    reference: TestDistribution
        Probability distribution of the test statistic under the null hypothesis.

    statistic: float
        Value of the test statistic.

    q: int
        The number of linearly independent constraints defining the null.

    rdof: float
        Residual degrees of freedom.

    p_value: float
        P-value.
    """

    constraint: ConstraintEquation
    reference: TestDistribution
    statistic: float
    q: int
    rdof: float
    p_value: float

    def __repr__(self) -> str:
        return (
            f"WaldTestResult({self.reference.name},"
            f" {self.reference.statistic}={self.statistic:#.3g},"
            f" p_value={self.p_value:#.3g})"
        )

    def summary(self) -> None:
        message = (
            "Wald Test\n"
            f"  Null Hypothesis: {self.constraint.description}\n"
            f"  Constraints:     q = {self.q}\n"
            f"  Statistic:       {self.reference.statistic} = {self.statistic:#.3g}\n"
            f"  Reference:       {self.reference.name}\n"
            f"  p-value:         {self.p_value:#.3g}\n"
        )
        print(message)


def _wald(model: GLM, constraint: ConstraintEquation) -> WaldTestResult:
    """Core calculation for Wald tests."""
    if (
        model.coef_ is None 
        or model.information_matrix_ is None
        or model.n is None
        or model.p is None
    ):
        raise ValueError("Model is not fit.")
    if model.alpha > 0:
        raise ValueError("Penalized models do not support hypothesis testing.")
    C, c = constraint.C, constraint.c
    q, rdof = C.shape[0], model.n - model.p
    r = C @ model.coef_ - c
    W = float(
        r.T @ np.linalg.solve(C @ model.coef_covariance_matrix_ @ C.T, r)
    )
    statistic = model.family.joint_test_statistic(W, q)
    null = model.family.joint_test(q, rdof)
    p_value = float(null.distribution.sf(statistic))
    return WaldTestResult(constraint, null, statistic, q, rdof, p_value)


class HypothesisTests:
    """Encapsulates various tests of linear constraint null hypotheses on model
    coefficients.

    Each public method here provides a specific flavor of test: wald
    (implemented), likelihood (pending), and score/rao (pending). All methods
    share the same interface, which provides three ways to specify each test:

    Method 1: Constraint equations:
    Specify explicit constraint equations with the `hypothesis` keyword, which
    is also the first argument when passing by position. For example:

        model.tests.wald('x1 = x2, x2 = x3')
        model.tests.wald('x1 = 0, x2 = 0')
        model.tests.wald('x1, x2')              <- Zero right hand side inferred
        model.tests.wald('C(x)[T.a] = 0')       <- Category level `a`
        model.tests.wald('cs(x, df=3)[1] = 0')  <- One spline basis coefficient
    
    Shorthands that use multiple = signs in one constraint equation are not
    supported. These are invalid:

        model.tests.wald('x1 = x2 = x3')
        model.tests.wald('x1 = x2 = 0')
    
    The special case model.tests.wald('x1 = 0') is the usual null test on single
    coefficients.
    
    Method 2: Nested Models:
    Specify a formula for a sub-model of a full model using the `null` keyword.
    This method requires re-supplying the training data for the full model, as
    it is needed to compute the implied constraint equations. For example:

        model = GLM(Poisson()).fit(X, formula='y ~ x1 + x2 + x3')
        model.tests.wald(null='y ~ x1', X=X)
    
    Testing that an effect is linear:

        model = GLM(Poisson()).fit(
            X, formula='y ~ cs(x, df=5, constraints="center")'
        )
        model.tests.wald(null='y ~ x', X=X)
    
    Method 3: Explicit constraint arrays:
    This is intended to be an uncommonly used escape hatch. You may construct
    your own constraint equation Cβ = c:

        model.tests.wald(constraint=ConstraintEquation.from_arrays(C, c))
    
    For details of each flavor of test, see the corresponding method.
    """

    def __init__(self, model: GLM):
        self.model = model

    def wald(
        self, 
        hypothesis: ConstraintFormula | None = None, 
        *, 
        null: ModelFormula | None = None,
        constraint: ConstraintEquation | None = None,
        X: DesignFrame | None = None
    ) -> WaldTestResult:
        """A Wald test whose null hypothesis is a linear constraint equation on
        model coefficients Cβ = c.

        Assuming that β̂ is asymptotically Gaussian(β, V), then the wald test
        statistic:

                W = (Cβ̂ − c)ᵗ (C V Cᵗ)⁻¹ (Cβ̂ − c)

        is asymptotically χ²(q) distributed, where q is the count of linearly
        independent constraints. In case the model estimates a dispersion, the
        test statistic is W/q, which is asymptotically F(q, n - p) distributed
        under the null.

        Exactly one of the parameters `hypothesis`, `null`, or `constraint` must
        be supplied.

        Parameters
        ----------
        hypothesis: ConstraintFormula | None
            Constraint equations. For example `x1 = 0, x2 = 0`.

        null: ModelFormula | None
            Formula specification of null model nested inside of full model.

        constraint: ConstraintEquation | None
            Explicit arrays for constraint equation Cβ = c.

        X: DesignFrame | None
            Training data for full model, needed to construct the constraint
            equations in the nested case.
        
        Returns
        -------
        result: WaldTestResult
            Encapsulates the test result.
        """
        if not self.model.is_fit():
            raise ValueError("Hypothesis tests require a fit model.")
        constraint = self._construct_constraints(hypothesis, null, constraint, X)
        return _wald(self.model, constraint)

    def _construct_constraints(
        self, 
        hypothesis: ConstraintFormula | None = None, 
        null: ModelFormula | None = None,
        constraint: ConstraintEquation | None = None,
        X: DesignFrame | None = None
    ) -> ConstraintEquation:
        which_args = [
            hypothesis is not None,
            null is not None,
            constraint is not None
        ]
        if sum(which_args) != 1:
            raise ValueError(
               "Exactly one of hypothesis, null, or constraint must be supplied."
            )
        if constraint is not None:
            if X is not None:
                warnings.warn(
                    "The data X is not needed when `constraint` is supplied, so is"
                    " being ignored."
                )
        if hypothesis is not None:
            if X is not None:
                warnings.warn(
                    "The data X is not needed when `hypothesis` is supplied, so is"
                    " being ignored."
                )
            assert self.model.X_names is not None
            constraint = ConstraintEquation.from_constraint_formula(
                hypothesis, 
                names=self.model.X_names
            )
        if null is not None:
            if X is None:
                raise ValueError(
                    "Test of nested models requires X to construct the"
                    " constraint equations."
                )
            constraint = ConstraintEquation.from_null_model_formula(
                model=self.model,
                formula=null,
                X=X
            )
        assert constraint is not None
        return constraint