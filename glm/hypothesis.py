"""This module encapsulates verious hypothesis tests associated with generalized
linear models.
"""

import numpy as np

from typing import Self
from dataclasses import dataclass
from formulaic.utils.constraints import LinearConstraints

from glm.types import (
    ConstraintMatrix,
    ConstraintValues,
    NestingMatrix,
    DesignMatrix,
    CoefficientVector,
    ReferenceDistribution,
    ReferenceDistributionName,
    TestStatisticName,
)
from glm.utils import check_constraint_formula


@dataclass(frozen=True)
class TestDistribution:
    """The reference distribution for a tests of linear hypothesis. Encapsulates
    the test statistic's name, the reference distrivution's name, and its
    distribution under the linear hypothesis."""

    statistic: TestStatisticName
    name: ReferenceDistributionName
    distribution: ReferenceDistribution


@dataclass(frozen=True, eq=False)
class ConstraintEquation:
    """Encapuslates a linear constraint equation  Cβ = c."""

    C: ConstraintMatrix
    c: ConstraintValues

    def __post_init__(self) -> None:
        C, c = self.C, self.c
        if C.ndim != 2:
            raise ValueError(f"C is not a matrix, its shape is {C.shape}.")
        if C.ndim != 1:
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
                "Constraint matrix C is not full rank, rank = {rC}."
                " A constraint may be repeated."
            )

    @classmethod
    def from_arrays(cls, C: ConstraintMatrix, c: ConstraintValues | None) -> Self:
        if not c:
            c = np.zeros(C.shape[0])
        return cls(C, c)

    @classmethod
    def from_constraint_formula(cls, formula: str, names: list[str]) -> Self:
        check_constraint_formula(formula)
        lc = LinearConstraints.from_spec(formula, variable_names=names)
        return cls(lc.constraint_matrix, lc.constraint_values)

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
        Probability distribution of the test statsic under the null hypothesis.

    statistic: float
        Value of the test statistic.

    q: float
        The number of linearly independent constraints defining the null.

    rdof: float
        Residual Degrees of freedom.

    p_value: float
        P-value.
    """

    constraint: ConstraintEquation
    reference: TestDistribution
    statistic: float
    q: int
    rdof: float | None
    p_value: float

    def summary(self) -> None:
        # TODO: Make this nice.
        print(f"Wald Test.")
