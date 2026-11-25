import pytest

from glm.types import ConstraintFormula
from glm.utils import check_constraint_formula

# Formulas that chain "=" within a single constraint, which formulaic would
# silently misread.
CHAINED_FORMULAS: list[ConstraintFormula] = [
    "x1 = x2 = x3",
    "x1 = x2 = 0, x3",
    "x1, x2 = x3 = 0",
]

# Valid formulas, including ones that a check on the raw string would get
# wrong.
VALID_FORMULAS: list[ConstraintFormula] = [
    "x1 = 0",
    "x1 = 0, x2 = 0",
    "x1, x2",
    "x1 = x2, x2 = x3",
    "x + `bs(x, df=3)[0]` = 0",
    "bs(x, df=3)[0] + x= 0",
]


@pytest.mark.parametrize("formula", CHAINED_FORMULAS)
def test_chained_equals_raises(formula: ConstraintFormula) -> None:
    with pytest.raises(ValueError, match="chains"):
        check_constraint_formula(formula)


@pytest.mark.parametrize("formula", VALID_FORMULAS)
def test_valid_formula_passes(formula: ConstraintFormula) -> None:
    check_constraint_formula(formula)
