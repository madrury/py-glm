import numpy as np
import pytest

from glm.extras.confidence_intervals import ConfidenceIntervals


def test_covers_includes_the_endpoints() -> None:
    cis = ConfidenceIntervals(
        lower=np.array([-1.0, 0.0, 1.0, 2.0]),
        upper=np.array([1.0, 2.0, 3.0, 4.0]),
        level=0.95,
    )
    np.testing.assert_array_equal(cis.covers(1.0), [True, True, True, False])


def test_raises_on_mismatched_endpoint_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ConfidenceIntervals(
            lower=np.array([0.0, 1.0]),
            upper=np.array([1.0]),
            level=0.95,
        )


@pytest.mark.parametrize("level", [0.0, 1.0, 95.0])
def test_raises_on_a_level_outside_the_unit_interval(level: float) -> None:
    with pytest.raises(ValueError, match="level must be in"):
        ConfidenceIntervals(
            lower=np.array([0.0]),
            upper=np.array([1.0]),
            level=level,
        )
