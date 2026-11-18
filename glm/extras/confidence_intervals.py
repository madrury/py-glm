from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConfidenceIntervals:
    """Confidence intervals for a scalar quantity, one for each sample.

    Attributes
    ----------
    lower : NDArray[np.float64]
        The lower endpoint of each interval.
    upper : NDArray[np.float64]
        The upper endpoint of each interval, the same shape as lower.
    level : float
        The confidence level, the intended probability that an interval
        covers the true value, e.g. 0.95.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    level: float

    def __post_init__(self) -> None:
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                "The lower and upper endpoints must have the same shape, got "
                f"{self.lower.shape} and {self.upper.shape}."
            )
        if not 0 < self.level < 1:
            raise ValueError(f"The level must be in (0, 1), got {self.level}.")

    def covers(self, value: float) -> NDArray[np.bool_]:
        """Whether each interval contains a value, endpoints included.

        Parameters
        ----------
        value : float
            The value to check, usually the true value of the quantity.

        Returns
        -------
        NDArray[np.bool_]
            True for each interval that contains value.
        """
        return (self.lower <= value) & (value <= self.upper)
