import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_random_state() -> None:
    """Seed numpy's global random state, which the data generators use, so
    that every test run sees the same simulated data."""
    np.random.seed(154)
