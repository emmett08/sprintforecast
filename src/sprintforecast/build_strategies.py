
from typing import Sequence

import numpy as np
from .distributions import EmpiricalDistribution
from .strategies import CapacityStrategy, ExecutionStrategy, ReviewStrategy

def build_strategies(
    dev_hours: Sequence[float],
    review_hours: Sequence[float],
    sprint_caps: Sequence[float],
) -> tuple[ExecutionStrategy, ReviewStrategy, CapacityStrategy]:
    dev = np.asarray(dev_hours, dtype=float)
    review = np.asarray(review_hours, dtype=float)
    caps = np.asarray(sprint_caps, dtype=float)

    exec_err = np.log(dev / dev.mean())
    return (
        ExecutionStrategy(EmpiricalDistribution(exec_err)),
        ReviewStrategy(EmpiricalDistribution(review)),
        CapacityStrategy(EmpiricalDistribution(caps)),
    )

