from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ._rng import rng

@dataclass(frozen=True, slots=True)
class CapacityPosterior:
    mu: float
    sigma: float
    def sample(self, size: int = 1) -> np.ndarray:
        return np.exp(rng.normal(self.mu, self.sigma, size))
