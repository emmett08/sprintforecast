from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ._rng import rng

@dataclass(frozen=True, slots=True)
class Pert:
    o: float
    m: float
    p: float
    def _alpha_beta(self) -> tuple[float, float]:
        a = 1 + 4 * (self.m - self.o) / (self.p - self.o)
        b = 1 + 4 * (self.p - self.m) / (self.p - self.o)
        return a, b
    def sample(self, size: int = 1) -> np.ndarray:
        a, b = self._alpha_beta()
        return self.o + (self.p - self.o) * rng.beta(a, b, size)
