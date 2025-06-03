from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ._rng import rng

@dataclass(slots=True)
class MomentumModel:
    caps: np.ndarray
    hours_per_sprint: float
    def _fit(self) -> tuple[float, float, float]:
        s = np.arange(1, len(self.caps) + 1)
        y = np.log(self.caps)
        b, a = np.polyfit(s, y, 1)
        resid = y - (a + b * s)
        tau = resid.std(ddof=1)
        return a, b, tau
    def forecast(self, size: int = 1) -> np.ndarray:
        a, b, tau = self._fit()
        k = len(self.caps) + 1
        mean = a + b * k
        var = tau ** 2 + tau ** 2 / k
        return np.exp(rng.normal(mean, np.sqrt(var), size))
    def momentum(self) -> float:
        _, b, _ = self._fit()
        return float(np.exp(b * self.hours_per_sprint))
