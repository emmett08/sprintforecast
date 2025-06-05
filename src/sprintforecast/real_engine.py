from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence
from numpy.random import Generator
from .rng_singleton import RNGSingleton
from .forecast import ForecastEngine, ForecastResult

@dataclass(slots=True, frozen=True)
class RealSprintForecastEngine(ForecastEngine):
    durations: Sequence[float]
    remaining_hours: float
    rng: Generator = RNGSingleton.rng()

    def _sample(self, draws: int) -> np.ndarray:
        base = np.asarray(self.durations, dtype=float)
        if draws == 1:
            return base[None, :]
        return self.rng.choice(base, size=(draws, base.size), replace=True)

    def forecast(self, draws: int = 1_000) -> ForecastResult:
        if not self.durations:
            raise ValueError("no empirical duration data")
        dur = self._sample(draws)
        span = dur.max(axis=1)
        prob = float((span <= self.remaining_hours).mean())
        carry = float((dur > self.remaining_hours).sum(axis=1).mean())
        return ForecastResult(prob, carry)
