from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
import typing as t
from ._rng import rng
from .error import ErrorDistribution
from .pert import Pert

@dataclass(slots=True)
class IntakePlanner:
    ticket_pool: pd.DataFrame
    error_dist: ErrorDistribution
    next_capacity: np.ndarray
    paths: int = 10_000

    @staticmethod
    def _bucket_idx(x: np.ndarray) -> np.ndarray:
        return np.searchsorted([2, 4, 8], x, side="right")

    def _ticket_sizes(self) -> np.ndarray:
        triads = self.ticket_pool[["o", "m", "p"]].values
        idx = rng.integers(0, len(triads), self.paths)
        x = np.array([Pert(*triads[i]).sample() for i in idx]).ravel()
        e = self.error_dist.sample(self.paths)
        return np.exp(e) * x

    def plan(self, gamma: float = 0.8) -> dict[str, t.Any]:
        cap = rng.choice(self.next_capacity, self.paths)
        load = self._ticket_sizes()
        count = np.ones_like(cap, dtype=int)
        buckets = np.zeros((self.paths, 4), dtype=int)
        np.add.at(buckets, (np.arange(self.paths), self._bucket_idx(load)), 1)
        while True:
            extra = self._ticket_sizes()
            fit = load + extra <= cap
            if not fit.any():
                break
            load[fit] += extra[fit]
            count[fit] += 1
            np.add.at(buckets, (np.flatnonzero(fit), self._bucket_idx(extra[fit])), 1)
        c = count.astype(int)
        n50 = int(np.quantile(c, 0.50, method="nearest"))
        n80 = int(np.quantile(c, 0.20, method="nearest"))
        n95 = int(np.quantile(c, 0.05, method="nearest"))
        rec = int(np.quantile(c, 1 - gamma, method="lower"))
        mix = buckets.mean(axis=0)
        mix /= mix.sum()
        return {
            "n50": n50,
            "n80": n80,
            "n95": n95,
            "recommended": rec,
            "size_mix": {
                "XS": float(mix[0]),
                "S": float(mix[1]),
                "M": float(mix[2]),
                "L": float(mix[3]),
            },
        }
