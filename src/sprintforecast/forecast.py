from __future__ import annotations
import typing as t
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .error import ErrorDistribution
from .capacity import CapacityPosterior
from .pert import Pert

@dataclass(slots=True)
class SprintForecaster:
    error_dist: ErrorDistribution
    capacity_post: CapacityPosterior
    triads: pd.DataFrame
    paths: int = 10_000
    def _simulate(self, sprint_hours: float) -> np.ndarray:
        triad_objs = self.triads[["o", "m", "p"]].apply(lambda r: Pert(*r), axis=1).tolist()
        x_prior = np.vstack([t.sample(self.paths) for t in triad_objs]).T
        errs = self.error_dist.sample(x_prior.size).reshape(self.paths, -1)
        effort = np.exp(errs) * x_prior
        total = effort.sum(axis=1)
        cap = self.capacity_post.sample(self.paths)
        return sprint_hours * total / cap
    @staticmethod
    def _crps(samples: np.ndarray) -> float:
        x = np.sort(samples)
        n = x.size
        coef = 2 * np.arange(1, n + 1) - n - 1
        mean_abs = (2.0 / (n * n)) * np.sum(coef * x)
        return 0.5 * mean_abs
    def summary(self, sprint_hours: float) -> dict[str, t.Any]:
        t_complete = self._simulate(sprint_hours)
        prob = (t_complete <= sprint_hours).mean()
        brier = prob * (1 - prob)
        crps = self._crps(t_complete)
        p50, p80, p95 = np.percentile(t_complete, [50, 80, 95])
        return {
            "p50": float(p50),
            "p80": float(p80),
            "p95": float(p95),
            "P_goal": float(prob),
            "Brier": float(brier),
            "CRPS": float(crps),
        }
