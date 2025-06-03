from __future__ import annotations
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
    def _simulate(self) -> np.ndarray:
        triad_objs = self.triads[["o", "m", "p"]].apply(lambda r: Pert(*r), axis=1).tolist()
        x_prior = np.vstack([t.sample(self.paths) for t in triad_objs]).T
        errs = self.error_dist.sample(x_prior.size).reshape(self.paths, -1)
        effort = np.exp(errs) * x_prior
        total = effort.sum(axis=1)
        cap = self.capacity_post.sample(self.paths)
        return total / cap
    def summary(self, sprint_hours: float) -> dict[str, t.Any]:
        t_complete = self._simulate()
        prob = (t_complete <= sprint_hours).mean()
        brier = prob * (1 - prob)
        a = np.abs(t_complete[:, None] - t_complete[None, :]).mean()
        b = 0.5 * np.abs(t_complete[:, None] - t_complete[None, :]).mean()
        crps = a - b
        p50, p80, p95 = np.percentile(t_complete, [50, 80, 95])
        return {"p50": p50, "p80": p80, "p95": p95, "P_goal": prob, "Brier": brier, "CRPS": crps}
