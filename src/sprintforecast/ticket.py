from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from .distributions import BetaDistribution


@dataclass(slots=True, frozen=True)
class Ticket:
    number: int
    optimistic: float
    mode: float
    pessimistic: float
    dependencies: FrozenSet[int] = frozenset()

    def beta_params(self) -> tuple[float, float]:
        if not self.optimistic < self.mode < self.pessimistic:
            raise ValueError("optimistic < mode < pessimistic must hold")
        a = 1 + 4 * (self.mode - self.optimistic) / (self.pessimistic - self.optimistic)
        b = 1 + 4 * (self.pessimistic - self.mode) / (self.pessimistic - self.optimistic)
        return a, b

    def base_distribution(self) -> BetaDistribution:
        a, b = self.beta_params()
        return BetaDistribution(a, b, self.optimistic, self.pessimistic)
