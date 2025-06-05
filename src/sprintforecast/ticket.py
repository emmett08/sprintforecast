from .distributions import BetaDistribution

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Ticket:
    optimistic: float
    mode: float
    pessimistic: float

    def beta_params(self) -> tuple[float, float]:
        if not self.optimistic < self.mode < self.pessimistic:
            raise ValueError("optimistic < mode < pessimistic must hold")
        a = 1 + 4 * (self.mode - self.optimistic) / (self.pessimistic - self.optimistic)
        b = 1 + 4 * (self.pessimistic - self.mode) / (self.pessimistic - self.optimistic)
        return a, b

    def base_distribution(self) -> BetaDistribution:
        a, b = self.beta_params()
        return BetaDistribution(a, b, self.optimistic, self.pessimistic)