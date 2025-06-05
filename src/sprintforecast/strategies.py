import numpy as np
from .distributions import Distribution
from .types import Sample

from numpy.random import Generator
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class ExecutionStrategy:
    distribution: Distribution

    def sample(self, size: int, *, rng: Generator) -> Sample:
        return self.distribution.sample(size, rng=rng)


@dataclass(slots=True, frozen=True)
class ReviewStrategy:
    distribution: Distribution

    def sample(self, size: int, *, rng: Generator) -> Sample:
        return self.distribution.sample(size, rng=rng)


@dataclass(slots=True, frozen=True)
class CapacityStrategy:
    distribution: Distribution

    def sample(self, *, rng: Generator) -> float:
        return float(np.asarray(self.distribution.sample(size=(), rng=rng)))