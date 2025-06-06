from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

import numpy as np
from numpy.random import Generator

from .queue_simulator import QueueSimulator
from .rng_singleton import RNGSingleton
from .size import Size
from .sprint_intake import SprintIntake
from .strategies import CapacityStrategy, ExecutionStrategy, ReviewStrategy
from .ticket import Ticket


@dataclass(slots=True, frozen=True)
class ForecastResult:
    probability: float
    expected_carry: float


class ForecastEngine(abc.ABC):
    @abc.abstractmethod
    def forecast(self, draws: int) -> ForecastResult: ...


@dataclass(slots=True, frozen=True)
class SprintForecastEngine(ForecastEngine):
    tickets: Sequence[Ticket]
    exec_strategy: ExecutionStrategy
    review_strategy: ReviewStrategy
    capacity_strategy: CapacityStrategy
    simulator: QueueSimulator
    remaining_hours: float
    rng: Generator = RNGSingleton.rng()

    def _sample_durations(self) -> np.ndarray:
        base = np.array(
            [t.base_distribution().sample(rng=self.rng)[0] for t in self.tickets],
            dtype=float,
        )
        error = self.exec_strategy.sample(len(self.tickets), rng=self.rng)
        review = self.review_strategy.sample(len(self.tickets), rng=self.rng)
        return np.exp(error) * base + review

    def forecast(self, draws: int = 2_000) -> ForecastResult:
        success = 0
        carry = 0.0
        for _ in range(draws):
            durations = self._sample_durations()
            finish, span = self.simulator.simulate(durations)
            if span <= self.remaining_hours:
                success += 1
            carry += (finish > self.remaining_hours).sum()
        return ForecastResult(success / draws, carry / draws)

    def _mean_hours(self, t: Ticket) -> float:
        return (t.optimistic + 4 * t.mode + t.pessimistic) / 6

    def _bucket(self, h: float) -> Size:
        for s in Size:
            if h <= s.value:
                return s
        return Size.XXL

    def suggested_intake(
        self,
        next_capacity_hours: float,
        backlog: Sequence[Ticket],
        carry_hours: float = 0.0,
        allocation: Mapping[Size, float] | None = None,
        resolved: Collection[int] = (),
    ) -> SprintIntake:
        alloc = allocation or {
            Size.XS: 0.05,
            Size.S: 0.15,
            Size.M: 0.35,
            Size.L: 0.30,
            Size.XL: 0.15,
        }
        avail = next_capacity_hours - carry_hours
        buckets: dict[Size, int] = {s: 0 for s in Size}
        ready = [t for t in backlog if t.dependencies.issubset(resolved)]
        sorted_backlog = sorted(ready, key=self._mean_hours)
        used = 0.0
        quota = {s: avail * alloc[s] for s in alloc}
        for t in sorted_backlog:
            h = self._mean_hours(t)
            b = self._bucket(h)
            if used + h > avail or b is Size.XXL:
                continue
            if buckets[b] * h >= quota[b]:
                continue
            buckets[b] += 1
            used += h
            if used >= avail:
                break
        return SprintIntake(buckets, used)
