import numpy as np

import heapq
from dataclasses import dataclass
from typing import Sequence

@dataclass(slots=True, frozen=True)
class QueueSimulator:
    workers: int

    def simulate(self, durations: Sequence[float]) -> tuple[np.ndarray, float]:
        n = len(durations)
        finish = np.empty(n)
        heap: list[tuple[float, int]] = [(0.0, w) for w in range(self.workers)]
        order = np.argsort(-np.asarray(durations))
        for idx in order:
            t, w = heapq.heappop(heap)
            t += durations[idx]
            heapq.heappush(heap, (t, w))
            finish[idx] = t
        makespan = max(t for t, _ in heap)
        return finish, makespan