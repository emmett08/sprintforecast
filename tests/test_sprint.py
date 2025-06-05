from __future__ import annotations

import math
import sympy as sp
import numpy as np
import pytest

from sprintforecast.distribution_factory import DistributionFactory
from sprintforecast.distributions import BetaDistribution, SkewTDistribution
from sprintforecast.forecast import SprintForecastEngine
from sprintforecast.queue_simulator import QueueSimulator
from sprintforecast.rng_singleton import RNGSingleton
from sprintforecast.sprint import (
    Size,
)
from sprintforecast.strategies import CapacityStrategy, ExecutionStrategy, ReviewStrategy
from sprintforecast.symbolic_metrics import SymbolicMetrics
from sprintforecast.ticket import Ticket


def test_size_classification():
    assert Size.classify(0.1) is Size.XS
    assert Size.classify(2.0) is Size.XS
    assert Size.classify(2.01) is Size.S
    assert Size.classify(24.0) is Size.XL
    assert Size.classify(24.1) is Size.XXL


def test_ticket_beta_params():
    t = Ticket(1.0, 2.0, 5.0)
    a, b = t.beta_params()
    assert math.isclose(a, 1 + 4 * (2 - 1) / 4)
    assert math.isclose(b, 1 + 4 * (5 - 2) / 4)


def test_distribution_factory():
    dist = DistributionFactory.create("beta", 2, 3, 0.0, 1.0)
    assert isinstance(dist, BetaDistribution)
    with pytest.raises(ValueError):
        DistributionFactory.create("beta", -1, 3, 0.0, 1.0)  # α must be > 0


def test_beta_distribution_bounds():
    rng = RNGSingleton.rng()
    bd = BetaDistribution(2, 2, 10.0, 20.0)
    samp = bd.sample(1000, rng=rng)
    assert np.all(samp >= 10.0)
    assert np.all(samp <= 20.0)


def test_queue_simulator_single_worker():
    qs = QueueSimulator(workers=1)
    d = np.array([1.0, 2.0, 3.0])
    finish, span = qs.simulate(d)
    assert np.allclose(finish, [6.0, 5.0, 3.0])
    assert span == pytest.approx(6.0)

def test_queue_simulator_multi_worker_makespan():
    qs = QueueSimulator(workers=2)
    d = np.array([4.0, 3.0, 2.0, 1.0])
    _, span = qs.simulate(d)
    assert span == pytest.approx(5.0)  # 4+1 and 3+2


def test_sprint_intake_never_exceeds_capacity():
    rng = RNGSingleton.rng()
    backlog = [Ticket(1, 2, 3) for _ in range(30)]
    eng = SprintForecastEngine(
        tickets=[],
        exec_strategy=ExecutionStrategy(SkewTDistribution(0, 0.1, 0, 5)),
        review_strategy=ReviewStrategy(BetaDistribution(1, 1, 0, 1)),
        capacity_strategy=CapacityStrategy(BetaDistribution(1, 1, 39, 41)),
        simulator=QueueSimulator(workers=1),
        remaining_hours=0.0,
        rng=rng,
    )
    intake = eng.suggested_intake(next_capacity_hours=40.0, backlog=backlog)
    assert intake.hours <= 40.0
    assert sum(intake.totals.values()) > 0


def test_symbolic_metrics_brier_derivative():
    p = sp.symbols("p", positive=True)
    assert sp.simplify(SymbolicMetrics.brier_derivative() - (1 - 2 * p)) == 0
