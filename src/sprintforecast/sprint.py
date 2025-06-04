from __future__ import annotations

import abc
from datetime import time, datetime
import heapq
from dataclasses import dataclass
import math
from typing import Callable, Iterator, MutableMapping, Protocol, Sequence, TypeAlias, Sequence, Mapping, Any, runtime_checkable
import numpy.typing as npt
import numpy as np
import sympy as sp
import threading
import requests
from enum import Enum
from numpy.random import Generator, default_rng
from scipy.stats import beta as _beta
from scipy.stats import gamma as _gamma

class Size(Enum):
    XS = 2      # ≤ 2 h
    S  = 4      # >2 – 4 h
    M  = 8      # >4 – 8 h
    L  = 16     # >8 – 16 h
    XL = 24     # >16 – 24 h
    XXL = math.inf  # >24 h

Sample: TypeAlias = npt.NDArray[np.floating]

@dataclass(slots=True, frozen=True)
class SprintIntake:
    totals: Mapping[Size, int]
    hours: float

@dataclass(slots=True, frozen=True)
class GitHubClient:
    token: str
    base_url: str = "https://api.github.com"

    def __post_init__(self) -> None:
        s = requests.Session()
        s.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            }
        )
        object.__setattr__(self, "_s", s)

    def get(self, path_or_url: str, **kw) -> requests.Response:
        url = (
            path_or_url
            if path_or_url.startswith("http")
            else f"{self.base_url.rstrip('/')}/{path_or_url.lstrip('/')}"
        )
        return self._s.get(url, **kw)

    def post(self, path: str, **kw) -> requests.Response:
        return self._s.post(f"{self.base_url.rstrip('/')}/{path.lstrip('/')}", **kw)


@dataclass(slots=True, frozen=True)
class IssueFetcher:
    client: GitHubClient
    owner: str
    repo: str
    per_page: int = 100

    def __post_init__(self) -> None:
        if not 1 <= self.per_page <= 100:
            raise ValueError("per_page must be in 1...100")

    def _base(self) -> str:
        return f"repos/{self.owner}/{self.repo}/issues"

    def iter_issues(
        self,
        *,
        state: str = "open",
        labels: Sequence[str] | None = None,
        **extra: Any,
    ) -> Iterator[Mapping[str, Any]]:
        params: dict[str, Any] = {"state": state, "per_page": self.per_page, **extra}
        if labels:
            params["labels"] = ",".join(labels)

        url: str | None = self._base()
        while url:
            r = self.client.get(url, params=params)
            r.raise_for_status()
            yield from r.json()
            url = r.links.get("next", {}).get("url")
            params = None

    def fetch(
        self,
        *,
        state: str = "open",
        labels: Sequence[str] | None = None,
        **extra: Any,
    ) -> list[Mapping[str, Any]]:
        return list(self.iter_issues(state=state, labels=labels, **extra))

class TimelineFetcher:
    def __init__(
        self,
        gh: GitHubClient,
        owner: str,
        repo: str,
        *,
        accept: str = "application/vnd.github.mockingbird-preview+json",
        per_page: int = 100,
    ) -> None:
        self._gh = gh
        self._root = f"/repos/{owner}/{repo}/issues/{{}}/timeline"
        self._hdr = {"Accept": accept}
        self._params = {"per_page": min(per_page, 100)}

    def iter_events(
        self,
        num: int,
        *,
        types: Sequence[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        url: str | None = self._root.format(num)
        while url:
            r = self._gh.get(url, headers=self._hdr, params=self._params)
            if r.status_code == 403 and int(r.headers.get("X-RateLimit-Remaining", 1)) == 0:
                reset = int(r.headers["X-RateLimit-Reset"])
                time.sleep(max(0, reset - time.time()) + 1)
                continue
            r.raise_for_status()
            for ev in r.json():
                if types and ev.get("event") not in types:
                    continue
                ts = ev.get("created_at")
                if ts:
                    t = datetime.fromisoformat(ts.rstrip("Z"))
                    if since and t < since:
                        continue
                    if until and t > until:
                        continue
                yield ev
            url = r.links.get("next", {}).get("url")

class ProjectBoard:
    def __init__(self, client: GitHubClient, column_id: int) -> None:
        self._c = client
        self._path = f"/projects/columns/{column_id}/cards"
        self._hdr = {"Accept": "application/vnd.github.inertia-preview+json"}

    def post_note(self, note: str):
        r = self._c.post(self._path, json={"note": note}, headers=self._hdr)
        r.raise_for_status()
        return r.json()

class RNGSingleton:
    _lock = threading.Lock()
    _rng: Generator | None = None

    @classmethod
    def rng(cls) -> Generator:
        if cls._rng is None:
            with cls._lock:
                if cls._rng is None:
                    cls._rng = default_rng()
        return cls._rng

@runtime_checkable
class Distribution(Protocol):
    def sample(self, size: int | tuple[int, ...] = 1, *, rng: Generator) -> Sample: ...


@dataclass(slots=True, frozen=True)
class BetaDistribution(Distribution):
    alpha: float
    beta: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("shape parameters must be positive")
        if self.lower >= self.upper:
            raise ValueError("lower bound must be < upper bound")

    def sample(self, size: int | tuple[int, ...] = 1, *, rng: Generator) -> Sample:
        x = _beta.rvs(self.alpha, self.beta, size=size, random_state=rng)
        return self.lower + (self.upper - self.lower) * x

@dataclass(slots=True, frozen=True)
class SkewTDistribution(Distribution):
    xi: float           # location  (ξ)
    omega: float        # scale     (ω)  > 0
    alpha: float        # skewness  (α)
    nu: float           # d.o.f.    (ν)  > 0

    def __post_init__(self) -> None:
        if self.omega <= 0 or self.nu <= 0:
            raise ValueError("scale and degrees-of-freedom must be positive")

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        rng: Generator,
    ) -> Sample:
        delta = self.alpha / math.sqrt(1.0 + self.alpha * self.alpha)

        w = rng.chisquare(self.nu, size) / self.nu

        z0 = rng.standard_normal(size)
        z1 = rng.standard_normal(size)

        t = (delta * np.abs(z0) + math.sqrt(1.0 - delta * delta) * z1) / np.sqrt(w)

        return self.xi + self.omega * t

@dataclass(slots=True, frozen=True)
class GammaDistribution(Distribution):
    shape: float
    scale: float

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("shape and scale must be positive")

    def sample(self, size: int | tuple[int, ...] = 1, *, rng: Generator) -> Sample:
        return _gamma.rvs(self.shape, loc=0.0, scale=self.scale, size=size, random_state=rng)

@dataclass(slots=True, frozen=True)
class LogNormalDistribution(Distribution):
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def sample(self, size: int | tuple[int, ...] = 1, *, rng: Generator) -> Sample:
        return rng.lognormal(self.mu, self.sigma, size=size)

class DistributionFactory:
    _registry: MutableMapping[str, type[Distribution]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[Distribution]], type[Distribution]]:
        def decorator(dist_cls: type[Distribution]) -> type[Distribution]:
            cls._registry[name.lower()] = dist_cls
            return dist_cls

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Distribution:
        dist_cls = cls._registry[name.lower()]
        return dist_cls(*args, **kwargs)


DistributionFactory.register("beta")(BetaDistribution)
DistributionFactory.register("skewt")(SkewTDistribution)
DistributionFactory.register("gamma")(GammaDistribution)
DistributionFactory.register("lognormal")(LogNormalDistribution)

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
        base = np.array([t.base_distribution().sample(rng=self.rng)[0] for t in self.tickets])
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
        sorted_backlog = sorted(backlog, key=self._mean_hours)
        used = 0.0
        quota = {s: avail * alloc[s] for s in alloc}
        for t in sorted_backlog:
            h = self._mean_hours(t)
            b = self._bucket(h)
            if used + h > avail or b == Size.XXL:
                continue
            if buckets[b] * h >= quota[b]:
                continue
            buckets[b] += 1
            used += h
            if used >= avail:
                break
        return SprintIntake(buckets, used)

class SymbolicMetrics:
    p = sp.symbols("p", positive=True)
    brier = p * (1 - p)

    @classmethod
    def brier_derivative(cls) -> sp.Expr:
        return sp.diff(cls.brier, cls.p)
