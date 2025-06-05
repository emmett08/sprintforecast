import math
import numpy as np
from scipy.stats import beta as _beta, gamma as _gamma
from dataclasses import dataclass
from .types import Sample
from numpy.random import Generator
from typing import Protocol, runtime_checkable
from typing import Callable, MutableMapping

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
