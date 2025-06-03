from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import stats, optimize
from ._rng import rng

class ErrorDistribution:  # interface
    def sample(self, size: int = 1) -> np.ndarray: ...
    def logpdf(self, x: np.ndarray) -> np.ndarray: ...
    def to_dict(self) -> dict: ...
    @staticmethod
    def from_dict(d: dict) -> "ErrorDistribution": ...

@dataclass(frozen=True, slots=True)
class LogNormalError(ErrorDistribution):
    mu: float
    sigma: float
    def sample(self, size: int = 1) -> np.ndarray:
        return rng.lognormal(self.mu, self.sigma, size)
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return stats.lognorm(self.sigma, scale=np.exp(self.mu)).logpdf(x)
    def to_dict(self) -> dict:
        return {"type": "lognormal", "mu": self.mu, "sigma": self.sigma}
    @staticmethod
    def from_dict(d: dict) -> "LogNormalError":
        return LogNormalError(float(d["mu"]), float(d["sigma"]))
    @staticmethod
    def fit(residuals: np.ndarray) -> "LogNormalError":
        mu = float(residuals.mean())
        sigma = float(residuals.std(ddof=1))
        return LogNormalError(mu, sigma)

@dataclass(frozen=True, slots=True)
class SkewTError(ErrorDistribution):
    xi: float
    omega: float
    alpha: float
    df: float
    def sample(self, size: int = 1) -> np.ndarray:
        return stats.skewt(self.df, self.alpha, loc=self.xi, scale=self.omega).rvs(size, random_state=rng)
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return stats.skewt(self.df, self.alpha, loc=self.xi, scale=self.omega).logpdf(x)
    def to_dict(self) -> dict:
        return {"type": "skewt", "xi": self.xi, "omega": self.omega, "alpha": self.alpha, "df": self.df}
    @staticmethod
    def from_dict(d: dict) -> "SkewTError":
        return SkewTError(float(d["xi"]), float(d["omega"]), float(d["alpha"]), float(d["df"]))
    @staticmethod
    def fit(data: np.ndarray) -> "SkewTError":
        def nll(params):
            xi, omega, alpha, df = params
            if omega <= 0 or df <= 2:
                return np.inf
            return -np.sum(stats.skewt(df, alpha, loc=xi, scale=omega).logpdf(data))
        x0 = (np.median(data), np.std(data, ddof=1), 1.0, 6.0)
        res = optimize.minimize(nll, x0, method="Nelder-Mead")
        xi, omega, alpha, df = res.x
        return SkewTError(float(xi), float(omega), float(alpha), float(df))

def choose_error_family(residuals: np.ndarray) -> ErrorDistribution:
    q = np.percentile(residuals, [80, 90, 95])
    test = (q[2] - q[0]) / (q[1] - q[0])
    return SkewTError.fit(residuals) if test > 1.6 else LogNormalError.fit(residuals)
