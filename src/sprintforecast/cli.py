from __future__ import annotations
import json, pathlib, numpy as np, pandas as pd
from .githubio import GitHubIssues, TriadCSV, ActualsCSV
import typer
from . import (
    rng, choose_error_family, CapacityPosterior,
    ErrorDistribution, LogNormalError, SkewTError,
    SprintForecaster, MomentumModel, IntakePlanner,
)

app = typer.Typer(add_completion=False)

def _gh(owner: str, repo: str, token: str | None):
    return GitHubIssues(owner, repo, token)

def _ni_gamma(history: np.ndarray, m0: float = 0, k0: float = 1, a0: float = 2, b0: float = 0.5) -> CapacityPosterior:
    n = len(history)
    mean = history.mean()
    var = history.var(ddof=1)
    k_n = k0 + n
    mu_n = (k0 * m0 + n * mean) / k_n
    a_n = a0 + n / 2
    b_n = b0 + 0.5 * (n * var + k0 * n * (mean - m0) ** 2 / k_n)
    sigma = np.sqrt(b_n / (a_n - 1))
    return CapacityPosterior(float(mu_n), float(sigma))


def _load_triads(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path).astype({"o": float, "m": float, "p": float})


def _load_actuals(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path).astype({"actual": float})


def _residuals(df: pd.DataFrame) -> np.ndarray:
    mu = (df["o"] + 4 * df["m"] + df["p"]) / 6
    return np.log(df["actual"].values / mu.values)


def _save_dist(d: ErrorDistribution, path: pathlib.Path):
    path.write_text(json.dumps(d.to_dict()))


def _load_dist(path: pathlib.Path) -> ErrorDistribution:
    raw = json.loads(path.read_text())
    return {"lognormal": LogNormalError.from_dict, "skewt": SkewTError.from_dict}[raw["type"]](raw)


@app.command()
def fit_error(triads: pathlib.Path, actuals: pathlib.Path, out: pathlib.Path):
    df = pd.concat([_load_triads(triads), _load_actuals(actuals)], axis=1)
    dist = choose_error_family(_residuals(df))
    _save_dist(dist, out)
    typer.echo(out)


@app.command()
def forecast(triads: pathlib.Path, capacity_hist: pathlib.Path, dist_file: pathlib.Path, sprint_hours: float = 80):
    triad_df = _load_triads(triads)
    cap_hist = np.loadtxt(capacity_hist)
    cap_post = _ni_gamma(np.log(cap_hist))
    dist = _load_dist(dist_file)
    f = SprintForecaster(dist, cap_post, triad_df)
    typer.echo(json.dumps(f.summary(sprint_hours), indent=2))


@app.command()
def plan_intake(pool: pathlib.Path, capacity_hist: pathlib.Path, dist_file: pathlib.Path, hours_per_sprint: float = 80, gamma: float = 0.8):
    caps = np.loadtxt(capacity_hist)
    mom = MomentumModel(caps, hours_per_sprint)
    next_cap = mom.forecast(10_000)
    pool_df = _load_triads(pool)
    dist = _load_dist(dist_file)
    planner = IntakePlanner(pool_df, dist, next_cap)
    res = planner.plan(gamma)
    res["momentum"] = mom.momentum()
    typer.echo(json.dumps(res, indent=2))

@app.command()
def triads(
    owner: str,
    repo: str,
    outfile: pathlib.Path = pathlib.Path("triads.csv"),
    token: str | None = typer.Option(None, envvar="GITHUB_TOKEN"),
):
    gh = _gh(owner, repo, token)
    TriadCSV(gh.fetch("open")).write(str(outfile))
    typer.echo(outfile)

@app.command()
def actuals(
    owner: str,
    repo: str,
    outfile: pathlib.Path = pathlib.Path("actuals.csv"),
    token: str | None = typer.Option(None, envvar="GITHUB_TOKEN"),
):
    gh = _gh(owner, repo, token)
    ActualsCSV(gh.fetch("closed")).write(str(outfile))
    typer.echo(outfile)

@app.command()
def pool(
    owner: str,
    repo: str,
    outfile: pathlib.Path = pathlib.Path("pool.csv"),
    token: str | None = typer.Option(None, envvar="GITHUB_TOKEN"),
):
    """
    Export triads for the backlog (“pool”)—all open issues that are *not*
    already assigned to a sprint milestone.
    """
    gh = _gh(owner, repo, token)
    issues = [
        i for i in gh.fetch("open")
        if not i.get("milestone")
        and "o=" in str(i["labels"])
    ]
    TriadCSV(issues).write(str(outfile))
    typer.echo(outfile)
