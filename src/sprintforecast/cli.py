from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import typer
from rich import print

from .sprint import (
    GitHubClient,
    IssueFetcher,
    ProjectBoard,
    Ticket,
    SprintForecastEngine,
    ExecutionStrategy,
    ReviewStrategy,
    CapacityStrategy,
    QueueSimulator,
    SkewTDistribution,
    BetaDistribution,
    RNGSingleton,
)


def _require_token(token: str | None) -> str:
    if token:
        return token
    env = os.getenv("GITHUB_TOKEN")
    if env:
        return env
    typer.echo("[bold red]GitHub token missing – pass --token or set GITHUB_TOKEN[/]")
    raise typer.Exit(1)


def _simple_triads(issues: List[dict]) -> List[Tuple[int, str, Ticket]]:
    import re

    pat = re.compile(r"PERT:\s*(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)")
    out: List[Tuple[int, str, Ticket]] = []
    for iss in issues:
        body = iss.get("body") or ""
        m = pat.search(body)
        if not m:
            continue
        o, m_, p = map(float, m.groups())
        out.append((iss["number"], iss["title"], Ticket(o, m_, p)))
    return out


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def plan(
    owner: str = typer.Option(...),
    repo: str = typer.Option(...),
    project: int = typer.Option(...),
    team: int = typer.Option(...),
    length: int = typer.Option(...),
    token: str | None = typer.Option(None),
):
    token = _require_token(token)
    gh = GitHubClient(token)
    open_issues = IssueFetcher(gh, owner, repo).fetch(state="open")

    cap_hours = team * length * 6

    def mean(t: Ticket) -> float:
        return (t.optimistic + 4 * t.mode + t.pessimistic) / 6

    triads = _simple_triads(open_issues)
    triads.sort(key=lambda x: mean(x[2]))

    chosen: List[Tuple[int, str, Ticket]] = []
    used = 0.0
    for num, title, tk in triads:
        m = mean(tk)
        if used + m > cap_hours:
            break
        chosen.append((num, title, tk))
        used += m

    print("[bold]Recommended tickets:[/]")
    for num, title, tk in chosen:
        print(f"  • #{num}: {title} — PERT {tk.optimistic}/{tk.mode}/{tk.pessimistic} h")
    print(f"Sprint capacity used: {used:.1f} / {cap_hours} h")


@app.command()
def forecast(
    owner: str = typer.Option(...),
    repo: str = typer.Option(...),
    project: int = typer.Option(...),
    remaining: float = typer.Option(...),
    workers: int = typer.Option(3),
    draws: int = typer.Option(2000),
    token: str | None = typer.Option(None),
):
    token = _require_token(token)
    gh = GitHubClient(token)
    open_issues = IssueFetcher(gh, owner, repo).fetch(state="open")
    triads = [tk for _, _, tk in _simple_triads(open_issues)]
    if not triads:
        print("[yellow]No issues with PERT triads found – aborting.[/]")
        raise typer.Exit(1)

    exec_err = SkewTDistribution(0, 0.25, 2, 5)
    review_lag = BetaDistribution(2, 5, 0.1, 1.5)
    capacity = BetaDistribution(8, 2, 40, 55)

    engine = SprintForecastEngine(
        tickets=triads,
        exec_strategy=ExecutionStrategy(exec_err),
        review_strategy=ReviewStrategy(review_lag),
        capacity_strategy=CapacityStrategy(capacity),
        simulator=QueueSimulator(workers),
        remaining_hours=remaining,
        rng=RNGSingleton.rng(),
    )

    res = engine.forecast(draws)
    print(f"Probability of finishing on time: {res.probability:.1%}")
    print(f"Expected carry‑over: {res.expected_carry:.1f} tickets")


@app.command("post-note")
def post_note(
    column_id: int = typer.Option(...),
    note_path: Path = typer.Option(..., exists=True, readable=True),
    token: str | None = typer.Option(None),
):
    token = _require_token(token)
    gh = GitHubClient(token)
    ProjectBoard(gh, column_id).post_note(note_path.read_text())
    print("✅ Note posted.")


def run() -> None:
    app()

if __name__ == "__main__":
    run()
