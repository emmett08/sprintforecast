"""Command‑line interface for SprintForecast.

Sub‑commands
------------
plan       – pick a backlog subset that fits the next‑sprint capacity
forecast   – Monte‑Carlo probability and expected carry‑over for current sprint
post‑note  – push a markdown digest to a GitHub project column
"""
from __future__ import annotations

import os
from enum import Enum
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
    Size,
)

def _require_token(tok: str | None) -> str:
    if tok:
        return tok
    env = os.getenv("GITHUB_TOKEN")
    if env:
        return env
    typer.echo("[bold red]GitHub token missing – use --token or set GITHUB_TOKEN[/]")
    raise typer.Exit(1)


PERT_RE = __import__("re").compile(
    r"PERT:\s*(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)",
    __import__("re").I,
)

def _extract_triads(issues: List[dict]) -> List[Tuple[int, str, Ticket]]:
    out: List[Tuple[int, str, Ticket]] = []
    for iss in issues:
        m = PERT_RE.search(iss.get("body", ""))
        if not m:
            continue
        o, m_, p = map(float, m.groups())
        out.append((iss["number"], iss["title"], Ticket(o, m_, p)))
    return out

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def plan(
    owner: str = typer.Option(..., help="GitHub org/user"),
    repo: str = typer.Option(..., help="Repository"),
    project: int = typer.Option(..., help="Project‑board number (unused for now)"),
    team: int = typer.Option(..., help="Developer head‑count"),
    length: int = typer.Option(..., help="Sprint length in days"),
    token: str | None = typer.Option(None, help="PAT or env GITHUB_TOKEN"),
):
    token = _require_token(token)
    gh = GitHubClient(token)
    issues = IssueFetcher(gh, owner, repo).fetch(state="open")
    triads = _extract_triads(issues)
    if not triads:
        print("[yellow]No issues with PERT triads found.[/]")
        raise typer.Exit(1)

    cap = team * length * 6  # TODO: make this configurable 

    def mean(t: Ticket) -> float:
        return (t.optimistic + 4 * t.mode + t.pessimistic) / 6

    triads.sort(key=lambda x: mean(x[2]))

    buckets: dict[Size, List[Tuple[int, str, Ticket]]] = {s: [] for s in Size}
    used = 0.0
    for num, title, tk in triads:
        hrs = mean(tk)
        if used + hrs > cap:
            continue
        sz = Size.classify(hrs)
        if sz is Size.XXL:
            continue
        buckets[sz].append((num, title, tk))
        used += hrs
        if used >= cap:
            break

    print("[bold]Intake suggestion[/] (capacity {:.1f} / {} h)".format(used, cap))
    for sz in [Size.XS, Size.S, Size.M, Size.L, Size.XL]:
        lst = buckets[sz]
        if not lst:
            continue
        print(f"[cyan]{sz.name}[/]  × {len(lst)}")
        for num, title, tk in lst:
            print(f"   • #{num} {title}  ({mean(tk):.1f} h)")


@app.command()
def forecast(
    owner: str = typer.Option(...),
    repo: str = typer.Option(...),
    project: int = typer.Option(...),
    remaining: float = typer.Option(..., help="Hours left in sprint"),
    workers: int = typer.Option(3),
    draws: int = typer.Option(2000),
    token: str | None = typer.Option(None),
):
    token = _require_token(token)
    gh = GitHubClient(token)
    triads = [tk for _, _, tk in _extract_triads(IssueFetcher(gh, owner, repo).fetch(state="open"))]
    if not triads:
        print("[yellow]No issues with PERT triads found.[/]")
        raise typer.Exit(1)

    engine = SprintForecastEngine(
        tickets=triads,
        exec_strategy=ExecutionStrategy(SkewTDistribution(0, 0.25, 2, 5)),
        review_strategy=ReviewStrategy(BetaDistribution(2, 5, 0.1, 1.5)),
        capacity_strategy=CapacityStrategy(BetaDistribution(8, 2, 40, 55)),
        simulator=QueueSimulator(workers),
        remaining_hours=remaining,
        rng=RNGSingleton.rng(),
    )
    res = engine.forecast(draws)
    print(f"Probability of finishing: [bold]{res.probability:.1%}[/]")
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
    print("✅ Note posted")

def run() -> None:
    app()

if __name__ == "__main__":
    run()
