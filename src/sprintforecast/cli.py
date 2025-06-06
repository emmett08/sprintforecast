from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import typer
from rich import print

from .queue_simulator import QueueSimulator
from .strategies import CapacityStrategy, ExecutionStrategy, ReviewStrategy
from .ticket import Ticket
from .distributions import (
    BetaDistribution,
    SkewTDistribution,
    LogNormalDistribution,
)
from .rng_singleton import RNGSingleton
from .project_board import ProjectBoard
from .github_client import GitHubClient
from .size import Size
from .triad_fetcher import TriadFetcher
from .label_durations import LabelDurationExtractor
from .forecast import SprintForecastEngine


def _require_token(tok: str | None) -> str:
    if tok:
        return tok
    env = os.getenv("GITHUB_TOKEN")
    if env:
        return env
    typer.echo("[bold red]GitHub token missing – use --token or set GITHUB_TOKEN[/]")
    raise typer.Exit(1)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def plan(
    owner: str = typer.Option(..., help="GitHub org/user"),
    repo: str = typer.Option(..., help="Repository"),
    project: int = typer.Option(..., help="Project-board number (unused for now)"),
    team: int = typer.Option(..., help="Developer head-count"),
    length: int = typer.Option(..., help="Sprint length in days"),
    token: str | None = typer.Option(None, help="PAT or env GITHUB_TOKEN"),
):
    token = _require_token(token)
    gh = GitHubClient(token)
    triads = TriadFetcher(gh, owner, repo, project).fetch()
    if not triads:
        print("[yellow]No issues with PERT triads found.[/]")
        raise typer.Exit(1)

    cap = team * length * 6

    def mean(t: Ticket) -> float:
        return (t.optimistic + 4 * t.mode + t.pessimistic) / 6

    triads.sort(key=lambda t: mean(t.ticket))

    buckets: dict[Size, List[Tuple[int, str, Ticket]]] = {s: [] for s in Size}
    used = 0.0
    for tr in triads:
        hrs = mean(tr.ticket)
        if used + hrs > cap:
            continue
        sz = Size.classify(hrs)
        if sz is Size.XXL:
            continue
        buckets[sz].append(tr)
        used += hrs
        if used >= cap:
            break

    print(f"[bold]Intake suggestion[/] (capacity {used:.1f} / {cap} h)")
    for sz in (Size.XS, Size.S, Size.M, Size.L, Size.XL):
        group = buckets[sz]
        if not group:
            continue
        print(f"[cyan]{sz.name}[/] × {len(group)}")
        for tr in group:
            print(f"   • #{tr.number} {tr.title} ({mean(tr.ticket):.1f} h)")


@app.command()
def forecast(
    owner: str = typer.Option(...),
    repo: str = typer.Option(...),
    project: int = typer.Option(...),
    remaining: float = typer.Option(..., help="Hours left in sprint"),
    workers: int = typer.Option(3),
    draws: int = typer.Option(2000),
    empirical: bool = typer.Option(False, help="Use empirical queue times"),
    token: str | None = typer.Option(None),
):
    token = _require_token(token)
    gh = GitHubClient(token)

    if empirical:
        triads = TriadFetcher(gh, owner, repo, project).fetch()
        nums = [t.number for t in triads]
        dev, rev = LabelDurationExtractor(gh, owner, repo).extract(nums)
        if dev.size + rev.size == 0:
            print("[yellow]No dev/review label history yet – move a card or run without --empirical[/]")
            raise typer.Exit(1)

        exec_strategy = ExecutionStrategy(
            LogNormalDistribution(float(np.mean(dev or [1])), float(np.std(dev or [1], ddof=1) or 1))
        )
        review_strategy = ReviewStrategy(
            LogNormalDistribution(float(np.mean(rev or [1])), float(np.std(rev or [1], ddof=1) or 1))
        )
        tickets = [tr.ticket for tr in triads]
    else:
        triads = TriadFetcher(gh, owner, repo, project).fetch()
        if not triads:
            print("[yellow]No issues with PERT triads found.[/]")
            raise typer.Exit(1)
        tickets = [tr.ticket for tr in triads]
        exec_strategy = ExecutionStrategy(SkewTDistribution(0, 0.25, 2, 5))
        review_strategy = ReviewStrategy(BetaDistribution(2, 5, 0.1, 1.5))

    engine = SprintForecastEngine(
        tickets=tickets,
        exec_strategy=exec_strategy,
        review_strategy=review_strategy,
        capacity_strategy=CapacityStrategy(BetaDistribution(8, 2, 40, 55)),
        simulator=QueueSimulator(workers),
        remaining_hours=remaining,
        rng=RNGSingleton.rng(),
    )

    res = engine.forecast(draws)
    print(f"Probability of finishing: [bold]{res.probability:.1%}[/]")
    print(f"Expected carry-over: {res.expected_carry:.1f} tickets")


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
