from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from .github_client import GitHubClient
from .timeline_fetcher import TimelineFetcher


def extract_durations(
    gh: GitHubClient,
    owner: str,
    repo: str,
    issue_numbers: Sequence[int],
):
    tf = TimelineFetcher(gh, owner, repo)
    dev, review = [], []
    for n in issue_numbers:
        di = do = ri = ro = None
        for ev in tf.iter_events(n, types=["moved_columns_in_project"]):
            info = ev.get("project_card", {})
            col_from = info.get("previous_column_name")
            col_to   = info.get("column_name")
            ts       = ev["created_at"]
            if col_to == "Dev":
                di = ts
            if col_from == "Dev":
                do = ts
            if col_to == "Review":
                ri = ts
            if col_from == "Review":
                ro = ts
        if di and do:
            d = (
                datetime.fromisoformat(do.rstrip("Z"))
                - datetime.fromisoformat(di.rstrip("Z"))
            ).total_seconds() / 3600
            dev.append(d)
        if ri and ro:
            r = (
                datetime.fromisoformat(ro.rstrip("Z"))
                - datetime.fromisoformat(ri.rstrip("Z"))
            ).total_seconds() / 3600
            review.append(r)
    return np.asarray(dev), np.asarray(review)
