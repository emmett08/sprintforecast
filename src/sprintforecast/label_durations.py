from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from .github_client import GitHubClient
from .timeline_fetcher import TimelineFetcher


class LabelDurationExtractor:
    def __init__(self, gh: GitHubClient, owner: str, repo: str) -> None:
        self._tf = TimelineFetcher(gh, owner, repo)

    def extract(self, issue_numbers: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        dev, rev = [], []
        for n in issue_numbers:
            di = do = ri = ro = None
            for ev in self._tf.iter_events(n, types=["labeled", "unlabeled"]):
                lab = ev.get("label", {}).get("name", "").lower()
                ts = ev["created_at"]
                if lab == "dev":
                    if ev["event"] == "labeled":
                        di = ts
                    else:
                        do = ts
                if lab == "review":
                    if ev["event"] == "labeled":
                        ri = ts
                    else:
                        ro = ts
            if di and do:
                dev.append(
                    (
                        datetime.fromisoformat(do.rstrip("Z"))
                        - datetime.fromisoformat(di.rstrip("Z"))
                    ).total_seconds()
                    / 3600
                )
            if ri and ro:
                rev.append(
                    (
                        datetime.fromisoformat(ro.rstrip("Z"))
                        - datetime.fromisoformat(ri.rstrip("Z"))
                    ).total_seconds()
                    / 3600
                )
        return np.asarray(dev), np.asarray(rev)
