from __future__ import annotations
import datetime as dt
import typing as t
import requests, pandas as pd

__all__ = [
    "GitHubIssues",
    "TriadCSV",
    "ActualsCSV",
]

class GitHubIssues:
    def __init__(self, owner: str, repo: str, token: str | None = None):
        self.owner = owner
        self.repo = repo
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"token {token}"
        self.session.headers["Accept"] = "application/vnd.github+json"

    def _get(self, url: str, **params) -> list[dict]:
        res = self.session.get(url, params=params, timeout=30)
        res.raise_for_status()
        return res.json()

    def fetch(self, state: str = "all") -> list[dict]:
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues"
        page, out = 1, []
        while True:
            batch = self._get(url, state=state, per_page=100, page=page)
            if not batch:
                break
            out.extend(i for i in batch if "pull_request" not in i)
            page += 1
        return out


class _BaseCSV:
    issues: list[dict]

    @staticmethod
    def _label_value(issue: dict, key: str) -> float | None:
        for lab in issue["labels"]:
            name: str = lab["name"]
            if name.startswith(f"{key}="):
                return float(name.split("=", 1)[1])
        return None

    def _frame(self) -> pd.DataFrame: ...

    def write(self, path: str | None = None) -> pd.DataFrame:
        df = self._frame()
        if path:
            df.to_csv(path, index=False)
        return df


class TriadCSV(_BaseCSV):
    def __init__(self, issues: list[dict]):
        self.issues = issues

    def _frame(self) -> pd.DataFrame:
        rows: list[dict[str, t.Any]] = []
        for i in self.issues:
            o = self._label_value(i, "o")
            m = self._label_value(i, "m")
            p = self._label_value(i, "p")
            if None not in (o, m, p):
                rows.append({"id": i["number"], "o": o, "m": m, "p": p})
        return pd.DataFrame(rows, dtype=float)


class ActualsCSV(_BaseCSV):
    def __init__(self, issues: list[dict]):
        self.issues = issues

    def _frame(self) -> pd.DataFrame:
        rows: list[dict[str, t.Any]] = []
        for i in self.issues:
            if i["state"] != "closed":
                continue
            start = pd.to_datetime(i["created_at"], utc=True)
            end = pd.to_datetime(i["closed_at"], utc=True)
            hours = (end - start) / pd.Timedelta(hours=1)
            rows.append({"id": i["number"], "actual": hours})
        return pd.DataFrame(rows, dtype=float)
