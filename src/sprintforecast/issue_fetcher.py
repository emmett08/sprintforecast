from .github_client import GitHubClient

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

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