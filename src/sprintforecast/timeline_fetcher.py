from .github_client import GitHubClient

from datetime import datetime, time
from typing import Any, Iterator, Mapping, Sequence


class TimelineFetcher:
    def __init__(
        self,
        gh: GitHubClient,
        owner: str,
        repo: str,
        *,
        accept: str = "application/vnd.github.mockingbird-preview+json",
        per_page: int = 100,
    ) -> None:
        self._gh = gh
        self._root = f"/repos/{owner}/{repo}/issues/{{}}/timeline"
        self._hdr = {"Accept": accept}
        self._params = {"per_page": min(per_page, 100)}

    def iter_events(
        self,
        num: int,
        *,
        types: Sequence[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        url: str | None = self._root.format(num)
        while url:
            r = self._gh.get(url, headers=self._hdr, params=self._params)
            if r.status_code == 403 and int(r.headers.get("X-RateLimit-Remaining", 1)) == 0:
                reset = int(r.headers["X-RateLimit-Reset"])
                time.sleep(max(0, reset - time.time()) + 1)
                continue
            r.raise_for_status()
            for ev in r.json():
                if types and ev.get("event") not in types:
                    continue
                ts = ev.get("created_at")
                if ts:
                    t = datetime.fromisoformat(ts.rstrip("Z"))
                    if since and t < since:
                        continue
                    if until and t > until:
                        continue
                yield ev
            url = r.links.get("next", {}).get("url")