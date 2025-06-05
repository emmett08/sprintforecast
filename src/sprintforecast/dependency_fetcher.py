from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .github_client import GitHubClient

@dataclass(slots=True, frozen=True)
class DependencyFetcher:
    client: GitHubClient
    owner: str
    repo: str
    page: int = 100

    _Q = """
    query($owner:String!,$repo:String!,$num:Int!,$first:Int!,$after:String){
      repository(owner:$owner,name:$repo){
        issue(number:$num){
          timelineItems(first:$first,after:$after,itemTypes:[CONNECTED_EVENT]){
            pageInfo{endCursor,hasNextPage}
            nodes{
              ... on ConnectedEvent{
                source  { ... on Issue{ number } }
                subject { ... on Issue{ number } }
              }
            }
          }
        }
      }
    }
    """

    def fetch(self, num: int) -> set[int]:
        deps: set[int] = set()
        after: str | None = None
        hdr = {"Accept": "application/vnd.github.hawkgirl-preview+json"}
        while True:
            r = self.client.post(
                "graphql",
                json={
                    "query": self._Q,
                    "variables": {
                        "owner": self.owner,
                        "repo": self.repo,
                        "num": num,
                        "first": self.page,
                        "after": after,
                    },
                },
                headers=hdr,
            )
            r.raise_for_status()
            payload: dict[str, Any] = r.json()
            if "errors" in payload:
                raise RuntimeError(payload["errors"])
            items = payload["data"]["repository"]["issue"]["timelineItems"]
            for ev in items["nodes"]:
                if ev["source"]["number"] == num:
                    deps.add(ev["subject"]["number"])
            if not items["pageInfo"]["hasNextPage"]:
                break
            after = items["pageInfo"]["endCursor"]
        return deps
