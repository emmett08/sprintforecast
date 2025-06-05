from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .github_client import GitHubClient

from .ticket import Ticket
from .dependency_fetcher import DependencyFetcher

@dataclass(slots=True, frozen=True)
class Triad:
    number: int
    title: str
    ticket: Ticket
    deps: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class TriadFetcher:
    client: GitHubClient
    owner: str
    repo: str
    project: int
    page: int = 50

    _Q = """
    query($owner:String!,$repo:String!,$first:Int!,$after:String){
      repository(owner:$owner,name:$repo){
        issues(states:OPEN,first:$first,after:$after){
          pageInfo{endCursor,hasNextPage}
          nodes{
            number
            title
            projectItems(first:10){
              nodes{
                project{number}
                fieldValues(first:20){
                  nodes{
                    ... on ProjectV2ItemFieldNumberValue{
                      number
                      field{ ... on ProjectV2FieldCommon{ name } }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    def _run(self, after: str | None) -> dict[str, Any]:
        r = self.client.post(
            "graphql",
            json={
                "query": self._Q,
                "variables": {
                    "owner": self.owner,
                    "repo": self.repo,
                    "first": self.page,
                    "after": after,
                },
            },
        )
        r.raise_for_status()
        payload: dict[str, Any] = r.json()
        if "errors" in payload:
            raise RuntimeError(payload["errors"])
        return payload["data"]["repository"]["issues"]

    @staticmethod
    def _num(it: dict[str, Any], k: str) -> float | None:
        for n in it.get("fieldValues", {}).get("nodes", []):
            fld = n.get("field") or {}
            if fld.get("name", "").lower() == k and n.get("number") is not None:
                return float(n["number"])
        return None

    def fetch(self) -> list[Triad]:
        out: list[Triad] = []
        dep_f = DependencyFetcher(self.client, self.owner, self.repo)
        after: str | None = None
        while True:
            page = self._run(after)
            for n in page["nodes"]:
                triad: tuple[float, float, float] | None = None
                for it in n["projectItems"]["nodes"]:
                    if it["project"]["number"] != self.project:
                        continue
                    o = self._num(it, "o")
                    m = self._num(it, "m")
                    p = self._num(it, "p")
                    if None not in (o, m, p):
                        triad = (o, m, p)
                        break
                if triad:
                    deps = dep_f.fetch(n["number"])
                    out.append(
                        Triad(
                            number=n["number"],
                            title=n["title"],
                            ticket=Ticket(*triad),
                            deps=tuple(deps),
                        )
                    )
            if not page["pageInfo"]["hasNextPage"]:
                break
            after = page["pageInfo"]["endCursor"]
        return out
