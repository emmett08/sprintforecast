# triad_fetcher.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .sprint import GitHubClient, Ticket

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
                      field{
                        ... on ProjectV2FieldCommon{ name }
                      }
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
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        payload = r.json()
        if "errors" in payload:
            raise RuntimeError(f"GraphQL error: {payload['errors']}")
        repo = payload.get("data", {}).get("repository")
        if not repo:
            return {"nodes": [], "pageInfo": {"hasNextPage": False}}
        return repo["issues"]

    @staticmethod
    def _num(item: dict[str, Any], key: str) -> float | None:
        for n in item.get("fieldValues", {}).get("nodes", []):
            fld = n.get("field") or {}
            if (fld.get("name", "").lower() == key) and (n.get("number") is not None):
                return float(n["number"])
        return None

    def fetch(self) -> list[tuple[int, str, Ticket]]:
        out: list[tuple[int, str, Ticket]] = []
        after: str | None = None
        while True:
            page = self._run(after)
            for iss in page["nodes"]:
                for it in iss["projectItems"]["nodes"]:
                    if it["project"]["number"] != self.project:
                        continue
                    o = self._num(it, "o")
                    m = self._num(it, "m")
                    p = self._num(it, "p")
                    if None not in (o, m, p):
                        out.append((iss["number"], iss["title"], Ticket(o, m, p)))
                        break
            if not page["pageInfo"]["hasNextPage"]:
                break
            after = page["pageInfo"]["endCursor"]
        return out
