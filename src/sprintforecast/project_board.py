from .github_client import GitHubClient

class ProjectBoard:
    def __init__(self, client: GitHubClient, column_id: int) -> None:
        self._c = client
        self._path = f"/projects/columns/{column_id}/cards"
        self._hdr = {"Accept": "application/vnd.github.inertia-preview+json"}

    def post_note(self, note: str):
        r = self._c.post(self._path, json={"note": note}, headers=self._hdr)
        r.raise_for_status()
        return r.json()