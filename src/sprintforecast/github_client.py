import requests

from dataclasses import dataclass, field

@dataclass(slots=True, frozen=True)
class GitHubClient:
    token: str
    base_url: str = "https://api.github.com"
    _s: requests.Session = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        s = requests.Session()
        s.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
            }
        )
        object.__setattr__(self, "_s", s)

    def get(self, path_or_url: str, **kw) -> requests.Response:
        url = (
            path_or_url
            if path_or_url.startswith("http")
            else f"{self.base_url.rstrip('/')}/{path_or_url.lstrip('/')}"
        )
        return self._s.get(url, **kw)

    def post(self, path: str, **kw) -> requests.Response:
        return self._s.post(f"{self.base_url.rstrip('/')}/{path.lstrip('/')}", **kw)