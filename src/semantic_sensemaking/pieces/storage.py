"""Read and write piece inputs and outputs by location.

Two kinds of location are understood:

- ``tapis://<system>/<path>`` — a file or folder on a Tapis system, reached
  through the Tapis Files API with the caller's token.
- a local path (or ``file://`` URL) — for tests and for running a piece on
  your own machine.

A location may name one file or a folder; folders are copied recursively.
"""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urlparse

import httpx

from semantic_sensemaking.errors import SemanticSensemakingError

DEFAULT_TAPIS_BASE_URL = "https://portals.tapis.io"
_PAGE = 1000


class StorageError(SemanticSensemakingError):
    """A location could not be read or written."""


def is_tapis(uri: str) -> bool:
    return uri.startswith("tapis://")


def join(uri: str, name: str) -> str:
    """``uri`` with ``name`` appended as a path segment."""
    return uri.rstrip("/") + "/" + name.lstrip("/")


def _split_tapis(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    system, path = parsed.netloc, unquote(parsed.path).strip("/")
    if not system:
        raise StorageError(f"Not a tapis://<system>/<path> location: {uri!r}")
    return system, path


def _local(uri: str) -> Path:
    return Path(unquote(urlparse(uri).path)) if uri.startswith("file://") else Path(uri)


class Storage:
    """Fetches and publishes files for one piece run."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._token = token
        self._base = (base_url or DEFAULT_TAPIS_BASE_URL).rstrip("/")
        # Redirects are never followed: that would replay the token to another host.
        self._client = client or httpx.Client(timeout=300.0, follow_redirects=False)

    # ── reading ───────────────────────────────────────────────────────────

    def fetch(self, uri: str, dest: Path) -> Path:
        """Copy the file or folder at ``uri`` under ``dest`` and return its local path."""
        dest.mkdir(parents=True, exist_ok=True)
        if is_tapis(uri):
            return self._fetch_tapis(uri, dest)
        source = _local(uri)
        if not source.exists():
            raise StorageError(f"Nothing found at {uri!r}")
        target = dest / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
        return target

    def _fetch_tapis(self, uri: str, dest: Path) -> Path:
        system, path = _split_tapis(uri)
        rows = self._list(system, path)
        files = [r for r in rows if str(r.get("type") or "file").lower() != "dir"]
        name = PurePosixPath(path).name or system
        if len(files) == 1 and str(files[0].get("path", "")).strip("/") == path:
            target = dest / name
            self._download(system, path, target)
            return target
        if not files:
            raise StorageError(f"No files found at {uri!r}")
        root = dest / name
        for row in files:
            remote = str(row.get("path", "")).strip("/")
            relative = PurePosixPath(remote).relative_to(path) if path else PurePosixPath(remote)
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._download(system, remote, target)
        return root

    def _list(self, system: str, path: str) -> list[dict]:
        rows: list[dict] = []
        while True:
            response = self._request(
                "GET",
                f"ops/{system}/{path}",
                params={"recurse": "true", "limit": str(_PAGE), "offset": str(len(rows))},
            )
            page = response.json().get("result") or []
            rows.extend(r for r in page if isinstance(r, dict))
            if len(page) < _PAGE:
                return rows

    def _download(self, system: str, path: str, target: Path) -> None:
        response = self._request("GET", f"content/{system}/{path}")
        target.write_bytes(response.content)

    # ── writing ───────────────────────────────────────────────────────────

    def put(self, local: Path, uri: str) -> str:
        """Write the local file to ``uri`` and return ``uri``."""
        if is_tapis(uri):
            system, path = _split_tapis(uri)
            with local.open("rb") as handle:
                self._request(
                    "POST", f"ops/{system}/{path}", files={"file": (local.name, handle)}
                )
            return uri
        target = _local(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != local.resolve():
            shutil.copy2(local, target)
        return uri

    # ── transport ─────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if not self._token:
            raise StorageError("A Tapis token is required to read or write tapis:// locations")
        url = f"{self._base}/v3/files/{quote(path, safe='/')}"
        try:
            response = self._client.request(
                method, url, headers={"X-Tapis-Token": self._token}, **kwargs
            )
        except httpx.HTTPError as exc:
            raise StorageError(f"Tapis Files {method} {path} failed: {type(exc).__name__}") from None
        if response.status_code >= 300:
            raise StorageError(f"Tapis Files {method} {path} answered {response.status_code}")
        return response
