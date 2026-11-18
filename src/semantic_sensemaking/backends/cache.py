"""On-disk embedding cache, so reruns over the same corpus don't re-embed it."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from semantic_sensemaking.backends.base import EmbeddingBackend, EmbedKind


class CachedEmbeddingBackend:
    """Wrap any :class:`EmbeddingBackend` with a SQLite cache keyed by model, kind, and text."""

    def __init__(self, backend: EmbeddingBackend, cache_dir: str | Path) -> None:
        self._backend = backend
        self.name = f"cached:{backend.name}"
        self.model = backend.model
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_dir / "embeddings.sqlite3")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
        )

    def embed(self, texts: Sequence[str], *, kind: EmbedKind = "document") -> np.ndarray:
        keys = [self._key(text, kind) for text in texts]
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            rows = self._db.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(chunk))})",
                chunk,
            ).fetchall()
            found.update((k, np.frombuffer(v, dtype=np.float32)) for k, v in rows)

        missing = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in found))
        if missing:
            computed = self._backend.embed([texts[i] for i in missing], kind=kind)
            with self._db:
                for i, vector in zip(missing, computed):
                    found[keys[i]] = vector.astype(np.float32)
                    self._db.execute(
                        "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                        (keys[i], vector.astype(np.float32).tobytes()),
                    )
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([found[k] for k in keys])

    def close(self) -> None:
        self._db.close()
        self._backend.close()

    def _key(self, text: str, kind: str) -> str:
        return hashlib.sha256(f"{self.model}\0{kind}\0{text}".encode()).hexdigest()
