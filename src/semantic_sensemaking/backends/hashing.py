"""Deterministic, dependency-free embeddings for tests and offline dry runs.

Uses the signed hashing trick over lowercase word tokens, so texts sharing
words are similar and everything else is near-orthogonal. This is *not* a
semantic model — it exists so pipelines, notebooks, and CI can exercise the
embedding code path without a GPU, a network, or model downloads.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from semantic_sensemaking.backends.base import EmbedKind, l2_normalize

_TOKEN = re.compile(r"\w+")


class DeterministicHashEmbedding:
    name = "hashing"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.model = f"hashing-{dimensions}"

    def embed(self, texts: Sequence[str], *, kind: EmbedKind = "document") -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                vectors[row, value % self.dimensions] += 1.0 if (value >> 63) & 1 else -1.0
        return l2_normalize(vectors)

    def close(self) -> None:
        pass
