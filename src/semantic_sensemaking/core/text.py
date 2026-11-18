"""Small text utilities shared across layers."""

from __future__ import annotations

import re

_UNSAFE_FILENAME_CHARS = re.compile(r"[^a-zA-Z0-9_\-]")


def word_count(text: str) -> int:
    """Whitespace-delimited word count (the prototype's ``len(text.split())``)."""
    return len(text.split())


def safe_filename(name: str, max_len: int = 100) -> str:
    """Replace anything outside ``[A-Za-z0-9_-]`` with ``_`` and truncate."""
    return _UNSAFE_FILENAME_CHARS.sub("_", name)[:max_len]


def chunk_words(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split ``text`` into passages of ``size`` words overlapping by ``overlap``.

    Used to embed long documents passage by passage. Always returns at least one
    passage for non-empty text.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    passages = []
    for start in range(0, len(words), step):
        passages.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return passages
