"""Package data: built-in lexicons, taxonomies, JSON schemas, and a sample corpus.

The sample corpus is synthetic text written for examples and tests; it contains
no real transcripts.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def builtin_path(name: str) -> Path:
    path = Path(str(files(__name__) / name))
    if not path.exists():
        raise FileNotFoundError(f"No built-in data file named {name!r}")
    return path


def sample_corpus_path() -> Path:
    """Directory of synthetic documents for trying the workflows."""
    return builtin_path("sample_corpus")
