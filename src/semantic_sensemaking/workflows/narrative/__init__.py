"""Semantic narrative analysis — planned. See ``README.md`` in this package."""

from __future__ import annotations

from semantic_sensemaking.core.config import RunConfig

DESCRIPTION = "Extract narrative frames from a corpus and project them onto a science backbone (planned)."
IMPLEMENTED = False


def run(config: RunConfig) -> None:
    raise NotImplementedError(
        "The narrative workflow is planned but not implemented yet; see "
        "src/semantic_sensemaking/workflows/narrative/README.md for its design."
    )
