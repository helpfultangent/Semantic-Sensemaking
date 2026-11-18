"""Narrative analysis: find the frames a corpus uses and place them on a science backbone.

Documents are split into passages, embedded, and clustered; each cluster becomes
a named frame (by a chat model when one is configured, otherwise by its
distinctive keywords). Frames are exported as a lexicon, mapped onto the
taxonomy, and projected exactly like dictionary groups, so the two workflows'
results share one reference frame. See ``README.md`` in this package.
"""

from semantic_sensemaking.workflows.narrative.cluster import Clustering, cluster_passages
from semantic_sensemaking.workflows.narrative.labeling import Frame, frames_to_lexicon
from semantic_sensemaking.workflows.narrative.pipeline import (
    NarrativeRun,
    analyze,
    run,
    segment_corpus,
)
from semantic_sensemaking.workflows.narrative.result import NarrativeResult, Passage

DESCRIPTION = "Extract narrative frames from a corpus and project them onto a science backbone."
IMPLEMENTED = True

__all__ = [
    "DESCRIPTION",
    "IMPLEMENTED",
    "Clustering",
    "Frame",
    "NarrativeResult",
    "NarrativeRun",
    "Passage",
    "analyze",
    "cluster_passages",
    "frames_to_lexicon",
    "run",
    "segment_corpus",
]
