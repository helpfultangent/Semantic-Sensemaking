"""Semantic Sensemaking: layered tools for semantic workflows on TACC.

Tool clusters (each importable on its own):

- ``core``       — data types, run configuration, run context
- ``corpus``     — read files and directories into a corpus
- ``lexicon``    — user-defined dictionaries: load, match, score (keyword or semantic)
- ``taxonomy``   — science backbones: load, map concepts onto them, project weights
- ``backends``   — embedding and chat backends behind one protocol
- ``viz``        — backbone overlay, comparison and heatmap figures
- ``report``     — CSV, Markdown summary and manifest writers
- ``workflows``  — end-to-end pipelines: ``dictionary``, ``narrative``
  and ``formulation``
- ``pieces``     — the workflows cut into steps a workflow engine runs one at a time
"""

from semantic_sensemaking._version import __version__
from semantic_sensemaking.core.config import RunConfig
from semantic_sensemaking.core.types import (
    Corpus,
    Document,
    Lexicon,
    LexiconEntry,
    Taxonomy,
    TaxonomyNode,
)
from semantic_sensemaking.corpus.loader import load_corpus
from semantic_sensemaking.data import sample_corpus_path
from semantic_sensemaking.lexicon.io import load_lexicon
from semantic_sensemaking.taxonomy.io import load_taxonomy

__all__ = [
    "Corpus",
    "Document",
    "Lexicon",
    "LexiconEntry",
    "RunConfig",
    "Taxonomy",
    "TaxonomyNode",
    "__version__",
    "load_corpus",
    "load_lexicon",
    "load_taxonomy",
    "sample_corpus_path",
]
