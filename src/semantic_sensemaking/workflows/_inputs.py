"""Input loading shared by workflows."""

from __future__ import annotations

import logging

from semantic_sensemaking.core.config import RunConfig
from semantic_sensemaking.core.types import Corpus
from semantic_sensemaking.corpus.loader import load_corpus
from semantic_sensemaking.data import sample_corpus_path
from semantic_sensemaking.errors import ConfigError, CorpusError

log = logging.getLogger(__name__)


def load_configured_corpus(config: RunConfig) -> Corpus:
    """Load ``inputs.corpus`` (or the sample corpus when allowed); fail if nothing loads."""
    inputs = config.inputs
    if inputs.corpus is not None:
        corpus_path = inputs.corpus
    elif inputs.use_sample_corpus:
        corpus_path = sample_corpus_path()
        log.warning("No corpus given; using the built-in synthetic sample corpus")
    else:
        raise ConfigError(
            "No corpus: set inputs.corpus (or --corpus), or inputs.use_sample_corpus: true"
        )
    corpus = load_corpus(corpus_path, config.corpus)
    if not len(corpus):
        reasons = "; ".join(f"{s.path}: {s.reason}" for s in corpus.skipped[:5])
        raise CorpusError(f"No documents loaded from {corpus_path}. {reasons}".strip())
    return corpus
