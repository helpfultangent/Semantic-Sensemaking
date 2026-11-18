"""Dictionary workflow orchestration: load inputs, analyze, write outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from semantic_sensemaking.backends.base import EmbeddingBackend
from semantic_sensemaking.backends.factory import get_embedding_backend
from semantic_sensemaking.core.config import RunConfig, ScoringConfig
from semantic_sensemaking.core.context import RunContext
from semantic_sensemaking.core.types import Corpus, Lexicon, Taxonomy
from semantic_sensemaking.errors import ConfigError
from semantic_sensemaking.lexicon.io import load_lexicon
from semantic_sensemaking.lexicon.scoring import score_corpus
from semantic_sensemaking.lexicon.semantic import score_corpus_semantic
from semantic_sensemaking.taxonomy.io import load_taxonomy
from semantic_sensemaking.taxonomy.mapping import map_lexicon_to_taxonomy
from semantic_sensemaking.taxonomy.projection import project_weights
from semantic_sensemaking.workflows._inputs import load_configured_corpus
from semantic_sensemaking.workflows.dictionary.outputs import write_outputs
from semantic_sensemaking.workflows.dictionary.result import DictionaryResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DictionaryRun:
    result: DictionaryResult
    context: RunContext


def analyze(
    corpus: Corpus,
    lexicon: Lexicon,
    taxonomy: Taxonomy,
    scoring: ScoringConfig | None = None,
    *,
    embedding_backend: EmbeddingBackend | None = None,
) -> DictionaryResult:
    """Pure analysis with no file I/O — the entry point for notebooks."""
    scoring = scoring or ScoringConfig()
    needs_backend = scoring.method == "semantic" or scoring.mapping_method == "embedding"
    if needs_backend and embedding_backend is None:
        raise ConfigError(
            "scoring.method 'semantic' and mapping_method 'embedding' need an embedding "
            "backend: set embedding.kind (e.g. openai_compatible, huggingface_local, hashing)"
        )

    mapping = map_lexicon_to_taxonomy(
        lexicon,
        taxonomy,
        top_n=scoring.top_n_assignments,
        method=scoring.mapping_method,
        backend=embedding_backend,
    )
    if scoring.method == "semantic":
        assert embedding_backend is not None
        scores = score_corpus_semantic(
            corpus,
            lexicon,
            embedding_backend,
            threshold=scoring.semantic_threshold,
            chunk_size=scoring.chunk_words,
            chunk_overlap=scoring.chunk_overlap_words,
        )
    else:
        scores = score_corpus(corpus, lexicon, scoring)
    projection = project_weights(scores.group_weights(), mapping, unmapped=scoring.unmapped_groups)
    return DictionaryResult(corpus, lexicon, taxonomy, mapping, scores, projection, scoring)


def load_inputs(config: RunConfig) -> tuple[Corpus, Lexicon, Taxonomy]:
    inputs = config.inputs
    corpus = load_configured_corpus(config)
    lexicon = load_lexicon(inputs.lexicon)
    taxonomy = load_taxonomy(inputs.taxonomy, layer=inputs.taxonomy_layer)
    return corpus, lexicon, taxonomy


def run(config: RunConfig) -> DictionaryRun:
    """Load inputs, analyze, and write every output into a new run directory."""
    corpus, lexicon, taxonomy = load_inputs(config)
    backend = get_embedding_backend(config.embedding)
    try:
        result = analyze(corpus, lexicon, taxonomy, config.scoring, embedding_backend=backend)
    finally:
        if backend is not None:
            backend.close()
    context = RunContext.create(config.output.dir, "dictionary", config.output.run_id)
    write_outputs(result, context, config)
    log.info("Wrote %d artifacts to %s", len(context.artifacts), context.run_dir)
    return DictionaryRun(result, context)
