"""Narrative workflow orchestration: segment, embed, cluster, label, map, project."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from semantic_sensemaking.backends.base import ChatBackend, EmbeddingBackend
from semantic_sensemaking.backends.factory import get_chat_backend, get_embedding_backend
from semantic_sensemaking.core.config import NarrativeConfig, RunConfig, ScoringConfig
from semantic_sensemaking.core.context import RunContext
from semantic_sensemaking.core.text import chunk_words
from semantic_sensemaking.core.types import Corpus, Taxonomy
from semantic_sensemaking.errors import ConfigError, CorpusError
from semantic_sensemaking.taxonomy.io import load_taxonomy
from semantic_sensemaking.taxonomy.mapping import map_lexicon_to_taxonomy
from semantic_sensemaking.taxonomy.projection import project_weights
from semantic_sensemaking.workflows._inputs import load_configured_corpus
from semantic_sensemaking.workflows.narrative.cluster import cluster_passages
from semantic_sensemaking.workflows.narrative.keywords import cluster_keywords
from semantic_sensemaking.workflows.narrative.labeling import frames_to_lexicon, label_frames
from semantic_sensemaking.workflows.narrative.outputs import write_outputs
from semantic_sensemaking.workflows.narrative.result import NarrativeResult, Passage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NarrativeRun:
    result: NarrativeResult
    context: RunContext


def segment_corpus(corpus: Corpus, words: int, overlap: int = 0) -> tuple[Passage, ...]:
    """Split every document into passages of ``words`` words."""
    if overlap >= words:
        raise ConfigError("narrative.segment_overlap_words must be smaller than segment_words")
    return tuple(
        Passage(doc.doc_id, i, text)
        for doc in corpus
        for i, text in enumerate(chunk_words(doc.text, words, overlap))
    )


def analyze(
    corpus: Corpus,
    taxonomy: Taxonomy,
    embedding_backend: EmbeddingBackend,
    narrative: NarrativeConfig | None = None,
    scoring: ScoringConfig | None = None,
    *,
    chat_backend: ChatBackend | None = None,
) -> NarrativeResult:
    """Pure analysis with no file I/O — the entry point for notebooks."""
    narrative = narrative or NarrativeConfig()
    scoring = scoring or ScoringConfig()
    if embedding_backend is None:
        raise ConfigError(
            "The narrative workflow needs an embedding backend: set embedding.kind "
            "(openai_compatible, huggingface_local, or hashing for a quick offline test)"
        )

    passages = segment_corpus(corpus, narrative.segment_words, narrative.segment_overlap_words)
    if not passages:
        raise CorpusError("No passages to analyze: every document is empty")
    texts = [p.text for p in passages]
    log.info("Embedding %d passages from %d documents", len(passages), len(corpus))
    vectors = np.asarray(embedding_backend.embed(texts, kind="document"))

    clustering = cluster_passages(
        vectors,
        narrative.max_frames,
        min_size=narrative.min_frame_passages,
        seed=narrative.seed,
    )
    keywords = cluster_keywords(texts, clustering.labels, clustering.k, top_n=narrative.terms_per_frame)
    frames = label_frames(
        [(clustering.members(c), keywords[c]) for c in range(clustering.k)],
        texts,
        chat=chat_backend,
        representative=narrative.representative_passages,
    )
    log.info("Found %d frames (labeled by %s)", len(frames), "chat" if chat_backend else "keywords")

    lexicon = frames_to_lexicon(frames)
    mapping = map_lexicon_to_taxonomy(
        lexicon,
        taxonomy,
        top_n=scoring.top_n_assignments,
        method=scoring.mapping_method,
        backend=embedding_backend,
    )
    weights = {f.name: float(f.n_passages) for f in frames}
    projection = project_weights(weights, mapping, unmapped=scoring.unmapped_groups)
    return NarrativeResult(
        corpus=corpus,
        passages=passages,
        passage_frames=tuple(int(c) for c in clustering.labels),
        passage_similarity=tuple(float(s) for s in clustering.similarity),
        frames=tuple(frames),
        lexicon=lexicon,
        taxonomy=taxonomy,
        mapping=mapping,
        projection=projection,
        narrative=narrative,
        scoring=scoring,
    )


def run(config: RunConfig) -> NarrativeRun:
    """Load inputs, analyze, and write every output into a new run directory."""
    corpus = load_configured_corpus(config)
    taxonomy = load_taxonomy(config.inputs.taxonomy, layer=config.inputs.taxonomy_layer)
    if config.inputs.lexicon is not None:
        log.warning("inputs.lexicon is ignored by the narrative workflow (frames replace it)")
    embedding = get_embedding_backend(config.embedding)
    chat = None
    try:
        chat = get_chat_backend(config.chat)
        result = analyze(
            corpus, taxonomy, embedding, config.narrative, config.scoring, chat_backend=chat
        )
    finally:
        for backend in (embedding, chat):
            if backend is not None:
                backend.close()
    context = RunContext.create(config.output.dir, "narrative", config.output.run_id)
    write_outputs(result, context, config)
    log.info("Wrote %d artifacts to %s", len(context.artifacts), context.run_dir)
    return NarrativeRun(result, context)
