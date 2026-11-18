"""Embedding-based lexicon matching.

Exact term counting misses paraphrase ("the aquifer is being drawn down faster
than it recharges" contains no ``depletion``). Here each document is split into
passages, and a passage counts toward a group when its embedding is close
enough to any of the group's terms. The output has the same shape as
:func:`~semantic_sensemaking.lexicon.scoring.score_corpus`, so projection,
figures and exports work unchanged.

Similarity scales differ between embedding models — calibrate ``threshold``
with :func:`explain_semantic_matches` on a few known documents before trusting
corpus-level numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from semantic_sensemaking.backends.base import EmbeddingBackend
from semantic_sensemaking.core.text import chunk_words
from semantic_sensemaking.core.types import Corpus, Document, Lexicon
from semantic_sensemaking.lexicon.scoring import DocGroupScore, LexiconScores


def score_corpus_semantic(
    corpus: Corpus,
    lexicon: Lexicon,
    backend: EmbeddingBackend,
    *,
    threshold: float = 0.45,
    chunk_size: int = 120,
    chunk_overlap: int = 20,
) -> LexiconScores:
    """Count, per document and group, the passages semantically matching the group."""
    terms, term_vectors = _embed_terms(lexicon, backend)
    rows = []
    for doc in corpus:
        best = _best_group_similarity(doc, lexicon, terms, term_vectors, backend, chunk_size, chunk_overlap)
        for group in lexicon.names():
            count = int((best[group] >= threshold).sum()) if best else 0
            rows.append(DocGroupScore(doc.doc_id, doc.label, group, count, doc.n_words))
    return LexiconScores(tuple(rows), lexicon.names(), corpus.ids(), "semantic")


def explain_semantic_matches(
    document: Document,
    lexicon: Lexicon,
    backend: EmbeddingBackend,
    *,
    top_k: int = 5,
    chunk_size: int = 120,
    chunk_overlap: int = 20,
) -> pd.DataFrame:
    """The ``top_k`` most similar passages per group, with the term that matched best.

    Use this to pick a ``threshold``: look at where genuine matches stop.
    """
    terms, term_vectors = _embed_terms(lexicon, backend)
    passages = chunk_words(document.text, chunk_size, chunk_overlap)
    columns = ["group", "rank", "similarity", "best_term", "passage"]
    if not passages:
        return pd.DataFrame(columns=columns)
    sims = backend.embed(passages, kind="document") @ term_vectors.T
    records = []
    for entry in lexicon:
        idx = [terms.index(t) for t in dict.fromkeys(entry.terms)]
        group_sims = sims[:, idx]
        best_per_passage = group_sims.max(axis=1)
        for rank, p in enumerate(np.argsort(-best_per_passage, kind="stable")[:top_k], start=1):
            records.append(
                {
                    "group": entry.name,
                    "rank": rank,
                    "similarity": float(best_per_passage[p]),
                    "best_term": terms[idx[int(group_sims[p].argmax())]],
                    "passage": passages[p],
                }
            )
    return pd.DataFrame(records, columns=columns)


def _embed_terms(lexicon: Lexicon, backend: EmbeddingBackend) -> tuple[list[str], np.ndarray]:
    terms = list(dict.fromkeys(t for entry in lexicon for t in entry.terms))
    return terms, backend.embed(terms, kind="query")


def _best_group_similarity(
    doc: Document,
    lexicon: Lexicon,
    terms: list[str],
    term_vectors: np.ndarray,
    backend: EmbeddingBackend,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, np.ndarray]:
    passages = chunk_words(doc.text, chunk_size, chunk_overlap)
    if not passages:
        return {}
    sims = backend.embed(passages, kind="document") @ term_vectors.T
    return {
        entry.name: sims[:, [terms.index(t) for t in dict.fromkeys(entry.terms)]].max(axis=1)
        for entry in lexicon
    }
