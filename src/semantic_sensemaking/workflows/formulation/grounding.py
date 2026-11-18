"""Check proposed terms against a seed corpus and keep the ones that occur in it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from semantic_sensemaking.backends.base import EmbeddingBackend
from semantic_sensemaking.core.config import ScoringConfig
from semantic_sensemaking.core.text import chunk_words
from semantic_sensemaking.core.types import Corpus, Lexicon, LexiconEntry
from semantic_sensemaking.lexicon.matching import TermMatcher


@dataclass(frozen=True)
class TermEvidence:
    group: str
    term: str
    count: int
    """Keyword occurrences across the corpus."""
    n_documents: int
    similarity: float
    """Best cosine similarity to any corpus passage; NaN when not computed."""
    grounded: bool


def ground_lexicon(
    lexicon: Lexicon,
    corpus: Corpus,
    *,
    scoring: ScoringConfig,
    min_count: int = 1,
    backend: EmbeddingBackend | None = None,
) -> tuple[TermEvidence, ...]:
    """Evidence for every term in ``lexicon``.

    A term is grounded when it occurs at least ``min_count`` times (keyword match
    with ``scoring.match_mode``), or — when ``backend`` is given — when some
    passage is at least ``scoring.semantic_threshold`` similar to it.
    """
    texts = [doc.text for doc in corpus]
    sims: dict[str, float] = {}
    if backend is not None:
        sims = _best_passage_similarity(lexicon, corpus, backend, scoring)

    evidence = []
    for entry in lexicon:
        matcher = TermMatcher(entry.terms, mode=scoring.match_mode, case_sensitive=scoring.case_sensitive)
        per_doc = [matcher.count_detailed(text) for text in texts]
        for term in dict.fromkeys(entry.terms):
            counts = [d.get(term, 0) for d in per_doc]
            count = int(sum(counts))
            sim = sims.get(term, math.nan)
            grounded = count >= min_count or (not math.isnan(sim) and sim >= scoring.semantic_threshold)
            evidence.append(
                TermEvidence(entry.name, term, count, sum(1 for c in counts if c), sim, grounded)
            )
    return tuple(evidence)


def grounded_lexicon(
    lexicon: Lexicon, evidence: tuple[TermEvidence, ...], *, min_terms: int = 1
) -> tuple[Lexicon, tuple[str, ...]]:
    """Keep grounded terms; drop groups left with fewer than ``min_terms``. Returns dropped names."""
    kept_terms: dict[str, list[str]] = {}
    for ev in evidence:
        if ev.grounded:
            kept_terms.setdefault(ev.group, []).append(ev.term)
    entries, dropped = [], []
    for entry in lexicon:
        terms = kept_terms.get(entry.name, [])
        if len(terms) >= min_terms:
            entries.append(LexiconEntry(entry.name, tuple(terms), entry.description))
        else:
            dropped.append(entry.name)
    return Lexicon(tuple(entries), name=lexicon.name, source=lexicon.source), tuple(dropped)


def evidence_frame(evidence: tuple[TermEvidence, ...]) -> pd.DataFrame:
    columns = ["group", "term", "count", "n_documents", "similarity", "grounded"]
    return pd.DataFrame(
        [
            {
                "group": e.group,
                "term": e.term,
                "count": e.count,
                "n_documents": e.n_documents,
                "similarity": None if math.isnan(e.similarity) else round(e.similarity, 4),
                "grounded": e.grounded,
            }
            for e in evidence
        ],
        columns=columns,
    )


def _best_passage_similarity(
    lexicon: Lexicon, corpus: Corpus, backend: EmbeddingBackend, scoring: ScoringConfig
) -> dict[str, float]:
    terms = list(dict.fromkeys(t for entry in lexicon for t in entry.terms))
    passages = [
        p for doc in corpus for p in chunk_words(doc.text, scoring.chunk_words, scoring.chunk_overlap_words)
    ]
    if not terms or not passages:
        return {}
    sims = backend.embed(terms, kind="query") @ np.asarray(backend.embed(passages, kind="document")).T
    return {term: float(row.max()) for term, row in zip(terms, sims)}
