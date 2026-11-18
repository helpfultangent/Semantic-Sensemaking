"""Score a corpus against a lexicon: the document × group count table."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from semantic_sensemaking.core.config import ScoringConfig
from semantic_sensemaking.core.types import Corpus, Lexicon
from semantic_sensemaking.lexicon.matching import TermMatcher

FRAME_COLUMNS = ["doc", "doc_id", "group", "count", "doc_n_words", "per_1k_words"]


@dataclass(frozen=True)
class DocGroupScore:
    doc_id: str
    doc: str
    group: str
    count: int
    doc_n_words: int

    @property
    def per_1k_words(self) -> float:
        return round(1000 * self.count / max(self.doc_n_words, 1), 2)


@dataclass(frozen=True)
class LexiconScores:
    """Long-format (document, group) match counts plus lookup helpers.

    ``method`` records how counts were produced: ``"keyword"`` counts term
    occurrences; ``"semantic"`` counts passages whose embedding similarity to the
    group cleared a threshold. Both share this shape so everything downstream
    (projection, figures, exports) is agnostic to the method.
    """

    rows: tuple[DocGroupScore, ...]
    groups: tuple[str, ...]
    documents: tuple[str, ...]
    method: str = "keyword"

    def to_frame(self) -> pd.DataFrame:
        """The prototype's ``overlay_doc_group_df`` (plus a ``doc_id`` column)."""
        return pd.DataFrame(
            [
                {
                    "doc": r.doc,
                    "doc_id": r.doc_id,
                    "group": r.group,
                    "count": r.count,
                    "doc_n_words": r.doc_n_words,
                    "per_1k_words": r.per_1k_words,
                }
                for r in self.rows
            ],
            columns=FRAME_COLUMNS,
        )

    def group_weights(self) -> dict[str, float]:
        """Total count per group, in lexicon order; groups with no matches are 0."""
        totals = {g: 0.0 for g in self.groups}
        for r in self.rows:
            totals[r.group] += r.count
        return totals

    def group_totals(self) -> pd.DataFrame:
        """The prototype's ``overlay_group_totals``: ``group, count`` sorted descending."""
        frame = pd.DataFrame(
            [{"group": g, "count": int(w)} for g, w in self.group_weights().items()],
            columns=["group", "count"],
        )
        return frame.sort_values("count", ascending=False, kind="stable").reset_index(drop=True)

    def total_matches(self) -> int:
        return sum(r.count for r in self.rows)

    def for_document(self, doc_id: str) -> LexiconScores:
        rows = tuple(r for r in self.rows if r.doc_id == doc_id)
        if not rows:
            raise KeyError(doc_id)
        return LexiconScores(rows, self.groups, (doc_id,), self.method)

    def top_groups(self, doc_id: str, n: int = 5) -> list[tuple[str, int]]:
        rows = [r for r in self.rows if r.doc_id == doc_id and r.count > 0]
        rows.sort(key=lambda r: r.count, reverse=True)
        return [(r.group, r.count) for r in rows[:n]]

    def pivot(self, value: str = "per_1k_words") -> pd.DataFrame:
        """Document × group matrix, columns ordered by overall group total (heatmap input)."""
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame()
        index = "doc" if frame["doc"].nunique() == len(self.documents) else "doc_id"
        matrix = frame.pivot(index=index, columns="group", values=value).fillna(0)
        order = self.group_totals()["group"].tolist()
        return matrix[[g for g in order if g in matrix.columns]]


def score_corpus(
    corpus: Corpus, lexicon: Lexicon, config: ScoringConfig | None = None
) -> LexiconScores:
    """Count every lexicon group's terms in every document."""
    config = config or ScoringConfig()
    matchers = {
        entry.name: TermMatcher(
            entry.terms, mode=config.match_mode, case_sensitive=config.case_sensitive
        )
        for entry in lexicon
    }
    rows = tuple(
        DocGroupScore(doc.doc_id, doc.label, group, matcher.count(doc.text), doc.n_words)
        for doc in corpus
        for group, matcher in matchers.items()
    )
    return LexiconScores(rows, lexicon.names(), corpus.ids(), "keyword")
