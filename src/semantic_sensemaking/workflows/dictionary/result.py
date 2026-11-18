"""The in-memory result of a dictionary analysis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from semantic_sensemaking.core.config import ScoringConfig
from semantic_sensemaking.core.types import Corpus, Document, Lexicon, Taxonomy
from semantic_sensemaking.corpus.select import resolve_document
from semantic_sensemaking.lexicon.scoring import LexiconScores
from semantic_sensemaking.taxonomy.mapping import TaxonomyMapping
from semantic_sensemaking.taxonomy.projection import (
    Projection,
    compare_distributions,
    project_weights,
)
from semantic_sensemaking.viz.overlay_data import BackboneOverlayData


@dataclass(frozen=True)
class DictionaryResult:
    corpus: Corpus
    lexicon: Lexicon
    taxonomy: Taxonomy
    mapping: TaxonomyMapping
    scores: LexiconScores
    projection: Projection
    scoring: ScoringConfig

    @cached_property
    def comparison(self) -> pd.DataFrame:
        return compare_distributions(self.mapping.base_distribution(), self.projection)

    def document(self, spec: int | str) -> Document:
        """Look up a document by index, id, label, or unique substring."""
        return resolve_document(self.corpus, spec)

    def document_projection(self, spec: int | str) -> tuple[LexiconScores, Projection]:
        doc = self.document(spec)
        doc_scores = self.scores.for_document(doc.doc_id)
        projection = project_weights(
            doc_scores.group_weights(), self.mapping, unmapped=self.scoring.unmapped_groups
        )
        return doc_scores, projection

    def to_overlay_data(self, document: int | str | None = None) -> BackboneOverlayData:
        """Figure data for the whole corpus, or for one document when ``document`` is given."""
        noun = "passages matched" if self.scores.method == "semantic" else "match count"
        if document is None:
            outer = {}
            for doc in self.corpus:
                top = "<br>".join(f"  • {g}: {c}" for g, c in self.scores.top_groups(doc.doc_id))
                outer[doc.label] = f"<b>{doc.label}</b><br>Top matched groups:<br>{top or '  (none)'}"
            return BackboneOverlayData(
                taxonomy=self.taxonomy,
                mapping=self.mapping,
                label_weights=self.scores.group_weights(),
                projection=self.projection,
                outer_nodes=outer,
                outer_edges=[(r.doc, r.group, float(r.count)) for r in self.scores.rows if r.count > 0],
                title="Backbone overlay projection",
                subtitle=(
                    f"Base: {self.taxonomy.source or self.taxonomy.name}  ·  "
                    f"Lexicon: {self.lexicon.name}  ·  Overlay: {len(self.corpus)} document(s)"
                ),
                label_noun=noun,
            )

        doc = self.document(document)
        doc_scores, projection = self.document_projection(doc.doc_id)
        weights = doc_scores.group_weights()
        active = sum(1 for w in weights.values() if w > 0)
        return BackboneOverlayData(
            taxonomy=self.taxonomy,
            mapping=self.mapping,
            label_weights=weights,
            projection=projection,
            title="Backbone overlay — single document",
            subtitle=(
                f"<b>{doc.label}</b> · {doc_scores.total_matches()} matches across "
                f"{active} active groups · {doc.n_words:,} words"
            ),
            show_weights_in_labels=True,
            label_noun=noun,
        )

    def frames(self) -> dict[str, pd.DataFrame]:
        """Every exportable table, keyed by output file stem."""
        return {
            "overlay_doc_group_counts": self.scores.to_frame(),
            "overlay_group_totals": self.scores.group_totals(),
            "overlay_domain_totals": self.projection.domain_totals(),
            "overlay_subdiscipline_totals": self.projection.subdiscipline_totals(),
            "comparison_base_vs_overlay": self.comparison,
            "group_mapping": self.mapping.to_frame(),
            "corpus_documents": self.corpus.describe(),
            "skipped_files": pd.DataFrame(
                [{"path": s.path, "reason": s.reason} for s in self.corpus.skipped],
                columns=["path", "reason"],
            ),
        }
