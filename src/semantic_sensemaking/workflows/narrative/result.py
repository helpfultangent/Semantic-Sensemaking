"""The in-memory result of a narrative analysis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd

from semantic_sensemaking.core.config import NarrativeConfig, ScoringConfig
from semantic_sensemaking.core.types import Corpus, Document, Lexicon, Taxonomy
from semantic_sensemaking.corpus.select import resolve_document
from semantic_sensemaking.taxonomy.mapping import TaxonomyMapping
from semantic_sensemaking.taxonomy.projection import (
    Projection,
    compare_distributions,
    project_weights,
)
from semantic_sensemaking.viz.overlay_data import BackboneOverlayData
from semantic_sensemaking.workflows.narrative.labeling import Frame

SNIPPET_WORDS = 40


@dataclass(frozen=True)
class Passage:
    doc_id: str
    index: int
    """Position of the passage within its document."""
    text: str


@dataclass(frozen=True)
class NarrativeResult:
    corpus: Corpus
    passages: tuple[Passage, ...]
    passage_frames: tuple[int, ...]
    """Frame id of each passage, aligned with ``passages``."""
    passage_similarity: tuple[float, ...]
    """Cosine similarity of each passage to its frame centroid."""
    frames: tuple[Frame, ...]
    lexicon: Lexicon
    taxonomy: Taxonomy
    mapping: TaxonomyMapping
    projection: Projection
    narrative: NarrativeConfig
    scoring: ScoringConfig

    # ── weights ──────────────────────────────────────────────────────────
    def frame_weights(self, doc_id: str | None = None) -> dict[str, float]:
        """Passages per frame, for the whole corpus or one document."""
        weights = {f.name: 0.0 for f in self.frames}
        for passage, fid in zip(self.passages, self.passage_frames):
            if doc_id is None or passage.doc_id == doc_id:
                weights[self.frames[fid].name] += 1.0
        return weights

    @cached_property
    def comparison(self) -> pd.DataFrame:
        return compare_distributions(self.mapping.base_distribution(), self.projection)

    def document(self, spec: int | str) -> Document:
        return resolve_document(self.corpus, spec)

    def document_projection(self, spec: int | str) -> tuple[dict[str, float], Projection]:
        doc = self.document(spec)
        weights = self.frame_weights(doc.doc_id)
        return weights, project_weights(weights, self.mapping, unmapped=self.scoring.unmapped_groups)

    # ── tables ───────────────────────────────────────────────────────────
    def frames_table(self) -> pd.DataFrame:
        rows = []
        for f in self.frames:
            docs = {self.passages[i].doc_id for i in f.passage_ids}
            central = self.passages[f.passage_ids[0]].text if f.passage_ids else ""
            rows.append(
                {
                    "frame_id": f.frame_id + 1,
                    "frame": f.name,
                    "description": f.description,
                    "n_passages": f.n_passages,
                    "n_documents": len(docs),
                    "primary_domain": self.mapping.primary_domain(f.name),
                    "terms": ", ".join(f.terms),
                    "keywords": ", ".join(f.keywords),
                    "labeled_by": f.labeled_by,
                    "central_passage": _snippet(central),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "frame_id", "frame", "description", "n_passages", "n_documents",
                "primary_domain", "terms", "keywords", "labeled_by", "central_passage",
            ],
        )

    def passages_table(self) -> pd.DataFrame:
        labels = {d.doc_id: d.label for d in self.corpus}
        return pd.DataFrame(
            [
                {
                    "doc_id": p.doc_id,
                    "doc": labels[p.doc_id],
                    "passage": p.index,
                    "frame": self.frames[fid].name,
                    "similarity": round(sim, 4),
                    "n_words": len(p.text.split()),
                    "text": _snippet(p.text),
                }
                for p, fid, sim in zip(self.passages, self.passage_frames, self.passage_similarity)
            ],
            columns=["doc_id", "doc", "passage", "frame", "similarity", "n_words", "text"],
        )

    def doc_frame_counts(self) -> pd.DataFrame:
        """Long table: ``doc_id, doc, frame, n_passages, share_pct`` for every pair."""
        rows = []
        for doc in self.corpus:
            weights = self.frame_weights(doc.doc_id)
            total = sum(weights.values())
            for frame, n in weights.items():
                rows.append(
                    {
                        "doc_id": doc.doc_id,
                        "doc": doc.label,
                        "frame": frame,
                        "n_passages": int(n),
                        "share_pct": 100.0 * n / total if total else 0.0,
                    }
                )
        return pd.DataFrame(rows, columns=["doc_id", "doc", "frame", "n_passages", "share_pct"])

    def pivot(self) -> pd.DataFrame:
        """Document × frame matrix of the share of each document's passages (%)."""
        counts = self.doc_frame_counts()
        if counts.empty:
            return pd.DataFrame()
        return counts.pivot_table(
            index="doc", columns="frame", values="share_pct", sort=False, aggfunc="sum"
        ).reindex(columns=[f.name for f in self.frames])

    def tables(self) -> dict[str, pd.DataFrame]:
        """Every exportable table, keyed by output file stem."""
        return {
            "frames": self.frames_table(),
            "passage_frames": self.passages_table(),
            "doc_frame_counts": self.doc_frame_counts(),
            "frame_mapping": self.mapping.to_frame().rename(columns={"group": "frame"}),
            "frame_domain_totals": self.projection.domain_totals(),
            "frame_subdiscipline_totals": self.projection.subdiscipline_totals(),
            "comparison_base_vs_frames": self.comparison,
            "corpus_documents": self.corpus.describe(),
            "skipped_files": pd.DataFrame(
                [{"path": s.path, "reason": s.reason} for s in self.corpus.skipped],
                columns=["path", "reason"],
            ),
        }

    # ── figures ──────────────────────────────────────────────────────────
    def to_overlay_data(self, document: int | str | None = None) -> BackboneOverlayData:
        """Figure data with frames in place of lexicon groups."""
        if document is None:
            outer, edges = {}, []
            for doc in self.corpus:
                weights = self.frame_weights(doc.doc_id)
                top = sorted(((n, f) for f, n in weights.items() if n > 0), reverse=True)[:5]
                lines = "<br>".join(f"  • {f}: {int(n)}" for n, f in top) or "  (none)"
                outer[doc.label] = f"<b>{doc.label}</b><br>Top frames:<br>{lines}"
                edges += [(doc.label, f, n) for f, n in weights.items() if n > 0]
            return BackboneOverlayData(
                taxonomy=self.taxonomy,
                mapping=self.mapping,
                label_weights=self.frame_weights(),
                projection=self.projection,
                outer_nodes=outer,
                outer_edges=edges,
                title="Narrative frames on the science backbone",
                subtitle=(
                    f"Base: {self.taxonomy.source or self.taxonomy.name}  ·  "
                    f"{len(self.frames)} frame(s) from {len(self.passages)} passages  ·  "
                    f"{len(self.corpus)} document(s)"
                ),
                label_noun="passages",
            )

        doc = self.document(document)
        weights, projection = self.document_projection(doc.doc_id)
        return BackboneOverlayData(
            taxonomy=self.taxonomy,
            mapping=self.mapping,
            label_weights=weights,
            projection=projection,
            title="Narrative frames — single document",
            subtitle=f"<b>{doc.label}</b> · {int(sum(weights.values()))} passages",
            show_weights_in_labels=True,
            label_noun="passages",
        )


def _snippet(text: str, words: int = SNIPPET_WORDS) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")
