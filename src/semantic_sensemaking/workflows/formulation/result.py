"""The in-memory result of a problem formulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from semantic_sensemaking.core.types import Corpus, Lexicon, Taxonomy, TaxonomyNode
from semantic_sensemaking.taxonomy.mapping import UNCATEGORIZED, TaxonomyMapping
from semantic_sensemaking.workflows.formulation.grounding import TermEvidence, evidence_frame


@dataclass(frozen=True)
class FormulationResult:
    problem_statement: str
    proposed: Lexicon
    """The model's draft, before grounding."""
    lexicon: Lexicon
    """The draft after grounding: what the dictionary workflow should read."""
    taxonomy: Taxonomy
    """The backbone, plus any proposed domains."""
    mapping: TaxonomyMapping
    evidence: tuple[TermEvidence, ...]
    """Empty when grounding was ``none``."""
    dropped_groups: tuple[str, ...]
    added_domains: tuple[TaxonomyNode, ...]
    corpus: Corpus | None

    def uncovered_groups(self) -> list[str]:
        return [g for g in self.mapping.labels() if self.mapping.primary_domain(g) == UNCATEGORIZED]

    def proposed_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"group": e.name, "description": e.description or "", "terms": ", ".join(e.terms)}
                for e in self.proposed
            ],
            columns=["group", "description", "terms"],
        )

    def lexicon_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"group": e.name, "terms": ", ".join(e.terms)} for e in self.lexicon],
            columns=["group", "terms"],
        )

    def coverage_table(self) -> pd.DataFrame:
        """One row per proposed group: how much survived grounding and where it maps."""
        kept = {e.name: e for e in self.lexicon}
        mapping = self.mapping.to_frame().set_index("group") if len(self.mapping.labels()) else None
        rows = []
        for entry in self.proposed:
            final = kept.get(entry.name)
            mapped = mapping is not None and entry.name in mapping.index
            rows.append(
                {
                    "group": entry.name,
                    "n_terms_proposed": len(entry.terms),
                    "n_terms_kept": len(final.terms) if final else 0,
                    "kept": final is not None,
                    "primary_domain": mapping.at[entry.name, "primary_domain"] if mapped else "",
                    "assignments": mapping.at[entry.name, "assignments"] if mapped else "",
                    "needs_review": (final is None)
                    or (mapped and mapping.at[entry.name, "primary_domain"] == UNCATEGORIZED),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "group", "n_terms_proposed", "n_terms_kept", "kept",
                "primary_domain", "assignments", "needs_review",
            ],
        )

    def tables(self) -> dict[str, pd.DataFrame]:
        tables = {
            "proposed_groups": self.proposed_table(),
            "group_coverage": self.coverage_table(),
        }
        if self.evidence:
            tables["term_grounding"] = evidence_frame(self.evidence)
        return tables
