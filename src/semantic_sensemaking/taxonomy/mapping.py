"""Map labeled term sets (lexicon groups, frames, clusters) onto a taxonomy."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from semantic_sensemaking.backends.base import EmbeddingBackend
from semantic_sensemaking.core.types import GENERAL_SUBDISCIPLINE, Lexicon, Taxonomy

UNCATEGORIZED = "Uncategorized"

MappingMethod = Literal["legacy", "subdiscipline_aware", "embedding"]


@dataclass(frozen=True)
class TaxonomyAssignment:
    domain: str
    subdiscipline: str
    score: float

    def as_tuple(self) -> tuple[str, str, float]:
        return (self.domain, self.subdiscipline, self.score)


UNCATEGORIZED_ASSIGNMENT = TaxonomyAssignment(UNCATEGORIZED, GENERAL_SUBDISCIPLINE, 0.0)


@dataclass(frozen=True)
class TaxonomyMapping:
    """Each label's ranked taxonomy assignments. The first assignment is primary."""

    assignments: Mapping[str, tuple[TaxonomyAssignment, ...]]
    method: str
    taxonomy_name: str = "taxonomy"
    terms: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def labels(self) -> tuple[str, ...]:
        return tuple(self.assignments)

    def __getitem__(self, label: str) -> tuple[TaxonomyAssignment, ...]:
        return self.assignments[label]

    def primary_domain(self, label: str) -> str:
        return self.assignments[label][0].domain

    def to_frame(self) -> pd.DataFrame:
        """The prototype's ``group_mapping_df`` with a readable ``assignments`` column."""
        return pd.DataFrame(
            [
                {
                    "group": label,
                    "primary_domain": assigned[0].domain,
                    "assignments": "; ".join(
                        f"{a.domain} / {a.subdiscipline} ({a.score:.2f})" for a in assigned
                    ),
                    "terms": ", ".join(self.terms.get(label, ())),
                }
                for label, assigned in self.assignments.items()
            ],
            columns=["group", "primary_domain", "assignments", "terms"],
        )

    def base_distribution(self) -> pd.DataFrame:
        """``domain, base_n_groups``: how many labels have each primary domain."""
        counts: dict[str, int] = {}
        for label in self.assignments:
            domain = self.primary_domain(label)
            counts[domain] = counts.get(domain, 0) + 1
        return pd.DataFrame(
            sorted(counts.items()), columns=["domain", "base_n_groups"]
        )


def map_terms_to_taxonomy(
    terms: Sequence[str],
    taxonomy: Taxonomy,
    *,
    top_n: int = 3,
    method: Literal["legacy", "subdiscipline_aware"] = "legacy",
) -> tuple[TaxonomyAssignment, ...]:
    """Rank ``(domain, subdiscipline)`` targets for a set of terms.

    A domain's score is the number of ``terms`` that contain any of the domain's
    indicator terms as a substring.

    ``legacy`` reproduces the prototype's ``map_terms_to_backbone``: every
    subdiscipline of a hit domain receives the *domain's* score, then the top
    ``top_n`` pairs are kept. Because subdisciplines always tie, the top 3 is
    usually one domain's first three subdisciplines in file order.

    ``subdiscipline_aware`` keeps the top ``top_n`` *domains* instead, and splits
    each domain's score across its subdisciplines, weighting subdisciplines whose
    names overlap the terms (e.g. "Coastal Science" for ``coastal``). With no
    name evidence the split is even rather than arbitrary.

    No hits returns a single ``Uncategorized / General`` assignment with score 0.
    """
    lowered = [str(t).lower() for t in terms]
    domain_scores = [
        (node, sum(1 for t in lowered if any(kw.lower() in t for kw in node.effective_terms)))
        for node in taxonomy
    ]

    if method == "legacy":
        hits = [
            TaxonomyAssignment(node.domain, sub, float(score))
            for node, score in domain_scores
            if score > 0
            for sub in node.effective_subdisciplines
        ]
        if not hits:
            return (UNCATEGORIZED_ASSIGNMENT,)
        return tuple(sorted(hits, key=lambda a: -a.score)[:top_n])

    if method == "subdiscipline_aware":
        ranked = sorted(
            ((n, s) for n, s in domain_scores if s > 0), key=lambda pair: -pair[1]
        )[:top_n]
        if not ranked:
            return (UNCATEGORIZED_ASSIGNMENT,)
        assignments = []
        for node, score in ranked:
            subs = node.effective_subdisciplines
            affinity = [1 + _name_overlap(sub, lowered) for sub in subs]
            total = sum(affinity)
            ranked_subs = sorted(zip(subs, affinity), key=lambda pair: -pair[1])
            assignments.extend(
                TaxonomyAssignment(node.domain, sub, score * a / total) for sub, a in ranked_subs
            )
        return tuple(assignments)

    raise ValueError(f"Unknown term mapping method: {method!r}")


def map_lexicon_to_taxonomy(
    lexicon: Lexicon,
    taxonomy: Taxonomy,
    *,
    top_n: int = 3,
    method: MappingMethod = "legacy",
    backend: EmbeddingBackend | None = None,
    min_similarity: float = 0.3,
) -> TaxonomyMapping:
    """Map every lexicon group onto the taxonomy."""
    if method == "embedding":
        if backend is None:
            raise ValueError("mapping_method 'embedding' requires an embedding backend")
        return map_embeddings_to_taxonomy(
            lexicon, taxonomy, backend, top_n=top_n, min_similarity=min_similarity
        )
    return TaxonomyMapping(
        {
            entry.name: map_terms_to_taxonomy(entry.terms, taxonomy, top_n=top_n, method=method)
            for entry in lexicon
        },
        method=method,
        taxonomy_name=taxonomy.name,
        terms={entry.name: entry.terms for entry in lexicon},
    )


def map_embeddings_to_taxonomy(
    lexicon: Lexicon,
    taxonomy: Taxonomy,
    backend: EmbeddingBackend,
    *,
    top_n: int = 3,
    min_similarity: float = 0.3,
) -> TaxonomyMapping:
    """Assign each group to the ``top_n`` most similar ``(domain, subdiscipline)`` pairs.

    Groups and taxonomy pairs are embedded as short descriptions; the score is
    cosine similarity. Pairs below ``min_similarity`` are dropped.
    """
    pairs = list(taxonomy.pairs())
    pair_texts = [
        f"{sub} ({domain}): {', '.join(taxonomy[domain].effective_terms)}" for domain, sub in pairs
    ]
    group_texts = [f"{e.name.replace('_', ' ')}: {', '.join(e.terms)}" for e in lexicon]
    sims = backend.embed(group_texts, kind="query") @ backend.embed(pair_texts, kind="document").T

    assignments: dict[str, tuple[TaxonomyAssignment, ...]] = {}
    for entry, row in zip(lexicon, sims):
        order = np.argsort(-row, kind="stable")[:top_n]
        chosen = tuple(
            TaxonomyAssignment(pairs[i][0], pairs[i][1], float(row[i]))
            for i in order
            if row[i] >= min_similarity
        )
        assignments[entry.name] = chosen or (UNCATEGORIZED_ASSIGNMENT,)
    return TaxonomyMapping(
        assignments,
        method="embedding",
        taxonomy_name=taxonomy.name,
        terms={entry.name: entry.terms for entry in lexicon},
    )


_NAME_STOPWORDS = {"science", "sciences", "engineering", "general", "studies"}


def _name_overlap(subdiscipline: str, lowered_terms: Sequence[str]) -> int:
    tokens = [
        w for w in re.findall(r"[a-z]+", subdiscipline.lower())
        if len(w) >= 4 and w not in _NAME_STOPWORDS
    ]
    return sum(
        1
        for term in lowered_terms
        if any(tok in term or (" " not in term and len(term) >= 4 and term in tok) for tok in tokens)
    )
