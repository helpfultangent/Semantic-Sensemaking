"""Project label weights onto a taxonomy and compare distributions.

Workflow-agnostic: a "label" can be a lexicon group, a narrative frame, or a
cluster — anything with a weight and a :class:`TaxonomyMapping`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from semantic_sensemaking.taxonomy.mapping import TaxonomyMapping


@dataclass(frozen=True)
class Projection:
    domain_weights: Mapping[str, float]
    subdiscipline_weights: Mapping[tuple[str, str], float]

    def domain_totals(self) -> pd.DataFrame:
        """``domain, weight`` sorted descending (the prototype's ``overlay_domain_totals``)."""
        frame = pd.DataFrame(
            list(self.domain_weights.items()), columns=["domain", "weight"]
        )
        return frame.sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)

    def subdiscipline_totals(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [(d, s, w) for (d, s), w in self.subdiscipline_weights.items()],
            columns=["domain", "subdiscipline", "weight"],
        )
        return frame.sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)

    def total_weight(self) -> float:
        return float(sum(self.domain_weights.values()))


def project_weights(
    weights: Mapping[str, float],
    mapping: TaxonomyMapping,
    *,
    unmapped: Literal["drop", "keep"] = "drop",
) -> Projection:
    """Split each label's weight across its assignments in proportion to score.

    A label whose assignments all score 0 — the ``Uncategorized`` fallback — is
    the prototype's third bug: with ``unmapped="drop"`` (legacy) its weight is
    multiplied by zero, so its matches vanish from domain totals while the
    domain still appears with weight 0. ``unmapped="keep"`` splits the weight
    evenly across those assignments instead, so the matches stay visible under
    ``Uncategorized``.
    """
    domain_weights: dict[str, float] = {}
    sub_weights: dict[tuple[str, str], float] = {}
    for label in mapping.labels():
        weight = float(weights.get(label, 0))
        assignments = mapping[label]
        if weight <= 0 or not assignments:
            continue
        total = sum(a.score for a in assignments)
        if total > 0:
            shares = [a.score / total for a in assignments]
        elif unmapped == "keep":
            shares = [1 / len(assignments)] * len(assignments)
        else:
            shares = [0.0] * len(assignments)
        for assignment, share in zip(assignments, shares):
            w = weight * share
            domain_weights[assignment.domain] = domain_weights.get(assignment.domain, 0.0) + w
            key = (assignment.domain, assignment.subdiscipline)
            sub_weights[key] = sub_weights.get(key, 0.0) + w
    return Projection(domain_weights, sub_weights)


def compare_distributions(base: pd.DataFrame, projection: Projection) -> pd.DataFrame:
    """Base-layer vs projected share of attention per domain.

    ``base`` is ``domain, base_n_groups`` (see ``TaxonomyMapping.base_distribution``).
    Returns ``domain, base_n_groups, base_pct, overlay_weight, overlay_pct,
    delta_pct`` sorted by ``overlay_pct`` descending.
    """
    base = base.copy()
    base["base_pct"] = 100 * base["base_n_groups"] / (base["base_n_groups"].sum() or 1)
    overlay = projection.domain_totals().rename(columns={"weight": "overlay_weight"})
    overlay["overlay_pct"] = 100 * overlay["overlay_weight"] / (overlay["overlay_weight"].sum() or 1.0)
    merged = pd.merge(base, overlay, on="domain", how="outer").fillna(0)
    merged["delta_pct"] = merged["overlay_pct"] - merged["base_pct"]
    columns = ["domain", "base_n_groups", "base_pct", "overlay_weight", "overlay_pct", "delta_pct"]
    return (
        merged[columns]
        .sort_values("overlay_pct", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
