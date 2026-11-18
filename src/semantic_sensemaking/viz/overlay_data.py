"""The neutral data contract consumed by the backbone overlay renderer.

Any workflow that can produce label weights, a taxonomy mapping and a
projection can draw the figure — the renderer never imports a workflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from semantic_sensemaking.core.types import Taxonomy
from semantic_sensemaking.taxonomy.mapping import TaxonomyMapping
from semantic_sensemaking.taxonomy.projection import Projection


@dataclass(frozen=True)
class BackboneOverlayData:
    taxonomy: Taxonomy
    mapping: TaxonomyMapping
    label_weights: Mapping[str, float]
    projection: Projection
    outer_nodes: Mapping[str, str] = field(default_factory=dict)
    """Outer-ring node label -> hover HTML (e.g. one node per document)."""
    outer_edges: Sequence[tuple[str, str, float]] = ()
    """``(outer node label, label, weight)`` connections."""
    title: str = ""
    subtitle: str = ""
    show_weights_in_labels: bool = False
    label_noun: str = "match count"

    def active_labels(self) -> list[str]:
        """Labels with positive weight, in mapping order; zero-weight labels are not drawn."""
        return [label for label in self.mapping.labels() if self.label_weights.get(label, 0) > 0]
