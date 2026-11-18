from __future__ import annotations

import pytest

from semantic_sensemaking.core.types import Lexicon
from semantic_sensemaking.taxonomy import (
    UNCATEGORIZED,
    TaxonomyAssignment,
    TaxonomyMapping,
    compare_distributions,
    map_lexicon_to_taxonomy,
    map_terms_to_taxonomy,
    project_weights,
)


def test_no_hits_maps_to_uncategorized(subside_taxonomy):
    (only,) = map_terms_to_taxonomy(["urban", "city"], subside_taxonomy)
    assert (only.domain, only.subdiscipline, only.score) == (UNCATEGORIZED, "General", 0.0)


def test_top_n_limits_legacy_assignments(subside_taxonomy):
    assert len(map_terms_to_taxonomy(["aquifer", "insar"], subside_taxonomy, top_n=2)) == 2


def test_projection_splits_weight_in_proportion_to_score():
    mapping = TaxonomyMapping(
        {
            "g": (TaxonomyAssignment("A", "a1", 3.0), TaxonomyAssignment("B", "b1", 1.0)),
            "unused": (TaxonomyAssignment("A", "a1", 1.0),),
        },
        method="test",
    )
    projection = project_weights({"g": 8, "unused": 0}, mapping)
    assert projection.domain_weights == {"A": 6.0, "B": 2.0}
    assert projection.total_weight() == 8.0
    assert projection.domain_totals()["domain"].tolist() == ["A", "B"]


def test_compare_distributions_percentages_sum_to_100(subside_lexicon, subside_taxonomy):
    mapping = map_lexicon_to_taxonomy(subside_lexicon, subside_taxonomy)
    weights = {g: 1.0 for g in subside_lexicon.names()}
    table = compare_distributions(
        mapping.base_distribution(), project_weights(weights, mapping, unmapped="keep")
    )
    assert table["base_pct"].sum() == pytest.approx(100)
    assert table["overlay_pct"].sum() == pytest.approx(100)
    assert (table["delta_pct"] == table["overlay_pct"] - table["base_pct"]).all()


def test_mapping_frame_is_readable(subside_taxonomy):
    lexicon = Lexicon.from_mapping({"Groundwater": ["aquifer"]})
    frame = map_lexicon_to_taxonomy(lexicon, subside_taxonomy).to_frame()
    assert frame.loc[0, "primary_domain"] == "Earth Sciences"
    assert frame.loc[0, "assignments"].startswith("Earth Sciences / Hydrology (1.00)")


def test_embedding_mapping_requires_backend(subside_lexicon, subside_taxonomy):
    with pytest.raises(ValueError, match="embedding backend"):
        map_lexicon_to_taxonomy(subside_lexicon, subside_taxonomy, method="embedding")
