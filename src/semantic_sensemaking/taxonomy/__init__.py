"""Science backbones: load, map concepts onto them, and project weights."""

from semantic_sensemaking.taxonomy.io import load_taxonomy, save_taxonomy, taxonomy_json_schema
from semantic_sensemaking.taxonomy.mapping import (
    UNCATEGORIZED,
    TaxonomyAssignment,
    TaxonomyMapping,
    map_embeddings_to_taxonomy,
    map_lexicon_to_taxonomy,
    map_terms_to_taxonomy,
)
from semantic_sensemaking.taxonomy.projection import (
    Projection,
    compare_distributions,
    project_weights,
)

__all__ = [
    "UNCATEGORIZED",
    "Projection",
    "TaxonomyAssignment",
    "TaxonomyMapping",
    "compare_distributions",
    "load_taxonomy",
    "map_embeddings_to_taxonomy",
    "map_lexicon_to_taxonomy",
    "map_terms_to_taxonomy",
    "project_weights",
    "save_taxonomy",
    "taxonomy_json_schema",
]
