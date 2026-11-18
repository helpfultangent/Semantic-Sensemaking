from __future__ import annotations

import pytest

from semantic_sensemaking.backends import DeterministicHashEmbedding
from semantic_sensemaking.core.config import ScoringConfig
from semantic_sensemaking.core.types import Corpus, Lexicon
from semantic_sensemaking.errors import ConfigError
from semantic_sensemaking.lexicon import explain_semantic_matches, score_corpus_semantic
from semantic_sensemaking.workflows.dictionary import analyze

LEXICON = Lexicon.from_mapping({"Groundwater": ["aquifer", "drawdown"], "Coastal": ["estuary", "shoreline"]})
CORPUS = Corpus.from_texts(
    {
        "wells": "aquifer " * 30 + "tidal " * 30,
        "coast": "shoreline estuary " * 20,
    }
)


def test_semantic_scores_count_matching_passages():
    backend = DeterministicHashEmbedding(256)
    scores = score_corpus_semantic(CORPUS, LEXICON, backend, threshold=0.5, chunk_size=20, chunk_overlap=0)
    counts = {(r.doc, r.group): r.count for r in scores.rows}
    assert scores.method == "semantic"
    assert counts[("wells", "Groundwater")] == 2  # the two all-"aquifer" passages
    assert counts[("wells", "Coastal")] == 0
    assert counts[("coast", "Coastal")] == 2
    assert counts[("coast", "Groundwater")] == 0


def test_threshold_above_one_matches_nothing():
    scores = score_corpus_semantic(CORPUS, LEXICON, DeterministicHashEmbedding(), threshold=1.01)
    assert scores.total_matches() == 0


def test_explain_semantic_matches_ranks_passages():
    table = explain_semantic_matches(CORPUS["coast"], LEXICON, DeterministicHashEmbedding(), top_k=1, chunk_size=10, chunk_overlap=0)
    assert list(table.columns) == ["group", "rank", "similarity", "best_term", "passage"]
    coastal = table[table["group"] == "Coastal"].iloc[0]
    assert coastal["best_term"] in {"estuary", "shoreline"}
    assert coastal["similarity"] > table[table["group"] == "Groundwater"].iloc[0]["similarity"]


def test_semantic_method_needs_a_backend(subside_taxonomy):
    with pytest.raises(ConfigError, match="embedding backend"):
        analyze(CORPUS, LEXICON, subside_taxonomy, ScoringConfig(method="semantic"))


def test_semantic_analysis_end_to_end(subside_taxonomy):
    result = analyze(
        CORPUS, LEXICON, subside_taxonomy,
        ScoringConfig(method="semantic", mapping_method="embedding", semantic_threshold=0.5),
        embedding_backend=DeterministicHashEmbedding(),
    )
    assert result.mapping.method == "embedding"
    assert result.scores.total_matches() > 0
