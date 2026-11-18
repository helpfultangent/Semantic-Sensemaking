"""Each prototype scoring bug, shown under legacy settings and fixed under the opt-in setting."""

from __future__ import annotations

import pytest

from semantic_sensemaking.core.config import MatchMode, ScoringConfig
from semantic_sensemaking.core.types import Corpus, Lexicon
from semantic_sensemaking.taxonomy import UNCATEGORIZED, map_terms_to_taxonomy
from semantic_sensemaking.workflows.dictionary import analyze

from . import prototype_reference as proto

SUBSIDENCE = ["subsidence", "land subsidence", "ground subsidence", "settlement", "compaction"]


class TestBug1OverlappingTermsDoubleCount:
    TEXT = "Land subsidence and ground subsidence were measured."

    def test_prototype_counts_each_phrase_twice(self):
        assert proto.count_group_in_text(self.TEXT, SUBSIDENCE) == 4

    def test_legacy_mode_reproduces_it(self):
        corpus = Corpus.from_texts({"d": self.TEXT})
        lexicon = Lexicon.from_mapping({"Subsidence": SUBSIDENCE})
        from semantic_sensemaking.lexicon import score_corpus

        assert score_corpus(corpus, lexicon).total_matches() == 4

    def test_longest_match_counts_each_phrase_once(self):
        corpus = Corpus.from_texts({"d": self.TEXT})
        lexicon = Lexicon.from_mapping({"Subsidence": SUBSIDENCE})
        from semantic_sensemaking.lexicon import score_corpus

        scores = score_corpus(corpus, lexicon, ScoringConfig(match_mode=MatchMode.LONGEST_MATCH))
        assert scores.total_matches() == 2

    def test_fix_lowers_sample_corpus_counts_only_where_terms_overlap(
        self, sample_corpus, subside_lexicon, subside_taxonomy
    ):
        legacy = analyze(sample_corpus, subside_lexicon, subside_taxonomy).scores.group_weights()
        fixed = analyze(
            sample_corpus, subside_lexicon, subside_taxonomy,
            ScoringConfig(match_mode=MatchMode.LONGEST_MATCH),
        ).scores.group_weights()
        assert fixed["Subsidence"] < legacy["Subsidence"]
        assert all(fixed[g] <= legacy[g] for g in legacy if g != "Coastal")


class TestBug2SubdisciplineScoresIgnoreSubdiscipline:
    COASTAL = ["coastal", "coast", "shoreline", "tidal", "sea level", "sea-level rise"]

    def test_legacy_picks_the_first_three_subdisciplines_in_file_order(self, subside_taxonomy):
        assignments = map_terms_to_taxonomy(self.COASTAL, subside_taxonomy)
        subs = subside_taxonomy["Earth Sciences"].subdisciplines
        assert [a.subdiscipline for a in assignments] == list(subs[:3])
        assert len({a.score for a in assignments}) == 1

    def test_subdiscipline_aware_prefers_named_subdiscipline_and_preserves_domain_score(
        self, subside_taxonomy
    ):
        assignments = map_terms_to_taxonomy(
            self.COASTAL, subside_taxonomy, method="subdiscipline_aware"
        )
        assert assignments[0].domain == "Earth Sciences"
        assert assignments[0].subdiscipline == "Coastal Science"
        domain_score = sum(a.score for a in assignments if a.domain == "Earth Sciences")
        legacy_domain_score = map_terms_to_taxonomy(self.COASTAL, subside_taxonomy)[0].score
        assert domain_score == pytest.approx(legacy_domain_score)


class TestBug3UnmappedGroupsVanish:
    def test_legacy_drops_uncategorized_matches(self, sample_corpus, subside_lexicon, subside_taxonomy):
        result = analyze(sample_corpus, subside_lexicon, subside_taxonomy)
        unmapped = [g for g in result.mapping.labels() if result.mapping.primary_domain(g) == UNCATEGORIZED]
        assert "Urban" in unmapped and result.scores.group_weights()["Urban"] > 0
        assert result.projection.domain_weights[UNCATEGORIZED] == 0.0

    def test_keep_preserves_their_weight(self, sample_corpus, subside_lexicon, subside_taxonomy):
        result = analyze(
            sample_corpus, subside_lexicon, subside_taxonomy, ScoringConfig(unmapped_groups="keep")
        )
        weights = result.scores.group_weights()
        unmapped = [g for g in result.mapping.labels() if result.mapping.primary_domain(g) == UNCATEGORIZED]
        assert result.projection.domain_weights[UNCATEGORIZED] == pytest.approx(
            sum(weights[g] for g in unmapped)
        )
        assert result.projection.total_weight() == pytest.approx(sum(weights.values()))
