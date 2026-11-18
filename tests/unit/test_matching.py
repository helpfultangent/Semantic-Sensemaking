from __future__ import annotations

from semantic_sensemaking.lexicon import MatchMode, TermMatcher, count_terms


def test_auto_uses_word_boundaries_for_single_words():
    assert count_terms("gas gases natural gas", ["gas"]) == 2


def test_auto_substring_matches_inside_words_for_multiword_and_hyphenated_terms():
    # Documented prototype behavior: no boundary guard on terms with spaces or hyphens.
    assert count_terms("the sea-level-rise report", ["sea-level"]) == 1
    assert count_terms("co2 storages", ["co2 storage"]) == 1


def test_auto_counts_duplicate_terms_twice():
    assert count_terms("aquifer", ["aquifer", "aquifer"]) == 2


def test_auto_is_case_insensitive_by_default_and_can_be_case_sensitive():
    assert count_terms("InSAR insar", ["insar"]) == 2
    assert count_terms("InSAR insar", ["insar"], case_sensitive=True) == 1


def test_word_boundary_matches_multiword_terms_across_line_breaks_but_not_inside_words():
    text = "land\nsubsidence, landsubsidence, co2 storages"
    assert count_terms(text, ["land subsidence"], mode=MatchMode.WORD_BOUNDARY) == 1
    assert count_terms(text, ["co2 storage"], mode=MatchMode.WORD_BOUNDARY) == 0


def test_word_boundary_still_counts_overlapping_terms_independently():
    text = "land subsidence"
    assert count_terms(text, ["subsidence", "land subsidence"], mode="word_boundary") == 2


def test_longest_match_consumes_text_once():
    matcher = TermMatcher(
        ["mine", "coal mine", "underground mine"], mode=MatchMode.LONGEST_MATCH
    )
    text = "The coal mine and the underground mine; a mine nearby."
    assert matcher.count_detailed(text) == {"mine": 1, "coal mine": 1, "underground mine": 1}


def test_longest_match_reports_terms_as_written():
    matcher = TermMatcher(["Sea Level"], mode=MatchMode.LONGEST_MATCH)
    assert matcher.count_detailed("sea   level and SEA LEVEL") == {"Sea Level": 2}
