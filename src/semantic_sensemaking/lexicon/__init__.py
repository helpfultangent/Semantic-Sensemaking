"""User-defined dictionaries: load, match terms, and score corpora."""

from semantic_sensemaking.lexicon.io import lexicon_json_schema, load_lexicon, save_lexicon
from semantic_sensemaking.lexicon.matching import MatchMode, TermMatcher, count_terms
from semantic_sensemaking.lexicon.scoring import DocGroupScore, LexiconScores, score_corpus
from semantic_sensemaking.lexicon.semantic import explain_semantic_matches, score_corpus_semantic

__all__ = [
    "DocGroupScore",
    "LexiconScores",
    "MatchMode",
    "TermMatcher",
    "count_terms",
    "explain_semantic_matches",
    "lexicon_json_schema",
    "load_lexicon",
    "save_lexicon",
    "score_corpus",
    "score_corpus_semantic",
]
