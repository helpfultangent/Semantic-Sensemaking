"""Formulation workflow: proposal parsing, grounding, coverage, domain proposals."""

from __future__ import annotations

import math

import pytest

from semantic_sensemaking.backends.hashing import DeterministicHashEmbedding
from semantic_sensemaking.core.config import FormulationConfig, ScoringConfig
from semantic_sensemaking.core.types import Corpus, Lexicon
from semantic_sensemaking.errors import BackendError, ConfigError
from semantic_sensemaking.taxonomy.mapping import UNCATEGORIZED
from semantic_sensemaking.workflows.formulation import (
    formulate,
    ground_lexicon,
    grounded_lexicon,
    propose_lexicon,
)
from semantic_sensemaking.workflows.formulation.propose import (
    ProposedDomain,
    ProposedDomains,
    ProposedGroup,
    ProposedLexicon,
)

STATEMENT = "How does groundwater policy interact with coastal subsidence research?"

CORPUS = Corpus.from_texts(
    {
        "a": "Groundwater pumping caused land subsidence. The aquifer drawdown continues.",
        "b": "The district adopted permit rules for groundwater withdrawal.",
    }
)

GROUPS = ProposedLexicon(
    groups=[
        ProposedGroup(name="Groundwater", description="water below ground", terms=["Groundwater", "aquifer", " Aquifer ", "drawdown"]),
        ProposedGroup(name="Subsidence", description="sinking land", terms=["land subsidence", "compaction"]),
        ProposedGroup(name="Governance", description="rules", terms=["permit", "regulation"]),
        ProposedGroup(name="Folklore", description="unrelated", terms=["dragon", "unicorn"]),
        ProposedGroup(name="groundwater", description="duplicate name", terms=["wells"]),
    ]
)


class FakeChat:
    name = "fake"
    model = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete_json(self, prompt, *, schema, system=None, max_tokens=4096):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        assert isinstance(reply, schema)
        return reply

    def close(self):
        pass


def test_propose_cleans_terms_dedupes_groups_and_caps(subside_taxonomy):
    chat = FakeChat(GROUPS)
    lexicon = propose_lexicon(chat, STATEMENT, subside_taxonomy, max_groups=10, max_terms_per_group=2)
    assert lexicon.names() == ("Groundwater", "Subsidence", "Governance", "Folklore")
    assert lexicon["Groundwater"].terms == ("groundwater", "aquifer")
    assert STATEMENT in chat.prompts[0] and "Earth Sciences" in chat.prompts[0]


def test_propose_rejects_empty_proposal(subside_taxonomy):
    empty = ProposedLexicon(groups=[ProposedGroup(name=" ", description="", terms=["x"])])
    with pytest.raises(BackendError):
        propose_lexicon(FakeChat(empty), STATEMENT, subside_taxonomy, max_groups=5, max_terms_per_group=5)


def test_keyword_grounding_keeps_terms_found_in_corpus():
    lexicon = Lexicon.from_mapping({"G": ["aquifer", "dragon"], "F": ["unicorn"]})
    evidence = ground_lexicon(lexicon, CORPUS, scoring=ScoringConfig())
    by_term = {e.term: e for e in evidence}
    assert by_term["aquifer"].grounded and by_term["aquifer"].count == 1
    assert not by_term["dragon"].grounded and math.isnan(by_term["dragon"].similarity)
    kept, dropped = grounded_lexicon(lexicon, evidence)
    assert kept.to_mapping() == {"G": ["aquifer"]}
    assert dropped == ("F",)


def test_semantic_grounding_records_similarity():
    lexicon = Lexicon.from_mapping({"G": ["permit rules"]})
    scoring = ScoringConfig(semantic_threshold=0.2, chunk_words=10, chunk_overlap_words=0)
    evidence = ground_lexicon(lexicon, CORPUS, scoring=scoring, backend=DeterministicHashEmbedding(256))
    assert evidence[0].similarity > 0.2 and evidence[0].grounded


def test_formulate_end_to_end_with_domain_proposal(subside_taxonomy):
    domains = ProposedDomains(
        domains=[
            ProposedDomain(domain="Earth Sciences", subdisciplines=["dup"], terms=["x"]),
            ProposedDomain(domain="Public Policy", subdisciplines=["Water Law"], terms=["permit"]),
        ]
    )
    chat = FakeChat(GROUPS, domains)
    result = formulate(
        subside_taxonomy,
        chat,
        FormulationConfig(problem_statement=STATEMENT, propose_domains=True),
        corpus=CORPUS,
    )
    assert "Folklore" in result.dropped_groups
    assert "Folklore" not in result.lexicon.names()
    assert [n.domain for n in result.added_domains] == ["Public Policy"]
    assert "Public Policy" in result.taxonomy.domains()

    coverage = result.coverage_table().set_index("group")
    assert not coverage.at["Folklore", "kept"] and coverage.at["Folklore", "needs_review"]
    assert coverage.at["Groundwater", "primary_domain"] == "Earth Sciences"
    assert set(result.tables()) == {"proposed_groups", "group_coverage", "term_grounding"}


def test_formulate_without_grounding_and_from_existing_lexicon(subside_taxonomy, subside_lexicon):
    result = formulate(
        subside_taxonomy, None, FormulationConfig(grounding="none"), proposed=subside_lexicon
    )
    assert result.lexicon is subside_lexicon and not result.evidence
    assert all(result.mapping.primary_domain(g) for g in result.lexicon.names())
    assert all(g in result.lexicon.names() for g in result.uncovered_groups())
    assert UNCATEGORIZED not in result.taxonomy.domains()


def test_formulate_errors_are_explicit(subside_taxonomy):
    with pytest.raises(ConfigError, match="problem_statement"):
        formulate(subside_taxonomy, FakeChat(), FormulationConfig())
    with pytest.raises(ConfigError, match="chat backend"):
        formulate(subside_taxonomy, None, FormulationConfig(problem_statement=STATEMENT))
    with pytest.raises(ConfigError, match="seed corpus"):
        formulate(subside_taxonomy, FakeChat(GROUPS), FormulationConfig(problem_statement=STATEMENT))
