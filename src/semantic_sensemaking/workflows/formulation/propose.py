"""Ask a chat model to draft concept groups and, optionally, missing backbone domains."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from semantic_sensemaking.backends.base import ChatBackend
from semantic_sensemaking.core.types import Lexicon, LexiconEntry, Taxonomy, TaxonomyNode
from semantic_sensemaking.errors import BackendError

SYSTEM_PROMPT = (
    "You are a research librarian helping an interdisciplinary team scope a problem. "
    "You produce controlled vocabularies: concept groups with the words and phrases "
    "that signal each concept in documents."
)


class ProposedGroup(BaseModel):
    name: str = Field(description="Short concept name, 1-4 words")
    description: str = Field(description="One sentence: what this concept covers for the problem")
    terms: list[str] = Field(
        min_length=1,
        description="Lowercase words or short phrases (synonyms, variants, jargon) that signal it",
    )


class ProposedLexicon(BaseModel):
    groups: list[ProposedGroup] = Field(min_length=1)


class ProposedDomain(BaseModel):
    domain: str = Field(description="Name of a research domain (discipline-level)")
    subdisciplines: list[str] = Field(description="2-6 subdisciplines within the domain")
    terms: list[str] = Field(min_length=1, description="Lowercase indicator terms for the domain")


class ProposedDomains(BaseModel):
    domains: list[ProposedDomain]


def propose_lexicon(
    chat: ChatBackend,
    problem_statement: str,
    taxonomy: Taxonomy,
    *,
    max_groups: int,
    max_terms_per_group: int,
    name: str = "formulated_lexicon",
) -> Lexicon:
    """Draft concept groups for ``problem_statement``, capped at the configured sizes."""
    prompt = (
        f"Problem statement:\n{problem_statement.strip()}\n\n"
        "The team will analyze documents (papers, transcripts, reports) about this problem "
        "and place what they find on a science backbone with these domains: "
        f"{', '.join(taxonomy.domains())}.\n\n"
        f"Propose up to {max_groups} concept groups that together cover the problem: the "
        "physical processes, data and methods, actors and institutions, and impacts involved. "
        f"Give each group up to {max_terms_per_group} terms as they would literally appear in "
        "documents (include common abbreviations and plural or spelling variants). Groups must "
        "not overlap."
    )
    proposal = chat.complete_json(prompt, schema=ProposedLexicon, system=SYSTEM_PROMPT, max_tokens=4096)
    entries: list[LexiconEntry] = []
    used: set[str] = set()
    for group in proposal.groups[:max_groups]:
        terms = _clean_terms(group.terms)[:max_terms_per_group]
        group_name = " ".join(group.name.split())
        if not terms or not group_name or group_name.lower() in used:
            continue
        used.add(group_name.lower())
        entries.append(LexiconEntry(group_name, terms, group.description.strip() or None))
    if not entries:
        raise BackendError("The chat model proposed no usable concept groups")
    return Lexicon(tuple(entries), name=name, source="formulation workflow (model draft)")


def propose_domains(
    chat: ChatBackend,
    problem_statement: str,
    taxonomy: Taxonomy,
    uncovered: Sequence[LexiconEntry],
) -> tuple[TaxonomyNode, ...]:
    """Suggest backbone domains for concept groups the taxonomy does not cover."""
    listing = "\n".join(f"- {e.name}: {', '.join(e.terms)}" for e in uncovered)
    prompt = (
        f"Problem statement:\n{problem_statement.strip()}\n\n"
        f"A science backbone has these domains: {', '.join(taxonomy.domains())}.\n"
        f"These concept groups match none of them:\n{listing}\n\n"
        "Propose the fewest new domains (usually 1-3) that would cover these groups. Do not "
        "repeat an existing domain. Indicator terms should be words that appear in the groups' "
        "terms."
    )
    reply = chat.complete_json(prompt, schema=ProposedDomains, system=SYSTEM_PROMPT, max_tokens=2048)
    existing = {d.lower() for d in taxonomy.domains()}
    nodes = []
    for d in reply.domains:
        name = " ".join(d.domain.split())
        if not name or name.lower() in existing:
            continue
        existing.add(name.lower())
        nodes.append(
            TaxonomyNode(
                name,
                tuple(" ".join(s.split()) for s in d.subdisciplines if s.strip()),
                _clean_terms(d.terms),
            )
        )
    return tuple(nodes)


def _clean_terms(terms: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(" ".join(t.split()).lower() for t in terms if t and t.strip()))
