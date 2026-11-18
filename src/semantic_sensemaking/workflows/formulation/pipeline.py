"""Formulation workflow orchestration: propose, ground, check coverage, write drafts."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from semantic_sensemaking.backends.base import ChatBackend, EmbeddingBackend
from semantic_sensemaking.backends.factory import get_chat_backend, get_embedding_backend
from semantic_sensemaking.core.config import FormulationConfig, RunConfig, ScoringConfig
from semantic_sensemaking.core.context import RunContext
from semantic_sensemaking.core.types import Corpus, Lexicon, Taxonomy
from semantic_sensemaking.errors import ConfigError
from semantic_sensemaking.taxonomy.io import load_taxonomy
from semantic_sensemaking.taxonomy.mapping import (
    UNCATEGORIZED,
    TaxonomyMapping,
    map_lexicon_to_taxonomy,
)
from semantic_sensemaking.workflows._inputs import load_configured_corpus
from semantic_sensemaking.workflows.formulation.grounding import ground_lexicon, grounded_lexicon
from semantic_sensemaking.workflows.formulation.outputs import write_outputs
from semantic_sensemaking.workflows.formulation.propose import propose_domains, propose_lexicon
from semantic_sensemaking.workflows.formulation.result import FormulationResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FormulationRun:
    result: FormulationResult
    context: RunContext


def formulate(
    taxonomy: Taxonomy,
    chat_backend: ChatBackend,
    formulation: FormulationConfig,
    scoring: ScoringConfig | None = None,
    *,
    corpus: Corpus | None = None,
    embedding_backend: EmbeddingBackend | None = None,
    proposed: Lexicon | None = None,
) -> FormulationResult:
    """Pure analysis with no file I/O — the entry point for notebooks.

    Pass ``proposed`` to skip the model call and ground/check an existing draft
    lexicon instead (``chat_backend`` is then only used for ``propose_domains``).
    """
    scoring = scoring or ScoringConfig()
    statement = (formulation.problem_statement or "").strip()
    if proposed is None:
        if not statement:
            raise ConfigError("Set formulation.problem_statement to the question the team is scoping")
        if chat_backend is None:
            raise ConfigError(
                "The formulation workflow needs a chat backend to draft concept groups: set "
                "chat.kind (openai_compatible or anthropic)"
            )
        proposed = propose_lexicon(
            chat_backend,
            statement,
            taxonomy,
            max_groups=formulation.max_groups,
            max_terms_per_group=formulation.max_terms_per_group,
            name=formulation.lexicon_name,
        )
        log.info("Model proposed %d groups", len(proposed))

    evidence: tuple = ()
    dropped: tuple[str, ...] = ()
    lexicon = proposed
    if formulation.grounding != "none":
        if corpus is None:
            raise ConfigError(
                "Grounding needs a seed corpus: set inputs.corpus, or formulation.grounding: none"
            )
        if formulation.grounding == "semantic" and embedding_backend is None:
            raise ConfigError("formulation.grounding 'semantic' needs embedding.kind set")
        evidence = ground_lexicon(
            proposed,
            corpus,
            scoring=scoring,
            min_count=formulation.min_term_count,
            backend=embedding_backend if formulation.grounding == "semantic" else None,
        )
        lexicon, dropped = grounded_lexicon(
            proposed, evidence, min_terms=formulation.min_terms_per_group
        )
        log.info(
            "Grounding kept %d of %d terms; dropped %d group(s)",
            sum(e.grounded for e in evidence), len(evidence), len(dropped),
        )

    def coverage(tax: Taxonomy) -> TaxonomyMapping:
        return map_lexicon_to_taxonomy(
            lexicon,
            tax,
            top_n=scoring.top_n_assignments,
            method=scoring.mapping_method,
            backend=embedding_backend,
        )

    mapping = coverage(taxonomy)
    added: tuple = ()
    uncovered = [lexicon[g] for g in mapping.labels() if mapping.primary_domain(g) == UNCATEGORIZED]
    if formulation.propose_domains and uncovered:
        if chat_backend is None:
            raise ConfigError("formulation.propose_domains needs a chat backend")
        added = propose_domains(chat_backend, statement or "(not given)", taxonomy, uncovered)
        if added:
            taxonomy = Taxonomy(
                taxonomy.nodes + added,
                name=f"{taxonomy.name}_draft",
                layer=taxonomy.layer,
                source=f"{taxonomy.source or taxonomy.name} + {len(added)} proposed domain(s)",
            )
            mapping = coverage(taxonomy)
            log.info("Added %d proposed domain(s): %s", len(added), [n.domain for n in added])

    return FormulationResult(
        problem_statement=statement,
        proposed=proposed,
        lexicon=lexicon,
        taxonomy=taxonomy,
        mapping=mapping,
        evidence=evidence,
        dropped_groups=dropped,
        added_domains=added,
        corpus=corpus,
    )


def run(config: RunConfig) -> FormulationRun:
    """Load inputs, formulate, and write the drafts into a new run directory."""
    f = config.formulation
    taxonomy = load_taxonomy(config.inputs.taxonomy, layer=config.inputs.taxonomy_layer)
    corpus = load_configured_corpus(config) if f.grounding != "none" else None
    proposed = None
    if config.inputs.lexicon is not None:
        from semantic_sensemaking.lexicon.io import load_lexicon

        proposed = load_lexicon(config.inputs.lexicon)
        log.info("Checking the existing lexicon %s instead of drafting a new one", config.inputs.lexicon)

    needs_embedding = f.grounding == "semantic" or config.scoring.mapping_method == "embedding"
    embedding = get_embedding_backend(config.embedding) if needs_embedding else None
    chat = None
    try:
        chat = get_chat_backend(config.chat)
        result = formulate(
            taxonomy, chat, f, config.scoring,
            corpus=corpus, embedding_backend=embedding, proposed=proposed,
        )
    finally:
        for backend in (embedding, chat):
            if backend is not None:
                backend.close()
    context = RunContext.create(config.output.dir, "formulation", config.output.run_id)
    write_outputs(result, context, config)
    log.info("Wrote %d artifacts to %s", len(context.artifacts), context.run_dir)
    return FormulationRun(result, context)
