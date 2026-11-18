"""Problem formulation: draft the lexicon and taxonomy the dictionary workflow reads.

A chat model proposes concept groups for a problem statement; the terms are
checked against a seed corpus; the surviving groups are mapped onto a science
backbone to flag gaps (optionally proposing new domains for them); and the
result is written as editable draft files. See ``README.md`` in this package.
"""

from semantic_sensemaking.workflows.formulation.grounding import (
    TermEvidence,
    ground_lexicon,
    grounded_lexicon,
)
from semantic_sensemaking.workflows.formulation.pipeline import FormulationRun, formulate, run
from semantic_sensemaking.workflows.formulation.propose import propose_domains, propose_lexicon
from semantic_sensemaking.workflows.formulation.result import FormulationResult

DESCRIPTION = "Draft a lexicon and taxonomy for a problem statement, grounded in a seed corpus."
IMPLEMENTED = True

__all__ = [
    "DESCRIPTION",
    "IMPLEMENTED",
    "FormulationResult",
    "FormulationRun",
    "TermEvidence",
    "formulate",
    "ground_lexicon",
    "grounded_lexicon",
    "propose_domains",
    "propose_lexicon",
    "run",
]
