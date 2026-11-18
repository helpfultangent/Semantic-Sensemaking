"""User-defined dictionary-driven semantic analysis.

Scores a corpus against a lexicon (keyword or embedding matching), maps the
lexicon groups onto a science backbone, and projects the corpus's attention onto
that backbone — the SUBSIDE Backbone Overlay prototype as a reusable workflow.
"""

from semantic_sensemaking.workflows.dictionary.pipeline import (
    DictionaryRun,
    analyze,
    load_inputs,
    run,
)
from semantic_sensemaking.workflows.dictionary.result import DictionaryResult

DESCRIPTION = "Score a corpus against a user-defined dictionary and project it onto a science backbone."
IMPLEMENTED = True

__all__ = ["DESCRIPTION", "IMPLEMENTED", "DictionaryResult", "DictionaryRun", "analyze", "load_inputs", "run"]
