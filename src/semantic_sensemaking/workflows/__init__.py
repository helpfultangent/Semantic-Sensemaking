"""End-to-end workflows built from the shared tool clusters.

Each workflow package exposes ``DESCRIPTION``, ``IMPLEMENTED`` and
``run(config) -> ...``. Workflows import tool clusters; tool clusters never
import workflows.
"""

from __future__ import annotations

import importlib
from types import ModuleType

WORKFLOWS: dict[str, str] = {
    "dictionary": "semantic_sensemaking.workflows.dictionary",
    "narrative": "semantic_sensemaking.workflows.narrative",
    "formulation": "semantic_sensemaking.workflows.formulation",
}


def get_workflow(name: str) -> ModuleType:
    try:
        return importlib.import_module(WORKFLOWS[name])
    except KeyError:
        raise KeyError(f"Unknown workflow {name!r}; choose from {sorted(WORKFLOWS)}") from None
