"""Architectural rules for the tool-cluster hierarchy."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import semantic_sensemaking
from semantic_sensemaking.cli import schema_documents

SRC = Path(semantic_sensemaking.__file__).parent
TOOL_CLUSTERS = ["core", "corpus", "lexicon", "taxonomy", "backends", "viz", "report"]


def python_files(*parts: str):
    return sorted((SRC.joinpath(*parts)).rglob("*.py"))


@pytest.mark.parametrize("cluster", TOOL_CLUSTERS)
def test_tool_clusters_never_import_workflows(cluster):
    offenders = [p for p in python_files(cluster) if "semantic_sensemaking.workflows" in p.read_text()]
    assert not offenders


def test_core_depends_on_nothing_above_it():
    pattern = re.compile(r"semantic_sensemaking\.(corpus|lexicon|taxonomy|backends|viz|report|workflows)")
    assert not [p for p in python_files("core") if pattern.search(p.read_text())]


def test_library_never_blocks_on_input_or_opens_figures():
    pattern = re.compile(r"(?<![\w.])input\(|\.show\(")
    assert not [p for p in python_files() if pattern.search(p.read_text())]


@pytest.mark.parametrize("name", ["formulation"])
def test_planned_workflows_are_importable_and_explicit(name):
    from semantic_sensemaking.core.config import RunConfig
    from semantic_sensemaking.workflows import get_workflow

    module = get_workflow(name)
    assert module.IMPLEMENTED is False
    assert (SRC / "workflows" / name / "README.md").is_file()
    with pytest.raises(NotImplementedError, match="README"):
        module.run(RunConfig.load())


def test_committed_json_schemas_are_current():
    for name, schema in schema_documents().items():
        committed = SRC / "data" / "schemas" / f"{name}.schema.json"
        assert json.loads(committed.read_text()) == schema, (
            f"{committed.name} is stale; run: semantic-sensemaking schema src/semantic_sensemaking/data/schemas"
        )


def test_builtin_data_validates_against_schemas():
    from semantic_sensemaking.lexicon.io import LexiconFileModel
    from semantic_sensemaking.taxonomy.io import TaxonomyFileModel

    LexiconFileModel.model_validate_json((SRC / "data" / "lexicon_subside_v1.json").read_text())
    TaxonomyFileModel.model_validate_json((SRC / "data" / "taxonomy_subside_v1.json").read_text())
