"""The pieces as registered for DYNAMO flows."""

import json

import pytest

from semantic_sensemaking.pieces import flow_catalog
from semantic_sensemaking.pieces.entrypoints import PIECES


@pytest.mark.parametrize("slug", sorted(flow_catalog.PIECES))
def test_each_piece_has_a_source_that_compiles_and_names_a_library_piece(slug):
    piece = flow_catalog.PIECES[slug][0]
    assert piece in PIECES
    source = flow_catalog.source(slug)
    compile(source, slug, "exec")
    assert flow_catalog.REF in source and flow_catalog.ARCHIVE_SHA256 in source
    assert "--require-hashes" in source and "--no-deps" in source
    assert len(source.encode()) < 64_000


@pytest.mark.parametrize("slug", sorted(flow_catalog.PIECES))
def test_each_registration_numbers_its_inputs_and_outputs_from_zero(slug):
    row = flow_catalog.registration(slug)
    for role in ("input", "output"):
        positions = [c["position"] for c in row["contracts"]["data"] if c["role"] == role]
        assert positions == list(range(len(positions))) and positions
    runtime = row["runtime_json"]
    assert runtime["env_from_args"]["OUTPUT_URI"] == "output_uri"
    assert runtime["parameters_schema_json"]["required"] == ["output_uri"]
    json.dumps(row)


def test_every_locked_dependency_is_pinned_with_a_hash():
    lines = [line for line in flow_catalog.LOCK.splitlines() if line.strip()]
    assert lines and all("==" in line and "--hash=sha256:" in line for line in lines)


def test_a_piece_refuses_to_run_with_an_input_left_unlinked():
    scope: dict = {}
    exec(compile(flow_catalog.source("map-and-project"), "piece", "exec"), scope)
    with pytest.raises(ValueError, match="LEXICON_URI"):
        scope["transform"]({"SOURCE_0": "term_counts.csv", "OUTPUT_URI": "out"})
