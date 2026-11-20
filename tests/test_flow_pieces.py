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


@pytest.mark.parametrize("slug", sorted(flow_catalog.SECRETS))
def test_a_piece_that_calls_a_model_server_declares_the_credential_not_settings(slug):
    runtime = flow_catalog.registration(slug)["runtime_json"]
    settings = set(runtime["parameters_schema_json"]["properties"])
    filled = set(runtime["secrets"][0]["env"])
    # Nothing a credential fills can also be a setting or a run argument.
    assert not {name.upper() for name in settings} & filled
    assert not set(runtime["env_from_args"]) & filled
    assert {"model_api_key", "chat_base_url", "embedding_base_url"}.isdisjoint(settings)
    assert runtime["sends_data_to"] and flow_catalog.registration(slug)["id"].endswith("-r2")


def test_a_piece_that_calls_no_model_server_declares_no_credential():
    runtime = flow_catalog.registration("score-lexicon")["runtime_json"]
    assert "secrets" not in runtime and "sends_data_to" not in runtime
    assert not flow_catalog.registration("score-lexicon")["id"].endswith("-r2")


@pytest.mark.parametrize("slug", sorted(flow_catalog.PIECES))
def test_each_committed_source_file_matches_the_template(slug):
    """DYNAMO registers a piece from its committed file on GitHub, so the file must be
    exactly what source(slug) produces. If this fails, run
    ``python -m semantic_sensemaking.pieces.flow_catalog --write-sources``."""
    with open(flow_catalog.source_path(slug), encoding="utf-8") as handle:
        committed = handle.read()
    assert committed == flow_catalog.source(slug)
    assert "def transform(inputs):" in committed
    assert len(committed.encode()) < 64_000
