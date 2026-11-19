"""The pieces: storage, each entry function, chaining, and the registered source."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from semantic_sensemaking import sample_corpus_path
from semantic_sensemaking.errors import ConfigError
from semantic_sensemaking.pieces import (
    PIECES,
    SPECS,
    Storage,
    StorageError,
    registration,
    run_piece,
    stored_source,
)
from semantic_sensemaking.pieces.entrypoints import COUNT_COLUMNS, read_corpus

N_DOCS = 5


def run(name: str, **inputs) -> dict:
    return json.loads(run_piece(name, inputs))


@pytest.fixture
def out(tmp_path) -> str:
    return str(tmp_path / "run")


@pytest.fixture
def ingested(out) -> dict:
    return run("ingest_corpus", SOURCE_URI=str(sample_corpus_path()), OUTPUT_URI=out, MIN_CHARS="0")


# ── each piece ──────────────────────────────────────────────────────────────


def test_ingest_writes_one_normalised_corpus_file(ingested, out):
    assert ingested["piece"] == "ingest_corpus" and ingested["output_uri"] == out
    assert ingested["documents"] == N_DOCS
    corpus = read_corpus(Path(ingested["corpus_uri"]))
    assert len(corpus) == N_DOCS and all(doc.text for doc in corpus)


def test_score_lexicon_counts_the_built_in_lexicon(ingested):
    result = run("score_lexicon", SOURCE_RESULT=json.dumps(ingested))
    counts = pd.read_csv(result["term_counts_uri"], encoding="utf-8-sig")
    assert list(counts.columns) == COUNT_COLUMNS
    assert counts["doc_id"].nunique() == N_DOCS
    assert result["total_matches"] == int(counts["count"].sum()) > 0
    assert Path(result["lexicon_uri"]).is_file()
    assert result["corpus_uri"] == ingested["corpus_uri"]  # carried forward


def test_dictionary_flow_chains_through_results(ingested):
    scored = run("score_lexicon", SOURCE_RESULT=json.dumps(ingested))
    projected = run("map_and_project", SOURCE_RESULT=json.dumps(scored))
    report = run("render_report", SOURCE_RESULT=json.dumps(projected))

    weights = pd.read_csv(projected["domain_weights_uri"], encoding="utf-8-sig")
    assert list(weights.columns) == ["domain", "weight"] and weights["weight"].sum() > 0
    assignment = pd.read_csv(projected["domain_assignment_uri"], encoding="utf-8-sig")
    assert {"group", "primary_domain"} <= set(assignment.columns)
    assert projected["total_weight"] == pytest.approx(weights["weight"].sum())

    summary = Path(report["report_uri"]).read_text(encoding="utf-8")
    assert "## Science backbone" in summary and "## Groups" in summary
    assert f"**Documents**: {N_DOCS}" in summary
    # The last result still names everything produced along the way.
    for key in ("corpus_uri", "term_counts_uri", "lexicon_uri", "domain_weights_uri", "report_uri"):
        assert Path(report[key]).is_file()


def test_dictionary_flow_writes_every_table_the_library_workflow_does(ingested):
    scored = run("score_lexicon", SOURCE_RESULT=json.dumps(ingested))
    projected = run("map_and_project", SOURCE_RESULT=json.dumps(scored))

    documents = pd.read_csv(projected["corpus_documents_uri"], encoding="utf-8-sig")
    assert len(documents) == N_DOCS and {"doc_id", "n_words"} <= set(documents.columns)
    skipped = pd.read_csv(projected["skipped_files_uri"], encoding="utf-8-sig")
    assert list(skipped.columns) == ["path", "reason"] and len(skipped) == ingested["skipped_files"]
    totals = pd.read_csv(projected["group_totals_uri"], encoding="utf-8-sig")
    counts = pd.read_csv(projected["term_counts_uri"], encoding="utf-8-sig")
    assert int(totals["count"].sum()) == int(counts["count"].sum())
    comparison = pd.read_csv(projected["comparison_uri"], encoding="utf-8-sig")
    assert list(comparison.columns) == [
        "domain", "base_n_groups", "base_pct", "overlay_weight", "overlay_pct", "delta_pct"
    ]
    assert comparison["overlay_weight"].sum() == pytest.approx(projected["total_weight"])


def test_each_piece_records_its_settings_without_credentials(out):
    result = run(
        "ingest_corpus", SOURCE_URI=str(sample_corpus_path()), OUTPUT_URI=out, MIN_CHARS="0",
        TAPIS_TOKEN="not-a-real-token", CHAT_API_KEY="not-a-real-key",
    )
    text = Path(result["settings_ingest_corpus_uri"]).read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["piece"] == "ingest_corpus" and record["library_version"]
    assert record["settings"]["MIN_CHARS"] == "0" and record["settings"]["OUTPUT_URI"] == out
    assert "not-a-real" not in text and "SOURCE_RESULT" not in record["settings"]


def test_narrative_flow_feeds_the_same_projection_piece(ingested):
    frames = run(
        "discover_frames", SOURCE_RESULT=json.dumps(ingested),
        EMBEDDING_KIND="hashing", SEGMENT_WORDS="40", MAX_FRAMES="4",
    )
    table = pd.read_csv(frames["frames_uri"], encoding="utf-8-sig")
    assert 1 <= len(table) <= 4 and frames["frames"] == len(table)
    counts = pd.read_csv(frames["term_counts_uri"], encoding="utf-8-sig")
    assert list(counts.columns) == COUNT_COLUMNS
    assert int(counts["count"].sum()) == frames["passages"]
    passages = pd.read_csv(frames["passage_frames_uri"], encoding="utf-8-sig")
    assert len(passages) == frames["passages"] and set(passages["frame"]) <= set(table["frame"])

    projected = run("map_and_project", SOURCE_RESULT=json.dumps(frames))
    report = run("render_report", SOURCE_RESULT=json.dumps(projected))
    assert "## Frames" in Path(report["report_uri"]).read_text(encoding="utf-8")


def test_discover_frames_needs_an_embedding_backend(ingested):
    with pytest.raises(ConfigError, match="EMBEDDING_KIND"):
        run("discover_frames", SOURCE_RESULT=json.dumps(ingested))


def test_draft_lexicon_checks_an_existing_lexicon_without_a_chat_model(ingested, tmp_path):
    lexicon = tmp_path / "mine.json"
    lexicon.write_text(json.dumps({"Subsidence": ["subsidence", "zzz-not-there"], "Empty": ["qqqq"]}))
    result = run("draft_lexicon", SOURCE_RESULT=json.dumps(ingested), LEXICON_URI=str(lexicon))
    assert result["review_required"] is True
    draft = json.loads(Path(result["lexicon_uri"]).read_text())
    assert "zzz-not-there" not in json.dumps(draft) and "subsidence" in json.dumps(draft)
    assert "Empty" in result["dropped_groups"]
    assert Path(result["taxonomy_uri"]).is_file() and Path(result["group_coverage_uri"]).is_file()


def test_draft_lexicon_without_a_lexicon_needs_a_chat_model(ingested):
    with pytest.raises(ConfigError, match="CHAT_KIND"):
        run("draft_lexicon", SOURCE_RESULT=json.dumps(ingested), PROBLEM_STATEMENT="x")


# ── inputs and chaining rules ───────────────────────────────────────────────


def test_an_explicit_location_wins_over_the_carried_one(ingested, tmp_path):
    lexicon = tmp_path / "mine.json"
    lexicon.write_text(json.dumps({"Only": ["aquifer"]}))
    result = run("score_lexicon", SOURCE_RESULT=json.dumps({**ingested, "lexicon_uri": "/nowhere"}),
                 LEXICON_URI=str(lexicon))
    counts = pd.read_csv(result["term_counts_uri"], encoding="utf-8-sig")
    assert set(counts["group"]) == {"Only"}


def test_output_location_is_required_on_the_first_piece():
    with pytest.raises(ConfigError, match="OUTPUT_URI"):
        run("ingest_corpus", SOURCE_URI=str(sample_corpus_path()))


def test_a_missing_required_input_is_named(out):
    with pytest.raises(ConfigError, match="CORPUS_URI"):
        run("score_lexicon", OUTPUT_URI=out)


def test_a_non_json_source_result_is_ignored(out):
    result = run("ingest_corpus", SOURCE_URI=str(sample_corpus_path()), OUTPUT_URI=out,
                 SOURCE_RESULT="tapis://system/some/file", MIN_CHARS="0")
    assert result["documents"] == N_DOCS


def test_unknown_piece():
    with pytest.raises(ConfigError, match="Unknown piece"):
        run_piece("nope", {})


# ── storage ─────────────────────────────────────────────────────────────────


def tapis(files: dict[str, bytes], uploads: dict[str, bytes]) -> Storage:
    """A Storage talking to an in-memory stand-in for the Tapis Files API on system ``sys``."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Tapis-Token"] == "tok"
        path = request.url.path.removeprefix("/v3/files/")
        kind, _, rest = path.partition("/sys/")
        if kind == "ops" and request.method == "GET":
            rows = [{"path": p, "type": "file"} for p in sorted(files) if p == rest or p.startswith(rest + "/")]
            return httpx.Response(200, json={"result": rows})
        if kind == "content":
            return httpx.Response(200, content=files[rest]) if rest in files else httpx.Response(404)
        if kind == "ops" and request.method == "POST":
            uploads[rest] = request.read()
            return httpx.Response(200, json={"status": "success"})
        return httpx.Response(400)
    return Storage("tok", "https://tapis.test", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_tapis_fetch_of_one_file_and_of_a_folder(tmp_path):
    files = {"data/corpus/a.txt": b"alpha", "data/corpus/sub/b.txt": b"beta", "data/lexicon.json": b"{}"}
    storage = tapis(files, {})
    one = storage.fetch("tapis://sys/data/lexicon.json", tmp_path / "one")
    assert one.name == "lexicon.json" and one.read_bytes() == b"{}"
    folder = storage.fetch("tapis://sys/data/corpus", tmp_path / "many")
    assert folder.name == "corpus"
    assert (folder / "a.txt").read_bytes() == b"alpha" and (folder / "sub" / "b.txt").read_bytes() == b"beta"


def test_tapis_put_uploads_to_the_named_path(tmp_path):
    uploads: dict[str, bytes] = {}
    local = tmp_path / "summary.md"
    local.write_text("hello")
    assert tapis({}, uploads).put(local, "tapis://sys/out/run-1/summary.md") == "tapis://sys/out/run-1/summary.md"
    assert b"hello" in uploads["out/run-1/summary.md"]


def test_tapis_errors_are_reported_without_the_token(tmp_path):
    with pytest.raises(StorageError, match="No files found"):
        tapis({}, {}).fetch("tapis://sys/missing", tmp_path)
    with pytest.raises(StorageError, match="token is required"):
        Storage(None).fetch("tapis://sys/x", tmp_path)
    with pytest.raises(StorageError, match="answered 404") as excinfo:
        tapis({}, {})._request("GET", "content/sys/gone")
    assert "tok" not in str(excinfo.value)


def test_a_piece_runs_against_tapis_locations(tmp_path):
    files = {f"in/corpus/{p.name}": p.read_bytes() for p in sample_corpus_path().iterdir() if p.is_file()}
    uploads: dict[str, bytes] = {}
    result = json.loads(run_piece(
        "ingest_corpus",
        {"SOURCE_URI": "tapis://sys/in/corpus", "OUTPUT_URI": "tapis://sys/out/run-1", "MIN_CHARS": "0"},
        storage=tapis(files, uploads),
    ))
    assert result["corpus_uri"] == "tapis://sys/out/run-1/corpus.json"
    assert result["documents"] == N_DOCS and "out/run-1/corpus.json" in uploads


# ── what gets registered ────────────────────────────────────────────────────


def test_every_piece_has_a_contract_and_an_entry_function():
    assert set(SPECS) == set(PIECES) == {
        "ingest_corpus", "score_lexicon", "map_and_project", "discover_frames", "draft_lexicon", "render_report"}
    for spec in SPECS.values():
        assert spec.inputs and spec.outputs and "output_uri" in spec.arguments


@pytest.mark.parametrize("name", sorted(SPECS))
def test_stored_source_is_a_small_script_that_defines_transform(name):
    source = stored_source(name, "0123abc")
    assert len(source) < 1000
    code = compile(source, "piece.py", "exec")  # syntax only: executing would pip-install
    assert "transform" in code.co_names
    assert "archive/0123abc.tar.gz" in source and f'run_piece("{name}", inputs)' in source


def test_stored_source_rejects_a_ref_that_could_alter_the_install_line():
    with pytest.raises(ValueError):
        stored_source("ingest_corpus", "main; rm -rf /")


def test_registration_matches_the_dynamo_tool_and_binds_no_other_secret():
    args = registration("score_lexicon", "0123abc")
    assert set(args) == {"label", "description", "inputs", "outputs", "python_source", "entrypoint",
                         "env_from_args", "parameters", "version"}
    assert args["inputs"][0] == {
        "standard_variable_uri": "https://w3id.org/okn/i/mint/document_collection__text", "format": "json"}
    assert args["env_from_args"] == {"TAPIS_TOKEN": "tapis_token", "OUTPUT_URI": "output_uri"}
    assert args["parameters"]["required"] == ["output_uri"]
    assert set(args["parameters"]["properties"]) == {"output_uri"}
    assert not {"signal_token", "token_fetch_token", "ckan_token", "credentials_url"} & set(
        args["env_from_args"].values())


def test_only_required_arguments_are_bound_unless_a_setting_is_asked_for():
    assert registration("ingest_corpus", "0123abc")["env_from_args"] == {
        "TAPIS_TOKEN": "tapis_token", "SOURCE_URI": "source_uri", "OUTPUT_URI": "output_uri"}
    args = registration("score_lexicon", "0123abc", optional=("lexicon_uri", "match_mode"))
    assert args["env_from_args"]["LEXICON_URI"] == "lexicon_uri"
    assert set(args["parameters"]["properties"]) == {"output_uri", "lexicon_uri", "match_mode"}
    assert args["parameters"]["required"] == ["output_uri"]
    with pytest.raises(ValueError, match="no argument named"):
        registration("score_lexicon", "0123abc", optional=("nope",))


def test_contracts_chain_each_flow_by_first_output_and_single_input():
    def chain(*names):
        for earlier, later in zip(names, names[1:]):
            assert SPECS[earlier].outputs[0] == SPECS[later].inputs[0], (earlier, later)
    for spec in SPECS.values():
        assert len(spec.inputs) == 1, spec.name
    chain("ingest_corpus", "score_lexicon", "map_and_project", "render_report")
    chain("ingest_corpus", "discover_frames", "map_and_project", "render_report")
    chain("ingest_corpus", "draft_lexicon")


def test_engine_supplied_arguments_are_bound_but_not_declared():
    args = registration("ingest_corpus", "0123abc")
    assert args["env_from_args"]["SOURCE_URI"] == "source_uri"
    assert set(args["parameters"]["properties"]) == {"output_uri"}
    assert args["parameters"]["required"] == ["output_uri"]


def test_stored_source_can_install_from_a_fork():
    source = stored_source("score_lexicon", "0123abc", repository="someone/Semantic-Sensemaking")
    assert "github.com/someone/Semantic-Sensemaking/archive/0123abc.tar.gz" in source
    assert "github.com/helpfultangent/Semantic-Sensemaking/archive/0123abc.tar.gz" in stored_source(
        "score_lexicon", "0123abc")
    with pytest.raises(ValueError, match="owner/name"):
        stored_source("score_lexicon", "0123abc", repository="evil.example/x/y")
