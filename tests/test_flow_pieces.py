"""The pieces as registered for DYNAMO flows."""

import json
from typing import get_args

import pytest

from semantic_sensemaking.core.config import (
    BackendConfig,
    CorpusConfig,
    FormulationConfig,
    MatchMode,
    NarrativeConfig,
    ScoringConfig,
)
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
    assert runtime["sends_data_to"] and flow_catalog.registration(slug)["id"].endswith(f"-r{flow_catalog.REVISION[slug]}")


def test_a_piece_that_calls_no_model_server_declares_no_credential():
    runtime = flow_catalog.registration("score-lexicon")["runtime_json"]
    assert "secrets" not in runtime and "sends_data_to" not in runtime
    assert flow_catalog.registration("score-lexicon")["id"].endswith("-r2")


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


# ── settings: titles, types, choices, defaults and bounds ───────────────────

SLUGS = sorted(flow_catalog.PIECES)


def _properties(slug):
    return flow_catalog.registration(slug)["runtime_json"]["parameters_schema_json"]["properties"]


def _all_settings():
    return [(slug, name, spec) for slug in SLUGS for name, spec in _properties(slug).items()]


def _field(model, name):
    return model.model_fields[name]


def _literal(model, name):
    return list(get_args(_field(model, name).annotation))


def _bound(model, name, kind):
    return next((getattr(m, kind) for m in _field(model, name).metadata if hasattr(m, kind)), None)


# setting -> (the library's allowed values, its default), read from the config models
CHOICES = {
    "match_mode": ([m.value for m in MatchMode], ScoringConfig.model_fields["match_mode"].default.value),
    # the credential-free piece offers the methods that compare words; 'embedding' is its own piece
    "mapping_method": (
        [m for m in _literal(ScoringConfig, "mapping_method") if m != "embedding"],
        _field(ScoringConfig, "mapping_method").default,
    ),
    "unmapped_groups": (_literal(ScoringConfig, "unmapped_groups"), _field(ScoringConfig, "unmapped_groups").default),
    "grounding": (_literal(FormulationConfig, "grounding"), _field(FormulationConfig, "grounding").default),
}
# setting -> (config model, field); the integer settings the library bounds
INTEGERS = {
    "max_files": (CorpusConfig, "max_files"), "min_chars": (CorpusConfig, "min_chars"),
    "max_frames": (NarrativeConfig, "max_frames"), "segment_words": (NarrativeConfig, "segment_words"),
}
# Backend kinds a piece can use: the library's kinds that give embeddings or chat, none being "not set".
EMBEDDING_KINDS = {"openai_compatible", "huggingface_local", "hashing"}
CHAT_KINDS = {"openai_compatible", "anthropic"}


@pytest.mark.parametrize(("slug", "name", "spec"), _all_settings())
def test_every_setting_has_a_title_a_description_and_a_type(slug, name, spec):
    assert spec["title"].strip() and spec["description"].strip()
    assert spec["title"][0].isupper() and name not in spec["title"]
    assert spec["type"] in {"string", "integer", "number", "boolean"}


@pytest.mark.parametrize(("slug", "name", "spec"), _all_settings())
def test_every_choice_is_explained_and_the_default_is_one_of_them(slug, name, spec):
    choices = spec.get("oneOf", [])
    for choice in choices:
        assert choice["title"].strip() and choice["description"].strip()
        assert "const" in choice
    if choices and "default" in spec:
        assert spec["default"] in [c["const"] for c in choices]


@pytest.mark.parametrize("name", sorted(CHOICES))
def test_choices_and_defaults_are_the_librarys(name):
    allowed, default = CHOICES[name]
    spec = flow_catalog.SETTINGS[name]
    assert [c["const"] for c in spec["oneOf"]] == allowed
    assert spec["default"] == default


@pytest.mark.parametrize("name", sorted(INTEGERS))
def test_counts_are_integers_with_the_librarys_default_and_bounds(name):
    model, field = INTEGERS[name]
    spec = flow_catalog.SETTINGS[name]
    assert spec["type"] == "integer" and isinstance(spec["default"], int)
    assert spec["default"] == _field(model, field).default
    assert spec.get("minimum") == _bound(model, field, "ge")
    assert "maximum" not in spec


def test_backend_kinds_are_kinds_the_library_has_and_the_factory_accepts():
    kinds = set(get_args(BackendConfig.model_fields["kind"].annotation))
    for name, expected in (("embedding_kind", EMBEDDING_KINDS), ("chat_kind", CHAT_KINDS)):
        offered = {c["const"] for c in flow_catalog.SETTINGS[name]["oneOf"]}
        assert offered == expected and offered <= kinds
        # Unset means kind "none", which is the library's default, so no default is stated.
        assert "default" not in flow_catalog.SETTINGS[name]
    assert BackendConfig.model_fields["kind"].default == "none"


def test_model_names_have_no_default_because_the_library_has_none():
    assert BackendConfig.model_fields["model"].default is None
    for name in ("embedding_model", "chat_model"):
        assert "default" not in flow_catalog.SETTINGS[name]


def test_advanced_settings_are_the_rarely_changed_ones():
    advanced = {n for n, spec in flow_catalog.SETTINGS.items() if spec.get("x-advanced")}
    assert advanced == {
        "output_uri", "unmapped_groups", "max_files", "min_chars", "segment_words",
        "embedding_kind", "embedding_model", "chat_kind", "chat_model",
    }


def test_a_setting_two_pieces_share_is_identical_in_both():
    seen: dict[str, dict] = {}
    for _, name, spec in _all_settings():
        assert seen.setdefault(name, spec) == spec
    users = {name: [slug for slug in SLUGS if name in _properties(slug)] for name in seen}
    assert len(users["match_mode"]) == 2 and len(users["chat_kind"]) == 2 and len(users["output_uri"]) == len(SLUGS)


@pytest.mark.parametrize("slug", SLUGS)
def test_each_piece_takes_only_settings_that_are_defined(slug):
    assert set(flow_catalog.PIECES[slug][5]) <= set(flow_catalog.SETTINGS)
    assert set(_properties(slug)) == set(flow_catalog.PIECES[slug][5])
    runtime = flow_catalog.registration(slug)["runtime_json"]
    assert {n for n in runtime["env_from_args"] if n != "TAPIS_TOKEN"} == {n.upper() for n in _properties(slug)}


def test_every_piece_has_a_revision_and_only_the_new_piece_is_on_its_first():
    assert set(flow_catalog.REVISION) == set(flow_catalog.PIECES)
    assert [s for s, r in flow_catalog.REVISION.items() if r == 1] == ["map-and-project-embedding"]


@pytest.mark.parametrize("name", ["embedding_kind", "chat_kind"])
def test_the_model_server_choice_is_titled_tejas_and_keeps_the_value_the_code_reads(name):
    choice = next(c for c in flow_catalog.SETTINGS[name]["oneOf"] if c["const"] == "openai_compatible")
    assert choice["title"] == "Tejas"
    assert "model_server" in choice["description"] and "OpenAI-compatible" in choice["description"]


# ── the pieces that call an embedding model ─────────────────────────────────

KEYWORD_ONLY = ["ingest-corpus", "score-lexicon", "map-and-project", "render-report", "render-frames-report"]


def _secret_env(slug):
    return flow_catalog.registration(slug)["runtime_json"]["secrets"][0]["env"]


@pytest.mark.parametrize("slug", KEYWORD_ONLY)
def test_a_piece_that_only_counts_or_maps_by_words_declares_no_credential(slug):
    runtime = flow_catalog.registration(slug)["runtime_json"]
    assert "secrets" not in runtime and "sends_data_to" not in runtime


def test_map_and_project_no_longer_offers_embedding_and_cannot_be_asked_for_it():
    props = _properties("map-and-project")
    assert [c["const"] for c in props["mapping_method"]["oneOf"]] == ["legacy", "subdiscipline_aware"]
    assert not {"embedding_kind", "embedding_model"} & set(props)
    assert "FIXED" not in flow_catalog.source("map-and-project") and "_FIXED" not in flow_catalog.source("map-and-project")


def test_the_embedding_piece_has_the_same_contract_as_map_and_project_and_fixes_the_method():
    plain = flow_catalog.registration("map-and-project")
    embed = flow_catalog.registration("map-and-project-embedding")
    assert embed["contracts"] == plain["contracts"]
    props = _properties("map-and-project-embedding")
    assert set(props) == {"output_uri", "taxonomy_uri", "unmapped_groups", "embedding_kind", "embedding_model"}
    assert "mapping_method" not in props
    assert flow_catalog.FIXED["map-and-project-embedding"] == {"MAPPING_METHOD": "embedding"}
    assert "given.update(_FIXED)" in flow_catalog.source("map-and-project-embedding")
    assert embed["id"].endswith("-r1")


def test_the_embedding_piece_declares_the_model_server_credential_and_what_it_sends():
    row = flow_catalog.registration("map-and-project-embedding")["runtime_json"]
    assert row["secrets"][0]["name"] == "model_server"
    assert row["secrets"][0]["env"] == {"MODEL_API_KEY": "api_key", "EMBEDDING_BASE_URL": "embedding_base_url"}
    sent = row["sends_data_to"][0]["what"]
    assert "group names and terms" in sent and "domain and subdiscipline names" in sent


def test_the_fixed_method_reaches_the_library_function(monkeypatch):
    seen = {}
    scope: dict = {}
    exec(compile(flow_catalog.source("map-and-project-embedding"), "piece", "exec"), scope)
    scope["_load_library"] = lambda: None

    def fake(piece, given):
        seen.update(piece=piece, given=given)
        return json.dumps({"domain_weights_uri": "a", "domain_assignment_uri": "b", "taxonomy_uri": "c"})

    import semantic_sensemaking.pieces as pieces
    monkeypatch.setattr(pieces, "run_piece", fake)
    scope["transform"]({"SOURCE_0": "t.csv", "SOURCE_1": "l.json", "OUTPUT_URI": "out", "MAPPING_METHOD": "legacy"})
    assert seen["piece"] == "map_and_project" and seen["given"]["MAPPING_METHOD"] == "embedding"


def test_draft_lexicon_can_build_an_embedding_backend_for_semantic_grounding():
    props = _properties("draft-lexicon")
    assert {"embedding_kind", "embedding_model"} <= set(props)
    assert _secret_env("draft-lexicon")["EMBEDDING_BASE_URL"] == "embedding_base_url"
    assert "cannot" not in json.dumps(props["grounding"]) and "stops with an error" not in json.dumps(props["grounding"])
    semantic = next(c for c in props["grounding"]["oneOf"] if c["const"] == "semantic")
    assert "model_server" in semantic["description"]
    assert "embedding_base_url" in semantic["description"]
    # Only keyword grounding can run without the embedding address; the pieces that
    # always embed keep it required.
    assert flow_catalog.SECRETS["draft-lexicon"][0]["optional"] == ["embedding_base_url"]
    assert "optional" not in flow_catalog.SECRETS["discover-frames"][0]
    assert "optional" not in flow_catalog.SECRETS["map-and-project-embedding"][0]
    assert flow_catalog.REVISION["draft-lexicon"] == 4
    assert flow_catalog.registration("draft-lexicon")["id"].endswith("-r4")
    sent = flow_catalog.registration("draft-lexicon")["runtime_json"]["sends_data_to"][0]["what"]
    assert "semantic grounding" in sent and "passages" in sent
