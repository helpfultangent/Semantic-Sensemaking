from __future__ import annotations

import json

import pytest

from semantic_sensemaking.errors import InputFormatError
from semantic_sensemaking.lexicon import load_lexicon, save_lexicon
from semantic_sensemaking.taxonomy import load_taxonomy, save_taxonomy


def test_builtin_lexicon_and_taxonomy(subside_lexicon, subside_taxonomy):
    assert len(subside_lexicon) == 18
    assert subside_lexicon["InSAR"].terms[0] == "insar"
    assert subside_lexicon.source == "built-in 'subside_v1'"
    assert subside_taxonomy.domains()[0] == "Earth Sciences"
    assert len(subside_taxonomy) == 5


def test_flat_lexicon_json_with_comment_keys(tmp_path):
    path = tmp_path / "keyword_groups_20260512.json"
    path.write_text(json.dumps({"_comment": "ignored", "Groundwater": ["aquifer", "drawdown"]}))
    lexicon = load_lexicon(path)
    assert lexicon.to_mapping() == {"Groundwater": ["aquifer", "drawdown"]}
    assert lexicon.name == "keyword_groups_20260512"


def test_csv_lexicon_preserves_group_order_and_descriptions(tmp_path):
    path = tmp_path / "terms.csv"
    path.write_text(
        "group,term,description\nFlood,flood,Flooding terms\nUrban,city,\nFlood,inundation,\n,,\n"
    )
    lexicon = load_lexicon(path)
    assert lexicon.names() == ("Flood", "Urban")
    assert lexicon["Flood"].terms == ("flood", "inundation")
    assert lexicon["Flood"].description == "Flooding terms"


@pytest.mark.parametrize("suffix", [".json", ".csv"])
def test_lexicon_round_trip(tmp_path, subside_lexicon, suffix):
    path = save_lexicon(subside_lexicon, tmp_path / f"out{suffix}")
    assert load_lexicon(path).to_mapping() == subside_lexicon.to_mapping()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ({"Groundwater": "aquifer"}, "list of term strings"),
        ({"Groundwater": []}, "has no terms"),
        ({"schema_version": 1, "groups": [{"name": "A", "terms": ["x"], "typo": 1}]}, "groups.0.typo"),
        ({"schema_version": 1, "groups": [{"name": "A", "terms": ["x"]}, {"name": "A", "terms": ["y"]}]}, "Duplicate"),
        (["not", "an", "object"], "expected an object"),
    ],
)
def test_invalid_lexicons_name_the_problem(tmp_path, content, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(content))
    with pytest.raises(InputFormatError, match=message):
        load_lexicon(path)


def test_csv_lexicon_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("name,word\nA,x\n")
    with pytest.raises(InputFormatError, match="'group' and 'term'"):
        load_lexicon(path)


LAYERED = {
    "selected_layer": "merged",
    "produced_at": "2026-05-12T11:24:15",
    "layers": {
        "ucsd": {"Biology": {"subdisciplines": ["Ecology"], "terms": ["species"]}},
        "merged": {
            "Earth Sciences": {"subdisciplines": ["Hydrology"], "terms": ["aquifer"], "_n_papers": 12},
        },
    },
}


def test_layered_backbone_uses_selected_layer_and_strips_metadata(tmp_path):
    path = tmp_path / "science_backbone.json"
    path.write_text(json.dumps(LAYERED))
    taxonomy = load_taxonomy(path)
    assert taxonomy.domains() == ("Earth Sciences",)
    assert taxonomy.source == "science_backbone.json (layer='merged', produced 2026-05-12)"
    assert load_taxonomy(path, layer="ucsd").domains() == ("Biology",)


def test_layered_backbone_falls_back_like_the_prototype(tmp_path):
    payload = {**LAYERED, "selected_layer": "missing", "layers": {"ucsd": LAYERED["layers"]["ucsd"]}}
    path = tmp_path / "science_backbone.json"
    path.write_text(json.dumps(payload))
    assert load_taxonomy(path).layer == "ucsd"
    with pytest.raises(InputFormatError, match="available"):
        load_taxonomy(path, layer="merged")


def test_flat_taxonomy_and_default_subdiscipline(tmp_path):
    path = tmp_path / "flat.json"
    path.write_text(json.dumps({"Humanities": {"terms": ["history"]}}))
    taxonomy = load_taxonomy(path)
    assert list(taxonomy.pairs()) == [("Humanities", "General")]


def test_taxonomy_round_trip(tmp_path, subside_taxonomy):
    path = save_taxonomy(subside_taxonomy, tmp_path / "tax.json")
    assert load_taxonomy(path).to_mapping() == subside_taxonomy.to_mapping()


def test_invalid_taxonomy(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"Earth Sciences": {"terms": "aquifer"}}))
    with pytest.raises(InputFormatError, match="Earth Sciences.terms"):
        load_taxonomy(path)
