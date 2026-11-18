"""The library with legacy settings reproduces the prototype notebook's numbers.

Two guards: a live comparison against the verbatim oracle in
``prototype_reference.py``, and a frozen golden file so that neither the oracle
nor the library can drift unnoticed. Regenerate the golden file only for an
intentional change: ``python tests/parity/make_golden.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from semantic_sensemaking import sample_corpus_path
from semantic_sensemaking.core.config import ScoringConfig
from semantic_sensemaking.data import builtin_path
from semantic_sensemaking.workflows.dictionary import analyze

from . import prototype_reference as proto

GOLDEN = Path(__file__).with_name("golden_sample_corpus.json")
LEGACY = ScoringConfig()  # defaults are the prototype's behavior


@pytest.fixture(scope="module")
def prototype():
    return proto.run_prototype(
        sample_corpus_path(),
        builtin_path("lexicon_subside_v1.json"),
        builtin_path("taxonomy_subside_v1.json"),
    )


@pytest.fixture(scope="module")
def library(sample_corpus, subside_lexicon, subside_taxonomy):
    return analyze(sample_corpus, subside_lexicon, subside_taxonomy, LEGACY)


def test_legacy_settings_are_the_defaults():
    assert LEGACY.match_mode.value == "auto"
    assert LEGACY.mapping_method == "legacy"
    assert LEGACY.unmapped_groups == "drop"


def test_same_documents_loaded(prototype, library):
    assert {d.label: d.text for d in library.corpus} == prototype["docs"]


def test_same_group_mapping(prototype, library):
    for _, row in prototype["group_mapping_df"].iterrows():
        assert [a.as_tuple() for a in library.mapping[row["group"]]] == [
            (d, s, float(sc)) for d, s, sc in row["backbone_mapping"]
        ]


def test_same_doc_group_counts(prototype, library):
    columns = ["doc", "group", "count", "doc_n_words", "per_1k_words"]
    ours = library.scores.to_frame()[columns].sort_values(["doc", "group"]).reset_index(drop=True)
    theirs = prototype["doc_group"][columns].sort_values(["doc", "group"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False)


def test_same_group_totals(prototype, library):
    theirs = dict(zip(prototype["group_totals"]["group"], prototype["group_totals"]["count"]))
    assert library.scores.group_weights() == pytest.approx(theirs)


def test_same_projection(prototype, library):
    assert dict(library.projection.domain_weights) == pytest.approx(prototype["domain_weights"])
    assert dict(library.projection.subdiscipline_weights) == pytest.approx(prototype["sub_weights"])


def test_same_comparison(prototype, library):
    columns = ["domain", "base_n_groups", "base_pct", "overlay_weight", "overlay_pct", "delta_pct"]
    ours = library.comparison[columns].sort_values("domain").reset_index(drop=True)
    theirs = prototype["comparison"][columns].sort_values("domain").reset_index(drop=True)
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False)


def test_matches_frozen_golden(library):
    golden = json.loads(GOLDEN.read_text())
    assert library.scores.group_weights() == pytest.approx(golden["group_totals"])
    assert dict(library.projection.domain_weights) == pytest.approx(golden["domain_weights"])
    assert {f"{d} :: {s}": w for (d, s), w in library.projection.subdiscipline_weights.items()} == (
        pytest.approx(golden["subdiscipline_weights"])
    )
