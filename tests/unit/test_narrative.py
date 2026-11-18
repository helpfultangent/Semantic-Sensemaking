"""Narrative workflow: segmentation, clustering, keywords, labeling, projection."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_sensemaking.backends.hashing import DeterministicHashEmbedding
from semantic_sensemaking.core.config import NarrativeConfig
from semantic_sensemaking.core.types import Corpus
from semantic_sensemaking.errors import BackendError, ConfigError
from semantic_sensemaking.lexicon.io import load_lexicon, save_lexicon
from semantic_sensemaking.workflows.narrative import (
    analyze,
    cluster_passages,
    segment_corpus,
)
from semantic_sensemaking.workflows.narrative.keywords import cluster_keywords, tokenize
from semantic_sensemaking.workflows.narrative.labeling import FrameLabel, label_frames

PUMPING = "Groundwater pumping drew down the aquifer and the land subsided near the wells. "
POLICY = "The district board adopted permit rules and the legislature debated water policy. "


def two_topic_corpus() -> Corpus:
    return Corpus.from_texts(
        {
            "a": (PUMPING * 4 + POLICY * 4),
            "b": (POLICY * 4 + PUMPING * 4),
            "c": PUMPING * 8,
        }
    )


def test_segment_corpus_keeps_document_and_position():
    passages = segment_corpus(Corpus.from_texts({"x": "one two three four five"}), 2)
    assert [(p.doc_id, p.index, p.text) for p in passages] == [
        ("x", 0, "one two"), ("x", 1, "three four"), ("x", 2, "five"),
    ]
    with pytest.raises(ConfigError):
        segment_corpus(Corpus.from_texts({"x": "a b"}), 5, overlap=5)


def test_clustering_separates_clear_groups_and_is_deterministic():
    rng = np.random.default_rng(1)
    centers = np.eye(3, 16)
    x = np.vstack([c + 0.05 * rng.standard_normal((6, 16)) for c in centers])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    first = cluster_passages(x, 3, seed=0)
    assert first.k == 3
    assert {tuple(sorted(set(first.labels[i * 6 : (i + 1) * 6]))) for i in range(3)} == {(0,), (1,), (2,)}
    again = cluster_passages(x, 3, seed=0)
    assert np.array_equal(first.labels, again.labels)


def test_clustering_respects_min_size_and_orders_largest_first():
    rng = np.random.default_rng(2)
    big = np.array([1.0, 0, 0]) + 0.01 * rng.standard_normal((8, 3))
    lone = np.array([[0, 1.0, 0]])
    x = np.vstack([big, lone])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    result = cluster_passages(x, 5, min_size=3)
    sizes = np.bincount(result.labels)
    assert sizes.min() >= 3
    assert list(sizes) == sorted(sizes, reverse=True)
    assert len(result.members(0)) == sizes[0]


def test_keywords_are_distinctive_and_skip_stopwords():
    texts = [PUMPING, PUMPING, POLICY, POLICY]
    kw = cluster_keywords(texts, [0, 0, 1, 1], 2, top_n=5)
    assert "aquifer" in kw[0] and "aquifer" not in kw[1]
    assert any("permit" in t for t in kw[1])
    assert "the" not in tokenize("The aquifer and the wells")
    # bigrams never bridge a stopword
    assert not any(t == "aquifer land" for t in kw[0])


class FakeChat:
    name = "fake"
    model = "fake"

    def __init__(self, fail_on: set[int] = frozenset()):
        self.calls = 0
        self.fail_on = fail_on

    def complete_json(self, prompt, *, schema, system=None, max_tokens=4096):
        self.calls += 1
        assert schema is FrameLabel and "Passage 1" in prompt
        if self.calls in self.fail_on:
            raise BackendError("boom")
        return FrameLabel(name="Same Name", description="d", terms=["Pumping", "pumping"])

    def close(self):
        pass


def test_label_frames_with_chat_dedupes_names_and_falls_back_per_frame():
    texts = [PUMPING, POLICY, PUMPING]
    chat = FakeChat(fail_on={3})
    frames = label_frames([([0], ["aquifer"]), ([1], ["permit"]), ([2], ["wells"])], texts, chat=chat)
    assert [f.name for f in frames] == ["Same Name", "Same Name (2)", "F3: wells"]
    assert frames[0].terms == ("pumping", "aquifer")
    assert [f.labeled_by for f in frames] == ["chat", "chat", "keywords"]


def test_analyze_end_to_end_offline(subside_taxonomy, tmp_path):
    corpus = two_topic_corpus()
    result = analyze(
        corpus,
        subside_taxonomy,
        DeterministicHashEmbedding(256),
        NarrativeConfig(segment_words=12, max_frames=4, min_frame_passages=2),
    )
    assert len(result.passage_frames) == len(result.passages)
    assert sum(f.n_passages for f in result.frames) == len(result.passages)
    assert sum(result.frame_weights().values()) == len(result.passages)
    assert result.lexicon.names() == tuple(f.name for f in result.frames)

    counts = result.doc_frame_counts()
    assert len(counts) == len(corpus) * len(result.frames)
    for _, group in counts.groupby("doc_id"):
        assert group["share_pct"].sum() == pytest.approx(100.0)

    tables = result.tables()
    assert {"frames", "passage_frames", "frame_mapping", "comparison_base_vs_frames"} <= set(tables)
    assert result.to_overlay_data().active_labels()

    # frames survive a lexicon round trip so they can feed the dictionary workflow
    path = save_lexicon(result.lexicon, tmp_path / "frames.json")
    assert load_lexicon(path).to_mapping() == result.lexicon.to_mapping()


def test_analyze_requires_embedding_backend(subside_taxonomy):
    with pytest.raises(ConfigError, match="embedding backend"):
        analyze(two_topic_corpus(), subside_taxonomy, None)
