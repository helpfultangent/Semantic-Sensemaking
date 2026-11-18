from __future__ import annotations

import json

import pytest

from semantic_sensemaking.core.config import CorpusConfig
from semantic_sensemaking.core.types import Corpus
from semantic_sensemaking.corpus import (
    extract_json_text,
    load_corpus,
    read_document,
    resolve_document,
)
from semantic_sensemaking.errors import CorpusError, DocumentLookupError, MissingExtraError


def test_sample_corpus_loads_every_file(sample_corpus):
    assert len(sample_corpus) == 5
    assert {d.source_kind for d in sample_corpus} == {"md", "json", "txt"}
    assert not sample_corpus.skipped


def test_edge_corpus_reads_all_formats_and_reports_skips(edge_corpus):
    corpus = load_corpus(edge_corpus)
    assert set(corpus.ids()) == {
        "a.md", "a.txt", "chunks.json", "content.json", "memo.docx", "plan.pdf",
        "speakers.json", "text.json",
    }
    reasons = {s.path: s.reason for s in corpus.skipped}
    assert reasons == {"table.xlsx": "unsupported format (.xlsx)", "tiny.txt": "too short (9 chars)"}
    assert "groundwater pumping" in corpus["plan.pdf"].text.lower()


def test_shared_stems_get_unique_labels_instead_of_overwriting(edge_corpus):
    corpus = load_corpus(edge_corpus)
    assert corpus["a.txt"].label == "a.txt"
    assert corpus["a.md"].label == "a.md"
    assert corpus["memo.docx"].label == "memo"


def test_max_files_truncates_and_records_the_rest(edge_corpus):
    corpus = load_corpus(edge_corpus, CorpusConfig(max_files=3, min_chars=0))
    assert len(corpus) == 3
    assert sum("max_files" in s.reason for s in corpus.skipped) == 7


def test_recursive_discovery_ignores_hidden_directories(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("x" * 300)
    (tmp_path / ".ipynb_checkpoints" / "copy.txt").write_text("x" * 300)
    assert load_corpus(tmp_path).ids() == ()
    assert load_corpus(tmp_path, CorpusConfig(recursive=True)).ids() == ("sub/deep.txt",)


def test_single_file_and_missing_path(tmp_path):
    f = tmp_path / "one.txt"
    f.write_text("y" * 250)
    assert load_corpus(f).ids() == ("one.txt",)
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "missing")


def test_missing_optional_reader_becomes_skip_reason_with_install_hint(tmp_path, monkeypatch):
    import semantic_sensemaking.corpus.readers as readers

    def fail(module, extra, purpose=None):
        raise MissingExtraError(module, extra, purpose)

    monkeypatch.setattr(readers, "optional_import", fail)
    path = tmp_path / "x.docx"
    path.write_bytes(b"")
    text, reason = read_document(path)
    assert text is None
    assert "pip install 'semantic-sensemaking[readers]'" in reason


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"text": "t" * 101}, "t" * 101),
        ({"text": "short", "chunks": [{"text": "a"}, {"text": "b"}]}, "a b"),
        ({"speakers": [{"speaker": "x", "text": "hi"}]}, "hi"),
        ({"content": "body"}, "body"),
        ("plain", "plain"),
    ],
)
def test_extract_json_text_shapes(data, expected):
    assert extract_json_text(data) == expected


def test_extract_json_text_falls_back_to_serialized_json():
    assert json.loads(extract_json_text({"other": 1})) == {"other": 1}


def test_resolve_document():
    corpus = Corpus.from_texts({"20230810_brown_bag": "a", "20230901_brown_bag": "b", "interview": "c"})
    assert resolve_document(corpus, 2).label == "interview"
    assert resolve_document(corpus, "interview").label == "interview"
    assert resolve_document(corpus, "0810").label == "20230810_brown_bag"
    assert resolve_document(corpus, "20230810").label == "20230810_brown_bag"  # out of range, so a name
    assert resolve_document(corpus, "1").label == "20230901_brown_bag"  # in range, so an index
    with pytest.raises(DocumentLookupError, match="matched 2"):
        resolve_document(corpus, "brown_bag")
    with pytest.raises(DocumentLookupError, match="out of range"):
        resolve_document(corpus, 9)
    with pytest.raises(DocumentLookupError, match="No document"):
        resolve_document(corpus, "999")


def test_corpus_rejects_duplicate_ids():
    from semantic_sensemaking.core.types import Document

    with pytest.raises(ValueError, match="Duplicate"):
        Corpus((Document("a", "x", "a"), Document("a", "y", "a")))
