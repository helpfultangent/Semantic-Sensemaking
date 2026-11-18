from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from semantic_sensemaking import load_corpus, load_lexicon, load_taxonomy, sample_corpus_path


@pytest.fixture(autouse=True)
def _isolate_ssm_environment(monkeypatch):
    """Keep a developer's SSM_* variables from leaking into config tests."""
    for key in list(os.environ):
        if key.startswith("SSM_") and key != "SSM_TEST_BASE_URL":
            monkeypatch.delenv(key)


@pytest.fixture(scope="session")
def sample_corpus():
    return load_corpus(sample_corpus_path())


@pytest.fixture(scope="session")
def subside_lexicon():
    return load_lexicon()


@pytest.fixture(scope="session")
def subside_taxonomy():
    return load_taxonomy()


LONG_TEXT = (
    "Groundwater pumping lowered the aquifer and caused land subsidence across the delta. "
    "InSAR and GPS measurements confirm the settlement pattern near the coast. "
) * 3


def write_pdf(path: Path, text: str) -> Path:
    """Write a minimal single-page PDF with a text layer (text must not contain parentheses)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def edge_corpus(tmp_path: Path) -> Path:
    """A directory exercising every reader and every skip/ignore rule."""
    import docx

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text(LONG_TEXT)
    (root / "a.md").write_text("# Same stem\n\n" + LONG_TEXT)
    (root / "text.json").write_text(json.dumps({"text": LONG_TEXT}))
    (root / "chunks.json").write_text(json.dumps({"chunks": [{"text": LONG_TEXT}, {"text": "More."}]}))
    (root / "speakers.json").write_text(json.dumps({"speakers": [{"speaker": "A", "text": LONG_TEXT}]}))
    (root / "content.json").write_text(json.dumps({"content": LONG_TEXT}))
    document = docx.Document()
    document.add_paragraph(LONG_TEXT)
    document.save(root / "memo.docx")
    write_pdf(root / "plan.pdf", " ".join([LONG_TEXT.replace(".", "")] * 1))
    (root / ".hidden.txt").write_text(LONG_TEXT)
    (root / "~$lock.docx").write_text("lock file")
    (root / "notes-checkpoint.txt").write_text(LONG_TEXT)
    (root / "tiny.txt").write_text("too short")
    (root / "table.xlsx").write_bytes(b"not really a spreadsheet")
    return root
