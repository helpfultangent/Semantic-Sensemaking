"""File readers keyed by suffix. Register more with :func:`register_reader`."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from semantic_sensemaking.corpus.json_text import extract_json_text
from semantic_sensemaking.errors import MissingExtraError, optional_import

Reader = Callable[[Path, str], str]


def _read_plain(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding, errors="replace")


def _read_json(path: Path, encoding: str) -> str:
    with open(path, encoding=encoding, errors="replace") as fh:
        return extract_json_text(json.load(fh))


def _read_docx(path: Path, encoding: str) -> str:
    docx = optional_import("docx", "readers", "Reading .docx files")
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def _read_pdf(path: Path, encoding: str) -> str:
    pypdf = optional_import("pypdf", "readers", "Reading .pdf files")
    reader = pypdf.PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


_READERS: dict[str, tuple[str, Reader]] = {
    ".txt": ("txt", _read_plain),
    ".md": ("md", _read_plain),
    ".json": ("json", _read_json),
    ".docx": ("docx", _read_docx),
    ".pdf": ("pdf", _read_pdf),
}


def register_reader(suffix: str, kind: str, reader: Reader) -> None:
    """Add or replace the reader for a file suffix, e.g. ``.vtt`` subtitles."""
    suffix = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    _READERS[suffix] = (kind, reader)


def supported_suffixes() -> tuple[str, ...]:
    return tuple(sorted(_READERS))


def read_document(path: str | Path, *, encoding: str = "utf-8") -> tuple[str | None, str]:
    """Return ``(text, kind)``, or ``(None, reason)`` if the file can't be read.

    Never raises for a bad file: one corrupt PDF shouldn't abort a batch job.
    A missing optional dependency is reported as the reason, with the pip command.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    entry = _READERS.get(suffix)
    if entry is None:
        return None, f"unsupported format ({suffix})"
    kind, reader = entry
    try:
        return reader(path, encoding), kind
    except MissingExtraError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — any reader failure becomes a skip reason
        return None, f"{type(exc).__name__}: {exc}"
