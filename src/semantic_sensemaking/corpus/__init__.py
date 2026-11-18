"""Turn files and directories into a :class:`~semantic_sensemaking.core.types.Corpus`."""

from semantic_sensemaking.corpus.json_text import extract_json_text
from semantic_sensemaking.corpus.loader import load_corpus
from semantic_sensemaking.corpus.readers import read_document, register_reader, supported_suffixes
from semantic_sensemaking.corpus.select import resolve_document

__all__ = [
    "extract_json_text",
    "load_corpus",
    "read_document",
    "register_reader",
    "resolve_document",
    "supported_suffixes",
]
