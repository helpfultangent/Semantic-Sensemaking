"""Resolve a user-supplied document spec to one document."""

from __future__ import annotations

from semantic_sensemaking.core.types import Corpus, Document
from semantic_sensemaking.errors import DocumentLookupError


def resolve_document(corpus: Corpus, spec: int | str) -> Document:
    """Find a document by index, exact ``doc_id``/``label``, or unique substring.

    Resolution order: an ``int`` is an index; a string is tried as an exact name,
    then as an index if it is all digits and in range, then as a substring. So
    ``"2"`` means the third document, while a date-like ``"20230810"`` (out of
    range as an index) still finds ``20230810_brown_bag`` — the prototype's
    ``_resolve_doc`` tried the index first and reported that as out of range.
    """
    if not len(corpus):
        raise DocumentLookupError("The corpus is empty")
    if isinstance(spec, int):
        return _by_index(corpus, spec)

    for doc in corpus:
        if spec in (doc.doc_id, doc.label):
            return doc
    if spec.strip().isdigit() and int(spec) < len(corpus):
        return corpus[int(spec)]
    needle = spec.lower()
    matches = [d for d in corpus if needle in d.label.lower() or needle in d.doc_id.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = "\n  ".join(d.label for d in matches)
        raise DocumentLookupError(
            f"'{spec}' matched {len(matches)} documents:\n  {names}\n"
            "Use a more specific substring or an index."
        )
    raise DocumentLookupError(f"No document matches '{spec}'")


def _by_index(corpus: Corpus, index: int) -> Document:
    if 0 <= index < len(corpus):
        return corpus[index]
    raise DocumentLookupError(f"Index {index} out of range (0..{len(corpus) - 1})")
