"""Core data types shared by every tool cluster and workflow.

These are frozen dataclasses with no I/O and no optional dependencies, so any
layer can accept or return them without importing another layer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

GENERAL_SUBDISCIPLINE = "General"


# ── Corpus ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Document:
    """One text unit in a corpus.

    ``doc_id`` is unique within a corpus (a POSIX path relative to the corpus
    root when loaded from disk). ``label`` is the human-facing name used in
    tables and figures; it defaults to the file stem, matching the prototype.
    """

    doc_id: str
    text: str
    label: str
    source_kind: str = "inline"
    source_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @cached_property
    def n_chars(self) -> int:
        return len(self.text)

    @cached_property
    def n_words(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class SkippedFile:
    """A file the loader saw but did not turn into a :class:`Document`."""

    path: str
    reason: str


@dataclass(frozen=True)
class Corpus:
    """An ordered, immutable collection of documents plus loader diagnostics."""

    documents: tuple[Document, ...]
    skipped: tuple[SkippedFile, ...] = ()
    root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "documents", tuple(self.documents))
        object.__setattr__(self, "skipped", tuple(self.skipped))
        seen: set[str] = set()
        for doc in self.documents:
            if doc.doc_id in seen:
                raise ValueError(f"Duplicate doc_id in corpus: {doc.doc_id!r}")
            seen.add(doc.doc_id)

    @classmethod
    def from_texts(cls, texts: Mapping[str, str]) -> Corpus:
        """Build an in-memory corpus from ``{name: text}``, e.g. in a notebook."""
        return cls(tuple(Document(doc_id=k, text=v, label=k) for k, v in texts.items()))

    @cached_property
    def _by_id(self) -> dict[str, Document]:
        return {doc.doc_id: doc for doc in self.documents}

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def __getitem__(self, key: int | str) -> Document:
        if isinstance(key, int):
            return self.documents[key]
        return self._by_id[key]

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._by_id

    def ids(self) -> tuple[str, ...]:
        return tuple(doc.doc_id for doc in self.documents)

    def subset(self, doc_ids: Iterable[str]) -> Corpus:
        wanted = set(doc_ids)
        return Corpus(
            tuple(d for d in self.documents if d.doc_id in wanted), self.skipped, self.root
        )

    def total_words(self) -> int:
        return sum(doc.n_words for doc in self.documents)

    def describe(self) -> pd.DataFrame:
        """Tabular digest of the corpus (replaces the prototype's ``list_overlay_docs``)."""
        return pd.DataFrame(
            [
                {
                    "doc_id": d.doc_id,
                    "label": d.label,
                    "source_kind": d.source_kind,
                    "n_chars": d.n_chars,
                    "n_words": d.n_words,
                }
                for d in self.documents
            ],
            columns=["doc_id", "label", "source_kind", "n_chars", "n_words"],
        )


# ── Lexicon (user-defined dictionary) ────────────────────────────────────


@dataclass(frozen=True)
class LexiconEntry:
    """A named concept group and the terms (synonyms, variants) that signal it."""

    name: str
    terms: tuple[str, ...]
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.name or not self.name.strip():
            raise ValueError("Lexicon entry name must be a non-empty string")
        if not self.terms:
            raise ValueError(f"Lexicon entry {self.name!r} has no terms")
        for term in self.terms:
            if not isinstance(term, str) or not term.strip():
                raise ValueError(f"Lexicon entry {self.name!r} has an empty or non-string term")


@dataclass(frozen=True)
class Lexicon:
    """An ordered set of :class:`LexiconEntry` groups — the user-defined dictionary."""

    entries: tuple[LexiconEntry, ...]
    name: str = "lexicon"
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        names = [e.name for e in self.entries]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Duplicate lexicon group names: {dupes}")

    @classmethod
    def from_mapping(
        cls,
        groups: Mapping[str, Sequence[str]],
        *,
        name: str = "lexicon",
        source: str | None = None,
    ) -> Lexicon:
        return cls(
            tuple(LexiconEntry(group, tuple(terms)) for group, terms in groups.items()),
            name=name,
            source=source,
        )

    def to_mapping(self) -> dict[str, list[str]]:
        return {e.name: list(e.terms) for e in self.entries}

    def names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self.entries)

    def __getitem__(self, name: str) -> LexiconEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LexiconEntry]:
        return iter(self.entries)


# ── Taxonomy (science backbone) ──────────────────────────────────────────


@dataclass(frozen=True)
class TaxonomyNode:
    """A top-level domain with its subdisciplines and indicator terms.

    Empty ``subdisciplines`` or ``terms`` are stored as given so files round-trip;
    use the ``effective_*`` properties, which apply the prototype's defaults.
    """

    domain: str
    subdisciplines: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subdisciplines", tuple(self.subdisciplines))
        object.__setattr__(self, "terms", tuple(self.terms))

    @property
    def effective_subdisciplines(self) -> tuple[str, ...]:
        return self.subdisciplines or (GENERAL_SUBDISCIPLINE,)

    @property
    def effective_terms(self) -> tuple[str, ...]:
        return self.terms or (self.domain.lower(),)


@dataclass(frozen=True)
class Taxonomy:
    """An ordered set of domains that corpora and concepts are projected onto."""

    nodes: tuple[TaxonomyNode, ...]
    name: str = "taxonomy"
    layer: str | None = None
    produced_at: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        domains = [n.domain for n in self.nodes]
        dupes = sorted({d for d in domains if domains.count(d) > 1})
        if dupes:
            raise ValueError(f"Duplicate taxonomy domains: {dupes}")

    @classmethod
    def from_mapping(
        cls,
        domains: Mapping[str, Mapping[str, Any]],
        *,
        name: str = "taxonomy",
        layer: str | None = None,
        produced_at: str | None = None,
        source: str | None = None,
    ) -> Taxonomy:
        """Build from ``{domain: {"subdisciplines": [...], "terms": [...]}}``.

        Keys beginning with ``_`` inside a domain node are metadata and ignored,
        as in the prototype.
        """
        nodes = []
        for domain, node in domains.items():
            node = {k: v for k, v in node.items() if not str(k).startswith("_")}
            nodes.append(
                TaxonomyNode(
                    domain=domain,
                    subdisciplines=tuple(node.get("subdisciplines") or ()),
                    terms=tuple(node.get("terms") or ()),
                )
            )
        return cls(tuple(nodes), name=name, layer=layer, produced_at=produced_at, source=source)

    def to_mapping(self) -> dict[str, dict[str, list[str]]]:
        return {
            n.domain: {"subdisciplines": list(n.subdisciplines), "terms": list(n.terms)}
            for n in self.nodes
        }

    def domains(self) -> tuple[str, ...]:
        return tuple(n.domain for n in self.nodes)

    def pairs(self) -> Iterator[tuple[str, str]]:
        """Yield every ``(domain, subdiscipline)`` in taxonomy order."""
        for node in self.nodes:
            for sub in node.effective_subdisciplines:
                yield node.domain, sub

    def __getitem__(self, domain: str) -> TaxonomyNode:
        for node in self.nodes:
            if node.domain == domain:
                return node
        raise KeyError(domain)

    def __contains__(self, domain: object) -> bool:
        return any(n.domain == domain for n in self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[TaxonomyNode]:
        return iter(self.nodes)
