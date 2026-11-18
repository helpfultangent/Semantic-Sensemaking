"""Read and write user-defined dictionaries (lexicons).

Accepted formats:

- **Structured JSON** (``schema_version: 1``), the format :func:`save_lexicon` writes::

      {"schema_version": 1, "name": "subside",
       "groups": [{"name": "Groundwater", "terms": ["aquifer", "drawdown"],
                   "description": "optional"}]}

- **Flat JSON**, as produced by the SUBSIDE setup notebooks
  (``keyword_groups_*.json``): ``{"Groundwater": ["aquifer", "drawdown"]}``.
  Top-level keys starting with ``_`` are treated as comments.

- **CSV** with ``group`` and ``term`` columns (one row per term) and an
  optional ``description`` column — convenient for dictionaries curated in a
  spreadsheet.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from semantic_sensemaking.core.types import Lexicon, LexiconEntry
from semantic_sensemaking.data import builtin_path
from semantic_sensemaking.errors import InputFormatError

DEFAULT_BUILTIN_LEXICON = "subside_v1"


class LexiconGroupModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    terms: list[str] = Field(min_length=1)
    description: str | None = None


class LexiconFileModel(BaseModel):
    """JSON Schema source of truth for structured lexicon files."""

    model_config = ConfigDict(extra="forbid", title="Semantic Sensemaking lexicon")

    schema_version: Literal[1] = 1
    name: str = "lexicon"
    description: str | None = None
    groups: list[LexiconGroupModel] = Field(min_length=1)


def lexicon_json_schema() -> dict[str, Any]:
    return LexiconFileModel.model_json_schema()


def load_lexicon(
    path: str | Path | None = None, *, builtin: str = DEFAULT_BUILTIN_LEXICON
) -> Lexicon:
    """Load a lexicon file, or the named built-in lexicon when ``path`` is None."""
    if path is None:
        lexicon = _load_json(builtin_path(f"lexicon_{builtin}.json"))
        return Lexicon(lexicon.entries, name=lexicon.name, source=f"built-in '{builtin}'")

    path = Path(path)
    if not path.is_file():
        raise InputFormatError(f"Lexicon file not found: {path}")
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    return _load_json(path)


def save_lexicon(lexicon: Lexicon, path: str | Path) -> Path:
    """Write structured JSON, or CSV when ``path`` ends in ``.csv``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["group", "term", "description"])
            for entry in lexicon:
                for term in entry.terms:
                    writer.writerow([entry.name, term, entry.description or ""])
        return path
    model = LexiconFileModel(
        name=lexicon.name,
        groups=[
            LexiconGroupModel(name=e.name, terms=list(e.terms), description=e.description)
            for e in lexicon
        ],
    )
    path.write_text(model.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    return path


def _load_json(path: Path) -> Lexicon:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"{path} is not valid JSON: {exc}") from exc

    if isinstance(data, dict) and "groups" in data:
        try:
            model = LexiconFileModel.model_validate(data)
        except ValidationError as exc:
            raise InputFormatError(_describe(path, exc)) from exc
        entries = [LexiconEntry(g.name, tuple(g.terms), g.description) for g in model.groups]
        return _build(entries, name=model.name, source=path.name, path=path)

    if isinstance(data, dict):
        entries = []
        for group, terms in data.items():
            if str(group).startswith("_"):
                continue
            if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
                raise InputFormatError(
                    f"{path}: group '{group}' must map to a list of term strings"
                )
            entries.append((group, terms))
        if not entries:
            raise InputFormatError(f"{path}: no keyword groups found")
        return _build(
            [_entry(path, g, t) for g, t in entries], name=path.stem, source=path.name, path=path
        )

    raise InputFormatError(
        f"{path}: expected an object of groups, e.g. {{\"Group\": [\"term\", ...]}}"
    )


def _load_csv(path: Path) -> Lexicon:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = {f.strip().lower() for f in (reader.fieldnames or [])}
        if not {"group", "term"} <= fields:
            raise InputFormatError(f"{path}: CSV lexicons need 'group' and 'term' columns")
        groups: dict[str, list[str]] = {}
        descriptions: dict[str, str] = {}
        for row in reader:
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            group, term = row.get("group", ""), row.get("term", "")
            if not group and not term:
                continue
            if not group or not term:
                raise InputFormatError(
                    f"{path}, line {reader.line_num}: both 'group' and 'term' are required"
                )
            groups.setdefault(group, []).append(term)
            if row.get("description"):
                descriptions.setdefault(group, row["description"])
    if not groups:
        raise InputFormatError(f"{path}: no keyword groups found")
    return _build(
        [_entry(path, g, t, descriptions.get(g)) for g, t in groups.items()],
        name=path.stem,
        source=path.name,
        path=path,
    )


def _entry(path: Path, group: str, terms: list[str], description: str | None = None) -> LexiconEntry:
    try:
        return LexiconEntry(group, tuple(terms), description)
    except ValueError as exc:
        raise InputFormatError(f"{path}: {exc}") from exc


def _build(entries: list[LexiconEntry], *, name: str, source: str, path: Path) -> Lexicon:
    try:
        return Lexicon(tuple(entries), name=name, source=source)
    except ValueError as exc:
        raise InputFormatError(f"{path}: {exc}") from exc


def _describe(path: Path, exc: ValidationError) -> str:
    lines = [f"{path} does not match the lexicon schema:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
