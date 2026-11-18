"""Read and write taxonomies (science backbones).

Accepted formats:

- **Structured JSON** (``schema_version: 1``), written by :func:`save_taxonomy`::

      {"schema_version": 1, "name": "ucsd", "layer": "ucsd",
       "domains": [{"domain": "Earth Sciences",
                    "subdisciplines": ["Hydrology"], "terms": ["aquifer"]}]}

- **Layered backbone JSON**, as produced by the SUBSIDE backbone notebooks
  (``science_backbone.json``)::

      {"selected_layer": "merged", "produced_at": "2026-05-12T...",
       "layers": {"merged": {"Earth Sciences": {"subdisciplines": [...],
                                                "terms": [...], "_meta": ...}}}}

  Without an explicit ``layer``, the prototype's resolution order is used:
  ``selected_layer``, then ``merged``, then ``ucsd``.

- **Flat JSON**: ``{"Earth Sciences": {"subdisciplines": [...], "terms": [...]}}``.

Keys starting with ``_`` inside a domain are metadata and ignored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from semantic_sensemaking.core.types import Taxonomy
from semantic_sensemaking.data import builtin_path
from semantic_sensemaking.errors import InputFormatError

DEFAULT_BUILTIN_TAXONOMY = "subside_v1"


class TaxonomyDomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    subdisciplines: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)


class TaxonomyFileModel(BaseModel):
    """JSON Schema source of truth for structured taxonomy files."""

    model_config = ConfigDict(extra="forbid", title="Semantic Sensemaking taxonomy")

    schema_version: Literal[1] = 1
    name: str = "taxonomy"
    description: str | None = None
    layer: str | None = None
    produced_at: str | None = None
    domains: list[TaxonomyDomainModel] = Field(min_length=1)


def taxonomy_json_schema() -> dict[str, Any]:
    return TaxonomyFileModel.model_json_schema()


def load_taxonomy(
    path: str | Path | None = None,
    *,
    layer: str | None = None,
    builtin: str = DEFAULT_BUILTIN_TAXONOMY,
) -> Taxonomy:
    """Load a taxonomy file, or the named built-in taxonomy when ``path`` is None."""
    if path is None:
        tax = _load(builtin_path(f"taxonomy_{builtin}.json"), layer)
        return Taxonomy(
            tax.nodes, name=tax.name, layer=tax.layer, produced_at=tax.produced_at,
            source=f"built-in '{builtin}'",
        )
    path = Path(path)
    if not path.is_file():
        raise InputFormatError(f"Taxonomy file not found: {path}")
    return _load(path, layer)


def save_taxonomy(taxonomy: Taxonomy, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = TaxonomyFileModel(
        name=taxonomy.name,
        layer=taxonomy.layer,
        produced_at=taxonomy.produced_at,
        domains=[
            TaxonomyDomainModel(
                domain=n.domain, subdisciplines=list(n.subdisciplines), terms=list(n.terms)
            )
            for n in taxonomy
        ],
    )
    path.write_text(model.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    return path


def _load(path: Path, layer: str | None) -> Taxonomy:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InputFormatError(f"{path}: expected a JSON object")

    if "domains" in data:
        if layer is not None and data.get("layer") not in (None, layer):
            raise InputFormatError(f"{path} holds layer '{data.get('layer')}', not '{layer}'")
        try:
            model = TaxonomyFileModel.model_validate(data)
        except ValidationError as exc:
            raise InputFormatError(_describe(path, exc)) from exc
        return _build(
            path,
            {d.domain: d.model_dump() for d in model.domains},
            name=model.name,
            layer=model.layer,
            produced_at=model.produced_at,
            source=path.name,
        )

    if "layers" in data:
        layers = data["layers"]
        if not isinstance(layers, dict) or not layers:
            raise InputFormatError(f"{path}: 'layers' must be a non-empty object")
        if layer is not None:
            if layer not in layers:
                raise InputFormatError(
                    f"{path}: layer '{layer}' not found; available: {sorted(layers)}"
                )
            used = layer
        else:
            preferred = data.get("selected_layer") or "merged"
            used = next((name for name in (preferred, "merged", "ucsd") if layers.get(name)), None)
            if used is None:
                raise InputFormatError(
                    f"{path}: none of the layers '{preferred}', 'merged', 'ucsd' exist; "
                    f"pass a layer explicitly from {sorted(layers)}"
                )
        produced_at = data.get("produced_at")
        return _build(
            path,
            layers[used],
            name=used,
            layer=used,
            produced_at=produced_at,
            source=f"{path.name} (layer='{used}', produced {str(produced_at or '?')[:10]})",
        )

    return _build(path, data, name=path.stem, layer=layer, produced_at=None, source=path.name)


def _build(path: Path, domains: Any, **kwargs: Any) -> Taxonomy:
    if not isinstance(domains, Mapping) or not domains:
        raise InputFormatError(f"{path}: expected a non-empty object of domains")
    for domain, node in domains.items():
        if not isinstance(node, Mapping):
            raise InputFormatError(f"{path}: domain '{domain}' must be an object")
        for key in ("subdisciplines", "terms"):
            value = node.get(key, [])
            if value is not None and (
                not isinstance(value, list) or not all(isinstance(v, str) for v in value)
            ):
                raise InputFormatError(f"{path}: '{domain}.{key}' must be a list of strings")
    try:
        return Taxonomy.from_mapping(domains, **kwargs)
    except ValueError as exc:
        raise InputFormatError(f"{path}: {exc}") from exc


def _describe(path: Path, exc: ValidationError) -> str:
    lines = [f"{path} does not match the taxonomy schema:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
