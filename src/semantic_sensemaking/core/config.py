"""Run configuration shared by the CLI, notebooks, and Tapis apps.

Every surface builds one :class:`RunConfig`. Values are layered, lowest to
highest precedence:

1. model defaults
2. a YAML file (``RunConfig.load("job.yaml")`` / ``--config``)
3. ``SSM_*`` environment variables, nested with ``__``
   (``SSM_CORPUS__MAX_FILES=50``) — this is how Tapis app parameters arrive
4. explicit overrides (CLI flags, ``--set corpus.max_files=50``)

Unknown keys are rejected so a typo in a job file fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from semantic_sensemaking.errors import ConfigError


class MatchMode(str, Enum):
    """How lexicon terms are counted in text.

    ``AUTO`` reproduces the prototype exactly: substring counts for terms
    containing a space or hyphen, word-boundary regex otherwise, and every term
    counted independently — so "land subsidence" also counts as "subsidence".
    ``LONGEST_MATCH`` fixes that double counting.
    """

    AUTO = "auto"
    WORD_BOUNDARY = "word_boundary"
    LONGEST_MATCH = "longest_match"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputsConfig(_Section):
    corpus: Path | None = None
    use_sample_corpus: bool = False
    lexicon: Path | None = None
    taxonomy: Path | None = None
    taxonomy_layer: str | None = None


class CorpusConfig(_Section):
    glob_patterns: list[str] = Field(default_factory=lambda: ["*"])
    recursive: bool = False
    max_files: int = Field(200, ge=1)
    min_chars: int = Field(200, ge=0)
    encoding: str = "utf-8"
    ignore_patterns: list[str] = Field(default_factory=lambda: [".*", "~$*", "*checkpoint*"])


class ScoringConfig(_Section):
    method: Literal["keyword", "semantic"] = "keyword"
    match_mode: MatchMode = MatchMode.AUTO
    case_sensitive: bool = False
    mapping_method: Literal["legacy", "subdiscipline_aware", "embedding"] = "legacy"
    top_n_assignments: int = Field(3, ge=1)
    unmapped_groups: Literal["drop", "keep"] = "drop"
    semantic_threshold: float = Field(0.45, ge=-1.0, le=1.0)
    chunk_words: int = Field(120, ge=10)
    chunk_overlap_words: int = Field(20, ge=0)


BackendKind = Literal["none", "openai_compatible", "huggingface_local", "anthropic", "hashing"]


class BackendConfig(_Section):
    kind: BackendKind = "none"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    batch_size: int = Field(32, ge=1)
    timeout_s: float = Field(120.0, gt=0)
    max_retries: int = Field(3, ge=0)
    cache_dir: Path | None = None
    device: str | None = None
    dimensions: int = Field(256, ge=8)


class VizConfig(_Section):
    enabled: bool = True
    per_document_figures: bool = False
    max_per_document_figures: int = Field(50, ge=0)
    include_plotlyjs: Literal["cdn", "inline"] = "cdn"


class OutputConfig(_Section):
    dir: Path = Path("results")
    run_id: str | None = None


class NarrativeConfig(_Section):
    """Settings reserved for the planned narrative-analysis workflow."""

    segment_words: int = Field(200, ge=10)
    max_frames: int = Field(12, ge=1)


class FormulationConfig(_Section):
    """Settings reserved for the planned problem-formulation workflow."""

    problem_statement: str | None = None
    max_groups: int = Field(20, ge=1)
    max_terms_per_group: int = Field(12, ge=1)


WorkflowName = Literal["dictionary", "narrative", "formulation"]

_YAML_PATH: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "semantic_sensemaking_yaml_path", default=None
)


class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SSM_", env_nested_delimiter="__", extra="forbid"
    )

    workflow: WorkflowName = "dictionary"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    inputs: InputsConfig = Field(default_factory=InputsConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    embedding: BackendConfig = Field(default_factory=BackendConfig)
    chat: BackendConfig = Field(default_factory=BackendConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    narrative: NarrativeConfig = Field(default_factory=NarrativeConfig)
    formulation: FormulationConfig = Field(default_factory=FormulationConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        yaml_path = _YAML_PATH.get()
        if yaml_path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        return tuple(sources)

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
        set_values: Iterable[str] = (),
    ) -> RunConfig:
        """Build a config from an optional YAML file, env vars, and overrides.

        ``overrides`` is a nested mapping (``{"corpus": {"max_files": 50}}``);
        ``set_values`` are ``dotted.key=value`` strings whose values are parsed
        as YAML scalars. ``set_values`` win over ``overrides``.
        """
        yaml_path: Path | None = None
        if config_path is not None:
            yaml_path = Path(config_path).expanduser().resolve()
            if not yaml_path.is_file():
                raise ConfigError(f"Config file not found: {config_path}")
        merged = deep_merge(dict(overrides or {}), parse_set_values(set_values))
        token = _YAML_PATH.set(yaml_path)
        try:
            return cls(**merged)
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(exc, yaml_path)) from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse YAML config {yaml_path}: {exc}") from exc
        finally:
            _YAML_PATH.reset(token)

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe dump for manifests. Holds env var *names*, never secret values."""
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.snapshot(), sort_keys=False)


def parse_set_values(items: Iterable[str]) -> dict[str, Any]:
    """Turn ``["corpus.max_files=50"]`` into ``{"corpus": {"max_files": 50}}``."""
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects dotted.key=value, got {item!r}")
        value = yaml.safe_load(raw) if raw.strip() else None
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"--set key {key!r} conflicts with another --set value")
        node[parts[-1]] = value
    return result


def deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base``."""
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _format_validation_error(exc: ValidationError, yaml_path: Path | None) -> str:
    where = f" (config file: {yaml_path})" if yaml_path else ""
    lines = [f"Invalid configuration{where}:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
