"""Backend protocols shared by every workflow.

Workflows depend only on these protocols, never on a vendor SDK, so the same
pipeline runs against a vLLM/Ollama endpoint on a TACC node, a local Hugging
Face model, or the Claude API.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

import numpy as np
from pydantic import BaseModel, ValidationError

from semantic_sensemaking.errors import BackendError

ModelT = TypeVar("ModelT", bound=BaseModel)
EmbedKind = Literal["document", "query"]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ChatResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    model: str

    def embed(self, texts: Sequence[str], *, kind: EmbedKind = "document") -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 array of L2-normalized vectors."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class ChatBackend(Protocol):
    name: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
    ) -> ChatResponse:
        """Single-turn completion. ``temperature=None`` leaves the provider default."""
        ...

    def complete_json(
        self,
        prompt: str,
        *,
        schema: type[ModelT],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> ModelT:
        """Completion parsed and validated into ``schema``."""
        ...

    def close(self) -> None: ...


def l2_normalize(vectors: Any) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise BackendError(f"Expected a 2-D embedding array, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norms == 0, 1.0, norms)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_model(text: str, schema: type[ModelT]) -> ModelT:
    """Validate model output as ``schema``, tolerating code fences and surrounding prose."""
    candidate = _FENCE.sub("", text.strip())
    try:
        return schema.model_validate_json(candidate)
    except ValidationError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise
        return schema.model_validate_json(candidate[start : end + 1])


def json_instruction(schema: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object and nothing else. It must validate against "
        f"this JSON Schema:\n{json.dumps(schema.model_json_schema())}"
    )
