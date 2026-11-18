"""Backend for any OpenAI-compatible HTTP API: vLLM or Ollama on TACC, or a hosted service.

Talks to ``/embeddings`` and ``/chat/completions`` directly with ``httpx`` so
the TACC image needs no vendor SDK to reach a model served on a compute node.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import numpy as np
from pydantic import ValidationError

from semantic_sensemaking.backends.base import (
    ChatResponse,
    EmbedKind,
    ModelT,
    Usage,
    json_instruction,
    l2_normalize,
    parse_json_model,
)
from semantic_sensemaking.errors import BackendError

log = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleBackend:
    """Embeddings and chat against ``base_url`` (e.g. ``http://c123-456:8000/v1``)."""

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        batch_size: int = 32,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s, transport=transport
        )

    def embed(self, texts: Sequence[str], *, kind: EmbedKind = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            payload = self._post("/embeddings", {"model": self.model, "input": batch})
            try:
                data = sorted(payload["data"], key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in data)
            except (KeyError, TypeError) as exc:
                raise BackendError(f"Unexpected embeddings response: {payload!r:.300}") from exc
            if len(data) != len(batch):
                raise BackendError(f"Asked for {len(batch)} embeddings, received {len(data)}")
        return l2_normalize(vectors)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ChatResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        if stop:
            body["stop"] = list(stop)
        body.update(extra or {})
        payload = self._post("/chat/completions", body)
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"Unexpected chat response: {payload!r:.300}") from exc
        usage = payload.get("usage") or {}
        return ChatResponse(
            text=text,
            model=payload.get("model", self.model),
            usage=Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            stop_reason=choice.get("finish_reason"),
        )

    def complete_json(
        self,
        prompt: str,
        *,
        schema: type[ModelT],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> ModelT:
        """Request schema-constrained JSON; re-ask with the validation error on failure.

        ``response_format`` with a JSON schema is honored by vLLM and recent Ollama.
        Servers that ignore it still get the schema in the system prompt, and the
        output is validated either way.
        """
        system_text = "\n\n".join(filter(None, [system, json_instruction(schema)]))
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
        }
        attempt_prompt = prompt
        for attempt in range(self.max_retries + 1):
            response = self.complete(
                attempt_prompt,
                system=system_text,
                max_tokens=max_tokens,
                extra={"response_format": response_format},
            )
            try:
                return parse_json_model(response.text, schema)
            except ValidationError as exc:
                if attempt == self.max_retries:
                    raise BackendError(
                        f"Model output did not match {schema.__name__} after "
                        f"{self.max_retries + 1} attempts: {exc}"
                    ) from exc
                attempt_prompt = (
                    f"{prompt}\n\nYour previous reply was not valid for the schema:\n{exc}\n"
                    "Reply again with only the corrected JSON object."
                )
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._client.close()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(path, json=body)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise BackendError(f"POST {path} failed: {exc}") from exc
                self._backoff(attempt, f"transport error: {exc}")
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                self._backoff(attempt, f"HTTP {response.status_code}")
                continue
            if response.is_error:
                raise BackendError(
                    f"POST {path} returned HTTP {response.status_code}: {response.text[:500]}"
                )
            return response.json()
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int, reason: str) -> None:
        delay = min(2.0**attempt, 30.0)
        log.warning("Retrying after %s (attempt %d, sleeping %.1fs)", reason, attempt + 1, delay)
        self._sleep(delay)
