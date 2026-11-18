"""Claude chat backend (``pip install 'semantic-sensemaking[llm]'``).

Anthropic does not offer an embeddings endpoint; pair this with the
OpenAI-compatible or Hugging Face backend for embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from semantic_sensemaking.backends.base import ChatResponse, ModelT, Usage
from semantic_sensemaking.errors import BackendError, optional_import

DEFAULT_MODEL = "claude-opus-5"

# Server-side fallback re-runs a request declined by a safety classifier on a
# model Anthropic picks by refusal category, instead of returning the refusal.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5-1")


class AnthropicClaudeBackend:
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        max_retries: int = 2,
        server_side_fallback: bool = True,
    ) -> None:
        anthropic = optional_import("anthropic", "llm", "The Claude chat backend")
        self.model = model
        self._anthropic = anthropic
        # With api_key=None the SDK resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile.
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=max_retries
        )
        self._fallback = server_side_fallback and model in _FALLBACK_MODELS

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
        stop: Sequence[str] | None = None,
    ) -> ChatResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            params["system"] = system
        if stop:
            params["stop_sequences"] = list(stop)
        # Current Claude models reject non-default sampling parameters; only send
        # temperature when a caller deliberately sets it (for older models).
        if temperature is not None:
            params["temperature"] = temperature

        try:
            if self._fallback:
                response = self._client.beta.messages.create(
                    betas=[_FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = self._client.messages.create(**params)
        except self._anthropic.APIError as exc:
            raise BackendError(f"Claude API request failed: {exc}") from exc

        _raise_on_refusal(response)
        text = "".join(block.text for block in response.content if block.type == "text")
        return ChatResponse(
            text=text,
            model=response.model,
            usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
            stop_reason=response.stop_reason,
        )

    def complete_json(
        self,
        prompt: str,
        *,
        schema: type[ModelT],
        system: str | None = None,
        max_tokens: int = 16000,
    ) -> ModelT:
        """Structured output validated against ``schema`` by the SDK's ``messages.parse``."""
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_format": schema,
        }
        if system:
            params["system"] = system
        try:
            response = self._client.messages.parse(**params)
        except self._anthropic.APIError as exc:
            raise BackendError(f"Claude API request failed: {exc}") from exc
        _raise_on_refusal(response)
        if response.parsed_output is None:
            raise BackendError(f"Claude returned no parsable {schema.__name__} output")
        return response.parsed_output

    def close(self) -> None:
        self._client.close()


def _raise_on_refusal(response: Any) -> None:
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise BackendError(f"Claude declined the request (refusal category: {category})")
