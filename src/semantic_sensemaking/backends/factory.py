"""Build backends from :class:`~semantic_sensemaking.core.config.BackendConfig`."""

from __future__ import annotations

import os

from semantic_sensemaking.backends.base import ChatBackend, EmbeddingBackend
from semantic_sensemaking.core.config import BackendConfig
from semantic_sensemaking.errors import ConfigError

_DEFAULT_KEY_ENV = {"openai_compatible": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def get_embedding_backend(config: BackendConfig) -> EmbeddingBackend | None:
    """Return the configured embedding backend, or ``None`` for ``kind: none``."""
    backend: EmbeddingBackend
    if config.kind == "none":
        return None
    if config.kind == "hashing":
        from semantic_sensemaking.backends.hashing import DeterministicHashEmbedding

        backend = DeterministicHashEmbedding(config.dimensions)
    elif config.kind == "openai_compatible":
        from semantic_sensemaking.backends.openai_compatible import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(
            _require(config, "base_url"),
            _require(config, "model"),
            api_key=_api_key(config),
            timeout_s=config.timeout_s,
            batch_size=config.batch_size,
            max_retries=config.max_retries,
        )
    elif config.kind == "huggingface_local":
        from semantic_sensemaking.backends.huggingface_local import (
            DEFAULT_MODEL,
            HuggingFaceLocalEmbedding,
        )

        backend = HuggingFaceLocalEmbedding(
            config.model or DEFAULT_MODEL, device=config.device, batch_size=config.batch_size
        )
    else:
        raise ConfigError(
            f"Backend kind '{config.kind}' does not provide embeddings; "
            "use openai_compatible, huggingface_local, or hashing"
        )

    if config.cache_dir is not None:
        from semantic_sensemaking.backends.cache import CachedEmbeddingBackend

        return CachedEmbeddingBackend(backend, config.cache_dir)
    return backend


def get_chat_backend(config: BackendConfig) -> ChatBackend | None:
    """Return the configured chat backend, or ``None`` for ``kind: none``."""
    if config.kind == "none":
        return None
    if config.kind == "openai_compatible":
        from semantic_sensemaking.backends.openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(
            _require(config, "base_url"),
            _require(config, "model"),
            api_key=_api_key(config),
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
        )
    if config.kind == "anthropic":
        from semantic_sensemaking.backends.anthropic_claude import (
            DEFAULT_MODEL,
            AnthropicClaudeBackend,
        )

        return AnthropicClaudeBackend(
            config.model or DEFAULT_MODEL,
            api_key=_api_key(config),
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
        )
    raise ConfigError(
        f"Backend kind '{config.kind}' does not provide chat; use openai_compatible or anthropic"
    )


def _require(config: BackendConfig, field: str) -> str:
    value = getattr(config, field)
    if not value:
        raise ConfigError(f"Backend kind '{config.kind}' requires '{field}'")
    return value


def _api_key(config: BackendConfig) -> str | None:
    env = config.api_key_env or _DEFAULT_KEY_ENV.get(config.kind)
    return os.environ.get(env) if env else None
