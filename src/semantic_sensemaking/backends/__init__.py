"""Embedding and chat backends behind shared protocols.

Vendor SDKs are imported only when a backend is constructed, so importing this
package never requires the ``embed`` or ``llm`` extras.
"""

from semantic_sensemaking.backends.anthropic_claude import AnthropicClaudeBackend
from semantic_sensemaking.backends.base import (
    ChatBackend,
    ChatResponse,
    EmbeddingBackend,
    Usage,
)
from semantic_sensemaking.backends.cache import CachedEmbeddingBackend
from semantic_sensemaking.backends.factory import get_chat_backend, get_embedding_backend
from semantic_sensemaking.backends.hashing import DeterministicHashEmbedding
from semantic_sensemaking.backends.huggingface_local import HuggingFaceLocalEmbedding
from semantic_sensemaking.backends.openai_compatible import OpenAICompatibleBackend

__all__ = [
    "AnthropicClaudeBackend",
    "CachedEmbeddingBackend",
    "ChatBackend",
    "ChatResponse",
    "DeterministicHashEmbedding",
    "EmbeddingBackend",
    "HuggingFaceLocalEmbedding",
    "OpenAICompatibleBackend",
    "Usage",
    "get_chat_backend",
    "get_embedding_backend",
]
