from __future__ import annotations

import json

import httpx
import numpy as np
import pytest
from pydantic import BaseModel

from semantic_sensemaking.backends import (
    CachedEmbeddingBackend,
    ChatBackend,
    DeterministicHashEmbedding,
    EmbeddingBackend,
    OpenAICompatibleBackend,
    get_chat_backend,
    get_embedding_backend,
)
from semantic_sensemaking.core.config import BackendConfig
from semantic_sensemaking.errors import BackendError, ConfigError, MissingExtraError


def test_hashing_embedding_is_deterministic_normalized_and_word_sensitive():
    backend = DeterministicHashEmbedding(64)
    a, b, c = backend.embed(["aquifer drawdown", "aquifer drawdown", "tidal estuary"])
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)
    related = backend.embed(["aquifer drawdown near the well"])[0]
    assert a @ related > a @ c
    assert isinstance(backend, EmbeddingBackend)


class Recorder:
    """httpx.MockTransport handler that replays scripted responses and records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        return response(request) if callable(response) else response


def embeddings_response(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(texts)]
    return httpx.Response(200, json={"data": list(reversed(data))})


def make_backend(recorder: Recorder, sleeps: list[float] | None = None, **kwargs) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        "http://node:8000/v1/",
        "test-model",
        transport=httpx.MockTransport(recorder),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
        **kwargs,
    )


def test_openai_embeddings_batch_reorder_and_normalize():
    recorder = Recorder(*[embeddings_response] * 3)
    backend = make_backend(recorder, batch_size=2, api_key="secret")
    vectors = backend.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert len(recorder.requests) == 3
    assert recorder.requests[0].url == "http://node:8000/v1/embeddings"
    assert recorder.requests[0].headers["authorization"] == "Bearer secret"
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors[1] @ np.array([2.0, 1.0]) / np.sqrt(5) == pytest.approx(1.0)


def test_openai_retries_transient_errors_then_raises_on_client_errors():
    sleeps: list[float] = []
    recorder = Recorder(httpx.Response(503), httpx.Response(429), embeddings_response)
    assert make_backend(recorder, sleeps).embed(["x"]).shape == (1, 2)
    assert sleeps == [1.0, 2.0]

    recorder = Recorder(httpx.Response(400, text="bad model"))
    with pytest.raises(BackendError, match="HTTP 400: bad model"):
        make_backend(recorder).embed(["x"])
    assert len(recorder.requests) == 1


def test_openai_gives_up_after_max_retries():
    recorder = Recorder(*[httpx.Response(500)] * 2)
    with pytest.raises(BackendError, match="HTTP 500"):
        make_backend(recorder, max_retries=1).embed(["x"])


def chat_reply(text: str):
    return httpx.Response(
        200,
        json={
            "model": "served-model",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )


def test_openai_chat_request_shape_and_response():
    recorder = Recorder(chat_reply("hello"))
    backend = make_backend(recorder)
    response = backend.complete("hi", system="be brief", stop=["\n"])
    body = json.loads(recorder.requests[0].content)
    assert body["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    assert body["stop"] == ["\n"] and "temperature" not in body
    assert (response.text, response.model, response.usage.input_tokens, response.stop_reason) == (
        "hello", "served-model", 7, "stop",
    )
    assert isinstance(backend, ChatBackend)


class Frame(BaseModel):
    label: str
    terms: list[str]


def test_openai_complete_json_reasks_after_invalid_output():
    recorder = Recorder(chat_reply("not json"), chat_reply('```json\n{"label": "Drought", "terms": ["dry"]}\n```'))
    result = make_backend(recorder).complete_json("frame it", schema=Frame)
    assert result == Frame(label="Drought", terms=["dry"])
    second = json.loads(recorder.requests[1].content)
    assert "not valid for the schema" in second["messages"][-1]["content"]
    assert second["response_format"]["json_schema"]["name"] == "Frame"


class CountingBackend(DeterministicHashEmbedding):
    def __init__(self):
        super().__init__(16)
        self.calls: list[list[str]] = []

    def embed(self, texts, *, kind="document"):
        self.calls.append(list(texts))
        return super().embed(texts, kind=kind)


def test_cache_only_embeds_new_texts(tmp_path):
    inner = CountingBackend()
    cached = CachedEmbeddingBackend(inner, tmp_path)
    first = cached.embed(["a", "b"])
    second = cached.embed(["b", "c", "a"])
    assert inner.calls == [["a", "b"], ["c"]]
    assert np.allclose(second[0], first[1]) and np.allclose(second[2], first[0])
    cached.embed(["a"], kind="query")
    assert inner.calls[-1] == ["a"]  # kind is part of the key
    cached.close()


def test_factory(tmp_path):
    assert get_embedding_backend(BackendConfig()) is None
    assert get_chat_backend(BackendConfig()) is None
    assert isinstance(get_embedding_backend(BackendConfig(kind="hashing")), DeterministicHashEmbedding)
    assert isinstance(
        get_embedding_backend(BackendConfig(kind="hashing", cache_dir=tmp_path)), CachedEmbeddingBackend
    )
    with pytest.raises(ConfigError, match="does not provide embeddings"):
        get_embedding_backend(BackendConfig(kind="anthropic"))
    with pytest.raises(ConfigError, match="requires 'base_url'"):
        get_embedding_backend(BackendConfig(kind="openai_compatible", model="m"))
    with pytest.raises(ConfigError, match="does not provide chat"):
        get_chat_backend(BackendConfig(kind="hashing"))


def test_missing_sdk_extras_name_the_install_command():
    pytest.importorskip("numpy")
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(MissingExtraError, match=r"semantic-sensemaking\[embed\]"):
            get_embedding_backend(BackendConfig(kind="huggingface_local"))
    try:
        import anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(MissingExtraError, match=r"semantic-sensemaking\[llm\]"):
            get_chat_backend(BackendConfig(kind="anthropic"))
