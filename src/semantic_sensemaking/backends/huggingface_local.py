"""Local sentence-transformers embeddings (``pip install 'semantic-sensemaking[embed]'``).

Pulls in PyTorch, so it is kept out of the TACC image; on TACC, serve the model
with vLLM and use the OpenAI-compatible backend instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from semantic_sensemaking.backends.base import EmbedKind, l2_normalize
from semantic_sensemaking.errors import optional_import

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HuggingFaceLocalEmbedding:
    name = "huggingface_local"

    def __init__(
        self, model: str = DEFAULT_MODEL, *, device: str | None = None, batch_size: int = 32
    ) -> None:
        st = optional_import("sentence_transformers", "embed", "Local Hugging Face embeddings")
        self.model = model
        self.batch_size = batch_size
        self._model = st.SentenceTransformer(model, device=device)

    def embed(self, texts: Sequence[str], *, kind: EmbedKind = "document") -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(vectors)

    def close(self) -> None:
        pass
