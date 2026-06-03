from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from marusya_qa.config import RetrieverConfig
from marusya_qa.device import resolve_device


class DenseEncoder:
    def __init__(self, config: RetrieverConfig, device: str | None = None) -> None:
        self.config = config
        self.device = resolve_device(device)
        self.model = SentenceTransformer(config.model_name_or_path, device=self.device)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self._encode(texts, self.config.query_prefix, batch_size)

    def encode_passages(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self._encode(texts, self.config.passage_prefix, batch_size)

    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int) -> np.ndarray:
        prepared = [f"{prefix}{text}" for text in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
