"""Turn text into vectors.

`OpenAIEmbedder` is the real one. `FakeEmbedder` needs no network and no key, so
tests and demos can run offline. It only matches shared words, so it is much
weaker than a real model and should never be used to judge quality.
"""

import hashlib
import math
import re
from typing import Protocol

from openai import OpenAI


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    BATCH = 96

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        # max_retries: free tiers have low rate limits, so wait and retry on 429 errors.
        self._client = OpenAI(api_key=api_key, base_url=base_url or None, max_retries=5)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors


class FakeEmbedder:
    """Hashed bag-of-words vectors. Same words in -> similar vectors out."""

    def __init__(self, dim: int = 512):
        self._dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) < 3:
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
