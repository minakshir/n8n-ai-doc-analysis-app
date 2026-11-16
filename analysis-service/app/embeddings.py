"""Text embeddings from the local Ollama server."""

import json
import urllib.error
import urllib.request

import numpy as np

from .errors import ServiceError

BATCH = 32
# Task prefixes the model was trained with; without them retrieval quality drops sharply.
PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float = 120):
        self.url = base_url.rstrip("/") + "/api/embed"
        self.model = model
        self.timeout = timeout
        self.doc_prefix, self.query_prefix = PREFIXES.get(model.split(":")[0], ("", ""))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        parts = [self._embed([self.doc_prefix + t for t in texts[i:i + BATCH]])
                 for i in range(0, len(texts), BATCH)]
        return np.vstack(parts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]

    def _embed(self, inputs: list[str]) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": inputs, "truncate": True}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                vectors = json.load(r).get("embeddings")
        except urllib.error.HTTPError as e:
            detail = e.read(300).decode(errors="replace")
            raise ServiceError("EMBEDDINGS_UNAVAILABLE", f"The embedding model returned HTTP {e.code}.",
                               hint=f"Is '{self.model}' installed in Ollama? {detail}", status=503)
        except (urllib.error.URLError, TimeoutError) as e:
            raise ServiceError("EMBEDDINGS_UNAVAILABLE", "The embedding server is unreachable.",
                               hint=f"Check that the ollama container is running ({e}).", status=503)
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise ServiceError("EMBEDDINGS_UNAVAILABLE", "The embedding server returned an unexpected response.",
                               status=503)
        return normalize(np.asarray(vectors, dtype=np.float32))


def normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
