"""Exact-match (SHA-256) disk cache — embeddings and LLM responses. NEVER semantic.

Two logical namespaces backed by `diskcache` under `config.cache_dir`:

- **embeddings** (ON by default): identical text -> identical vector across runs, so retrieval
  stays deterministic and rebuilds don't re-embed.
- **llm** (OFF by default): the headline experiment measures inter-run agreement across k
  INDEPENDENT calls of the SAME prompt. A warm LLM cache would replay run 1 for runs 2..k and
  force a trivial 100% agreement, silently invalidating the result. It is enabled only for the
  replay / cache-warm ablation.

The cache key is the SHA-256 of the canonical JSON of a payload dict (`sort_keys=True`, compact
separators) so it is identical across runs and machines. There is deliberately no similarity /
cosine / nearest-neighbour lookup surface anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.config import Config
from src.index import Embedder

Messages = list[dict[str, Any]]


# --- canonical key + payloads -----------------------------------------------------------------


def canonical_key(payload: dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of a payload's canonical JSON serialization."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def embedding_payload(text: str, config: Config) -> dict[str, Any]:
    """Canonical payload identifying one embedding request (pinned model + dimensions + text)."""
    return {
        "kind": "embedding",
        "model": config.embedding_model,
        "dimensions": config.embedding_dimensions,
        "text": text,
    }


def llm_payload(messages: Messages, schema_sha256: str | None, config: Config) -> dict[str, Any]:
    """Canonical payload identifying one LLM call (pinned model + decoding params + messages)."""
    return {
        "kind": "llm",
        "model": config.llm_model,
        "params": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.llm_seed,
        },
        "messages": messages,
        "schema_sha256": schema_sha256,
    }


def embedding_key(text: str, config: Config) -> str:
    """SHA-256 cache key for an embedding request."""
    return canonical_key(embedding_payload(text, config))


def llm_key(messages: Messages, schema_sha256: str | None, config: Config) -> str:
    """SHA-256 cache key for an LLM call."""
    return canonical_key(llm_payload(messages, schema_sha256, config))


# --- diskcache stores -------------------------------------------------------------------------


def open_embedding_store(config: Config) -> Any:
    """Open the `embeddings` diskcache namespace under `config.cache_dir`."""
    import diskcache

    return diskcache.Cache(str(config.cache_dir / "embeddings"))


def open_llm_store(config: Config) -> Any:
    """Open the `llm` diskcache namespace under `config.cache_dir`."""
    import diskcache

    return diskcache.Cache(str(config.cache_dir / "llm"))


# --- embedding cache (wrapping embedder) ------------------------------------------------------


class CachingEmbedder:
    """Wrap any `Embedder`, serving exact-match cached vectors and embedding only on misses.

    Satisfies the `Embedder` protocol, so it is a drop-in for `OpenAIEmbedder`. Returns vectors in
    the original input order. Keys on the exact text passed in (data.py already normalizes it).
    """

    def __init__(
        self,
        embedder: Embedder,
        config: Config,
        *,
        store: Any | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._inner = embedder
        self._config = config
        self._enabled = config.embedding_cache_enabled if enabled is None else enabled
        self._store = open_embedding_store(config) if store is None else store

    @property
    def enabled(self) -> bool:
        return self._enabled

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._enabled:
            return self._inner.embed_documents(texts)

        keys = [embedding_key(text, self._config) for text in texts]
        results: list[list[float] | None] = [self._store.get(key, default=None) for key in keys]

        miss_indices = [i for i, vec in enumerate(results) if vec is None]
        if miss_indices:
            computed = self._inner.embed_documents([texts[i] for i in miss_indices])
            for i, vector in zip(miss_indices, computed, strict=True):
                self._store.set(keys[i], vector)
                results[i] = vector

        return [list(vector) for vector in results]  # all misses are now filled

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# --- LLM-response cache (disabled by default) -------------------------------------------------


class LLMCache:
    """Exact-match LLM-response cache, gated OFF by default (`config.llm_cache_enabled`).

    When disabled, lookups always miss and writes no-op, so headline (cold) runs never replay or
    persist responses. `get_llm` returning a non-None value means the lookup hit the cache — that
    boolean is what provenance records as `cache_hit`.
    """

    def __init__(
        self,
        config: Config,
        *,
        store: Any | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._config = config
        self._enabled = config.llm_cache_enabled if enabled is None else enabled
        self._store = open_llm_store(config) if store is None else store

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_llm(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return the cached response for `payload`, or None on a miss / when disabled."""
        if not self._enabled:
            return None
        return self._store.get(canonical_key(payload), default=None)

    def set_llm(self, payload: dict[str, Any], response: dict[str, Any]) -> None:
        """Store a response under its exact-match key. No-op when the cache is disabled."""
        if not self._enabled:
            return
        self._store.set(canonical_key(payload), response)
