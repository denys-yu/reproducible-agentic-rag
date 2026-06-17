"""Offline tests for src.cache: key determinism, round-trips, embedding-miss behaviour,

the LLM cache being cold by default, and the absence of any semantic-cache surface. No API calls.
"""

from __future__ import annotations

import src.cache as cache_module
from src.cache import (
    CachingEmbedder,
    LLMCache,
    canonical_key,
    embedding_payload,
    llm_payload,
)
from src.config import Config


class CountingEmbedder:
    """Fake embedder that records how many texts it was asked to embed."""

    def __init__(self) -> None:
        self.calls = 0
        self.embedded_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.embedded_texts.extend(texts)
        # Deterministic vector derived from text length, distinct enough for assertions.
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _config(tmp_path, **overrides) -> Config:
    return Config(cache_dir=tmp_path, **overrides)


# --- key determinism + sensitivity ------------------------------------------------------------


def test_canonical_key_is_deterministic():
    payload = {"kind": "llm", "b": 2, "a": 1}
    assert canonical_key(payload) == canonical_key({"a": 1, "b": 2, "kind": "llm"})  # order-independent
    assert len(canonical_key(payload)) == 64


def test_embedding_key_changes_with_any_field(tmp_path):
    cfg = _config(tmp_path)
    base = canonical_key(embedding_payload("hello", cfg))
    assert base != canonical_key(embedding_payload("hello!", cfg))  # text
    assert base != canonical_key(embedding_payload("hello", _config(tmp_path, embedding_model="x")))
    assert base != canonical_key(
        embedding_payload("hello", _config(tmp_path, embedding_dimensions=256))
    )


def test_llm_key_changes_with_any_field(tmp_path):
    cfg = _config(tmp_path)
    messages = [{"role": "user", "content": "hi"}]
    base = canonical_key(llm_payload(messages, "sha", cfg))
    assert base != canonical_key(llm_payload([{"role": "user", "content": "bye"}], "sha", cfg))
    assert base != canonical_key(llm_payload(messages, "other-sha", cfg))
    assert base != canonical_key(llm_payload(messages, None, cfg))
    assert base != canonical_key(llm_payload(messages, "sha", _config(tmp_path, llm_seed=7)))
    assert base != canonical_key(llm_payload(messages, "sha", _config(tmp_path, temperature=0.5)))


# --- embedding cache round-trip + miss-only computation ---------------------------------------


def test_caching_embedder_only_computes_on_misses(tmp_path):
    cfg = _config(tmp_path)
    inner = CountingEmbedder()
    embedder = CachingEmbedder(inner, cfg)

    first = embedder.embed_documents(["alpha", "beta"])
    assert inner.calls == 1
    assert inner.embedded_texts == ["alpha", "beta"]

    # Same texts again: served from cache, underlying embedder not called.
    second = embedder.embed_documents(["alpha", "beta"])
    assert second == first
    assert inner.calls == 1  # no new call

    # New text mixed with cached ones: only the miss is embedded, order preserved.
    third = embedder.embed_documents(["alpha", "gamma", "beta"])
    assert inner.calls == 2
    assert inner.embedded_texts[-1:] == ["gamma"]  # only "gamma" was computed
    assert third[0] == first[0] and third[2] == first[1]


def test_caching_embedder_identical_text_identical_vector_across_instances(tmp_path):
    cfg = _config(tmp_path)
    v1 = CachingEmbedder(CountingEmbedder(), cfg).embed_query("same text")
    # A fresh wrapper over a fresh underlying embedder must return the cached vector.
    fresh_inner = CountingEmbedder()
    v2 = CachingEmbedder(fresh_inner, cfg).embed_query("same text")
    assert v1 == v2
    assert fresh_inner.calls == 0  # served entirely from disk cache


def test_caching_embedder_disabled_bypasses_store(tmp_path):
    cfg = _config(tmp_path, embedding_cache_enabled=False)
    inner = CountingEmbedder()
    embedder = CachingEmbedder(inner, cfg)
    embedder.embed_documents(["x"])
    embedder.embed_documents(["x"])
    assert inner.calls == 2  # no caching: every call hits the underlying embedder


# --- LLM cache: cold by default ---------------------------------------------------------------


def test_llm_cache_disabled_by_default_misses_and_does_not_persist(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.llm_cache_enabled is False
    llm = LLMCache(cfg)
    assert llm.enabled is False

    payload = llm_payload([{"role": "user", "content": "q"}], "sha", cfg)
    llm.set_llm(payload, {"answer": "cached"})  # no-op while disabled
    assert llm.get_llm(payload) is None  # cold: always a miss


def test_llm_cache_enabled_round_trip(tmp_path):
    cfg = _config(tmp_path, llm_cache_enabled=True)
    llm = LLMCache(cfg)
    assert llm.enabled is True

    payload = llm_payload([{"role": "user", "content": "q"}], "sha", cfg)
    assert llm.get_llm(payload) is None  # initial miss
    llm.set_llm(payload, {"answer": "42"})
    assert llm.get_llm(payload) == {"answer": "42"}  # hit


# --- no semantic-cache surface ----------------------------------------------------------------


def test_no_semantic_cache_surface():
    forbidden = ("semantic", "similar", "cosine", "nearest", "knn", "approximate")
    for name in dir(cache_module):
        lowered = name.lower()
        assert not any(token in lowered for token in forbidden), f"forbidden cache surface: {name}"
