"""SHA-256 exact-match LLM response cache (diskcache).

Exact-match ONLY. Cache key = SHA-256 of canonical JSON
`{model, model_kwargs, system_prompt, user_prompt, schema_hash}`. Never implement semantic /
cosine-similarity caching — it introduces non-determinism and contradicts the paper's central
claim. Headline runs use a COLD cache; warming is a separate ablation.
"""

from __future__ import annotations

from typing import Any

from src.config import Config


def cache_key(
    model: str,
    model_kwargs: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    schema_hash: str,
) -> str:
    """Return the SHA-256 hex digest of the canonical-JSON cache key."""
    raise NotImplementedError


def get_cache(config: Config) -> Any:
    """Open the diskcache store rooted at `config.cache_dir`."""
    raise NotImplementedError


def cache_get(cache: Any, key: str) -> dict[str, Any] | None:
    """Return the cached response for `key`, or None on a miss."""
    raise NotImplementedError


def cache_set(cache: Any, key: str, value: dict[str, Any]) -> None:
    """Store a response under its exact-match key."""
    raise NotImplementedError
