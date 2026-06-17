"""Pydantic v2 enum schemas for the structured-output (`enum`) arm.

Defines the strict structured-output models returned by every agent step in the treatment arm:
`answer: str`, `confidence: ConfidenceLevel {high, medium, low}`, `scope: AnswerScope
{full, partial, none}`, `supporting_doc_ids: list[str]`. OpenAI strict mode requires
`additionalProperties: false`, all fields required, no root `oneOf`, and a small schema (< 30
fields). `schema_sha256` provides the stable hash logged in provenance.
"""

from __future__ import annotations

from typing import Any

from src.config import SchemaVariant


def get_response_schema(variant: SchemaVariant) -> type:
    """Return the Pydantic model class for the given schema variant (the `enum`-arm response)."""
    raise NotImplementedError


def schema_sha256(schema: type) -> str:
    """Return the SHA-256 hex digest of a model's canonical JSON schema."""
    raise NotImplementedError


def to_openai_strict_schema(schema: type) -> dict[str, Any]:
    """Render a Pydantic model as an OpenAI strict structured-output JSON schema."""
    raise NotImplementedError
