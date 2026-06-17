"""Pydantic v2 enum schemas for the structured-output (`enum`) treatment arm.

The baseline (`free`) arm returns plain strings and needs nothing here. This module defines the
strict structured-output model(s) returned by every agent step in the `enum` arm, one per
`SchemaVariant` declared in `config.py`. Config currently declares a single variant
(`ANSWER_V1`), which is the fullest / headline schema:

    answer: str
    confidence: ConfidenceLevel  {high, medium, low}
    scope: AnswerScope           {full, partial, none}
    supporting_doc_ids: list[str]

The agent's non-final steps reuse the same enums via `ContextGrade` (grade) and `RewriteQuery`
(rewrite); `response_schema_for_node` maps each step to its model.

All fields are required with no defaults. `to_openai_strict_schema` renders the canonical
OpenAI strict-mode JSON schema (the one we log and verify), and `schema_sha256` hashes it
deterministically for the provenance `schema_sha256` field. No API calls happen here.
"""

from __future__ import annotations

import copy
import enum
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.config import SchemaVariant

# Same node identifiers used by the provenance log; kept local to avoid cross-module coupling.
NodeName = Literal["grade", "rewrite", "synthesize"]


class ConfidenceLevel(str, enum.Enum):
    """How confident the step is in its answer."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnswerScope(str, enum.Enum):
    """How much of the question the answer covers."""

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class AnswerV1(BaseModel):
    """Headline treatment-arm response: answer + enum judgements + supporting doc ids.

    `extra="forbid"` mirrors OpenAI strict mode's `additionalProperties: false` and makes the
    model reject any field the LLM hallucinates beyond the schema.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: ConfidenceLevel
    scope: AnswerScope
    supporting_doc_ids: list[str]


class ContextGrade(BaseModel):
    """Grade-step judgement on the retrieved context (reuses the shared enums).

    `scope` = how well the retrieved context covers the question; `confidence` = confidence in
    that judgement; `needs_more_context` = whether to rewrite the query and retrieve again.
    """

    model_config = ConfigDict(extra="forbid")

    scope: AnswerScope
    confidence: ConfidenceLevel
    needs_more_context: bool


class RewriteQuery(BaseModel):
    """Rewrite-step output: the reformulated search query."""

    model_config = ConfigDict(extra="forbid")

    query: str


# One model per declared SchemaVariant. The fullest variant maps to the headline model.
_SCHEMA_REGISTRY: dict[SchemaVariant, type[BaseModel]] = {
    SchemaVariant.ANSWER_V1: AnswerV1,
}


def get_response_schema(variant: SchemaVariant) -> type[BaseModel]:
    """Return the Pydantic model class for the given schema variant (the `enum`-arm response)."""
    try:
        return _SCHEMA_REGISTRY[variant]
    except KeyError as exc:  # fail loud on an unregistered variant
        raise ValueError(f"No response schema registered for variant {variant!r}") from exc


def response_schema_for_node(node: NodeName, variant: SchemaVariant) -> type[BaseModel]:
    """Return the enum-arm structured-output model for a given agent step.

    grade -> ContextGrade, rewrite -> RewriteQuery, synthesize -> the variant's response model.
    Fails loud on an unknown node.
    """
    if node == "grade":
        return ContextGrade
    if node == "rewrite":
        return RewriteQuery
    if node == "synthesize":
        return get_response_schema(variant)
    raise ValueError(f"Unknown agent node {node!r}")


def _is_object_node(node: dict[str, Any]) -> bool:
    """A JSON-schema node that describes an object (and thus needs strict-mode fixups)."""
    return node.get("type") == "object" or "properties" in node


def _enforce_strict(node: Any) -> None:
    """Recursively coerce a JSON-schema tree into OpenAI strict form, in place.

    On every object node: set `additionalProperties=false` and make `required` list every
    property (sorted for a canonical, machine-independent ordering). Strip any `default` keys
    (strict mode forbids them; our models declare none).
    """
    if isinstance(node, dict):
        node.pop("default", None)
        if _is_object_node(node):
            properties = node.get("properties", {})
            node["additionalProperties"] = False
            node["required"] = sorted(properties.keys())
        for value in node.values():
            _enforce_strict(value)
    elif isinstance(node, list):
        for item in node:
            _enforce_strict(item)


def to_openai_strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Render a Pydantic model as an OpenAI strict-mode JSON schema.

    Guarantees: `additionalProperties=false` on every object (including under `$defs`), every
    property present in `required`, no `default`s, and no root `oneOf`. This is the canonical
    schema logged for provenance and checked by `assert_strict_compliant`.
    """
    json_schema = copy.deepcopy(schema.model_json_schema())
    _enforce_strict(json_schema)
    return json_schema


def schema_sha256(schema: type[BaseModel]) -> str:
    """Return the SHA-256 of the canonical strict schema, hashed deterministically.

    Serialized with `sort_keys=True` and compact separators so the digest is identical across
    runs and machines — this is the `schema_sha256` recorded in the provenance log.
    """
    canonical = json.dumps(
        to_openai_strict_schema(schema),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_strict_compliant(strict_schema: dict[str, Any]) -> None:
    """Assert a rendered schema satisfies OpenAI strict mode; raise AssertionError otherwise.

    Checks: no root `oneOf`; every object node has `additionalProperties is False` and a
    `required` set equal to its property set; no `default` keys anywhere.
    """
    if "oneOf" in strict_schema:
        raise AssertionError("strict schema must not use `oneOf` at the root")

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if "default" in node:
                raise AssertionError(f"`default` is not allowed in strict mode (at {path})")
            if _is_object_node(node):
                if node.get("additionalProperties") is not False:
                    raise AssertionError(f"`additionalProperties` must be false (at {path})")
                properties = set(node.get("properties", {}).keys())
                required = set(node.get("required", []))
                if properties != required:
                    raise AssertionError(
                        f"`required` must cover every property (at {path}): "
                        f"properties={sorted(properties)} required={sorted(required)}"
                    )
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(strict_schema, "$")
