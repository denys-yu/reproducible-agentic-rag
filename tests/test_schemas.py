"""Offline tests for src.schemas: strict-mode compliance, enums, and deterministic hashing.

No API calls — these only exercise the Pydantic models and the strict-schema renderer.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from src.config import SchemaVariant
from src.schemas import (
    AnswerScope,
    AnswerV1,
    ConfidenceLevel,
    ContextGrade,
    RewriteQuery,
    assert_strict_compliant,
    get_response_schema,
    response_schema_for_node,
    schema_sha256,
    to_openai_strict_schema,
)


def _all_object_nodes(node, acc=None):
    """Collect every object-typed node in a JSON-schema tree."""
    acc = [] if acc is None else acc
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            acc.append(node)
        for value in node.values():
            _all_object_nodes(value, acc)
    elif isinstance(node, list):
        for item in node:
            _all_object_nodes(item, acc)
    return acc


# --- enums ------------------------------------------------------------------------------------


def test_confidence_level_members():
    assert {e.value for e in ConfidenceLevel} == {"high", "medium", "low"}


def test_answer_scope_members():
    assert {e.value for e in AnswerScope} == {"full", "partial", "none"}


# --- strict schema compliance (per declared variant) ------------------------------------------


@pytest.mark.parametrize("variant", list(SchemaVariant))
def test_strict_schema_is_compliant_for_every_variant(variant):
    model = get_response_schema(variant)
    strict = to_openai_strict_schema(model)

    # The dedicated checker must pass...
    assert_strict_compliant(strict)

    # ...and, independently, every object node must satisfy the two key rules.
    objects = _all_object_nodes(strict)
    assert objects, "expected at least the root object node"
    for obj in objects:
        assert obj.get("additionalProperties") is False
        assert set(obj.get("required", [])) == set(obj.get("properties", {}).keys())


def test_get_response_schema_rejects_unknown_variant():
    with pytest.raises(ValueError, match="No response schema"):
        get_response_schema("not-a-variant")  # type: ignore[arg-type]


# --- headline model shape ---------------------------------------------------------------------


def test_headline_model_has_exact_fields():
    assert set(AnswerV1.model_fields) == {"answer", "confidence", "scope", "supporting_doc_ids"}
    # All required, no defaults.
    assert all(field.is_required() for field in AnswerV1.model_fields.values())


def test_answer_v1_is_the_fullest_variant():
    assert get_response_schema(SchemaVariant.ANSWER_V1) is AnswerV1


# --- deterministic hashing --------------------------------------------------------------------


def test_schema_sha256_is_deterministic():
    first = schema_sha256(AnswerV1)
    second = schema_sha256(AnswerV1)
    assert first == second
    assert len(first) == 64 and all(c in "0123456789abcdef" for c in first)


def test_schema_sha256_differs_across_models():
    class _Other(BaseModel):
        model_config = ConfigDict(extra="forbid")

        foo: str

    assert schema_sha256(AnswerV1) != schema_sha256(_Other)


# --- validation behaviour ---------------------------------------------------------------------


def test_model_instantiates_on_valid_data():
    obj = AnswerV1(
        answer="Paris",
        confidence="high",
        scope="full",
        supporting_doc_ids=["abc", "def"],
    )
    assert obj.confidence is ConfidenceLevel.HIGH
    assert obj.scope is AnswerScope.FULL


def test_model_rejects_invalid_enum_value():
    with pytest.raises(ValidationError):
        AnswerV1(
            answer="x",
            confidence="certain",  # not a ConfidenceLevel member
            scope="full",
            supporting_doc_ids=[],
        )


def test_model_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AnswerV1(
            answer="x",
            confidence="low",
            scope="none",
            supporting_doc_ids=[],
            hallucinated="nope",  # extra="forbid"
        )


# --- agent step models (ContextGrade / RewriteQuery) ------------------------------------------


@pytest.mark.parametrize("model", [AnswerV1, ContextGrade, RewriteQuery])
def test_strict_schema_is_compliant_for_step_models(model):
    strict = to_openai_strict_schema(model)
    assert_strict_compliant(strict)
    objects = _all_object_nodes(strict)
    assert objects, "expected at least the root object node"
    for obj in objects:
        assert obj.get("additionalProperties") is False
        assert set(obj.get("required", [])) == set(obj.get("properties", {}).keys())


def test_context_grade_field_set_and_reuses_enums():
    assert set(ContextGrade.model_fields) == {"scope", "confidence", "needs_more_context"}
    assert all(field.is_required() for field in ContextGrade.model_fields.values())
    # The grade step reuses the shared enums rather than redefining them.
    assert ContextGrade.model_fields["scope"].annotation is AnswerScope
    assert ContextGrade.model_fields["confidence"].annotation is ConfidenceLevel
    assert ContextGrade.model_fields["needs_more_context"].annotation is bool


def test_rewrite_query_field_set():
    assert set(RewriteQuery.model_fields) == {"query"}
    assert all(field.is_required() for field in RewriteQuery.model_fields.values())
    assert RewriteQuery.model_fields["query"].annotation is str


def test_step_model_schema_hashes_are_deterministic_and_distinct():
    models = [AnswerV1, ContextGrade, RewriteQuery]
    # Deterministic per model.
    for model in models:
        assert schema_sha256(model) == schema_sha256(model)
    # Mutually distinct across the three models.
    hashes = {schema_sha256(model) for model in models}
    assert len(hashes) == len(models)


def test_context_grade_instantiates_and_validates():
    grade = ContextGrade(scope="partial", confidence="medium", needs_more_context=True)
    assert grade.scope is AnswerScope.PARTIAL
    assert grade.confidence is ConfidenceLevel.MEDIUM
    assert grade.needs_more_context is True

    with pytest.raises(ValidationError):
        ContextGrade(scope="kinda", confidence="medium", needs_more_context=False)  # bad enum
    with pytest.raises(ValidationError):
        ContextGrade(scope="full", confidence="high", needs_more_context=False, extra="x")


def test_rewrite_query_instantiates_and_validates():
    assert RewriteQuery(query="who founded Foo?").query == "who founded Foo?"
    with pytest.raises(ValidationError):
        RewriteQuery(query="x", extra="nope")  # extra="forbid"


def test_response_schema_for_node_maps_each_step():
    assert response_schema_for_node("grade", SchemaVariant.ANSWER_V1) is ContextGrade
    assert response_schema_for_node("rewrite", SchemaVariant.ANSWER_V1) is RewriteQuery
    assert response_schema_for_node("synthesize", SchemaVariant.ANSWER_V1) is AnswerV1


def test_response_schema_for_node_rejects_unknown_node():
    with pytest.raises(ValueError, match="Unknown agent node"):
        response_schema_for_node("plan", SchemaVariant.ANSWER_V1)  # type: ignore[arg-type]
