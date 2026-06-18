"""Offline tests for src.llm: model config, free-form parsing, and the call_llm wrapper.

A fake chat model stands in for ChatOpenAI — no API calls. call_llm is exercised in both arms with
a temp provenance logger and the default (cold) LLM cache.
"""

from __future__ import annotations

import json
import warnings

from src.config import Config, SchemaVariant
from src.llm import CallResult, call_llm, make_llm, parse_free_response
from src.provenance import ProvenanceLogger
from src.schemas import AnswerScope, AnswerV1, ConfidenceLevel, schema_sha256

_DOC_ID = "deadbeef" * 8  # a valid 64-hex token


# --- fakes ------------------------------------------------------------------------------------


class FakeMessage:
    def __init__(self, content: str, *, fingerprint="fp_test", tin=11, tout=7) -> None:
        self.content = content
        self.response_metadata = {
            "system_fingerprint": fingerprint,
            "token_usage": {"prompt_tokens": tin, "completion_tokens": tout},
        }
        self.usage_metadata = {"input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout}


class _FakeStructured:
    def __init__(self, raw: FakeMessage, parsed) -> None:
        self._raw = raw
        self._parsed = parsed

    def invoke(self, messages):
        return {"raw": self._raw, "parsed": self._parsed, "parsing_error": None}


class FakeChatModel:
    """Returns a fixed message for free calls and a fixed parsed model for structured calls."""

    def __init__(self, raw: FakeMessage, parsed=None) -> None:
        self._raw = raw
        self._parsed = parsed
        self.structured_calls: list[dict] = []

    def invoke(self, messages):
        return self._raw

    def with_structured_output(self, schema, *, strict, include_raw):
        assert strict is True and include_raw is True
        self.structured_calls.append({"schema": schema, "strict": strict, "include_raw": include_raw})
        return _FakeStructured(self._raw, self._parsed)


def _config(tmp_path, **overrides) -> Config:
    return Config(cache_dir=tmp_path, runs_dir=tmp_path, **overrides)


def _read_records(logger: ProvenanceLogger):
    return [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]


# --- make_llm ---------------------------------------------------------------------------------


def test_make_llm_uses_pinned_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = make_llm(Config())

    def param(name):
        kwargs = getattr(llm, "model_kwargs", {}) or {}
        return kwargs[name] if name in kwargs else getattr(llm, name, None)

    assert getattr(llm, "model_name", getattr(llm, "model", None)) == "gpt-4o-mini-2024-07-18"
    assert llm.temperature == 0.0
    assert param("top_p") == 1.0
    assert param("seed") == 42


# --- parse_free_response ----------------------------------------------------------------------


def test_parse_free_grade_extracts_and_handles_absence():
    grade = parse_free_response("grade", "Scope: partial\nConfidence: medium\nNeeds more context: yes")
    assert grade == {
        "scope": AnswerScope.PARTIAL,
        "confidence": ConfidenceLevel.MEDIUM,
        "needs_more_context": True,
    }
    empty = parse_free_response("grade", "   ")
    assert empty == {"scope": None, "confidence": None, "needs_more_context": None}


def test_parse_free_rewrite_uses_label_or_falls_back():
    assert parse_free_response("rewrite", "Rewritten query: who founded Foo?") == {
        "query": "who founded Foo?"
    }
    assert parse_free_response("rewrite", "  just the raw text  ") == {"query": "just the raw text"}


def test_parse_free_synthesize_extracts_fields_and_doc_ids():
    text = f"Answer: Paris\nConfidence: high\nScope: full\nSupporting: {_DOC_ID}"
    parsed = parse_free_response("synthesize", text)
    assert parsed["answer"] == "Paris"
    assert parsed["confidence"] is ConfidenceLevel.HIGH
    assert parsed["scope"] is AnswerScope.FULL
    assert parsed["supporting_doc_ids"] == [_DOC_ID]


def test_parse_free_synthesize_unsure_enums_are_none():
    parsed = parse_free_response("synthesize", "Paris is the capital.")
    assert parsed["answer"] == "Paris is the capital."
    assert parsed["confidence"] is None
    assert parsed["scope"] is None
    assert parsed["supporting_doc_ids"] == []


# --- call_llm: enum arm -----------------------------------------------------------------------


def test_call_llm_enum_parses_structured_and_logs(tmp_path):
    config = _config(tmp_path)
    parsed_model = AnswerV1(
        answer="Paris", confidence="high", scope="full", supporting_doc_ids=[_DOC_ID]
    )
    raw = FakeMessage(content=json.dumps(parsed_model.model_dump(mode="json")))
    fake = FakeChatModel(raw, parsed=parsed_model)
    logger = ProvenanceLogger.for_run("run-enum", config)

    result = call_llm(
        node="synthesize",
        arm="enum",
        variant=SchemaVariant.ANSWER_V1,
        messages=[{"role": "user", "content": "capital of France?"}],
        config=config,
        logger=logger,
        run_id="run-enum",
        question_id="q1",
        retrieved_ids=[_DOC_ID],
        retrieved_scores=[0.9],
        model=fake,
    )
    logger.close()

    assert isinstance(result, CallResult)
    assert result.answer == "Paris"
    assert result.confidence is ConfidenceLevel.HIGH
    assert result.scope is AnswerScope.FULL
    assert result.supporting_doc_ids == [_DOC_ID]
    assert result.cache_hit is False
    assert fake.structured_calls and fake.structured_calls[0]["schema"] is AnswerV1

    records = _read_records(logger)
    assert len(records) == 1
    rec = records[0]
    assert rec["arm"] == "enum"
    assert rec["node"] == "synthesize"
    assert rec["schema_sha256"] == schema_sha256(AnswerV1)
    assert rec["parsed"]["answer"] == "Paris"
    assert rec["system_fingerprint"] == "fp_test"
    assert rec["tokens_in"] == 11 and rec["tokens_out"] == 7
    assert rec["cache_hit"] is False


# --- call_llm: free arm -----------------------------------------------------------------------


def test_call_llm_free_parses_text_and_logs_null_schema(tmp_path):
    config = _config(tmp_path)
    raw = FakeMessage(content="Scope: partial\nConfidence: medium\nNeeds more context: yes")
    fake = FakeChatModel(raw)
    logger = ProvenanceLogger.for_run("run-free", config)

    result = call_llm(
        node="grade",
        arm="free",
        variant=SchemaVariant.ANSWER_V1,
        messages=[{"role": "user", "content": "is the context enough?"}],
        config=config,
        logger=logger,
        run_id="run-free",
        question_id="q1",
        retrieved_ids=["a", "b"],
        retrieved_scores=[0.5, 0.4],
        model=fake,
    )
    logger.close()

    assert result.scope is AnswerScope.PARTIAL
    assert result.confidence is ConfidenceLevel.MEDIUM
    assert result.needs_more_context is True
    assert result.cache_hit is False

    records = _read_records(logger)
    assert len(records) == 1
    rec = records[0]
    assert rec["arm"] == "free"
    assert rec["schema_sha256"] is None  # free arm logs a null schema hash
    assert rec["parsed"] == {"scope": "partial", "confidence": "medium", "needs_more_context": True}
    assert rec["system_fingerprint"] == "fp_test"
    assert rec["tokens_in"] == 11 and rec["tokens_out"] == 7
    assert rec["cache_hit"] is False


# --- hardened free-form parsing ---------------------------------------------------------------


def test_parse_free_handles_em_dash_en_dash_hyphen_and_colon():
    for delim in ("—", "–", "-", ":"):
        text = f"answer {delim} Maine\nconfidence {delim} high\nscope {delim} full"
        parsed = parse_free_response("synthesize", text)
        assert parsed["answer"] == "Maine"
        assert parsed["confidence"] is ConfidenceLevel.HIGH
        assert parsed["scope"] is AnswerScope.FULL


def test_parse_free_scope_synonyms():
    def scope(value):
        return parse_free_response("grade", f"scope — {value}")["scope"]

    assert scope("fully covers the question") is AnswerScope.FULL
    assert scope("completely") is AnswerScope.FULL
    assert scope("partially") is AnswerScope.PARTIAL
    assert scope("partly") is AnswerScope.PARTIAL
    assert scope("not at all") is AnswerScope.NONE
    assert scope("does not cover") is AnswerScope.NONE
    assert scope("unrelated prose") is None


def test_parse_free_confidence_synonyms():
    def conf(value):
        return parse_free_response("grade", f"confidence — {value}")["confidence"]

    assert conf("very confident") is ConfidenceLevel.HIGH
    assert conf("certain") is ConfidenceLevel.HIGH
    assert conf("moderate") is ConfidenceLevel.MEDIUM
    assert conf("somewhat") is ConfidenceLevel.MEDIUM
    assert conf("uncertain") is ConfidenceLevel.LOW
    assert conf("not confident") is ConfidenceLevel.LOW  # longest-match beats "confident"
    assert conf("no idea") is None


def test_parse_free_needs_more_context_yes_no():
    def needs(value):
        return parse_free_response("grade", f"needs_more_context — {value}")["needs_more_context"]

    assert needs("yes") is True
    assert needs("true") is True
    assert needs("no") is False
    assert needs("false") is False
    assert needs("maybe") is None


def test_parse_free_answer_is_isolated_not_whole_blob():
    text = "answer — The Maine Legislature\nconfidence — high\nscope — full"
    parsed = parse_free_response("synthesize", text)
    assert parsed["answer"] == "The Maine Legislature"  # not the whole blob
    assert parsed["scope"] is AnswerScope.FULL


def test_parse_free_answer_falls_back_when_unlabelled():
    parsed = parse_free_response("synthesize", "The capital is Paris.")
    assert parsed["answer"] == "The capital is Paris."
    assert parsed["confidence"] is None
    assert parsed["scope"] is None


# --- logged parsed is plain JSON strings, no serialization warning -----------------------------


def _enum_call(config, logger):
    parsed_model = AnswerV1(
        answer="Paris", confidence="high", scope="full", supporting_doc_ids=[_DOC_ID]
    )
    raw = FakeMessage(content=json.dumps(parsed_model.model_dump(mode="json")))
    return call_llm(
        node="synthesize",
        arm="enum",
        variant=SchemaVariant.ANSWER_V1,
        messages=[{"role": "user", "content": "capital of France?"}],
        config=config,
        logger=logger,
        run_id="run-json",
        question_id="q1",
        retrieved_ids=[_DOC_ID],
        retrieved_scores=[0.9],
        model=FakeChatModel(raw, parsed=parsed_model),
    )


def test_logged_parsed_contains_string_enum_values(tmp_path):
    config = _config(tmp_path)
    logger = ProvenanceLogger.for_run("run-json", config)
    _enum_call(config, logger)
    logger.close()

    parsed = _read_records(logger)[0]["parsed"]
    assert parsed["confidence"] == "high" and isinstance(parsed["confidence"], str)
    assert parsed["scope"] == "full" and isinstance(parsed["scope"], str)
    assert parsed["answer"] == "Paris"
    assert parsed["supporting_doc_ids"] == [_DOC_ID]


def test_logging_emits_no_pydantic_serialization_warning(tmp_path):
    config = _config(tmp_path)
    logger = ProvenanceLogger.for_run("run-json", config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _enum_call(config, logger)
        logger.close()

    serialization_warnings = [
        w for w in caught if "serializ" in str(w.message).lower() or "pydantic" in str(w.message).lower()
    ]
    assert not serialization_warnings, [str(w.message) for w in serialization_warnings]
