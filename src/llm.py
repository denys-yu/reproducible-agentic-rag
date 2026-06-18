"""Single-LLM-call layer: one logged, cached, structured-or-free call with a uniform result.

`call_llm` is the only place the graph touches a model. It builds the cache payload, consults the
(cold-by-default) LLM cache, invokes the pinned model — strict structured output for the `enum`
arm, plain text + deterministic parsing for the `free` arm — captures `system_fingerprint`, token
usage and latency, logs exactly one provenance record, and returns an arm-agnostic `CallResult`.

`make_llm` is the single place the model is configured (pinned snapshot, temperature=0, top_p=1,
seed=42). `parse_free_response` deterministically recovers the structured fields from free-form
text, using `None` to faithfully signal "not confidently present" rather than guessing a default.
"""

from __future__ import annotations

import contextlib
import enum
import re
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.cache import LLMCache, llm_payload
from src.config import Arm, Config, SchemaVariant
from src.provenance import NodeName, ProvenanceLogger, make_record, prompt_sha256
from src.schemas import AnswerScope, ConfidenceLevel, response_schema_for_node, schema_sha256

Messages = list[dict[str, Any]]

# 64-char hex tokens are SHA-256 doc_ids (see data.compute_doc_id).
_DOC_ID_RE = re.compile(r"\b[0-9a-f]{64}\b", re.IGNORECASE)


@contextlib.contextmanager
def _suppress_response_serializer_warning() -> Iterator[None]:
    """Silence langchain_openai's benign response-serialization warning during a structured call.

    For strict structured output, the OpenAI SDK populates `response.choices[0].message.parsed`
    with the raw Pydantic model. langchain_openai then calls `response.model_dump()`
    (`_create_chat_result`), and Pydantic warns "Expected `none` ... got ContextGrade" because the
    SDK's `parsed` field is `Optional[...]`. It is emitted synchronously inside our
    `invoke` (in a RunnableParallel worker thread) and is harmless — the parsed value we use is
    unaffected. We scope the filter to our own call so no global warning state leaks.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
        yield

# Free-arm responses echo the prompt's field names as "<field> <delim> <value>" lines. The model
# observed using em-dashes, so accept ":", "-", "—" (em) and "–" (en) as delimiters. Field labels
# match the schema field names case-insensitively (underscores or spaces).
_DELIMITERS = r"[:—–-]"  # contains literal em-dash (U+2014) and en-dash (U+2013)
_FIELD_LABEL_PATTERNS: dict[str, str] = {
    "answer": r"answer",
    "confidence": r"confidence",
    "scope": r"scope",
    "needs_more_context": r"needs[\s_]*more[\s_]*context",
    "supporting_doc_ids": r"supporting[\s_]*doc[\s_]*ids",
    "query": r"(?:rewritten\s+)?query",
}
_LABEL_RE = re.compile(
    r"(?:"
    + "|".join(rf"(?P<{field}>\b{pattern}\b)" for field, pattern in _FIELD_LABEL_PATTERNS.items())
    + r")"
    + rf"\s*{_DELIMITERS}\s*",
    re.IGNORECASE,
)

# Pre-registered enum synonym maps (explicit; we never interpret beyond these). Matched as whole
# words/phrases against the labelled value, longest key first so e.g. "not confident" beats
# "confident". Anything unmatched stays None.
_SCOPE_MAP: dict[str, AnswerScope] = {
    "full": AnswerScope.FULL,
    "fully": AnswerScope.FULL,
    "completely": AnswerScope.FULL,
    "partial": AnswerScope.PARTIAL,
    "partially": AnswerScope.PARTIAL,
    "partly": AnswerScope.PARTIAL,
    "none": AnswerScope.NONE,
    "not at all": AnswerScope.NONE,
    "no": AnswerScope.NONE,
    "doesn't cover": AnswerScope.NONE,
    "does not cover": AnswerScope.NONE,
}
_CONFIDENCE_MAP: dict[str, ConfidenceLevel] = {
    "high": ConfidenceLevel.HIGH,
    "very confident": ConfidenceLevel.HIGH,
    "certain": ConfidenceLevel.HIGH,
    "confident": ConfidenceLevel.HIGH,
    "medium": ConfidenceLevel.MEDIUM,
    "moderate": ConfidenceLevel.MEDIUM,
    "somewhat": ConfidenceLevel.MEDIUM,
    "low": ConfidenceLevel.LOW,
    "unsure": ConfidenceLevel.LOW,
    "uncertain": ConfidenceLevel.LOW,
    "not confident": ConfidenceLevel.LOW,
}
_NEEDS_MORE_MAP: dict[str, bool] = {
    "yes": True,
    "true": True,
    "no": False,
    "false": False,
}


# --- model construction -----------------------------------------------------------------------


def make_llm(config: Config) -> Any:
    """Construct the pinned ChatOpenAI — the single place decoding params are set."""
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": config.llm_model,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.llm_seed,
    }
    if config.openai_api_key is not None:
        kwargs["api_key"] = config.openai_api_key.get_secret_value()
    return ChatOpenAI(**kwargs)


# --- deterministic free-form parsing ----------------------------------------------------------


def _labelled_fields(text: str) -> dict[str, str]:
    """Split text into `field -> value` segments, each value running up to the next label.

    A field's value is the text between its delimiter and the start of the next recognized label
    (or end of text). The first occurrence of each field wins.
    """
    matches = list(_LABEL_RE.finditer(text))
    fields: dict[str, str] = {}
    for i, match in enumerate(matches):
        field = match.lastgroup  # the matched named label group
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if field is not None:
            fields.setdefault(field, text[match.end() : value_end].strip())
    return fields


def _match_synonym(value: str | None, mapping: dict[str, Any]) -> Any | None:
    """Map a labelled value to its pre-registered enum/bool, longest key first; else None."""
    if not value:
        return None
    lowered = value.lower()
    for key in sorted(mapping, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            return mapping[key]
    return None


def _extract_doc_ids(text: str) -> list[str]:
    """Extract any 64-hex doc_id tokens, lowercased and de-duplicated in order of appearance."""
    seen: dict[str, None] = {}
    for token in _DOC_ID_RE.findall(text):
        seen.setdefault(token.lower(), None)
    return list(seen)


def parse_free_response(node: NodeName, text: str) -> dict[str, Any]:
    """Deterministically recover the structured fields a node would emit, from free-form text.

    Returns the same field set as the corresponding schema, with None where a value is not
    confidently present (never a guessed default). No randomness, no model calls.
    """
    text = text or ""
    fields = _labelled_fields(text)
    if node == "grade":
        return {
            "scope": _match_synonym(fields.get("scope"), _SCOPE_MAP),
            "confidence": _match_synonym(fields.get("confidence"), _CONFIDENCE_MAP),
            "needs_more_context": _match_synonym(fields.get("needs_more_context"), _NEEDS_MORE_MAP),
        }
    if node == "rewrite":
        query = fields.get("query")
        return {"query": query if query else text.strip()}
    if node == "synthesize":
        answer = fields.get("answer")  # isolated to the "answer" segment, not the whole blob
        return {
            "answer": answer if answer else text.strip(),
            "confidence": _match_synonym(fields.get("confidence"), _CONFIDENCE_MAP),
            "scope": _match_synonym(fields.get("scope"), _SCOPE_MAP),
            "supporting_doc_ids": _extract_doc_ids(text),
        }
    raise ValueError(f"Unknown agent node {node!r}")


# --- uniform result ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallResult:
    """Arm-agnostic call result.

    `parsed` is a PLAIN JSON dict (string enum values, bools, lists, None) — it never holds a raw
    Pydantic structured model, so nothing typed lingers to be re-serialized when a CallResult sits
    in the LangGraph state. The convenience properties coerce string enum values back to typed
    members on read for the graph.
    """

    node: NodeName
    arm: str
    raw_text: str
    cache_hit: bool
    parsed: dict[str, Any]

    @property
    def scope(self) -> AnswerScope | None:
        value = self.parsed.get("scope")
        return AnswerScope(value) if value is not None else None

    @property
    def confidence(self) -> ConfidenceLevel | None:
        value = self.parsed.get("confidence")
        return ConfidenceLevel(value) if value is not None else None

    @property
    def needs_more_context(self) -> bool | None:
        return self.parsed.get("needs_more_context")

    @property
    def answer(self) -> str | None:
        return self.parsed.get("answer")

    @property
    def supporting_doc_ids(self) -> list[str] | None:
        return self.parsed.get("supporting_doc_ids")

    @property
    def query(self) -> str | None:
        return self.parsed.get("query")


# --- message metadata helpers -----------------------------------------------------------------


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return content blocks
        parts = [p if isinstance(p, str) else p.get("text", "") for p in content]
        return "".join(parts)
    return str(content)


def _extract_meta(message: Any) -> tuple[str | None, int | None, int | None]:
    """Return (system_fingerprint, tokens_in, tokens_out) from an AIMessage, None where absent."""
    metadata = getattr(message, "response_metadata", None) or {}
    fingerprint = metadata.get("system_fingerprint")

    usage = getattr(message, "usage_metadata", None)
    if usage:
        return fingerprint, usage.get("input_tokens"), usage.get("output_tokens")

    token_usage = metadata.get("token_usage") or {}
    return fingerprint, token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")


# --- parsed (de)serialization -----------------------------------------------------------------


def _jsonify_free(parsed: dict[str, Any]) -> dict[str, Any]:
    """Convert enum-bearing free-parse output to a JSON-serializable dict (enums -> values)."""
    return {
        key: (value.value if isinstance(value, enum.Enum) else value)
        for key, value in parsed.items()
    }


def _model_to_json(model: Any) -> dict[str, Any]:
    """Convert a structured-output model to a plain string-scalar dict WITHOUT Pydantic serialization.

    LangChain's strict parser constructs the model without coercing enum fields, so they may hold
    raw strings. Calling `model_dump(mode="json")` on such an instance emits a Pydantic serializer
    warning. We read each field directly instead (enum -> .value, anything else as-is), yielding a
    clean JSON dict and never triggering the serializer.
    """
    data: dict[str, Any] = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        data[name] = value.value if isinstance(value, enum.Enum) else value
    return data


# --- the core wrapper -------------------------------------------------------------------------


def call_llm(
    *,
    node: NodeName,
    arm: Arm | str,
    variant: SchemaVariant,
    messages: Messages,
    config: Config,
    logger: ProvenanceLogger,
    run_id: str,
    question_id: str,
    retrieved_ids: list[str],
    retrieved_scores: list[float],
    model: Any | None = None,
    llm_cache: LLMCache | None = None,
) -> CallResult:
    """Make one logged, cached, structured-or-free LLM call and return a uniform `CallResult`."""
    arm_value = arm.value if isinstance(arm, Arm) else str(arm)
    model = make_llm(config) if model is None else model
    llm_cache = LLMCache(config) if llm_cache is None else llm_cache

    schema = response_schema_for_node(node, variant) if arm_value == Arm.ENUM.value else None
    schema_hash = schema_sha256(schema) if schema is not None else None

    payload = llm_payload(messages, schema_hash, config)
    cached = llm_cache.get_llm(payload)
    cache_hit = cached is not None

    if cache_hit:  # only reachable when the cache is explicitly enabled (replay ablation)
        raw_text = cached["raw_text"]
        fingerprint = cached["system_fingerprint"]
        tokens_in, tokens_out = cached["tokens_in"], cached["tokens_out"]
        parsed_json = cached["parsed_json"]
        latency_ms = 0.0  # replay: no API latency (observational only)
    else:
        start = time.perf_counter()
        if schema is not None:
            structured = model.with_structured_output(schema, strict=True, include_raw=True)
            with _suppress_response_serializer_warning():
                result = structured.invoke(messages)
            raw_message = result["raw"]
            parsed_model = result["parsed"]
            if parsed_model is None:  # fail loud on malformed strict output
                raise ValueError(
                    f"strict structured output failed for node {node!r}: {result.get('parsing_error')}"
                )
            raw_text = _message_text(raw_message)
            parsed_json = _model_to_json(parsed_model)  # plain string scalars for logging
        else:
            raw_message = model.invoke(messages)
            raw_text = _message_text(raw_message)
            parsed_json = _jsonify_free(parse_free_response(node, raw_text))
        latency_ms = (time.perf_counter() - start) * 1000.0
        fingerprint, tokens_in, tokens_out = _extract_meta(raw_message)

        if llm_cache.enabled:
            llm_cache.set_llm(
                payload,
                {
                    "raw_text": raw_text,
                    "system_fingerprint": fingerprint,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "parsed_json": parsed_json,
                },
            )

    record = make_record(
        config,
        run_id=run_id,
        question_id=question_id,
        arm=arm_value,
        node=node,
        prompt_sha256=prompt_sha256(messages),
        schema_sha256=schema_hash,
        cache_hit=cache_hit,
        retrieved_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
        raw_response=raw_text,
        parsed=parsed_json,
        system_fingerprint=fingerprint,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )
    logger.log(record)

    return CallResult(
        node=node,
        arm=arm_value,
        raw_text=raw_text,
        cache_hit=cache_hit,
        parsed=parsed_json,  # plain JSON dict; accessors coerce enums on read
    )
