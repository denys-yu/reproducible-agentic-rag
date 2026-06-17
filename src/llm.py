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

import enum
import re
import time
from dataclasses import dataclass
from typing import Any

from src.cache import LLMCache, llm_payload
from src.config import Arm, Config, SchemaVariant
from src.provenance import NodeName, ProvenanceLogger, make_record, prompt_sha256
from src.schemas import AnswerScope, ConfidenceLevel, response_schema_for_node, schema_sha256

Messages = list[dict[str, Any]]

# 64-char hex tokens are SHA-256 doc_ids (see data.compute_doc_id).
_DOC_ID_RE = re.compile(r"\b[0-9a-f]{64}\b", re.IGNORECASE)
# Free-arm enum fields that must be coerced back from strings.
_FREE_ENUM_FIELDS: dict[str, type[enum.Enum]] = {
    "scope": AnswerScope,
    "confidence": ConfidenceLevel,
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


def _find_enum(text: str, enum_cls: type[enum.Enum], label: str) -> enum.Enum | None:
    """Find an enum member by its exact word: prefer a `label: member` form, else an unambiguous one."""
    members = [member.value for member in enum_cls]
    alternation = "|".join(re.escape(m) for m in members)

    labelled = re.search(rf"{label}\s*[:=\-]?\s*\b({alternation})\b", text, re.IGNORECASE)
    if labelled:
        return enum_cls(labelled.group(1).lower())

    present = {m for m in members if re.search(rf"\b{re.escape(m)}\b", text, re.IGNORECASE)}
    if len(present) == 1:
        return enum_cls(present.pop())
    return None  # absent or ambiguous -> faithfully unknown


def _find_bool(text: str, label: str) -> bool | None:
    """Find a labelled yes/no (true/false) boolean, or None when not present."""
    match = re.search(rf"{label}\s*[:=\-]?\s*\b(yes|true|no|false)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() in ("yes", "true")
    return None


def _extract_labelled_value(text: str, label: str) -> str | None:
    """Return the remainder of the first `label: value` line, stripped, or None."""
    match = re.search(rf"{label}\s*[:=\-]\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


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
    if node == "grade":
        return {
            "scope": _find_enum(text, AnswerScope, "scope"),
            "confidence": _find_enum(text, ConfidenceLevel, "confidence"),
            "needs_more_context": _find_bool(text, r"needs[\s_]*more[\s_]*context"),
        }
    if node == "rewrite":
        labelled = _extract_labelled_value(text, r"(?:rewritten\s+)?query")
        return {"query": labelled if labelled else text.strip()}
    if node == "synthesize":
        labelled_answer = _extract_labelled_value(text, "answer")
        return {
            "answer": labelled_answer if labelled_answer is not None else text.strip(),
            "confidence": _find_enum(text, ConfidenceLevel, "confidence"),
            "scope": _find_enum(text, AnswerScope, "scope"),
            "supporting_doc_ids": _extract_doc_ids(text),
        }
    raise ValueError(f"Unknown agent node {node!r}")


# --- uniform result ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallResult:
    """Arm-agnostic call result: node-relevant parsed fields plus raw text and cache flag.

    `parsed` holds enum-bearing values (AnswerScope/ConfidenceLevel) or None where free-form
    parsing was unsure; the convenience properties expose the per-node fields.
    """

    node: NodeName
    arm: str
    raw_text: str
    cache_hit: bool
    parsed: dict[str, Any]

    @property
    def scope(self) -> AnswerScope | None:
        return self.parsed.get("scope")

    @property
    def confidence(self) -> ConfidenceLevel | None:
        return self.parsed.get("confidence")

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


def _coerce_parsed(node: NodeName, arm: str, parsed_json: dict[str, Any], variant: SchemaVariant):
    """Rebuild the enum-bearing parsed dict from its JSON form (shared by fresh + cached paths)."""
    if arm == Arm.ENUM.value:
        schema_cls = response_schema_for_node(node, variant)
        model = schema_cls.model_validate(parsed_json)
        return {name: getattr(model, name) for name in schema_cls.model_fields}
    coerced = dict(parsed_json)
    for field, enum_cls in _FREE_ENUM_FIELDS.items():
        value = coerced.get(field)
        if isinstance(value, str):
            coerced[field] = enum_cls(value)
    return coerced


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
            result = structured.invoke(messages)
            raw_message = result["raw"]
            parsed_model = result["parsed"]
            if parsed_model is None:  # fail loud on malformed strict output
                raise ValueError(
                    f"strict structured output failed for node {node!r}: {result.get('parsing_error')}"
                )
            raw_text = _message_text(raw_message)
            parsed_json = parsed_model.model_dump(mode="json")
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
        parsed=_coerce_parsed(node, arm_value, parsed_json, variant),
    )
