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
import hashlib
import json
import re
import time
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from src.cache import LLMCache, llm_payload
from src.config import Arm, Condition, Config, SchemaVariant
from src.hashing import sha256_messages
from src.prompts import FREE_FORMAT_SUFFIX, suffixed_nodes
from src.provenance import NodeName, ProvenanceLogger, make_record
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


def required_request_params(config: Config) -> dict[str, Any]:
    """The exact params that MUST appear in the outbound request, name -> required value.

    This is the contract the run is pinned to: the profile's supported sampling params at their
    configured values, plus every mandatory extra param the profile declares.
    """
    profile = config.model_profile
    params: dict[str, Any] = {}
    if "temperature" in profile.supports:
        params["temperature"] = config.temperature
    if "top_p" in profile.supports:
        params["top_p"] = config.top_p
    if "seed" in profile.supports:
        params["seed"] = config.llm_seed
    params.update(profile.extra_params)
    return params


def resolve_request_payload(model: Any, messages: Messages) -> dict[str, Any]:
    """Return the payload langchain_openai would actually send to the API for `messages`.

    This is the resolved outbound request, NOT the values we intended to pass — the whole point
    is to catch params that the client drops on the way out. Uses langchain_openai's own
    request builder, so what we inspect is what the SDK receives.
    """
    build = getattr(model, "_get_request_payload", None)
    if build is None:  # unknown/incompatible client: refuse to guess
        raise RuntimeError(
            f"{type(model).__name__} exposes no _get_request_payload; cannot verify that the "
            "pinned decoding params survive into the outbound request."
        )
    return build(messages, stop=None)


def _missing_params(payload: Mapping[str, Any], required: Mapping[str, Any]) -> dict[str, Any]:
    """Return the required params absent from — or contradicted by — the resolved payload."""
    missing = {}
    for name, expected in required.items():
        if name not in payload or payload[name] != expected:
            missing[name] = expected
    return missing


# One probe message; only the request *shape* matters, and it is never sent.
_PROBE_MESSAGES: Messages = [{"role": "user", "content": "probe"}]


@dataclass(frozen=True)
class ParamGuardReport:
    """Outcome of the outbound-parameter check performed when the model is constructed."""

    model_id: str
    required: dict[str, Any]
    mechanism: str  # "direct" | "model_kwargs"
    rerouted: list[str]  # params that had to go through model_kwargs to survive
    payload: dict[str, Any]  # resolved payload for the probe request, messages stripped


def make_llm(config: Config, *, report: list[ParamGuardReport] | None = None) -> Any:
    """Construct the pinned ChatOpenAI — the single place decoding params are set.

    langchain_openai drops sampling params it believes a model family does not accept, silently
    and without raising (verified: for gpt-5.x it strips `temperature` outright, even when the
    model does accept it under `reasoning_effort="none"`). A dropped `temperature=0` would leave
    the run sampling at the API default with nothing in the log to show it — the run would look
    pinned and not be. So the constructed client is checked against the resolved outbound
    payload, any stripped param is rerouted through `model_kwargs`, and a param that still will
    not survive aborts construction rather than quietly downgrading the run.

    Pass `report` to receive the resulting `ParamGuardReport` (appended).
    """
    from langchain_openai import ChatOpenAI

    profile = config.model_profile
    required = required_request_params(config)

    base: dict[str, Any] = {"model": profile.model_id}
    if config.openai_api_key is not None:
        base["api_key"] = config.openai_api_key.get_secret_value()

    model = ChatOpenAI(**base, **required)
    payload = resolve_request_payload(model, _PROBE_MESSAGES)
    stripped = _missing_params(payload, required)

    mechanism = "direct"
    if stripped:
        # Reroute exactly the stripped params; the rest stay on their first-class fields.
        mechanism = "model_kwargs"
        direct = {k: v for k, v in required.items() if k not in stripped}
        with warnings.catch_warnings():
            # langchain warns that these "should be specified explicitly" — which is precisely
            # what we did, and precisely what it undid.
            warnings.filterwarnings("ignore", message=".*model_kwargs.*", category=UserWarning)
            model = ChatOpenAI(**base, **direct, model_kwargs=dict(stripped))
        payload = resolve_request_payload(model, _PROBE_MESSAGES)
        still_missing = _missing_params(payload, required)
        if still_missing:
            raise RuntimeError(
                f"Refusing to run: {sorted(still_missing)} do not survive into the outbound "
                f"request for model {profile.model_id!r}, via first-class fields or model_kwargs. "
                f"Resolved payload: {_payload_for_log(payload)}"
            )

    if report is not None:
        report.append(
            ParamGuardReport(
                model_id=profile.model_id,
                required=dict(required),
                mechanism=mechanism,
                rerouted=sorted(stripped),
                payload=_payload_for_log(payload),
            )
        )
    return model


def _payload_for_log(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The resolved payload with the message bodies replaced by a count (params are the point)."""
    logged = {k: v for k, v in payload.items() if k != "messages"}
    logged["messages"] = f"<{len(payload.get('messages', []))} messages>"
    return logged


def assert_outbound_params(config: Config, model: Any) -> dict[str, Any]:
    """Assert the pinned params are present in the resolved outbound request; return that payload.

    Called once at run start. Raises RuntimeError naming the offending params if any is missing,
    so a stripped param aborts the run instead of silently producing an unpinned stratum.
    """
    payload = resolve_request_payload(model, _PROBE_MESSAGES)
    missing = _missing_params(payload, required_request_params(config))
    if missing:
        raise RuntimeError(
            f"Outbound request for {config.llm_model!r} is missing required params "
            f"{sorted(missing)} (expected {missing}). Resolved payload: {_payload_for_log(payload)}"
        )
    return payload


def request_payload_sha256(model: Any, messages: Messages, schema_sha256: str | None) -> str | None:
    """SHA-256 of the resolved outbound request (params + messages + schema identity).

    Hashing the RESOLVED payload — rather than the params we meant to send — makes any silent
    client-side change to the request detectable after the fact, from the log alone.

    Returns None for a client that cannot render a payload (the injected fakes used in offline
    tests). That is not a silent downgrade of a real run: `make_llm` has already refused to build
    any real client whose payload could not be resolved and verified.
    """
    if getattr(model, "_get_request_payload", None) is None:
        return None
    payload = dict(resolve_request_payload(model, messages))
    payload["response_format_schema_sha256"] = schema_sha256
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- request-body interception ------------------------------------------------------------------
# Everything below reads the request body langchain_openai actually renders, never the values we
# asked it to send. The two are NOT the same: for gpt-5.x the client strips `temperature` outright.
# A log that reports config's intent would describe a run that did not happen.

#: The decoding params recorded per call, in a fixed order.
_LOGGED_PARAMS: tuple[str, ...] = ("temperature", "top_p", "seed", "reasoning_effort")


def intercepted_messages(model: Any, messages: Messages) -> list[dict[str, Any]]:
    """The messages as they appear in the resolved outbound body; the input list as a fallback.

    The fallback is only reachable for injected test fakes that render no payload — `make_llm`
    refuses to build a real client whose body cannot be resolved.
    """
    if getattr(model, "_get_request_payload", None) is None:
        return [dict(message) for message in messages]
    payload = resolve_request_payload(model, messages)
    body = payload.get("messages")
    if not isinstance(body, list):
        return [dict(message) for message in messages]
    return [dict(message) if isinstance(message, Mapping) else {"content": str(message)} for message in body]


def intercepted_prompt_sha256(model: Any, messages: Messages) -> str:
    """SHA-256 of the intercepted request-body messages, CRLF normalised to LF first.

    `core.autocrlf=true` on this machine, and prompt text reaches the body via source files; a
    digest taken without normalising would differ between a Windows and a Linux checkout of the
    identical prompt, which is the one thing a reproducibility hash must never do.
    """
    return sha256_messages(intercepted_messages(model, messages))


def intercepted_request_params(model: Any, messages: Messages) -> dict[str, Any]:
    """Read `temperature`, `top_p`, `seed` and `reasoning_effort` back off the outbound body.

    A param the client dropped is recorded as None — that absence is the finding, so it is logged
    as an observation rather than papered over with config's value. `source` records where the
    numbers came from, so no reader has to guess whether a null means "dropped" or "unavailable".
    """
    if getattr(model, "_get_request_payload", None) is None:
        return {name: None for name in _LOGGED_PARAMS} | {"source": "unavailable"}
    payload = resolve_request_payload(model, messages)
    params: dict[str, Any] = {name: payload.get(name) for name in _LOGGED_PARAMS}
    # Params rerouted through model_kwargs land at top level in the rendered body, but fall back
    # to the nested dict if a client version keeps them there.
    nested = payload.get("model_kwargs")
    if isinstance(nested, Mapping):
        for name in _LOGGED_PARAMS:
            if params[name] is None and name in nested:
                params[name] = nested[name]
    params["source"] = "intercepted_request_body"
    return params


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


@dataclass(frozen=True)
class CallMeta:
    """Per-call server-reported metadata. Every field is None when the provider omits it.

    `system_fingerprint` is null for models that publish no build identifier (gpt-5.6-luna) —
    recorded as null rather than dropped, so the log distinguishes "not offered" from "not read".
    The cache counters are observational: server-side prompt caching cannot be turned off, so
    the experiment observes it instead of pretending it is absent.
    """

    system_fingerprint: str | None
    tokens_in: int | None
    tokens_out: int | None
    cached_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce a usage-details value (dict, pydantic model, or None) to a plain dict."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    return dict(dump()) if callable(dump) else {}


def _extract_meta(message: Any) -> CallMeta:
    """Extract fingerprint, token counts, cache counters and reasoning tokens from an AIMessage."""
    metadata = getattr(message, "response_metadata", None) or {}
    token_usage = _as_mapping(metadata.get("token_usage"))
    prompt_details = _as_mapping(token_usage.get("prompt_tokens_details"))
    completion_details = _as_mapping(token_usage.get("completion_tokens_details"))

    usage = getattr(message, "usage_metadata", None) or {}
    tokens_in = usage.get("input_tokens", token_usage.get("prompt_tokens"))
    tokens_out = usage.get("output_tokens", token_usage.get("completion_tokens"))

    return CallMeta(
        system_fingerprint=metadata.get("system_fingerprint"),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_tokens=prompt_details.get("cached_tokens"),
        cache_write_tokens=prompt_details.get("cache_write_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
    )


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


def _assert_suffix_matches_wiring(
    node: NodeName,
    arm: str,
    condition: str,
    messages: Messages,
    suffix_nodes: list[str],
) -> None:
    """Assert the format appendix is in THIS call's system message exactly when the wiring says so.

    `format_suffix_nodes` is derived from the wiring table, but the value that matters is what the
    system message actually carried. Checking the two against each other on every call is what
    makes the logged field evidence rather than a restatement of an assumption: if the prompt text
    and the wiring table ever drift apart, the run stops on the first call instead of producing a
    manifest that confidently misdescribes itself.
    """
    from src.hashing import to_lf

    system = "".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    )
    carried = to_lf(FREE_FORMAT_SUFFIX) in to_lf(system)
    expected = node in suffix_nodes
    if carried != expected:
        raise RuntimeError(
            f"Format-appendix wiring violated on ({condition}, {arm}, {node}): the system prompt "
            f"{'CARRIES' if carried else 'does NOT carry'} FREE_FORMAT_SUFFIX, but the wiring "
            f"table says nodes {suffix_nodes} receive it. Refusing to log a record that "
            f"misdescribes the prompt actually sent."
        )


# --- the core wrapper -------------------------------------------------------------------------


def call_llm(
    *,
    node: NodeName,
    arm: Arm | str,
    variant: SchemaVariant,
    messages: Messages,
    config: Config,
    condition: Condition | str,
    logger: ProvenanceLogger,
    run_id: str,
    question_id: str,
    retrieved_ids: list[str],
    retrieved_scores: list[float],
    model: Any | None = None,
    llm_cache: LLMCache | None = None,
) -> CallResult:
    """Make one logged, cached, structured-or-free LLM call and return a uniform `CallResult`.

    `condition` is a required keyword. Every record must state the prompt wiring it ran under, and
    there is no defensible default to fall back to — a record that guessed its own condition would
    quietly misfile itself into the wrong arm of the experiment.
    """
    arm_value = arm.value if isinstance(arm, Arm) else str(arm)
    condition_value = Condition(condition).value
    model = make_llm(config) if model is None else model
    llm_cache = LLMCache(config) if llm_cache is None else llm_cache

    schema = response_schema_for_node(node, variant) if arm_value == Arm.ENUM.value else None
    schema_hash = schema_sha256(schema) if schema is not None else None

    payload = llm_payload(messages, schema_hash, config)
    cached = llm_cache.get_llm(payload)
    cache_hit = cached is not None

    payload_hash = request_payload_sha256(model, messages, schema_hash)
    # Intercepted BEFORE the call: what we hash and log is the body as rendered for the wire, not
    # config's intent. See `intercepted_request_params` for why those two can disagree.
    body_prompt_sha256 = intercepted_prompt_sha256(model, messages)
    body_params = intercepted_request_params(model, messages)
    suffix_nodes = suffixed_nodes(arm_value, condition_value)
    _assert_suffix_matches_wiring(node, arm_value, condition_value, messages, suffix_nodes)

    if cache_hit:  # only reachable when the cache is explicitly enabled (replay ablation)
        raw_text = cached["raw_text"]
        parsed_json = cached["parsed_json"]
        meta = CallMeta(
            system_fingerprint=cached["system_fingerprint"],
            tokens_in=cached["tokens_in"],
            tokens_out=cached["tokens_out"],
            cached_tokens=cached.get("cached_tokens"),
            cache_write_tokens=cached.get("cache_write_tokens"),
            reasoning_tokens=cached.get("reasoning_tokens"),
        )
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
        meta = _extract_meta(raw_message)

        if llm_cache.enabled:
            llm_cache.set_llm(
                payload,
                {
                    "raw_text": raw_text,
                    "system_fingerprint": meta.system_fingerprint,
                    "tokens_in": meta.tokens_in,
                    "tokens_out": meta.tokens_out,
                    "cached_tokens": meta.cached_tokens,
                    "cache_write_tokens": meta.cache_write_tokens,
                    "reasoning_tokens": meta.reasoning_tokens,
                    "parsed_json": parsed_json,
                },
            )

    record = make_record(
        config,
        run_id=run_id,
        question_id=question_id,
        arm=arm_value,
        node=node,
        prompt_sha256=body_prompt_sha256,
        schema_sha256=schema_hash,
        condition=condition_value,
        request_params=body_params,
        format_suffix_nodes=suffix_nodes,
        cache_hit=cache_hit,
        retrieved_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
        raw_response=raw_text,
        parsed=parsed_json,
        system_fingerprint=meta.system_fingerprint,
        tokens_in=meta.tokens_in,
        tokens_out=meta.tokens_out,
        cached_tokens=meta.cached_tokens,
        cache_write_tokens=meta.cache_write_tokens,
        reasoning_tokens=meta.reasoning_tokens,
        request_payload_sha256=payload_hash,
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
