"""Offline tests for src.agent: the retrieve -> grade -> [rewrite] -> synthesize graph.

A fake chat model (responding per node, detected from the system prompt) + a tiny ephemeral Chroma
collection over a fake deterministic embedder + a temp logger. No API calls.
"""

from __future__ import annotations

import json

import chromadb

from src.agent import (
    GRADE_PROMPT,
    REWRITE_PROMPT,
    SYNTHESIZE_PROMPT,
    build_agent,
    initial_state,
    run_question,
)
from src.cache import LLMCache
from src.config import Config, SchemaVariant
from src.data import Chunk
from src.index import build_index, retrieve
from src.provenance import ProvenanceLogger
from src.schemas import AnswerScope, AnswerV1, ConfidenceLevel, ContextGrade, RewriteQuery

_QID = "q1"


# --- fakes ------------------------------------------------------------------------------------


class FakeEmbedder:
    """Constant-vector embedder: enough for the graph to retrieve deterministically offline."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.response_metadata = {
            "system_fingerprint": "fp_test",
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        self.usage_metadata = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}


def _node_of(messages) -> str:
    system = messages[0]["content"]
    if system == GRADE_PROMPT:
        return "grade"
    if system == REWRITE_PROMPT:
        return "rewrite"
    if system == SYNTHESIZE_PROMPT:
        return "synthesize"
    raise AssertionError("unrecognized system prompt")


class _FakeStructured:
    def __init__(self, model: FakeChatModel) -> None:
        self._model = model

    def invoke(self, messages, config=None):
        node = _node_of(messages)
        return {"raw": FakeMessage(self._model.free_text[node]), "parsed": self._model.parsed[node], "parsing_error": None}


class FakeChatModel:
    """Responds per node; same instance serves both the free and enum arms."""

    def __init__(self, *, grade_needs_more: bool) -> None:
        yn = "yes" if grade_needs_more else "no"
        self.free_text = {
            "grade": f"Scope: partial\nConfidence: medium\nNeeds more context: {yn}",
            "rewrite": "Rewritten query: improved reformulated query",
            "synthesize": "Answer: 42\nConfidence: high\nScope: full",
        }
        self.parsed = {
            "grade": ContextGrade(
                scope=AnswerScope.PARTIAL,
                confidence=ConfidenceLevel.MEDIUM,
                needs_more_context=grade_needs_more,
            ),
            "rewrite": RewriteQuery(query="improved reformulated query"),
            "synthesize": AnswerV1(
                answer="42", confidence="high", scope="full", supporting_doc_ids=[]
            ),
        }

    def invoke(self, messages, config=None):  # free arm
        return FakeMessage(self.free_text[_node_of(messages)])

    def with_structured_output(self, schema, *, strict, include_raw):  # enum arm
        assert strict is True and include_raw is True
        return _FakeStructured(self)


# --- fixtures ---------------------------------------------------------------------------------


_FIXTURE: list[Chunk] = [
    Chunk(doc_id=f"id_{c}", question_id=_QID, title=f"T{c}", text=f"text {c}",
          is_gold=(c == "a"), para_index=0, chunk_index=0)
    for c in ("a", "b", "c")
]


def _config(tmp_path) -> Config:
    return Config(cache_dir=tmp_path, runs_dir=tmp_path)


def _collection(config):
    client = chromadb.EphemeralClient()
    return build_index(config, chunks=_FIXTURE, embedder=FakeEmbedder(), client=client)


def _build(config, *, grade_needs_more):
    collection = _collection(config)
    fake = FakeChatModel(grade_needs_more=grade_needs_more)
    graph = build_agent(
        config, collection=collection, embedder=FakeEmbedder(), model=fake, llm_cache=LLMCache(config)
    )
    return graph, collection


def _records(logger: ProvenanceLogger):
    return [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]


# --- no-rewrite path --------------------------------------------------------------------------


def test_no_rewrite_path_logs_two_records(tmp_path):
    config = _config(tmp_path)
    graph, _ = _build(config, grade_needs_more=False)
    logger = ProvenanceLogger.for_run("run-norew", config)
    result = run_question(
        graph, "who is X?", _QID, arm="enum", variant=SchemaVariant.ANSWER_V1,
        run_id="run-norew", logger=logger,
    )
    logger.close()

    records = _records(logger)
    assert [r["node"] for r in records] == ["grade", "synthesize"]
    assert result.answer == "42"
    assert result.rewrite_count == 0


# --- rewrite path -----------------------------------------------------------------------------


def test_rewrite_path_logs_three_records_and_changes_query(tmp_path):
    config = _config(tmp_path)
    graph, _ = _build(config, grade_needs_more=True)
    logger = ProvenanceLogger.for_run("run-rew", config)

    # Drive the compiled graph directly to inspect final state (current_query).
    state = initial_state("who is X?", _QID, arm="enum", variant=SchemaVariant.ANSWER_V1, run_id="run-rew")
    final = graph.invoke(state, config={"configurable": {"logger": logger}})
    logger.close()

    assert [r["node"] for r in _records(logger)] == ["grade", "rewrite", "synthesize"]
    assert final["rewrite_count"] == 1  # never more than one rewrite
    assert final["current_query"] != "who is X?"  # query was reformulated
    assert final["current_query"] == "improved reformulated query"


# --- both arms produce a result ---------------------------------------------------------------


def test_both_arms_run_through_the_graph(tmp_path):
    for arm in ("free", "enum"):
        config = _config(tmp_path / arm)
        graph, _ = _build(config, grade_needs_more=False)
        logger = ProvenanceLogger.for_run(f"run-{arm}", config)
        result = run_question(
            graph, "who is X?", _QID, arm=arm, variant=SchemaVariant.ANSWER_V1,
            run_id=f"run-{arm}", logger=logger,
        )
        logger.close()
        assert result.answer == "42"
        assert [r["node"] for r in _records(logger)] == ["grade", "synthesize"]


# --- retrieved_ids linkage --------------------------------------------------------------------


def test_logged_retrieved_ids_match_the_retrieval(tmp_path):
    config = _config(tmp_path)
    graph, collection = _build(config, grade_needs_more=False)
    logger = ProvenanceLogger.for_run("run-link", config)
    run_question(
        graph, "who is X?", _QID, arm="enum", variant=SchemaVariant.ANSWER_V1,
        run_id="run-link", logger=logger,
    )
    logger.close()

    expected = [hit["doc_id"] for hit in retrieve("who is X?", _QID, k=config.top_k, collection=collection, embedder=FakeEmbedder())]
    for record in _records(logger):
        assert record["retrieved_ids"] == expected
