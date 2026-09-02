"""LangGraph agentic loop: retrieve -> grade -> [rewrite + re-retrieve, max 1] -> synthesize.

This module is orchestration + prompts only. Every per-call concern — caching, provenance,
`system_fingerprint`, structured-vs-free parsing, arm-agnostic results — already lives in
`llm.call_llm`; the graph just strings `index.retrieve` and `call_llm` together. Both arms run
through the SAME graph; schema enforcement is handled inside
`call_llm` via the `arm` argument. The one prompt-level arm difference is `FREE_FORMAT_SUFFIX`:
the free arm's output-format contract has to be carried in prose because it has no schema to
carry it, so the suffix is appended to the `grade` and `synthesize` system prompts in the FREE
arm only. The enum arm's format contract is the schema, and its prompts stay unsuffixed.

The bounded wiring makes 2 LLM calls (grade + synthesize) or 3 (grade + rewrite + synthesize) and
NEVER grades twice — the second retrieval round skips straight to synthesize.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from src.config import Arm, Condition, Config, SchemaVariant
from src.index import RetrievedChunk, retrieve
from src.llm import CallResult, call_llm
from src.prompts import (
    FREE_FORMAT_SUFFIX as FREE_FORMAT_SUFFIX,  # re-exported: see the prompts note below
    GRADE_PROMPT as GRADE_PROMPT,
    REWRITE_PROMPT as REWRITE_PROMPT,
    SYNTHESIZE_PROMPT as SYNTHESIZE_PROMPT,
)
from src.prompts import compose_system_prompt
from src.schemas import AnswerScope, ConfidenceLevel

# --- prompts -----------------------------------------------------------------------------------
# The prompt text and the condition switch live in `src.prompts`, which holds the wording
# recovered verbatim from the publishing commit d6efcab. They are re-exported here so existing
# importers of `agent.GRADE_PROMPT` keep working, but this module no longer owns the text and
# must not fork it — `preflight.assert_condition_contract` hashes the `src.prompts` values
# against docs/published_series_reference.json.

def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as title + doc_id + text. Never exposes gold/supporting signals."""
    if not chunks:
        return "(no documents retrieved)"
    return "\n\n".join(
        f"[doc_id: {chunk['doc_id']}] {chunk['title']}\n{chunk['text']}" for chunk in chunks
    )


def _grade_messages(
    question: str, chunks: list[RetrievedChunk], arm: Arm | str, condition: Condition | str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": compose_system_prompt("grade", arm, condition)},
        {"role": "user", "content": f"Question: {question}\n\nRetrieved context:\n{_format_context(chunks)}"},
    ]


def _rewrite_messages(
    question: str, chunks: list[RetrievedChunk], arm: Arm | str, condition: Condition | str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": compose_system_prompt("rewrite", arm, condition)},
        {"role": "user", "content": f"Question: {question}\n\nContext retrieved so far:\n{_format_context(chunks)}"},
    ]


def _synthesize_messages(
    question: str, chunks: list[RetrievedChunk], arm: Arm | str, condition: Condition | str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": compose_system_prompt("synthesize", arm, condition)},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{_format_context(chunks)}"},
    ]


# --- graph state + result ---------------------------------------------------------------------


class AgentState(TypedDict):
    """Per-run graph state. Infra handles (config/collection/embedder/logger) are NOT stored here."""

    question: str
    question_id: str
    arm: str
    condition: str
    variant: SchemaVariant
    run_id: str
    current_query: str
    retrieved: list[RetrievedChunk]
    retrieved_ids: list[str]
    retrieved_scores: list[float]
    rewrite_count: int
    grade: CallResult | None
    answer: CallResult | None


@dataclass(frozen=True)
class AgentResult:
    """Convenience summary of one question's run.

    The canonical per-call record the metrics consume is the provenance JSONL; this is just a
    flattened view of the final answer fields, the grade fields, and the rewrite count.
    """

    question_id: str
    arm: str
    answer: str | None
    confidence: ConfidenceLevel | None
    scope: AnswerScope | None
    supporting_doc_ids: list[str] | None
    raw_answer_text: str
    grade_scope: AnswerScope | None
    grade_confidence: ConfidenceLevel | None
    grade_needs_more_context: bool | None
    rewrite_count: int


def initial_state(
    question: str,
    question_id: str,
    *,
    arm: Arm | str,
    condition: Condition | str,
    variant: SchemaVariant,
    run_id: str,
) -> AgentState:
    """Build the starting state for one question (current_query starts as the question).

    `condition` is required: the prompt wiring in force must be stated explicitly for every run,
    and it is carried in the state so each node composes under the same one.
    """
    return AgentState(
        question=question,
        question_id=question_id,
        arm=arm.value if isinstance(arm, Arm) else str(arm),
        condition=Condition(condition).value,
        variant=variant,
        run_id=run_id,
        current_query=question,
        retrieved=[],
        retrieved_ids=[],
        retrieved_scores=[],
        rewrite_count=0,
        grade=None,
        answer=None,
    )


def _logger_from(config: RunnableConfig) -> Any:
    """Fetch the per-run ProvenanceLogger injected through the invoke-time config."""
    return config["configurable"]["logger"]


# --- routing ----------------------------------------------------------------------------------


def _route_after_retrieve(state: AgentState) -> str:
    # First retrieval -> grade; the post-rewrite retrieval (count == 1) skips straight to synthesize.
    return "grade_context" if state["rewrite_count"] == 0 else "synthesize"


def _route_after_grade(state: AgentState) -> str:
    grade = state["grade"]
    # None needs_more_context (free-arm uncertainty) is treated as False -> synthesize.
    if grade is not None and grade.needs_more_context is True and state["rewrite_count"] == 0:
        return "rewrite"
    return "synthesize"


# --- public API -------------------------------------------------------------------------------


def build_agent(
    config: Config,
    *,
    collection: Any,
    embedder: Any,
    model: Any | None = None,
    llm_cache: Any | None = None,
) -> Any:
    """Compile the agent graph once; nodes close over config/collection/embedder/model/cache.

    `model`/`llm_cache` are injectable for offline tests; by default the pinned ChatOpenAI and the
    cold LLM cache are used. The per-run provenance logger is supplied at invoke time, not here.
    """
    agent_config = config  # closure alias; node params are named `config` (the RunnableConfig)
    if model is None:
        from src.llm import make_llm

        model = make_llm(agent_config)
    if llm_cache is None:
        from src.cache import LLMCache

        llm_cache = LLMCache(agent_config)

    # LangGraph injects the per-invoke RunnableConfig only into a param literally named `config`.
    def retrieve_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        hits = retrieve(
            query=state["current_query"],
            question_id=state["question_id"],
            k=agent_config.top_k,
            collection=collection,
            embedder=embedder,
        )
        return {
            "retrieved": hits,
            "retrieved_ids": [hit["doc_id"] for hit in hits],
            "retrieved_scores": [hit["similarity"] for hit in hits],
        }

    def _call(state: AgentState, config: RunnableConfig, node, messages) -> CallResult:
        return call_llm(
            node=node,
            arm=state["arm"],
            variant=state["variant"],
            messages=messages,
            config=agent_config,
            condition=state["condition"],  # carried in state so every node logs the same wiring
            logger=_logger_from(config),
            run_id=state["run_id"],
            question_id=state["question_id"],
            retrieved_ids=state["retrieved_ids"],
            retrieved_scores=state["retrieved_scores"],
            model=model,
            llm_cache=llm_cache,
        )

    def grade_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = _grade_messages(
            state["question"], state["retrieved"], state["arm"], state["condition"]
        )
        return {"grade": _call(state, config, "grade", messages)}

    def rewrite_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = _rewrite_messages(
            state["question"], state["retrieved"], state["arm"], state["condition"]
        )
        result = _call(state, config, "rewrite", messages)
        return {"current_query": result.query, "rewrite_count": 1}

    def synthesize_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = _synthesize_messages(
            state["question"], state["retrieved"], state["arm"], state["condition"]
        )
        return {"answer": _call(state, config, "synthesize", messages)}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_context", grade_node)  # not "grade": node names can't collide with state keys
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", _route_after_retrieve, {"grade_context": "grade_context", "synthesize": "synthesize"}
    )
    graph.add_conditional_edges(
        "grade_context", _route_after_grade, {"rewrite": "rewrite", "synthesize": "synthesize"}
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_question(
    graph: Any,
    question: str,
    question_id: str,
    *,
    arm: Arm | str,
    condition: Condition | str,
    variant: SchemaVariant,
    run_id: str,
    logger: Any,
) -> AgentResult:
    """Run one question through the compiled graph, logging provenance, and summarize the result.

    The provenance JSONL written via `logger` is the canonical per-call record the metrics
    consume; the returned AgentResult is only a convenience summary.
    """
    state = initial_state(
        question, question_id, arm=arm, condition=condition, variant=variant, run_id=run_id
    )
    final: AgentState = graph.invoke(state, config={"configurable": {"logger": logger}})

    answer = final.get("answer")
    grade = final.get("grade")
    return AgentResult(
        question_id=final["question_id"],
        arm=final["arm"],
        answer=answer.answer if answer else None,
        confidence=answer.confidence if answer else None,
        scope=answer.scope if answer else None,
        supporting_doc_ids=answer.supporting_doc_ids if answer else None,
        raw_answer_text=answer.raw_text if answer else "",
        grade_scope=grade.scope if grade else None,
        grade_confidence=grade.confidence if grade else None,
        grade_needs_more_context=grade.needs_more_context if grade else None,
        rewrite_count=final["rewrite_count"],
    )


# --- CLI: one real question end-to-end --------------------------------------------------------


def _run_smoke(question_id: str, arm: str, condition: str) -> None:
    from src.cache import CachingEmbedder
    from src.data import load_questions_by_ids
    from src.index import OpenAIEmbedder, open_collection
    from src.preflight import load_pinned_question_ids
    from src.provenance import ProvenanceLogger, run_manifest_path

    config = Config()
    # Pinned ids, never the sampler — the smoke path must address the same question set the
    # experiment does, or a green smoke run says nothing about the run that follows it.
    pinned_ids = load_pinned_question_ids()
    questions = {q["question_id"]: q for q in load_questions_by_ids(config, pinned_ids)}
    if question_id not in questions:
        raise SystemExit(f"question_id {question_id!r} is not in the pinned set / index.")
    question = questions[question_id]["question"]

    collection = open_collection(config)
    embedder = CachingEmbedder(OpenAIEmbedder(config), config)
    graph = build_agent(config, collection=collection, embedder=embedder)

    run_id = f"smoke-{arm}-{question_id}"
    logger = ProvenanceLogger.for_run(run_id, config)
    try:
        result = run_question(
            graph,
            question,
            question_id,
            arm=arm,
            condition=condition,
            variant=config.schema_variant,
            run_id=run_id,
            logger=logger,
        )
    finally:
        logger.close()

    print(f"Question [{question_id}] ({arm}): {question}")
    print(f"  answer      : {result.answer!r}")
    print(f"  confidence  : {result.confidence}")
    print(f"  scope       : {result.scope}")
    print(f"  support ids : {result.supporting_doc_ids}")
    print(f"  grade       : scope={result.grade_scope} confidence={result.grade_confidence} "
          f"needs_more={result.grade_needs_more_context}")
    print(f"  rewrites    : {result.rewrite_count}")

    records = [
        json.loads(line)
        for line in run_manifest_path(run_id, config).read_text(encoding="utf-8").splitlines()
    ]
    nodes = [record["node"] for record in records]
    print(f"  provenance  : {len(records)} records, nodes={nodes}")
    expected = (["grade", "synthesize"], ["grade", "rewrite", "synthesize"])
    assert nodes in expected, f"unexpected node sequence: {nodes}"
    print("Smoke OK: provenance manifest received the expected records.")


def main(argv: list[str] | None = None) -> None:
    """CLI: run ONE real question end-to-end against the persisted index (real OpenAI call)."""
    parser = argparse.ArgumentParser(
        prog="python -m src.agent",
        description="Run one real question end-to-end through the agent and check provenance.",
    )
    parser.add_argument("--smoke", metavar="QUESTION_ID", required=True, help="question_id to run")
    parser.add_argument("--arm", choices=[Arm.FREE.value, Arm.ENUM.value], default=Arm.FREE.value)
    parser.add_argument(
        "--condition",
        required=True,  # never defaulted: the prompt wiring must always be stated
        choices=[c.value for c in Condition],
    )
    args = parser.parse_args(argv)
    _run_smoke(args.smoke, args.arm, args.condition)


if __name__ == "__main__":
    main()
