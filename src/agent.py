"""LangGraph agentic loop: retrieve -> grade_context -> [rewrite_query (max 1 retry)] -> synthesize.

Builds the state machine that drives one question through the pipeline, making >=2 LLM calls so
the "agentic" label is justified and capping retrieval at `config.max_retrieval_rounds` rounds.
The same graph serves both arms; only the response schema (free string vs. strict Pydantic model)
differs, selected from config.
"""

from __future__ import annotations

from typing import Any

from src.config import Config


def build_graph(config: Config) -> Any:
    """Compile the LangGraph state machine for the configured arm."""
    raise NotImplementedError


def retrieve_node(state: dict[str, Any], config: Config) -> dict[str, Any]:
    """Retrieve top-k context for the current query and append it to the state."""
    raise NotImplementedError


def grade_context_node(state: dict[str, Any], config: Config) -> dict[str, Any]:
    """LLM call: judge whether the retrieved context is sufficient to answer."""
    raise NotImplementedError


def rewrite_query_node(state: dict[str, Any], config: Config) -> dict[str, Any]:
    """LLM call: rewrite the query for a second retrieval round (at most one retry)."""
    raise NotImplementedError


def synthesize_node(state: dict[str, Any], config: Config) -> dict[str, Any]:
    """LLM call: produce the final answer (free string or strict structured model)."""
    raise NotImplementedError


def run_pipeline(question: dict[str, Any], config: Config) -> dict[str, Any]:
    """Run one question end-to-end through the compiled graph; return the terminal state."""
    raise NotImplementedError
