"""Pinned prompt constants and the condition switch that composes them.

The prompt text here is the wording recovered from commit `d6efcab` — the commit that published
`results.json` — and is byte-verbatim, em-dashes and all. It is deliberately NOT modernised: the
published series ran with "fully, partially, or not at all (none)" and "(high, medium, low)", and
restoring anything else would silently change the condition being reproduced. The phrase
"exactly one of" appears nowhere in this experiment's history and must not be introduced.

Two conditions are selectable:

- `PUBLISHED` — no format suffix anywhere. Both arms send byte-identical system prompts at every
  node. This reproduces the published series.
- `ABLATION` — identical to PUBLISHED except that the FREE arm's GRADE prompt gains
  `FREE_FORMAT_SUFFIX`. The SYNTHESIZE node must NOT receive it, in either arm. That single
  containment property is what the ablation isolates, so it is asserted the hardest.

`condition` is a REQUIRED parameter of `compose_system_prompt` with no default. There is
deliberately no code path that composes a prompt without a condition being stated.
"""

from __future__ import annotations

from typing import Literal

from src.config import Arm, Condition
from src.hashing import sha256_text

NodeName = Literal["grade", "rewrite", "synthesize"]

#: Every node the agent composes a system prompt for.
NODES: tuple[NodeName, ...] = ("grade", "rewrite", "synthesize")


# --- pinned prompts: verbatim from d6efcab ----------------------------------------------------
# Verified against docs/published_series_reference.json by `preflight.assert_condition_contract`.

GRADE_PROMPT = (
    "You assess whether retrieved context can answer a question. "
    "Given the question and the retrieved context, judge: "
    "scope — whether the context covers the question fully, partially, or not at all (none); "
    "confidence — your confidence (high, medium, low) in that judgement; "
    "needs_more_context — whether the system should search again with a reformulated query. "
    "Base your judgement only on the provided context."
)

REWRITE_PROMPT = (
    "You reformulate search queries. Given the question and the context retrieved so far, "
    "write a single improved search query that would retrieve better context to answer the "
    "question. Provide only the reformulated query."
)

SYNTHESIZE_PROMPT = (
    "You answer questions using retrieved context. Given the question and the context, provide: "
    "answer — the answer to the question; "
    "confidence — your confidence (high, medium, low); "
    "scope — whether the context covered the question fully, partially, or none; "
    "supporting_doc_ids — the doc_ids of the documents you used. "
    "Use only the provided context."
)

BASE_PROMPTS: dict[NodeName, str] = {
    "grade": GRADE_PROMPT,
    "rewrite": REWRITE_PROMPT,
    "synthesize": SYNTHESIZE_PROMPT,
}

# The free arm has no schema to carry an output-format contract, so under ABLATION the contract is
# carried in prose — on the GRADE node only. Never appended under PUBLISHED, never at SYNTHESIZE,
# and never in the enum arm (whose contract is its schema).
FREE_FORMAT_SUFFIX = (
    " Write one line per field, in this exact form:\n"
    "field_name: value\n"
    "Begin each line with the field name itself. "
    "Do not use JSON, markdown, bullet points, asterisks, or bold text, "
    "and add no commentary."
)

#: (condition, arm, node) triples that receive FREE_FORMAT_SUFFIX. Exhaustive by construction:
#: anything not listed here gets the bare base prompt. Keeping this as data rather than nested
#: conditionals is what makes the containment property auditable at a glance.
_SUFFIXED: frozenset[tuple[Condition, str, str]] = frozenset(
    {(Condition.ABLATION, Arm.FREE.value, "grade")}
)


def compose_system_prompt(node: NodeName, arm: Arm | str, condition: Condition | str) -> str:
    """Compose the system prompt actually sent for one (arm, node) under one condition.

    `condition` is required and has no default — a prompt is never composed without one being
    stated. Fails loud on an unknown node, arm, or condition rather than falling back.
    """
    if node not in BASE_PROMPTS:
        raise ValueError(f"Unknown agent node {node!r}; expected one of {NODES}.")
    arm_value = arm.value if isinstance(arm, Arm) else Arm(arm).value
    condition_value = condition if isinstance(condition, Condition) else Condition(condition)

    base = BASE_PROMPTS[node]
    if (condition_value, arm_value, node) in _SUFFIXED:
        return base + FREE_FORMAT_SUFFIX
    return base


def suffixed_nodes(arm: Arm | str, condition: Condition | str) -> list[NodeName]:
    """The nodes that receive FREE_FORMAT_SUFFIX for one (arm, condition), sorted by NODES order.

    Read straight off `_SUFFIXED`, so this can never disagree with what `compose_system_prompt`
    actually does. This is the wiring fact each provenance record carries as `format_suffix_nodes`.
    """
    arm_value = arm.value if isinstance(arm, Arm) else Arm(arm).value
    condition_value = condition if isinstance(condition, Condition) else Condition(condition)
    return [
        node
        for node in NODES
        if (condition_value, arm_value, node) in _SUFFIXED
    ]


def suffixed_nodes_for_condition(condition: Condition | str) -> list[NodeName]:
    """The nodes receiving the suffix under one condition, across BOTH arms (deduped, NODES order).

    This is the condition-level wiring the run-start assertion checks: PUBLISHED must yield the
    empty list, ABLATION exactly ["grade"].
    """
    condition_value = condition if isinstance(condition, Condition) else Condition(condition)
    return [
        node
        for node in NODES
        if any(
            (condition_value, arm.value, node) in _SUFFIXED for arm in (Arm.ENUM, Arm.FREE)
        )
    ]


def composed_prompt_table(condition: Condition | str) -> dict[str, str]:
    """Every composed system prompt under one condition, keyed `"{arm}|{node}"`."""
    return {
        f"{arm.value}|{node}": compose_system_prompt(node, arm, condition)
        for arm in (Arm.ENUM, Arm.FREE)
        for node in NODES
    }


def composed_hash_table(condition: Condition | str) -> dict[str, str]:
    """`composed_prompt_table` reduced to LF-normalised sha256 digests."""
    return {key: sha256_text(text) for key, text in composed_prompt_table(condition).items()}
