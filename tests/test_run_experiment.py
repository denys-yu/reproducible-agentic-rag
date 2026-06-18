"""Offline tests for src.run_experiment: manifest layout, run_id scheme, resume, dry-run.

Fully offline — fake per-node chat model + fake embedder + ephemeral Chroma + temp runs dir.
No OpenAI calls.
"""

from __future__ import annotations

import json

import chromadb

from src.agent import GRADE_PROMPT, REWRITE_PROMPT, SYNTHESIZE_PROMPT
from src.config import Arm, Config
from src.data import Chunk
from src.index import build_index
from src.run_experiment import run_experiment
from src.schemas import AnswerScope, AnswerV1, ConfidenceLevel, ContextGrade, RewriteQuery

_QIDS = ["q1", "q2"]


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeMessage:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {"system_fingerprint": "fp", "token_usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        self.usage_metadata = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}


def _node_of(messages):
    system = messages[0]["content"]
    if system == GRADE_PROMPT:
        return "grade"
    if system == REWRITE_PROMPT:
        return "rewrite"
    if system == SYNTHESIZE_PROMPT:
        return "synthesize"
    raise AssertionError("unknown prompt")


class _FakeStructured:
    def __init__(self, model):
        self._model = model

    def invoke(self, messages, config=None):
        node = _node_of(messages)
        self._model.calls += 1
        return {"raw": FakeMessage("{}"), "parsed": self._model.parsed[node], "parsing_error": None}


class FakeChatModel:
    """Per-node responder; no rewrites (grade.needs_more_context=False). Counts invocations."""

    def __init__(self):
        self.calls = 0
        self.parsed = {
            "grade": ContextGrade(scope=AnswerScope.FULL, confidence=ConfidenceLevel.HIGH, needs_more_context=False),
            "rewrite": RewriteQuery(query="x"),
            "synthesize": AnswerV1(answer="42", confidence="high", scope="full", supporting_doc_ids=[]),
        }
        self.free_text = {
            "grade": "Scope: full\nConfidence: high\nNeeds more context: no",
            "rewrite": "Rewritten query: x",
            "synthesize": "Answer: 42\nConfidence: high\nScope: full",
        }

    def invoke(self, messages, config=None):  # free arm
        self.calls += 1
        return FakeMessage(self.free_text[_node_of(messages)])

    def with_structured_output(self, schema, *, strict, include_raw):
        return _FakeStructured(self)


def _config(tmp_path):
    return Config(runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache")


def _collection(config):
    chunks = [
        Chunk(doc_id=f"{qid}_d", question_id=qid, title="T", text="t", is_gold=False, para_index=0, chunk_index=0)
        for qid in _QIDS
    ]
    return build_index(config, chunks=chunks, embedder=FakeEmbedder(), client=chromadb.EphemeralClient())


def _questions():
    return [{"question_id": qid, "question": f"who is {qid}?"} for qid in _QIDS]


def _run(config, model, **kwargs):
    return run_experiment(
        config,
        collection=_collection(config),
        embedder=FakeEmbedder(),
        model=model,
        questions=_questions(),
        runs=2,
        **kwargs,
    )


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- manifest layout + run_id scheme ----------------------------------------------------------


def test_produces_one_manifest_per_arm_run_with_correct_counts(tmp_path):
    config = _config(tmp_path)
    summary = _run(config, FakeChatModel())

    assert summary.runs == 2 and summary.n_questions == 2
    # 2 arms x 2 runs = 4 (arm, run) passes, each over 2 questions -> 8 invocations.
    assert summary.invocations == 8
    assert summary.arms == ["free", "enum"]

    for arm in ("free", "enum"):
        for i in (1, 2):
            run_id = f"{arm}_run{i}"  # run_id scheme
            manifest = config.runs_dir / run_id / "run_manifest.jsonl"
            assert manifest.exists()
            records = _records(manifest)
            # 2 questions x (grade + synthesize), no rewrites -> 4 records.
            assert len(records) == 4
            assert [r["node"] for r in records] == ["grade", "synthesize", "grade", "synthesize"]
            assert {r["run_id"] for r in records} == {run_id}
            assert {r["arm"] for r in records} == {arm}
            assert (config.runs_dir / run_id / ".done").exists()


# --- resume -----------------------------------------------------------------------------------


def test_resume_skips_runs_with_done_marker(tmp_path):
    config = _config(tmp_path)
    # Pre-mark free_run1 as done with a sentinel manifest that must NOT be overwritten.
    done_dir = config.runs_dir / "free_run1"
    done_dir.mkdir(parents=True)
    (done_dir / ".done").write_text("")
    (done_dir / "run_manifest.jsonl").write_text("SENTINEL\n", encoding="utf-8")

    summary = _run(config, FakeChatModel(), resume=True)

    assert "free_run1" in summary.skipped_run_ids
    assert "free_run1" not in summary.executed_run_ids
    # Skipped manifest left untouched; the other three passes ran.
    assert (done_dir / "run_manifest.jsonl").read_text(encoding="utf-8") == "SENTINEL\n"
    assert summary.invocations == 6  # 3 remaining passes x 2 questions
    assert set(summary.executed_run_ids) == {"free_run2", "enum_run1", "enum_run2"}


# --- dry-run ----------------------------------------------------------------------------------


def test_dry_run_writes_nothing_and_makes_no_calls(tmp_path):
    config = _config(tmp_path)
    model = FakeChatModel()
    summary = run_experiment(
        config,
        collection=_collection(config),
        embedder=FakeEmbedder(),
        model=model,
        questions=_questions(),
        runs=2,
        dry_run=True,
    )

    assert summary.dry_run is True
    assert summary.invocations == 0
    assert summary.n_questions == 2
    assert model.calls == 0  # no API/model calls
    assert not config.runs_dir.exists()  # nothing written


def test_dry_run_respects_limit(tmp_path):
    config = _config(tmp_path)
    summary = run_experiment(config, questions=_questions(), runs=3, limit=1, dry_run=True)
    assert summary.n_questions == 1
    assert not config.runs_dir.exists()


# --- arm selection ----------------------------------------------------------------------------


def test_single_arm_and_limit(tmp_path):
    config = _config(tmp_path)
    summary = _run(config, FakeChatModel(), arms=[Arm.ENUM], limit=1)
    assert summary.arms == ["enum"]
    assert summary.n_questions == 1
    assert summary.invocations == 2  # 1 arm x 2 runs x 1 question
    assert set(summary.executed_run_ids) == {"enum_run1", "enum_run2"}
