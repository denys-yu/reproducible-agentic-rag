"""Offline tests for src.metrics: synthesize manifests with KNOWN patterns and check the math.

Fully offline and deterministic — no API, no BERTScore download (bertscore=False).
"""

from __future__ import annotations

import json

import pytest

from src.config import Config
from src.metrics import (
    cliffs_delta,
    compute_metrics,
    contains_gold,
    exact_match,
    normalize_answer,
    token_f1,
)

_GOLD = {"q1": "Paris", "q2": "London", "q3": "Berlin", "q4": "Rome"}
_COMPLETE = ["q1", "q2", "q3"]
_ENUM_SYN_SCOPE = {"q1": "full", "q2": "partial", "q3": "full", "q4": "full"}
_ENUM_GRADE_SCOPE = {"q1": "full", "q2": "full", "q3": "partial", "q4": "full"}
_FREE_CONF = {1: "high", 2: "low", 3: "high"}  # per-run pattern (q-independent)


def _write_manifest(runs_dir, arm, run, records):
    run_dir = runs_dir / f"{arm}_run{run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "run_id": f"{arm}_run{run}",
            "question_id": rec["question_id"],
            "arm": arm,
            "node": rec["node"],
            "parsed": rec.get("parsed"),
            "raw_response": rec.get("raw_response", ""),
        })
        for rec in records
    ]
    (run_dir / "run_manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _enum_records(run, qids):
    recs = []
    for q in qids:
        recs.append({"question_id": q, "node": "grade",
                     "parsed": {"scope": _ENUM_GRADE_SCOPE[q], "confidence": "high", "needs_more_context": False}})
        recs.append({"question_id": q, "node": "synthesize",
                     "parsed": {"answer": _GOLD[q], "confidence": "high", "scope": _ENUM_SYN_SCOPE[q], "supporting_doc_ids": []},
                     "raw_response": f"answer: {_GOLD[q]}"})
    return recs


def _free_records(run, qids):
    recs = []
    conf = _FREE_CONF[run]
    for q in qids:
        answer = "wrong" if run == 2 else _GOLD[q]
        recs.append({"question_id": q, "node": "grade",
                     "parsed": {"scope": "full", "confidence": conf, "needs_more_context": False}})
        recs.append({"question_id": q, "node": "synthesize",
                     "parsed": {"answer": answer, "confidence": conf, "scope": "full", "supporting_doc_ids": []},
                     "raw_response": f"answer: {answer}"})
        if q == "q1" and run == 2:  # one rewrite -> nonzero rewrite rate
            recs.append({"question_id": "q1", "node": "rewrite", "parsed": {"query": "x"}})
    return recs


@pytest.fixture
def results(tmp_path):
    runs_dir = tmp_path / "runs"
    all_qids = ["q1", "q2", "q3", "q4"]
    for run in (1, 2, 3):
        _write_manifest(runs_dir, "enum", run, _enum_records(run, all_qids))
        # q4 is omitted from free run 3 -> incomplete -> excluded.
        free_qids = ["q1", "q2", "q3"] if run == 3 else all_qids
        _write_manifest(runs_dir, "free", run, _free_records(run, free_qids))

    config = Config(runs_dir=runs_dir)
    questions = [{"question_id": q, "answer": _GOLD[q], "type": "bridge", "level": "easy"} for q in all_qids]
    return compute_metrics(config, runs_dir=runs_dir, questions=questions, bertscore=False)


# --- normalization / EM / F1 ------------------------------------------------------------------


def test_normalize_and_em_f1():
    assert normalize_answer("The Maine Legislature!") == "maine legislature"
    assert exact_match("the Paris", "Paris ") == 1.0
    assert token_f1("wrong", "Paris") == 0.0
    assert token_f1("Paris", "Paris") == 1.0


def test_cliffs_delta_dominance():
    assert cliffs_delta([1, 1, 1], [0, 0, 0]) == 1.0
    assert cliffs_delta([0, 0], [1, 1]) == -1.0


def test_contains_gold_token_subsequence():
    verbose = "The Maine Legislature convenes at the State House in Augusta, Maine."
    assert contains_gold(verbose, "Augusta Maine") == 1.0  # contiguous gold span present
    assert contains_gold(verbose, "Boston") == 0.0  # wrong answer not contained
    assert contains_gold("yesterday", "yes") == 0.0  # token-level, not raw substring
    assert contains_gold("yes I agree", "yes") == 1.0
    # non-contiguous tokens must NOT count (gold span must be contiguous)
    assert contains_gold("augusta is in maine", "augusta maine") == 0.0


# --- completeness -----------------------------------------------------------------------------


def test_excludes_incomplete_questions(results):
    meta = results["meta"]
    assert meta["k"] == 3
    assert sorted(meta["arms"]) == ["enum", "free"]
    assert meta["n_complete"] == 3
    assert meta["excluded_qids"] == ["q4"]


# --- enum arm: fully consistent, kappa degenerate handled -------------------------------------


def test_enum_arm_full_agreement_and_kappa(results):
    enum = results["per_arm"]["enum"]
    syn_conf = enum["fields"]["synthesize.confidence"]
    assert syn_conf["tar_a"] == 1.0
    assert syn_conf["ema"] == 1.0
    assert syn_conf["cohen_kappa"] is None  # constant field -> undefined
    assert syn_conf["fleiss_kappa"] is None

    syn_scope = enum["fields"]["synthesize.scope"]
    assert syn_scope["tar_a"] == 1.0
    assert syn_scope["cohen_kappa"] == 1.0  # varies across questions, constant across runs
    assert syn_scope["fleiss_kappa"] == 1.0

    assert enum["fields"]["grade.scope"]["cohen_kappa"] == 1.0
    assert enum["fields"]["grade.needs_more_context"]["cohen_kappa"] is None

    answer = enum["answer"]
    assert answer["tar_a_normalized"] == 1.0
    assert answer["ema_normalized"] == 1.0
    assert answer["tar_r_raw_answer"] == 1.0
    assert answer["tar_r_raw_response"] == 1.0
    assert answer["bertscore_f1"] is None  # bertscore disabled in tests

    assert enum["rewrite_rate"] == 0.0


def test_enum_quality_perfect(results):
    q = results["per_arm"]["enum"]["quality"]
    assert q["em_mean"] == 1.0 and q["em_std"] == 0.0
    assert q["f1_mean"] == 1.0
    # enum answers are exactly the gold -> always contained, single-token, gold char lengths.
    assert q["containment_mean"] == 1.0 and q["containment_std"] == 0.0
    assert q["answer_len_tokens_mean"] == 1.0
    assert q["answer_len_chars_mean"] == pytest.approx((5 + 6 + 6) / 3)  # Paris/London/Berlin


# --- free arm: deliberately varying -----------------------------------------------------------


def test_free_arm_varies(results):
    free = results["per_arm"]["free"]
    syn_conf = free["fields"]["synthesize.confidence"]
    assert syn_conf["tar_a"] == 0.0  # [high, low, high] -> never all identical
    assert syn_conf["ema"] == pytest.approx(1 / 3)  # only run1 == run3

    # constant free fields stay perfectly agreeing.
    assert free["fields"]["synthesize.scope"]["tar_a"] == 1.0
    assert free["fields"]["synthesize.scope"]["cohen_kappa"] is None

    assert free["answer"]["tar_a_normalized"] == 0.0
    assert free["answer"]["ema_normalized"] == pytest.approx(1 / 3)

    # one rewrite among 3 questions x 3 runs.
    assert free["rewrite_rate"] == pytest.approx(1 / 9)


def test_free_quality_partial(results):
    q = results["per_arm"]["free"]["quality"]
    # run2 answers are all wrong -> EM/F1/containment per run = [1, 0, 1].
    assert q["em_mean"] == pytest.approx(2 / 3)
    assert q["f1_mean"] == pytest.approx(2 / 3)
    assert q["em_per_run"] == [1.0, 0.0, 1.0]
    assert q["containment_mean"] == pytest.approx(2 / 3)
    assert q["containment_per_run"] == [1.0, 0.0, 1.0]
    assert q["answer_len_tokens_mean"] == 1.0  # every answer is a single token


# --- enum vs free comparison ------------------------------------------------------------------


def test_comparison_confidence_and_quality(results):
    comp = results["comparison"]["synthesize.confidence"]
    assert comp["delta_mean_agreement"] == pytest.approx(2 / 3)  # enum 1.0 - free 1/3
    assert comp["cliffs_delta"] == 1.0  # enum dominates on every question
    assert 0.0 < comp["wilcoxon_p"] <= 1.0
    lo, hi = comp["agreement_ci95"]
    assert lo == pytest.approx(2 / 3) and hi == pytest.approx(2 / 3)  # constant diff

    quality = results["comparison"]["quality"]
    assert quality["delta_f1"] == pytest.approx(1 / 3)  # enum 1.0 - free 2/3
    assert quality["cliffs_delta"] == 1.0
    # containment moves the same way here (enum 1.0 vs free 2/3).
    assert quality["delta_containment"] == pytest.approx(1 / 3)
    assert quality["containment_cliffs_delta"] == 1.0


def test_results_are_json_serializable_after_sanitize(results):
    from src.metrics import _sanitize

    json.dumps(_sanitize(results))  # must not raise
