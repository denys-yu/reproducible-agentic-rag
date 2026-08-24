"""Tests for the agreement convention fixed in preregistration section 6 (2026-08-24).

6.1 None is not a category: a pair where either side is None is excluded from kappa, EMA and TAR.
6.2 Coverage travels with every figure; below 0.5 it is flagged.
6.3 unparsed_rate is a first-class descriptive endpoint.
6.4 route_on_missing_signal cross-tabulates grade parse_status against the route actually taken.

Offline. No API calls, no manifests written.
"""

from __future__ import annotations

import math

import pytest

from src.metrics import (
    LOW_COVERAGE_THRESHOLD,
    Dataset,
    _pairwise_agreement,
    cohen_kappa_mean,
    ema_a,
    fleiss_kappa,
    pair_coverage,
    route_on_missing_signal,
    tar_a,
    unparsed_rate,
)

NONE = "None"


# --- 6.1 None is not a category ----------------------------------------------------------------


def test_two_failed_parses_are_not_agreement():
    """The defect this convention exists to remove: None == None scored as a match.

    A model that fails to parse on every run previously showed perfect stability. That inverts the
    quantity the paper measures, so a pair with a None on either side now carries no evidence.
    """
    assert _pairwise_agreement([NONE, NONE]) is None
    assert ema_a([[NONE, NONE]]) != 1.0  # NaN, not perfect agreement
    assert tar_a([[NONE, NONE]]) != 1.0


def test_a_pair_with_one_none_is_excluded_not_counted_as_disagreement():
    """Excluded, not scored zero — an absence of evidence is not evidence of instability."""
    # One question, 3 runs: full/full/None. The full-full pair is the only scored pair.
    assert _pairwise_agreement(["full", "full", NONE]) == 1.0
    # And a genuine disagreement still registers.
    assert _pairwise_agreement(["full", "none", NONE]) == 0.0


def test_ema_and_tar_drop_unscoreable_questions_from_the_denominator():
    per_q = [["full", "full"], [NONE, NONE], ["full", "none"]]
    # Only questions 0 and 2 are scoreable: one agrees, one does not.
    assert ema_a(per_q) == 0.5
    assert tar_a(per_q) == 0.5


def test_kappa_ignores_positions_where_either_run_is_none():
    # Run A and run B agree on every position that carries evidence.
    matrix = [["full", "none", NONE, "full"], ["full", "none", "full", NONE]]
    kappa = cohen_kappa_mean(matrix)
    assert kappa == 1.0  # scored on the two complete positions only


def test_kappa_is_undefined_when_nothing_survives_the_filter():
    assert cohen_kappa_mean([[NONE, NONE], [NONE, NONE]]) is None
    assert fleiss_kappa([[NONE, NONE], [NONE, NONE]]) is None


def test_fleiss_drops_a_question_unless_every_run_produced_a_label():
    """Fleiss is an all-rater statistic, so a partial rater set is not scored."""
    matrix = [["full", "full", NONE], ["full", "none", "full"]]
    # Only positions 0 and 1 are complete; position 2 is dropped whole.
    assert fleiss_kappa(matrix) is not None


# --- 6.2 coverage ------------------------------------------------------------------------------


def test_pair_coverage_counts_scored_over_available():
    assert pair_coverage([["a", "b"]]) == 1.0
    assert pair_coverage([[NONE, "b"]]) == 0.0
    # Two questions, one scoreable pair of two available.
    assert pair_coverage([["a", "b"], [NONE, "b"]]) == 0.5


def test_low_coverage_threshold_is_the_registered_one():
    assert LOW_COVERAGE_THRESHOLD == 0.5


def test_high_agreement_over_few_pairs_is_flagged(tmp_path):
    """The luna failure mode: EMA 1.000 on 2 of 15 questions must not read as stability."""
    from src.metrics import _field_metrics

    per_slot = {}
    for run in (1, 2):
        for i in range(15):
            qid = f"q{i}"
            # Only two questions parse; the rest are None on both runs.
            parsed = {"scope": "full"} if i < 2 else {"scope": None}
            per_slot[("free", qid, run)] = {
                "grade": parsed,
                "synthesize": {"answer": "x"},
                "synth_raw": "x",
                "rewrote": False,
                "grade_parse_status": "ok" if i < 2 else "unparsed",
            }
    ds = Dataset(["free"], [1, 2], [f"q{i}" for i in range(15)], [], per_slot, [])
    metrics = _field_metrics(ds, "free", "grade", "scope")
    assert metrics["ema"] == 1.0          # the surviving pairs do agree
    assert metrics["coverage"] < 0.5      # but on almost nothing
    assert metrics["low_coverage"] is True


# --- 6.6 usable_agreement ----------------------------------------------------------------------


def test_usable_agreement_is_coverage_times_ema():
    """The composite that stops sparse-but-consistent from outranking dense-and-consistent.

    The luna cell is the motivating case: EMA 1.000 at coverage 0.133 must not read as more stable
    than EMA 0.933 at coverage 1.000. Multiplying restores the ordering.
    """
    from src.metrics import usable_agreement

    assert usable_agreement(1.0, 0.9333) == pytest.approx(0.9333)
    sparse = usable_agreement(0.1333, 1.000)
    dense = usable_agreement(1.0, 0.9333)
    assert sparse == pytest.approx(0.1333)
    assert sparse < dense  # the whole point of the composite

    # NaN in either input propagates rather than silently scoring 0.
    assert math.isnan(usable_agreement(float("nan"), 1.0))
    assert math.isnan(usable_agreement(1.0, float("nan")))


def test_usable_agreement_is_not_applied_to_kappa():
    """6.6 applies the composite to EMA/TAR only — kappa is chance-corrected and does not compose."""
    from src.metrics import _field_metrics

    slots = {}
    for run in (1, 2):
        for i in range(4):
            slots[("free", f"q{i}", run)] = {
                "grade": {"scope": "full" if i % 2 else "none"},
                "synthesize": {"answer": "x"},
                "synth_raw": "x",
                "rewrote": False,
                "grade_parse_status": "ok",
            }
    ds = Dataset(["free"], [1, 2], [f"q{i}" for i in range(4)], [], slots, [])
    metrics = _field_metrics(ds, "free", "grade", "scope")
    assert "usable_agreement" in metrics
    assert metrics["usable_agreement"] == pytest.approx(
        metrics["coverage"] * metrics["ema"]
    )
    # There is deliberately no kappa-composite key.
    assert not any(key.startswith("usable") and "kappa" in key for key in metrics)


# --- 6.3 / 6.4 descriptive endpoints -----------------------------------------------------------


def _dataset_with_rows(rows, slots):
    qids = sorted({r["question_id"] for r in rows})
    return Dataset(["free"], [1], qids, [], slots, rows)


def test_unparsed_rate_is_reported_per_model_condition_arm_node():
    rows = [
        {"model": "m", "condition": "published", "arm": "free", "node": "grade",
         "run_index": 1, "question_id": "q1", "parse_status": "unparsed", "output_shape": "json_object"},
        {"model": "m", "condition": "published", "arm": "free", "node": "grade",
         "run_index": 1, "question_id": "q2", "parse_status": "ok", "output_shape": "labeled_lines"},
        {"model": "m", "condition": "published", "arm": "free", "node": "synthesize",
         "run_index": 1, "question_id": "q1", "parse_status": "ok", "output_shape": "prose"},
    ]
    out = unparsed_rate(_dataset_with_rows(rows, {}))
    grade = out["m|published|free|grade"]
    assert grade["n"] == 2
    assert grade["unparsed"] == 1
    assert grade["unparsed_rate"] == 0.5
    assert out["m|published|free|synthesize"]["unparsed_rate"] == 0.0


def test_route_on_missing_signal_crosstabs_parse_status_against_the_route_taken():
    """An unparsed grade yields needs_more_context=None, which the router reads as False."""
    slots = {
        ("free", "q1", 1): {"grade": {}, "synthesize": {}, "synth_raw": "",
                            "rewrote": False, "grade_parse_status": "unparsed"},
        ("free", "q2", 1): {"grade": {}, "synthesize": {}, "synth_raw": "",
                            "rewrote": False, "grade_parse_status": "unparsed"},
        ("free", "q3", 1): {"grade": {}, "synthesize": {}, "synth_raw": "",
                            "rewrote": True, "grade_parse_status": "ok"},
    }
    table = route_on_missing_signal(_dataset_with_rows([], slots))
    assert table["free"]["unparsed"]["n"] == 2
    assert table["free"]["unparsed"]["rewrote"] == 0
    assert table["free"]["unparsed"]["rewrite_rate"] == 0.0
    assert table["free"]["ok"]["rewrote"] == 1


def test_route_table_skips_slots_with_no_grade_record():
    slots = {
        ("free", "q1", 1): {"grade": None, "synthesize": {}, "synth_raw": "",
                            "rewrote": False, "grade_parse_status": None},
    }
    assert route_on_missing_signal(_dataset_with_rows([], slots)) == {}
