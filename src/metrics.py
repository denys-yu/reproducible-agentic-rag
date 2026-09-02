"""Agreement, quality, and statistical metrics over the provenance manifests.

Reads `runs/<arm>_run<i>/run_manifest.jsonl`, restricts to questions with COMPLETE data in all k
runs of BOTH arms, and computes — per arm — categorical inter-run agreement (TARa@k, EMA@k, Cohen's
and Fleiss' kappa, TARr@k), final-answer agreement, optional BERTScore-F1, and EM/token-F1 quality
vs HotpotQA gold. It then compares enum vs free PAIRED by question (Wilcoxon, bootstrap CI, Cliff's
delta) and reports the per-arm rewrite rate. Offline by default; only the optional `--cosine` flag
hits the API. This module reads runs and returns plain data — it never makes LLM calls itself.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config
from src.provenance import classify_parse_status

# Categorical fields whose inter-run agreement we report, as (node, parsed-key).
CATEGORICAL_FIELDS: list[tuple[str, str]] = [
    ("synthesize", "confidence"),
    ("synthesize", "scope"),
    ("grade", "confidence"),
    ("grade", "scope"),
    ("grade", "needs_more_context"),
]
_BOOTSTRAP_ITERS = 1000


def _field_label(node: str, key: str) -> str:
    return f"{node}.{key}"


# --- HotpotQA answer normalization ------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Official HotpotQA/SQuAD normalization: lowercase, strip punctuation/articles, fix whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    """Normalized exact match (1.0 / 0.0)."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """HotpotQA token-level F1 over normalized tokens."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_gold(prediction: str, gold: str) -> float:
    """1.0 if the gold's normalized tokens are a CONTIGUOUS sub-sequence of the answer's tokens.

    Token-level (not raw substring), so e.g. gold "yes" does not match inside "yesterday". This
    disambiguates a verbosity/format artifact (a fuller answer that still contains the gold span)
    from a real correctness loss.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return 1.0  # empty sequence is trivially contained
    span = len(gold_tokens)
    for start in range(len(pred_tokens) - span + 1):
        if pred_tokens[start : start + span] == gold_tokens:
            return 1.0
    return 0.0


# --- manifest loading -------------------------------------------------------------------------


def _parse_run_id(run_id: str) -> tuple[str, int]:
    arm, _, index = run_id.rpartition("_run")
    return arm, int(index)


@dataclass
class Dataset:
    """Per-(arm, question, run) parsed values, restricted to complete questions."""

    arms: list[str]
    run_indices: list[int]
    complete_qids: list[str]
    excluded_qids: list[str]
    _slots: dict[tuple[str, str, int], dict[str, Any]]
    #: One row per LLM call, for the descriptive endpoints of section 6.3/6.4. Kept separate from
    #: `_slots` because these are per-CALL facts (parse_status, node) that the agreement machinery
    #: deliberately does not see.
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.run_indices)

    def _slot(self, arm: str, qid: str, run: int) -> dict[str, Any]:
        return self._slots[(arm, qid, run)]

    def field_tokens(self, arm: str, node: str, key: str, qid: str) -> list[Any]:
        """Hashable token per run for a categorical field (a missing value -> `MISSING`)."""
        return [_token(self._slot(arm, qid, r)[node].get(key)) for r in self.run_indices]

    def answers(self, arm: str, qid: str, *, normalized: bool) -> list[str]:
        out = []
        for r in self.run_indices:
            ans = self._slot(arm, qid, r)["synthesize"].get("answer") or ""
            out.append(normalize_answer(ans) if normalized else ans)
        return out

    def synth_raw(self, arm: str, qid: str) -> list[str]:
        return [self._slot(arm, qid, r)["synth_raw"] or "" for r in self.run_indices]

    def rewrote(self, arm: str, qid: str) -> list[bool]:
        return [self._slot(arm, qid, r)["rewrote"] for r in self.run_indices]


class _Missing:
    """The absence of a model-produced value, as a unique object.

    This was a string (`"None"`) and collided by construction with a real label: `grade.scope`
    legitimately takes the value `none`, one capital letter away, and any node that ever emitted
    the literal text `None` would have been silently read as "no answer". A sentinel that lives in
    the same value space as the data cannot be distinguished from it. This one is a singleton with
    identity equality, so no string, number, or bool a model can emit is ever equal to it.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<MISSING>"


#: Sole instance. Compare with `is` or `==`; both are identity for this type.
MISSING = _Missing()


def _token(value: Any) -> Any:
    return MISSING if value is None else str(value)


def load_manifests(runs_dir: Path) -> Dataset:
    """Parse all manifests and restrict to questions complete in every run of every arm."""
    runs_dir = Path(runs_dir)
    slots: dict[tuple[str, str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    arms_seen: set[str] = set()
    runs_seen: set[int] = set()
    qids_seen: set[str] = set()

    # `run_manifest*.jsonl`, not `run_manifest.jsonl`: the published series shipped as
    # `run_manifest_enum_run1.jsonl` and `run_manifest__enum_run3.jsonl` (note the double
    # underscore). The narrow glob matched none of them and returned an EMPTY dataset without
    # raising — a silent zero, which is the worst way for an aggregator to fail. The `.jsonl`
    # suffix still excludes the run-level `run_manifest.json`.
    manifests = sorted(runs_dir.glob("*/run_manifest*.jsonl"))
    if not manifests:
        raise FileNotFoundError(
            f"No per-call manifests found under {runs_dir} (looked for */run_manifest*.jsonl). "
            "Refusing to return an empty dataset that would read as 'no disagreement'."
        )
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            arm, run = _parse_run_id(record["run_id"])
            qid = record["question_id"]
            arms_seen.add(arm)
            runs_seen.add(run)
            qids_seen.add(qid)
            slot = slots.setdefault(
                (arm, qid, run),
                {
                    "grade": None,
                    "synthesize": None,
                    "synth_raw": None,
                    "rewrote": False,
                    "grade_parse_status": None,
                },
            )
            rows.append(
                {
                    "model": record.get("model"),
                    "condition": record.get("condition"),
                    "arm": arm,
                    "node": record["node"],
                    "run_index": run,
                    "question_id": qid,
                    # The published series predates the `parse_status` field, so it is derived
                    # from the parsed payload when absent. Derivation uses the SAME classifier the
                    # live harness writes, so old and new manifests are directly comparable.
                    "parse_status": (
                        record.get("parse_status")
                        or classify_parse_status(record["node"], record.get("parsed"))
                    ),
                    "output_shape": record.get("output_shape"),
                }
            )
            node = record["node"]
            if node == "grade":
                slot["grade"] = record.get("parsed")
                slot["grade_parse_status"] = record.get("parse_status") or classify_parse_status(
                    "grade", record.get("parsed")
                )
            elif node == "synthesize":
                slot["synthesize"] = record.get("parsed")
                slot["synth_raw"] = record.get("raw_response")
            elif node == "rewrite":
                slot["rewrote"] = True

    arms = sorted(arms_seen)
    run_indices = sorted(runs_seen)

    complete, excluded = [], []
    for qid in sorted(qids_seen):
        ok = all(
            (arm, qid, run) in slots
            and slots[(arm, qid, run)]["grade"] is not None
            and slots[(arm, qid, run)]["synthesize"] is not None
            for arm in arms
            for run in run_indices
        )
        (complete if ok else excluded).append(qid)

    return Dataset(arms, run_indices, complete, excluded, slots, rows)


# --- agreement primitives ---------------------------------------------------------------------


# Pre-registration section 6.1: None is NOT a category. Two failed parses agree on nothing — they
# are two absences of evidence — yet scoring them as a match made a model that always fails look
# perfectly stable, inverting the quantity this paper measures. Every agreement primitive below
# therefore drops any PAIR in which either side is None, and reports coverage alongside the figure.
_NONE_TOKEN = MISSING  # `_token()` renders a missing value as this unique object


def _scored_pairs(tokens: Sequence[Any]) -> list[tuple[int, int]]:
    """Run-index pairs where NEITHER side is None — the only pairs that carry evidence."""
    return [
        (a, b)
        for a, b in combinations(range(len(tokens)), 2)
        if tokens[a] != _NONE_TOKEN and tokens[b] != _NONE_TOKEN
    ]


def pair_coverage(per_question: Sequence[Sequence[Any]]) -> float:
    """Scored pairs over available pairs (section 6.2). NaN when there are no pairs at all."""
    scored = total = 0
    for tokens in per_question:
        scored += len(_scored_pairs(tokens))
        total += len(list(combinations(range(len(tokens)), 2)))
    return float("nan") if total == 0 else scored / total


def _pairwise_agreement(tokens: Sequence[Any]) -> float | None:
    """Agreeing fraction over scored pairs; None when no pair carries evidence."""
    pairs = _scored_pairs(tokens)
    if not pairs:
        return None
    return sum(1 for a, b in pairs if tokens[a] == tokens[b]) / len(pairs)


def tar_a(per_question: Sequence[Sequence[Any]]) -> float:
    """TARa@k: fraction of SCORED questions whose non-None runs are all identical.

    A question contributes only if at least one pair survives the None filter; questions with no
    evidence are dropped from the denominator rather than counted as agreeing.
    """
    values = []
    for tokens in per_question:
        pairs = _scored_pairs(tokens)
        if not pairs:
            continue
        present = {tokens[i] for pair in pairs for i in pair}
        values.append(1.0 if len(present) == 1 else 0.0)
    return float(np.mean(values)) if values else float("nan")


def ema_a(per_question: Sequence[Sequence[Any]]) -> float:
    """EMA@k: mean over SCORED questions of the agreeing-pair fraction."""
    values = [v for v in (_pairwise_agreement(t) for t in per_question) if v is not None]
    return float(np.mean(values)) if values else float("nan")


def _drop_none_positions(a: Sequence[Any], b: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """Keep only the positions where BOTH raters produced a real label (section 6.1)."""
    kept = [(x, y) for x, y in zip(a, b, strict=True) if x != _NONE_TOKEN and y != _NONE_TOKEN]
    return [x for x, _ in kept], [y for _, y in kept]


def cohen_kappa_mean(label_matrix: Sequence[Sequence[Any]]) -> float | None:
    """Mean pairwise Cohen's kappa across runs; None when undefined (constant labels or no data)."""
    from sklearn.metrics import cohen_kappa_score

    values = []
    for r1, r2 in combinations(range(len(label_matrix)), 2):
        a, b = _drop_none_positions(label_matrix[r1], label_matrix[r2])
        if len(a) < 2 or len(set(a) | set(b)) <= 1:
            continue  # no evidence, or single category -> kappa undefined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = cohen_kappa_score(a, b)
        if not math.isnan(kappa):
            values.append(kappa)
    return float(np.mean(values)) if values else None


def _fast_cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's kappa via a numpy confusion matrix (identical to sklearn; used in hot bootstrap loop)."""
    categories = sorted(set(a) | set(b))
    index = {c: i for i, c in enumerate(categories)}
    confusion = np.zeros((len(categories), len(categories)))
    for x, y in zip(a, b, strict=True):
        confusion[index[x], index[y]] += 1
    total = confusion.sum()
    p_o = np.trace(confusion) / total
    p_e = float(((confusion.sum(axis=1) / total) * (confusion.sum(axis=0) / total)).sum())
    return float("nan") if math.isclose(p_e, 1.0) else (p_o - p_e) / (1 - p_e)


def _fast_cohen_kappa_mean(label_matrix: Sequence[Sequence[Any]]) -> float | None:
    """Mean pairwise Cohen's kappa using the fast numpy implementation; None when undefined."""
    values = []
    for r1, r2 in combinations(range(len(label_matrix)), 2):
        a, b = _drop_none_positions(label_matrix[r1], label_matrix[r2])
        if len(a) < 2 or len(set(a) | set(b)) <= 1:
            continue
        kappa = _fast_cohen_kappa(a, b)
        if not math.isnan(kappa):
            values.append(kappa)
    return float(np.mean(values)) if values else None


def fleiss_kappa(label_matrix: Sequence[Sequence[Any]]) -> float | None:
    """Fleiss' kappa over k raters (runs); None when degenerate (single category)."""
    # Fleiss is an all-rater statistic, so a question is usable only if EVERY run produced a real
    # label; a question with any None is dropped whole rather than scored on a partial rater set.
    keep = [
        q
        for q in range(len(label_matrix[0]))
        if all(run[q] != _NONE_TOKEN for run in label_matrix)
    ]
    label_matrix = [[run[q] for q in keep] for run in label_matrix]
    categories = sorted({token for run in label_matrix for token in run})
    if len(categories) <= 1:
        return None
    k = len(label_matrix)
    n = len(label_matrix[0]) if label_matrix else 0
    if k < 2 or n == 0:
        return None
    counts = np.zeros((n, len(categories)), dtype=float)
    index = {c: j for j, c in enumerate(categories)}
    for r in range(k):
        for q in range(n):
            counts[q, index[label_matrix[r][q]]] += 1
    p_j = counts.sum(axis=0) / (n * k)
    p_i = (np.square(counts).sum(axis=1) - k) / (k * (k - 1))
    p_bar = p_i.mean()
    p_e = float(np.square(p_j).sum())
    if math.isclose(p_e, 1.0):
        return None
    return float((p_bar - p_e) / (1 - p_e))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size for dominance of a over b."""
    if not a or not b:
        return float("nan")
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (greater - less) / (len(a) * len(b))


# --- multiplicity correction ------------------------------------------------------------------

#: The secondary family fixed in pre-registration section 1, after synthesize.confidence was
#: removed as a bijective duplicate of synthesize.scope. FIVE endpoints, not six.
SECONDARY_FAMILY: tuple[str, ...] = (
    "grade.needs_more_context",
    "grade.confidence",
    "grade.scope",
    "synthesize.scope",
    "answer.normalized",
)


def holm_correction(pvalues: dict[str, float], family: Sequence[str] = SECONDARY_FAMILY) -> dict[str, Any]:
    """Holm-Bonferroni step-down over the registered family. Returns per-endpoint verdicts.

    Sorted ascending, endpoint i (1-based) is compared against alpha / (m - i + 1). The procedure
    stops at the first non-rejection: everything after it is retained regardless of its own
    threshold, which is what makes Holm valid and what a naive per-endpoint comparison gets wrong.
    """
    alpha = 0.05
    present = [name for name in family if name in pvalues and pvalues[name] is not None]
    m = len(present)
    ordered = sorted(present, key=lambda name: pvalues[name])

    out: dict[str, Any] = {}
    still_rejecting = True
    for i, name in enumerate(ordered, start=1):
        threshold = alpha / (m - i + 1)
        if still_rejecting and pvalues[name] <= threshold:
            verdict = "reject"
        else:
            still_rejecting = False
            verdict = "retain"
        out[name] = {
            "raw_p": pvalues[name],
            "rank": i,
            "threshold": threshold,
            "holm_adjusted_p": min(1.0, max(
                pvalues[ordered[j]] * (m - j) for j in range(i)
            )),
            "verdict": verdict,
        }
    return {"family_size": m, "alpha": alpha, "endpoints": out}


# --- descriptive endpoints (pre-registration section 6.3 / 6.4) -------------------------------


def unparsed_rate(ds: Dataset) -> dict[str, dict[str, Any]]:
    """unparsed_rate per (model, condition, arm, node) — section 6.3.

    A first-class descriptive endpoint, NOT part of the corrected family: it takes no correction
    and supports no stability claim. It is computed over EVERY call, including questions the
    agreement machinery excluded, because a parse failure is exactly the event that would make a
    question incomplete and so would otherwise vanish from the record.
    """
    buckets: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for row in ds.rows:
        key = (row["model"], row["condition"], row["arm"], row["node"])
        buckets[key].append(row["parse_status"])
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(buckets, key=lambda k: tuple(str(x) for x in k)):
        statuses = buckets[key]
        unparsed = sum(1 for s in statuses if s == "unparsed")
        out["|".join(str(x) for x in key)] = {
            "model": key[0],
            "condition": key[1],
            "arm": key[2],
            "node": key[3],
            "n": len(statuses),
            "unparsed": unparsed,
            "unparsed_rate": unparsed / len(statuses) if statuses else float("nan"),
            "status_counts": dict(Counter(statuses)),
        }
    return out


def route_on_missing_signal(ds: Dataset) -> dict[str, Any]:
    """Cross-tabulate GRADE parse_status against the rewrite decision actually taken — section 6.4.

    When the grade output does not parse, `needs_more_context` is None and the router treats None
    as False, so the rewrite branch is never taken. That is a routing consequence of a parsing
    failure, and it is invisible in any agreement number: the arm simply looks decisive. This
    tabulates it directly, per arm, over every (question, run) slot in the cell.
    """
    table: dict[tuple[str, str, bool], int] = defaultdict(int)
    for (arm, _qid, _run), slot in ds._slots.items():
        status = slot.get("grade_parse_status")
        if status is None:
            continue
        table[(arm, status, bool(slot["rewrote"]))] += 1

    out: dict[str, Any] = {}
    for arm in sorted({key[0] for key in table}):
        rows: dict[str, Any] = {}
        for status in sorted({key[1] for key in table if key[0] == arm}):
            rewrote = table.get((arm, status, True), 0)
            not_rewrote = table.get((arm, status, False), 0)
            total = rewrote + not_rewrote
            rows[status] = {
                "rewrote": rewrote,
                "did_not_rewrite": not_rewrote,
                "n": total,
                "rewrite_rate": rewrote / total if total else float("nan"),
            }
        out[arm] = rows
    return out


# --- per-arm metrics --------------------------------------------------------------------------


def _matrix(per_question: list[list[Any]], k: int) -> list[list[Any]]:
    """Transpose per-question token lists into a runs x questions label matrix."""
    return [[per_question[qi][r] for qi in range(len(per_question))] for r in range(k)]


#: Below this, a figure is marked in every table and may not support a stability claim (6.2).
LOW_COVERAGE_THRESHOLD = 0.5


def usable_agreement(coverage: float, ema: float) -> float:
    """coverage x EMA — the probability a question yields a scorable pair that also agrees (6.6).

    Under 6.1 an arm that mostly fails to parse can post EMA 1.000 on the handful of pairs it does
    produce, which reads as MORE stable than an arm that answers every time. Multiplying by
    coverage restores the comparison: the luna free grade cell scores 1.000 x 0.133 = 0.133 against
    the enum arm's 0.933 x 1.000 = 0.933.

    Applied to EMA (and TAR) only. Kappa is chance-corrected and does not compose this way, so it
    is deliberately left alone.
    """
    if coverage != coverage or ema != ema:  # NaN in either input
        return float("nan")
    return coverage * ema


def _field_metrics(ds: Dataset, arm: str, node: str, key: str) -> dict[str, Any]:
    per_q = [ds.field_tokens(arm, node, key, qid) for qid in ds.complete_qids]
    matrix = _matrix(per_q, ds.k)
    coverage = pair_coverage(per_q)
    ema = ema_a(per_q)
    return {
        "tar_a": tar_a(per_q),
        "ema": ema,
        "cohen_kappa": cohen_kappa_mean(matrix),
        "fleiss_kappa": fleiss_kappa(matrix),
        # Section 6.2: coverage travels WITH the figure, never in a footnote. A high agreement over
        # a handful of surviving pairs is not the same claim as the same number over all of them.
        "coverage": coverage,
        "low_coverage": bool(coverage == coverage and coverage < LOW_COVERAGE_THRESHOLD),
        # Section 6.6: the probability that a question yields a scorable pair AND that pair agrees.
        # Deliberately NOT computed for kappa, which is chance-corrected and does not compose.
        "usable_agreement": usable_agreement(coverage, ema),
    }


def _answer_metrics(ds: Dataset, arm: str) -> dict[str, Any]:
    norm_q = [ds.answers(arm, qid, normalized=True) for qid in ds.complete_qids]
    raw_q = [ds.answers(arm, qid, normalized=False) for qid in ds.complete_qids]
    resp_q = [ds.synth_raw(arm, qid) for qid in ds.complete_qids]
    return {
        "tar_a_normalized": tar_a(norm_q),
        "ema_normalized": ema_a(norm_q),
        "cohen_kappa_normalized": cohen_kappa_mean(_matrix(norm_q, ds.k)),
        "tar_r_raw_answer": tar_a(raw_q),  # all k raw answer strings identical
        "tar_r_raw_response": tar_a(resp_q),  # all k raw response strings identical (most stringent)
        # An answer always exists (the free parser falls back to the raw text), so coverage here is
        # 1.0 by construction and usable_agreement collapses to EMA. Reported anyway so every
        # agreement figure in the output carries the same three columns.
        "coverage": pair_coverage(norm_q),
        "usable_agreement": usable_agreement(pair_coverage(norm_q), ema_a(norm_q)),
    }


def _quality(ds: Dataset, arm: str, gold: dict[str, str]) -> dict[str, Any]:
    em_per_run, f1_per_run, contain_per_run = [], [], []
    for r_idx in range(ds.k):
        ems, f1s, contains = [], [], []
        for qid in ds.complete_qids:
            answer = ds.answers(arm, qid, normalized=False)[r_idx]
            ems.append(exact_match(answer, gold[qid]))
            f1s.append(token_f1(answer, gold[qid]))
            contains.append(contains_gold(answer, gold[qid]))
        em_per_run.append(float(np.mean(ems)))
        f1_per_run.append(float(np.mean(f1s)))
        contain_per_run.append(float(np.mean(contains)))

    # Answer-length stats over every (run, question) answer, to quantify verbosity directly.
    norm_token_lengths, raw_char_lengths = [], []
    for qid in ds.complete_qids:
        for answer in ds.answers(arm, qid, normalized=False):
            norm_token_lengths.append(len(normalize_answer(answer).split()))
            raw_char_lengths.append(len(answer))

    return {
        "em_mean": float(np.mean(em_per_run)),
        "em_std": float(np.std(em_per_run)),
        "f1_mean": float(np.mean(f1_per_run)),
        "f1_std": float(np.std(f1_per_run)),
        "containment_mean": float(np.mean(contain_per_run)),
        "containment_std": float(np.std(contain_per_run)),
        "em_per_run": em_per_run,
        "f1_per_run": f1_per_run,
        "containment_per_run": contain_per_run,
        "answer_len_tokens_mean": float(np.mean(norm_token_lengths)) if norm_token_lengths else 0.0,
        "answer_len_chars_mean": float(np.mean(raw_char_lengths)) if raw_char_lengths else 0.0,
    }


def rewrite_rate(ds: Dataset, arm: str) -> float:
    """Fraction of pipeline runs (arm x question x run) that triggered a rewrite."""
    flags = [flag for qid in ds.complete_qids for flag in ds.rewrote(arm, qid)]
    return float(np.mean(flags)) if flags else 0.0


def bertscore_f1(ds: Dataset, arm: str, config: Config) -> float:
    """Mean pairwise BERTScore-F1 between run answers (per question, then over questions).

    Uses a `BERTScorer` so we can clamp the tokenizer's `model_max_length`: the
    deberta-xlarge-mnli tokenizer ships a sentinel-huge value that overflows the Rust tokenizer's
    truncation (`OverflowError: int too big to convert`). HotpotQA answers are short, so a 512
    bound never truncates meaningfully.
    """
    from bert_score import BERTScorer

    cands, refs, owner = [], [], []
    for qi, qid in enumerate(ds.complete_qids):
        answers = ds.answers(arm, qid, normalized=False)
        for i, j in combinations(range(len(answers)), 2):
            cands.append(answers[i])
            refs.append(answers[j])
            owner.append(qi)
    if not cands:
        return float("nan")

    scorer = BERTScorer(
        model_type=config.bert_score_model,
        device=config.bert_score_device,
        rescale_with_baseline=False,
    )
    tokenizer = getattr(scorer, "_tokenizer", None)
    if tokenizer is not None and (tokenizer.model_max_length is None or tokenizer.model_max_length > 4096):
        tokenizer.model_max_length = 512  # avoid the sentinel-overflow in enable_truncation

    _, _, f1 = scorer.score(cands, refs, verbose=False)
    per_q: dict[int, list[float]] = defaultdict(list)
    for owner_qi, score in zip(owner, f1.tolist(), strict=True):
        per_q[owner_qi].append(score)
    return float(np.mean([np.mean(scores) for scores in per_q.values()]))


# --- enum-vs-free comparison (paired by question) ---------------------------------------------


def _wilcoxon_p(diffs: np.ndarray) -> tuple[float, float]:
    if len(diffs) == 0 or np.allclose(diffs, 0.0):
        return 0.0, 1.0  # no differences -> no evidence
    from scipy.stats import wilcoxon

    try:
        stat, p = wilcoxon(diffs)
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def _bootstrap_ci(
    qids: list[str], stat_fn: Callable[[list[str]], float | None], seed: int, iters: int = _BOOTSTRAP_ITERS
) -> tuple[float | None, float | None]:
    n = len(qids)
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iters):
        sample = [qids[i] for i in rng.integers(0, n, n)]
        value = stat_fn(sample)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            values.append(value)
    if not values:
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _agreement_by_qid(ds: Dataset, arm: str, tokens_fn: Callable[[str, str], list[Any]]) -> dict[str, float]:
    return {qid: _pairwise_agreement(tokens_fn(arm, qid)) for qid in ds.complete_qids}


def _compare_agreement(
    ds: Dataset, tokens_fn: Callable[[str, str], list[Any]], seed: int, kappa_field: tuple[str, str] | None
) -> dict[str, Any]:
    enum = _agreement_by_qid(ds, "enum", tokens_fn)
    free = _agreement_by_qid(ds, "free", tokens_fn)
    # Section 6.1 leaves a question undefined in an arm when no pair survives the None filter. The
    # test is PAIRED, so such a question is dropped from BOTH arms — comparing a defined value
    # against an absent one would silently reintroduce the very None-as-data error 6.1 removes.
    # `paired_coverage` records how much of the question set the comparison actually rests on.
    qids = [q for q in ds.complete_qids if enum[q] is not None and free[q] is not None]
    dropped = [q for q in ds.complete_qids if enum[q] is None or free[q] is None]
    if not qids:
        return {
            "delta_mean_agreement": None,
            "enum_mean_agreement": None,
            "free_mean_agreement": None,
            "wilcoxon_stat": None,
            "wilcoxon_p": None,
            "agreement_ci95": [None, None],
            "cliffs_delta": None,
            "paired_coverage": 0.0,
            "n_paired": 0,
            "n_dropped_undefined": len(dropped),
            "low_coverage": True,
        }
    enum_arr = np.array([enum[q] for q in qids], dtype=float)
    free_arr = np.array([free[q] for q in qids], dtype=float)
    diffs = enum_arr - free_arr

    stat, p = _wilcoxon_p(diffs)
    ci_low, ci_high = _bootstrap_ci(
        qids, lambda s: float(np.mean([enum[q] - free[q] for q in s])), seed
    )

    result = {
        "delta_mean_agreement": float(enum_arr.mean() - free_arr.mean()),
        "enum_mean_agreement": float(enum_arr.mean()),
        "free_mean_agreement": float(free_arr.mean()),
        "paired_coverage": len(qids) / len(ds.complete_qids) if ds.complete_qids else float("nan"),
        "n_paired": len(qids),
        "n_dropped_undefined": len(dropped),
        "low_coverage": bool(
            ds.complete_qids and len(qids) / len(ds.complete_qids) < LOW_COVERAGE_THRESHOLD
        ),
        "wilcoxon_stat": stat,
        "wilcoxon_p": p,
        "agreement_ci95": [ci_low, ci_high],
        "cliffs_delta": cliffs_delta(enum_arr.tolist(), free_arr.tolist()),
    }

    if kappa_field is not None:
        node, key = kappa_field
        # Point estimate uses sklearn (the reported headline kappa); the bootstrap CI uses the fast
        # numpy kappa so resampling stays tractable.
        enum_k = cohen_kappa_mean(_matrix([ds.field_tokens("enum", node, key, q) for q in qids], ds.k))
        free_k = cohen_kappa_mean(_matrix([ds.field_tokens("free", node, key, q) for q in qids], ds.k))
        result["delta_kappa"] = None if enum_k is None or free_k is None else enum_k - free_k

        def delta_kappa_fast(sample: list[str]) -> float | None:
            ek = _fast_cohen_kappa_mean(_matrix([ds.field_tokens("enum", node, key, q) for q in sample], ds.k))
            fk = _fast_cohen_kappa_mean(_matrix([ds.field_tokens("free", node, key, q) for q in sample], ds.k))
            return None if ek is None or fk is None else ek - fk

        klo, khi = _bootstrap_ci(qids, delta_kappa_fast, seed)
        result["kappa_ci95"] = [klo, khi]

    return result


def _compare_quality(ds: Dataset, gold: dict[str, str], seed: int) -> dict[str, Any]:
    qids = ds.complete_qids

    def per_q(arm: str, metric: Callable[[str, str], float]) -> dict[str, float]:
        return {
            qid: float(np.mean([metric(a, gold[qid]) for a in ds.answers(arm, qid, normalized=False)]))
            for qid in qids
        }

    def paired(enum_map: dict[str, float], free_map: dict[str, float]) -> dict[str, Any]:
        diffs = np.array([enum_map[q] - free_map[q] for q in qids])
        _, p = _wilcoxon_p(diffs)
        lo, hi = _bootstrap_ci(qids, lambda s: float(np.mean([enum_map[q] - free_map[q] for q in s])), seed)
        return {
            "delta": float(np.mean(list(enum_map.values())) - np.mean(list(free_map.values()))),
            "wilcoxon_p": p,
            "ci95": [lo, hi],
            "cliffs_delta": cliffs_delta(list(enum_map.values()), list(free_map.values())),
        }

    f1 = paired(per_q("enum", token_f1), per_q("free", token_f1))
    containment = paired(per_q("enum", contains_gold), per_q("free", contains_gold))
    return {
        "delta_f1": f1["delta"],
        "wilcoxon_p": f1["wilcoxon_p"],
        "f1_ci95": f1["ci95"],
        "cliffs_delta": f1["cliffs_delta"],
        "delta_containment": containment["delta"],
        "containment_wilcoxon_p": containment["wilcoxon_p"],
        "containment_ci95": containment["ci95"],
        "containment_cliffs_delta": containment["cliffs_delta"],
    }


# --- top-level computation --------------------------------------------------------------------


def _gold_and_meta(questions: Sequence[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    gold = {q["question_id"]: q.get("answer", "") for q in questions}
    meta = {q["question_id"]: {"type": q.get("type", ""), "level": q.get("level", "")} for q in questions}
    return gold, meta


def _field_tokens_fn(ds: Dataset, node: str, key: str) -> Callable[[str, str], list[Any]]:
    return lambda arm, qid: ds.field_tokens(arm, node, key, qid)


def _safe_optional(name: str, fn: Callable[[], float]) -> float | None:
    """Run an optional/fragile metric; on ANY failure warn and return None (never fatal)."""
    try:
        return fn()
    except Exception as exc:  # optional metric must never take down the headline report
        print(f"WARNING: {name} skipped: {exc}", file=sys.stderr)
        return None


def compute_metrics(
    config: Config,
    *,
    runs_dir: Path | None = None,
    questions: Sequence[dict[str, Any]] | None = None,
    cosine: bool = False,
    strata: bool = False,
    bertscore: bool = False,
) -> dict[str, Any]:
    """Compute all metrics over the manifests and return a machine-readable results dict."""
    runs_dir = config.runs_dir if runs_dir is None else Path(runs_dir)
    ds = load_manifests(runs_dir)

    if questions is None:
        from src.data import load_sampled_questions

        questions = load_sampled_questions(config)
    gold, meta = _gold_and_meta(questions)

    seed = config.numpy_seed
    results: dict[str, Any] = {
        "meta": {
            "runs_dir": str(runs_dir),
            "arms": ds.arms,
            "k": ds.k,
            "n_complete": len(ds.complete_qids),
            "n_excluded": len(ds.excluded_qids),
            "excluded_qids": ds.excluded_qids,
        },
        "per_arm": {},
        "comparison": {},
    }

    for arm in ds.arms:
        fields = {_field_label(n, k): _field_metrics(ds, arm, n, k) for n, k in CATEGORICAL_FIELDS}
        answer = _answer_metrics(ds, arm)
        # _safe_optional invokes immediately, so capturing the loop var `arm` is correct here.
        answer["bertscore_f1"] = (
            _safe_optional("BERTScore", lambda a=arm: bertscore_f1(ds, a, config)) if bertscore else None
        )
        answer["cosine"] = (
            _safe_optional("cosine", lambda a=arm: _cosine_agreement(ds, a, config)) if cosine else None
        )
        results["per_arm"][arm] = {
            "fields": fields,
            "answer": answer,
            "quality": _quality(ds, arm, gold),
            "rewrite_rate": rewrite_rate(ds, arm),
        }

    if "enum" in ds.arms and "free" in ds.arms and ds.complete_qids:
        comparison = {}
        for node, key in CATEGORICAL_FIELDS:
            comparison[_field_label(node, key)] = _compare_agreement(
                ds, _field_tokens_fn(ds, node, key), seed, (node, key)
            )
        comparison["answer.normalized"] = _compare_agreement(
            ds, lambda arm, qid: ds.answers(arm, qid, normalized=True), seed, None
        )
        comparison["quality"] = _compare_quality(ds, gold, seed)
        results["comparison"] = comparison

    results["unparsed_rate"] = unparsed_rate(ds)
    results["route_on_missing_signal"] = route_on_missing_signal(ds)

    if strata:
        results["strata"] = _strata(ds, meta)

    return results


def _strata(ds: Dataset, meta: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Break per-arm normalized-answer agreement down by HotpotQA type and level."""
    out: dict[str, Any] = {}
    for dimension in ("type", "level"):
        groups: dict[str, list[str]] = defaultdict(list)
        for qid in ds.complete_qids:
            groups[meta.get(qid, {}).get(dimension, "")].append(qid)
        out[dimension] = {}
        for value, qids in sorted(groups.items()):
            per_arm = {}
            for arm in ds.arms:
                norm_q = [ds.answers(arm, qid, normalized=True) for qid in qids]
                per_arm[arm] = {"tar_a": tar_a(norm_q), "ema": ema_a(norm_q)}
            out[dimension][value] = {"n": len(qids), "per_arm": per_arm}
    return out


def _cosine_agreement(ds: Dataset, arm: str, config: Config) -> float:
    """Mean pairwise cosine similarity of answer embeddings (text-embedding-3-small). Needs API."""
    from src.index import OpenAIEmbedder

    embedder = OpenAIEmbedder(config)
    per_q = []
    for qid in ds.complete_qids:
        vecs = np.array(embedder.embed_documents(ds.answers(arm, qid, normalized=False)))
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit = vecs / np.clip(norms, 1e-12, None)
        sims = [float(unit[i] @ unit[j]) for i, j in combinations(range(len(vecs)), 2)]
        per_q.append(float(np.mean(sims)) if sims else 1.0)
    return float(np.mean(per_q)) if per_q else float("nan")


# --- reporting --------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.3f}"
    return str(value)


def print_report(results: dict[str, Any]) -> None:
    meta = results["meta"]
    print(f"Runs: {meta['runs_dir']}  arms={meta['arms']}  k={meta['k']}")
    print(f"Complete questions: {meta['n_complete']}  (excluded {meta['n_excluded']})")

    for arm, data in results["per_arm"].items():
        print(f"\n=== arm: {arm} ===  rewrite_rate={_fmt(data['rewrite_rate'])}")
        print(f"  {'field':<28}{'TARa':>8}{'EMA':>8}{'CohenK':>10}{'FleissK':>10}{'coverage':>10}")
        for label, m in data["fields"].items():
            flag = "  <LOW COVERAGE>" if m.get("low_coverage") else ""
            print(f"  {label:<28}{_fmt(m['tar_a']):>8}{_fmt(m['ema']):>8}"
                  f"{_fmt(m['cohen_kappa']):>10}{_fmt(m['fleiss_kappa']):>10}"
                  f"{_fmt(m.get('coverage')):>10}{flag}")
        a = data["answer"]
        print(f"  {'answer(norm)':<28}{_fmt(a['tar_a_normalized']):>8}{_fmt(a['ema_normalized']):>8}"
              f"{_fmt(a['cohen_kappa_normalized']):>10}")
        print(f"  answer TARr: raw={_fmt(a['tar_r_raw_answer'])} response={_fmt(a['tar_r_raw_response'])}"
              f"  BERTScore-F1={_fmt(a['bertscore_f1'])}")
        q = data["quality"]
        print(f"  quality vs gold: EM={_fmt(q['em_mean'])}+/-{_fmt(q['em_std'])}  "
              f"F1={_fmt(q['f1_mean'])}+/-{_fmt(q['f1_std'])}  "
              f"Contains={_fmt(q['containment_mean'])}+/-{_fmt(q['containment_std'])}")
        print(f"  answer length: {_fmt(q['answer_len_tokens_mean'])} tokens / "
              f"{_fmt(q['answer_len_chars_mean'])} chars (mean)")

    if results.get("unparsed_rate"):
        print("\n=== unparsed_rate (descriptive, section 6.3; not in the corrected family) ===")
        print(f"  {'model':<24}{'condition':<11}{'arm':<6}{'node':<12}{'n':>5}{'unparsed':>10}{'rate':>9}")
        for entry in results["unparsed_rate"].values():
            print(f"  {str(entry['model']):<24}{str(entry['condition']):<11}{entry['arm']:<6}"
                  f"{entry['node']:<12}{entry['n']:>5}{entry['unparsed']:>10}"
                  f"{_fmt(entry['unparsed_rate']):>9}")

    if results.get("route_on_missing_signal"):
        print("\n=== route_on_missing_signal (section 6.4): grade parse_status x rewrite taken ===")
        print(f"  {'arm':<6}{'grade parse_status':<20}{'n':>5}{'rewrote':>9}{'no rewrite':>12}{'rate':>8}")
        for arm, rows in results["route_on_missing_signal"].items():
            for status, cell in rows.items():
                print(f"  {arm:<6}{status:<20}{cell['n']:>5}{cell['rewrote']:>9}"
                      f"{cell['did_not_rewrite']:>12}{_fmt(cell['rewrite_rate']):>8}")

    if results["comparison"]:
        print("\n=== enum vs free (paired by question; delta = enum - free) ===")
        for label, c in results["comparison"].items():
            if label == "quality":
                print(f"  {'d_F1':<24}{_fmt(c['delta_f1']):>8}  p={_fmt(c['wilcoxon_p'])}  "
                      f"CI95={[_fmt(x) for x in c['f1_ci95']]}  cliffs_d={_fmt(c['cliffs_delta'])}")
                print(f"  {'d_containment':<24}{_fmt(c['delta_containment']):>8}  "
                      f"p={_fmt(c['containment_wilcoxon_p'])}  "
                      f"CI95={[_fmt(x) for x in c['containment_ci95']]}  "
                      f"cliffs_d={_fmt(c['containment_cliffs_delta'])}")
                continue
            line = (f"  {label:<24}{_fmt(c['delta_mean_agreement']):>8}  p={_fmt(c['wilcoxon_p'])}  "
                    f"CI95={[_fmt(x) for x in c['agreement_ci95']]}  cliffs_d={_fmt(c['cliffs_delta'])}")
            if "delta_kappa" in c:
                line += f"  d_kappa={_fmt(c['delta_kappa'])}"
            if "paired_coverage" in c:
                line += f"  cov={_fmt(c['paired_coverage'])}(n={c['n_paired']})"
                if c.get("low_coverage"):
                    line += "  <LOW COVERAGE>"
            print(line)


def _sanitize(obj: Any) -> Any:
    """Recursively replace nan/inf with None so results.json is valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main(argv: list[str] | None = None) -> None:
    """Compute metrics over the manifests, print a report, and write results.json."""
    parser = argparse.ArgumentParser(
        prog="python -m src.metrics",
        description="Compute inter-run agreement + quality metrics over the run manifests.",
    )
    parser.add_argument("--runs-dir", type=Path, default=None, help="manifests dir (default: config)")
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="also compute BERTScore-F1 (downloads deberta-xlarge-mnli; OFF by default)",
    )
    parser.add_argument("--cosine", action="store_true", help="also compute answer cosine (needs API)")
    parser.add_argument("--strata", action="store_true", help="break results down by type/level")
    args = parser.parse_args(argv)

    config = Config()
    results = compute_metrics(
        config,
        runs_dir=args.runs_dir,
        cosine=args.cosine,
        strata=args.strata,
        bertscore=args.bertscore,
    )
    print_report(results)
    args.out.write_text(json.dumps(_sanitize(results), indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
