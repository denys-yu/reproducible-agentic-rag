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
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config

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

    @property
    def k(self) -> int:
        return len(self.run_indices)

    def _slot(self, arm: str, qid: str, run: int) -> dict[str, Any]:
        return self._slots[(arm, qid, run)]

    def field_tokens(self, arm: str, node: str, key: str, qid: str) -> list[str]:
        """Hashable token per run for a categorical field (None -> 'None')."""
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


def _token(value: Any) -> str:
    return "None" if value is None else str(value)


def load_manifests(runs_dir: Path) -> Dataset:
    """Parse all manifests and restrict to questions complete in every run of every arm."""
    runs_dir = Path(runs_dir)
    slots: dict[tuple[str, str, int], dict[str, Any]] = {}
    arms_seen: set[str] = set()
    runs_seen: set[int] = set()
    qids_seen: set[str] = set()

    for manifest in sorted(runs_dir.glob("*/run_manifest.jsonl")):
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
                {"grade": None, "synthesize": None, "synth_raw": None, "rewrote": False},
            )
            node = record["node"]
            if node == "grade":
                slot["grade"] = record.get("parsed")
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

    return Dataset(arms, run_indices, complete, excluded, slots)


# --- agreement primitives ---------------------------------------------------------------------


def _pairwise_agreement(tokens: Sequence[Any]) -> float:
    pairs = list(combinations(range(len(tokens)), 2))
    if not pairs:
        return 1.0
    return sum(1 for a, b in pairs if tokens[a] == tokens[b]) / len(pairs)


def tar_a(per_question: Sequence[Sequence[Any]]) -> float:
    """TARa@k: fraction of questions whose k runs are all identical."""
    if not per_question:
        return float("nan")
    return float(np.mean([1.0 if len(set(tokens)) == 1 else 0.0 for tokens in per_question]))


def ema_a(per_question: Sequence[Sequence[Any]]) -> float:
    """EMA@k: mean over questions of the agreeing-pair fraction."""
    if not per_question:
        return float("nan")
    return float(np.mean([_pairwise_agreement(tokens) for tokens in per_question]))


def cohen_kappa_mean(label_matrix: Sequence[Sequence[Any]]) -> float | None:
    """Mean pairwise Cohen's kappa across runs; None when undefined (constant labels)."""
    from sklearn.metrics import cohen_kappa_score

    values = []
    for r1, r2 in combinations(range(len(label_matrix)), 2):
        a, b = label_matrix[r1], label_matrix[r2]
        if len(set(a) | set(b)) <= 1:
            continue  # single category -> kappa undefined
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
        a, b = label_matrix[r1], label_matrix[r2]
        if len(set(a) | set(b)) <= 1:
            continue
        kappa = _fast_cohen_kappa(a, b)
        if not math.isnan(kappa):
            values.append(kappa)
    return float(np.mean(values)) if values else None


def fleiss_kappa(label_matrix: Sequence[Sequence[Any]]) -> float | None:
    """Fleiss' kappa over k raters (runs); None when degenerate (single category)."""
    categories = sorted({token for run in label_matrix for token in run})
    if len(categories) <= 1:
        return None
    k = len(label_matrix)
    n = len(label_matrix[0])
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


# --- per-arm metrics --------------------------------------------------------------------------


def _matrix(per_question: list[list[Any]], k: int) -> list[list[Any]]:
    """Transpose per-question token lists into a runs x questions label matrix."""
    return [[per_question[qi][r] for qi in range(len(per_question))] for r in range(k)]


def _field_metrics(ds: Dataset, arm: str, node: str, key: str) -> dict[str, Any]:
    per_q = [ds.field_tokens(arm, node, key, qid) for qid in ds.complete_qids]
    matrix = _matrix(per_q, ds.k)
    return {
        "tar_a": tar_a(per_q),
        "ema": ema_a(per_q),
        "cohen_kappa": cohen_kappa_mean(matrix),
        "fleiss_kappa": fleiss_kappa(matrix),
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
    }


def _quality(ds: Dataset, arm: str, gold: dict[str, str]) -> dict[str, Any]:
    em_per_run, f1_per_run = [], []
    for r_idx in range(ds.k):
        ems, f1s = [], []
        for qid in ds.complete_qids:
            answer = ds.answers(arm, qid, normalized=False)[r_idx]
            ems.append(exact_match(answer, gold[qid]))
            f1s.append(token_f1(answer, gold[qid]))
        em_per_run.append(float(np.mean(ems)))
        f1_per_run.append(float(np.mean(f1s)))
    return {
        "em_mean": float(np.mean(em_per_run)),
        "em_std": float(np.std(em_per_run)),
        "f1_mean": float(np.mean(f1_per_run)),
        "f1_std": float(np.std(f1_per_run)),
        "em_per_run": em_per_run,
        "f1_per_run": f1_per_run,
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
    qids = ds.complete_qids
    enum_arr = np.array([enum[q] for q in qids])
    free_arr = np.array([free[q] for q in qids])
    diffs = enum_arr - free_arr

    stat, p = _wilcoxon_p(diffs)
    ci_low, ci_high = _bootstrap_ci(
        qids, lambda s: float(np.mean([enum[q] - free[q] for q in s])), seed
    )

    result = {
        "delta_mean_agreement": float(enum_arr.mean() - free_arr.mean()),
        "enum_mean_agreement": float(enum_arr.mean()),
        "free_mean_agreement": float(free_arr.mean()),
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
    def per_q_f1(arm: str) -> dict[str, float]:
        return {
            qid: float(np.mean([token_f1(a, gold[qid]) for a in ds.answers(arm, qid, normalized=False)]))
            for qid in ds.complete_qids
        }

    enum_f1, free_f1 = per_q_f1("enum"), per_q_f1("free")
    qids = ds.complete_qids
    diffs = np.array([enum_f1[q] - free_f1[q] for q in qids])
    stat, p = _wilcoxon_p(diffs)
    lo, hi = _bootstrap_ci(qids, lambda s: float(np.mean([enum_f1[q] - free_f1[q] for q in s])), seed)
    return {
        "delta_f1": float(np.mean(list(enum_f1.values())) - np.mean(list(free_f1.values()))),
        "wilcoxon_p": p,
        "f1_ci95": [lo, hi],
        "cliffs_delta": cliffs_delta(list(enum_f1.values()), list(free_f1.values())),
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
        print(f"  {'field':<28}{'TARa':>8}{'EMA':>8}{'CohenK':>10}{'FleissK':>10}")
        for label, m in data["fields"].items():
            print(f"  {label:<28}{_fmt(m['tar_a']):>8}{_fmt(m['ema']):>8}"
                  f"{_fmt(m['cohen_kappa']):>10}{_fmt(m['fleiss_kappa']):>10}")
        a = data["answer"]
        print(f"  {'answer(norm)':<28}{_fmt(a['tar_a_normalized']):>8}{_fmt(a['ema_normalized']):>8}"
              f"{_fmt(a['cohen_kappa_normalized']):>10}")
        print(f"  answer TARr: raw={_fmt(a['tar_r_raw_answer'])} response={_fmt(a['tar_r_raw_response'])}"
              f"  BERTScore-F1={_fmt(a['bertscore_f1'])}")
        q = data["quality"]
        print(f"  quality vs gold: EM={_fmt(q['em_mean'])}+/-{_fmt(q['em_std'])}  "
              f"F1={_fmt(q['f1_mean'])}+/-{_fmt(q['f1_std'])}")

    if results["comparison"]:
        print("\n=== enum vs free (paired by question; delta = enum - free) ===")
        for label, c in results["comparison"].items():
            if label == "quality":
                print(f"  {'d_F1':<24}{_fmt(c['delta_f1']):>8}  p={_fmt(c['wilcoxon_p'])}  "
                      f"CI95={[_fmt(x) for x in c['f1_ci95']]}  cliffs_d={_fmt(c['cliffs_delta'])}")
                continue
            line = (f"  {label:<24}{_fmt(c['delta_mean_agreement']):>8}  p={_fmt(c['wilcoxon_p'])}  "
                    f"CI95={[_fmt(x) for x in c['agreement_ci95']]}  cliffs_d={_fmt(c['cliffs_delta'])}")
            if "delta_kappa" in c:
                line += f"  d_kappa={_fmt(c['delta_kappa'])}"
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
