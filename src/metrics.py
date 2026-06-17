"""Agreement, quality, and statistical metrics over completed runs.

Computes the pre-registered metrics from the JSONL run manifests: Cohen's kappa per enum field,
TARa@5 / TARr@5 / EMA@5 on final answers, pairwise BERTScore-F1 and embedding cosine for semantic
agreement, Exact-Match and token-F1 vs HotpotQA gold for quality, and paired Wilcoxon + bootstrap
CIs for significance. Reads runs, returns plain data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.config import Config


def cohen_kappa_per_field(runs: Sequence[dict[str, Any]], field: str) -> float:
    """Mean pairwise Cohen's kappa for one enum field across the k runs of a question."""
    raise NotImplementedError


def tar_at_k(answers: Sequence[str], normalize: bool) -> float:
    """Top-Answer-Agreement@k: fraction of run pairs whose final answers agree."""
    raise NotImplementedError


def ema_at_k(answers: Sequence[str], gold: str) -> float:
    """Exact-Match-Agreement@k against the gold answer."""
    raise NotImplementedError


def bertscore_f1(candidates: Sequence[str], references: Sequence[str], config: Config) -> list[float]:
    """Pairwise BERTScore-F1 (CPU, pinned checkpoint) between run answers."""
    raise NotImplementedError


def exact_match(prediction: str, gold: str) -> float:
    """HotpotQA-style normalized exact match."""
    raise NotImplementedError


def token_f1(prediction: str, gold: str) -> float:
    """HotpotQA-style token-level F1."""
    raise NotImplementedError


def wilcoxon_test(deltas: Sequence[float]) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank on per-question deltas; return (statistic, p_value)."""
    raise NotImplementedError


def bootstrap_ci(values: Sequence[float], config: Config) -> tuple[float, float]:
    """95% bootstrap confidence interval (seeded from config)."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Resolve config from CLI/env and compute metrics over the run manifests."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
