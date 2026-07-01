"""Publication-quality figures for the ICSFTI 2026 agentic-RAG reproducibility paper.

Reads a pre-computed metrics summary (`results.json`, produced by `src.metrics`) and renders two
grouped bar charts, each saved as both a 300-dpi PNG and a vector PDF into `figures/`:

  * Figure 1 — inter-run Cohen's kappa per judgement (enum vs free), with Wilcoxon significance
    markers read from the paired comparison block.
  * Figure 2 — answer quality vs HotpotQA gold (EM, token-F1, containment), enum vs free.

This module NEVER re-runs the experiment or makes LLM calls; it only reads numbers from the JSON.
It is fully deterministic (no randomness, no sampling). Figures are legible in grayscale: the two
arms are distinguished by a colour-blind-safe colour AND a distinct hatch pattern, not colour alone,
and the styling is kept consistent across both figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless, deterministic backend
import matplotlib.pyplot as plt

# --- Arm styling (consistent across both figures) --------------------------------------------
# Colour-blind-safe (Wong / Okabe-Ito) plus distinct hatches so bars separate in grayscale.
ARM_STYLE: dict[str, dict[str, str]] = {
    "enum": {"label": "enum (structured)", "color": "#0072B2", "hatch": "///"},
    "free": {"label": "free (free-form)", "color": "#E69F00", "hatch": "..."},
}
ARM_ORDER: list[str] = ["enum", "free"]

_BAR_WIDTH = 0.38
_EDGE_COLOR = "black"

# Base font sizes (legible, ~11-12 pt).
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "hatch.linewidth": 0.8,
        "figure.dpi": 100,
        "savefig.dpi": 300,
    }
)


def load_results(path: Path) -> dict[str, Any]:
    """Load the metrics summary JSON."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def significance_marker(p: float | None) -> str:
    """Star coding for a Wilcoxon p-value: *** <0.001, ** <0.01, * <0.05, else 'n.s.'."""
    if p is None:
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _add_value_labels(ax: plt.Axes, bars: Any, values: list[float]) -> None:
    """Print each bar's value (two decimals) just above its top."""
    for rect, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Save `fig` as both PNG (300 dpi) and vector PDF; return the two paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # PDF is vector regardless of dpi
    plt.close(fig)
    return png_path, pdf_path


def _grouped_bars(
    ax: plt.Axes,
    values_by_arm: dict[str, list[float]],
    x_positions: list[float],
) -> dict[str, Any]:
    """Draw one clustered pair of bars (enum, free) per x position; return the bar containers."""
    containers: dict[str, Any] = {}
    offsets = {"enum": -_BAR_WIDTH / 2, "free": _BAR_WIDTH / 2}
    for arm in ARM_ORDER:
        style = ARM_STYLE[arm]
        positions = [x + offsets[arm] for x in x_positions]
        bars = ax.bar(
            positions,
            values_by_arm[arm],
            width=_BAR_WIDTH,
            color=style["color"],
            hatch=style["hatch"],
            edgecolor=_EDGE_COLOR,
            linewidth=0.8,
            label=style["label"],
        )
        _add_value_labels(ax, bars, values_by_arm[arm])
        containers[arm] = bars
    return containers


def make_figure1(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Figure 1: inter-run Cohen's kappa per judgement, enum vs free, with significance markers."""
    # (x-axis label, per-arm value accessor, comparison key) in fixed left-to-right order.
    judgements: list[tuple[str, str, str]] = [
        ("Confidence\n(grading)", "grade.confidence", "grade.confidence"),
        ("Context scope\n(grading)", "grade.scope", "grade.scope"),
        ("Re-retrieval decision\n(grading)", "grade.needs_more_context", "grade.needs_more_context"),
        ("Confidence\n(synthesis)", "synthesize.confidence", "synthesize.confidence"),
        ("Answer scope\n(synthesis)", "synthesize.scope", "synthesize.scope"),
        ("Final\nanswer", "__answer__", "answer.normalized"),
    ]

    def kappa(arm: str, field_key: str) -> float:
        arm_block = results["per_arm"][arm]
        if field_key == "__answer__":
            return float(arm_block["answer"]["cohen_kappa_normalized"])
        return float(arm_block["fields"][field_key]["cohen_kappa"])

    x_labels = [j[0] for j in judgements]
    values_by_arm = {
        arm: [kappa(arm, field_key) for _, field_key, _ in judgements] for arm in ARM_ORDER
    }
    markers = [
        significance_marker(results["comparison"][cmp_key].get("wilcoxon_p"))
        for _, _, cmp_key in judgements
    ]

    x_positions = list(range(len(judgements)))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    _grouped_bars(ax, values_by_arm, [float(x) for x in x_positions])

    # Significance marker centred above each enum/free pair.
    for x, marker in zip(x_positions, markers, strict=True):
        pair_top = max(values_by_arm["enum"][x], values_by_arm["free"][x])
        ax.annotate(
            marker,
            xy=(x, pair_top),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Cohen's κ (inter-run agreement)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)
    _strip_chartjunk(ax)

    fig.tight_layout()
    return _save(fig, out_dir, "fig1_agreement_kappa")


def make_figure2(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Figure 2: answer quality vs gold (EM, token-F1, containment), enum vs free."""
    # (x-axis label, quality key, significance annotation).
    metrics: list[tuple[str, str, str]] = [
        ("Exact match", "em_mean", "*"),
        ("Token-F1", "f1_mean", "***"),
        ("Containment", "containment_mean", "n.s."),
    ]

    x_labels = [m[0] for m in metrics]
    values_by_arm = {
        arm: [float(results["per_arm"][arm]["quality"][key]) for _, key, _ in metrics]
        for arm in ARM_ORDER
    }
    markers = [m[2] for m in metrics]

    x_positions = list(range(len(metrics)))
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _grouped_bars(ax, values_by_arm, [float(x) for x in x_positions])

    for x, marker in zip(x_positions, markers, strict=True):
        pair_top = max(values_by_arm["enum"][x], values_by_arm["free"][x])
        ax.annotate(
            marker,
            xy=(x, pair_top),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0.0, 0.8)
    ax.set_ylabel("Score (vs. gold)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4)
    # Upper-left: the "Containment" pair is tallest and its n.s. marker sits at upper-right.
    ax.legend(loc="upper left", framealpha=0.95)
    _strip_chartjunk(ax)

    fig.tight_layout()
    return _save(fig, out_dir, "fig2_quality")


def _strip_chartjunk(ax: plt.Axes) -> None:
    """Remove top/right spines for a clean, journal-style axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results.json"),
        help="Path to the metrics summary JSON (default: results.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to write figures into (default: figures/).",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    fig1_png, fig1_pdf = make_figure1(results, args.out_dir)
    fig2_png, fig2_pdf = make_figure2(results, args.out_dir)

    print("Wrote:")
    for path in (fig1_png, fig1_pdf, fig2_png, fig2_pdf):
        print(f"  {path}")


if __name__ == "__main__":
    main()
