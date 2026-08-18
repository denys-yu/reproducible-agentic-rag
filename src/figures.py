"""Print-ready, grayscale, bilingual figures for the agentic-RAG reproducibility paper.

Reads a pre-computed metrics summary (`results.json`, produced by `src.metrics`) and renders two
grouped bar charts. Each figure is produced in BOTH language variants (Ukrainian and English) in a
single run, and each variant is saved as a 600-dpi PNG and a vector PDF into `figures/`:

  * `kappa_by_field_<lang>` — mean pairwise Cohen's kappa per judgement field (structured vs free
    text), annotated with Holm-corrected paired-Wilcoxon p-values (pH) over the six endpoints.
  * `answer_quality_<lang>` — answer quality vs HotpotQA gold (Exact Match, token-F1, gold-answer
    containment), annotated with paired-Wilcoxon p-values (Exact Match is a descriptive comparison).

This module NEVER re-runs the experiment or makes LLM calls; it only reads numbers from the JSON,
and it never recomputes or mutates any statistic stored in `results.json`. The one derived quantity,
the Holm correction, is a display-only transform of the six raw p-values already in the file and is
applied identically for both languages.

Grayscale-safe by design (legible in print and when photocopied): the two arms are distinguished by
BOTH fill lightness AND hatch pattern with black edges — the enum arm is white with a `///` hatch,
the free-text arm is mid-gray (`0.65`) with a `...` hatch. No colour is used anywhere. All numbers
drawn on a figure follow the variant's locale (decimal comma for `uk`, decimal point for `en`).

Layout, axis limits, tick positions, bar widths, and data are identical between language variants;
only the text differs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless, deterministic backend
import matplotlib.pyplot as plt

from src.labels import FIGURE_IDS, LABELS, LANGS

# --- Arm styling (grayscale-safe, consistent across both figures) ----------------------------
# Distinguished by BOTH fill lightness AND hatch, with black edges, so bars separate in print and
# when photocopied. `label_key` names the per-figure legend string looked up in `labels.py`.
ARM_STYLE: dict[str, dict[str, str]] = {
    "enum": {"facecolor": "white", "hatch": "///"},
    "free": {"facecolor": "0.65", "hatch": "..."},
}
ARM_ORDER: list[str] = ["enum", "free"]

_BAR_WIDTH = 0.38
_EDGE_COLOR = "black"
_EDGE_WIDTH = 1.0
_GRID_COLOR = "0.8"  # light gray gridlines, behind the bars

# Everything ink-black; no colour anywhere.
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
        "savefig.dpi": 600,
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)


# --- Data / statistics helpers (read-only; no experiment statistic is recomputed) -------------
def load_results(path: Path) -> dict[str, Any]:
    """Load the metrics summary JSON."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment of a family of p-values, returned in the input order.

    This is a display-only multiple-comparison transform of p-values already present in
    `results.json`; it does not read from or write to any experiment metric.
    """
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(pvalues[idx] * (m - rank), 1.0))
        adjusted[idx] = running
    return adjusted


# --- Locale-aware number formatting -----------------------------------------------------------
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def format_value(value: float, lang: str) -> str:
    """Two-decimal bar value in the variant's locale (e.g. '0,91' / '0.91')."""
    return f"{value:.2f}".replace(".", LABELS[lang]["decimal_sep"])


def format_pvalue(p: float, lang: str) -> str:
    """Format a p-value to 3 significant figures in the variant's locale.

    Values below 1e-4 use scientific notation with a proper '×' and superscript exponent
    (e.g. '4,25×10⁻⁷' / '4.25×10⁻⁷'); larger values use plain decimals (e.g. '0,000236').
    """
    sep = LABELS[lang]["decimal_sep"]
    if p < 1e-4:
        exp = math.floor(math.log10(p))
        mantissa = p / (10.0**exp)
        if mantissa >= 10.0:  # guard against rounding to '10.00×10ⁿ'
            mantissa /= 10.0
            exp += 1
        mantissa_str = f"{mantissa:.2f}".replace(".", sep)
        return f"{mantissa_str}×10{str(exp).translate(_SUPERSCRIPT)}"
    exp = math.floor(math.log10(p))
    decimals = max(0, 2 - exp)  # 3 significant figures
    return f"{p:.{decimals}f}".replace(".", sep)


# --- Drawing helpers --------------------------------------------------------------------------
def _add_value_labels(ax: plt.Axes, bars: Any, values: list[float], lang: str) -> None:
    """Print each bar's value (locale-formatted) just above its top."""
    for rect, value in zip(bars, values, strict=True):
        ax.annotate(
            format_value(value, lang),
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _grouped_bars(
    ax: plt.Axes,
    values_by_arm: dict[str, list[float]],
    x_positions: list[float],
    arm_labels: dict[str, str],
    lang: str,
) -> None:
    """Draw one clustered pair of bars (enum, free) per x position, with value labels."""
    offsets = {"enum": -_BAR_WIDTH / 2, "free": _BAR_WIDTH / 2}
    for arm in ARM_ORDER:
        style = ARM_STYLE[arm]
        positions = [x + offsets[arm] for x in x_positions]
        bars = ax.bar(
            positions,
            values_by_arm[arm],
            width=_BAR_WIDTH,
            facecolor=style["facecolor"],
            hatch=style["hatch"],
            edgecolor=_EDGE_COLOR,
            linewidth=_EDGE_WIDTH,
            label=arm_labels[arm],
        )
        _add_value_labels(ax, bars, values_by_arm[arm], lang)


def _annotate_pair(ax: plt.Axes, x: float, pair_top: float, text: str) -> None:
    """Draw the p-value (or descriptive) line just above a bar pair, clear of the value labels."""
    ax.annotate(
        text,
        xy=(x, pair_top),
        xytext=(0, 16),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )


def _style_axes(ax: plt.Axes) -> None:
    """Light gridlines behind bars, black text, clean journal-style spines."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRID_COLOR, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_footnote(fig: plt.Figure, text: str) -> None:
    """Render a centered, small-print explanatory line below the axes (smaller than tick labels)."""
    fig.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=8, wrap=True)


def _save(fig: plt.Figure, out_dir: Path, figure_key: str, lang: str) -> tuple[Path, Path]:
    """Save `fig` as `<figure_id>_<lang>` in both PNG (600 dpi) and vector PDF; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{FIGURE_IDS[figure_key]}_{lang}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")  # PDF is vector regardless of dpi
    plt.close(fig)
    return png_path, pdf_path


# --- Figure 1: mean pairwise Cohen's kappa by field -------------------------------------------
# (label key, kappa field accessor, comparison key) in fixed left-to-right order (paper Table 2).
_FIG1_FIELDS: list[tuple[str, str, str]] = [
    ("fig1_cat_grade_conf", "grade.confidence", "grade.confidence"),
    ("fig1_cat_grade_scope", "grade.scope", "grade.scope"),
    ("fig1_cat_grade_needs", "grade.needs_more_context", "grade.needs_more_context"),
    ("fig1_cat_syn_conf", "synthesize.confidence", "synthesize.confidence"),
    ("fig1_cat_syn_scope", "synthesize.scope", "synthesize.scope"),
    ("fig1_cat_answer", "__answer__", "answer.normalized"),
]


def _fig1_data(results: dict[str, Any]) -> dict[str, Any]:
    """Extract Figure 1 numbers once (language-independent), including Holm-adjusted p-values."""

    def kappa(arm: str, field_key: str) -> float:
        arm_block = results["per_arm"][arm]
        if field_key == "__answer__":
            return float(arm_block["answer"]["cohen_kappa_normalized"])
        return float(arm_block["fields"][field_key]["cohen_kappa"])

    values_by_arm = {
        arm: [kappa(arm, field_key) for _, field_key, _ in _FIG1_FIELDS] for arm in ARM_ORDER
    }
    raw_p = [float(results["comparison"][cmp_key]["wilcoxon_p"]) for *_, cmp_key in _FIG1_FIELDS]
    return {
        "label_keys": [lk for lk, _, _ in _FIG1_FIELDS],
        "values_by_arm": values_by_arm,
        "pvalues": holm_adjust(raw_p),  # display-only Holm correction over the six endpoints
    }


def make_figure1(data: dict[str, Any], out_dir: Path, lang: str) -> tuple[Path, Path]:
    """Render Figure 1 (kappa by field) for one language."""
    L = LABELS[lang]
    values_by_arm = data["values_by_arm"]
    x_positions = [float(i) for i in range(len(_FIG1_FIELDS))]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    arm_labels = {"enum": L["fig1_legend_enum"], "free": L["fig1_legend_free"]}
    _grouped_bars(ax, values_by_arm, x_positions, arm_labels, lang)

    for x, p in zip(x_positions, data["pvalues"], strict=True):
        pair_top = max(values_by_arm["enum"][int(x)], values_by_arm["free"][int(x)])
        _annotate_pair(ax, x, pair_top, f"{L['fig1_annot_p_prefix']}{format_pvalue(p, lang)}")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(L["fig1_y_axis"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels([L[lk] for lk in data["label_keys"]])
    _style_axes(ax)
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    _add_footnote(fig, L["fig1_footnote"])
    return _save(fig, out_dir, "fig1", lang)


# --- Figure 2: answer quality vs gold ---------------------------------------------------------
# (label key, quality accessor, comparison p-value key or None for a descriptive comparison).
_FIG2_METRICS: list[tuple[str, str, str | None]] = [
    ("fig2_m_em", "em_mean", None),
    ("fig2_m_f1", "f1_mean", "wilcoxon_p"),
    ("fig2_m_containment", "containment_mean", "containment_wilcoxon_p"),
]


def _fig2_data(results: dict[str, Any]) -> dict[str, Any]:
    """Extract Figure 2 numbers once (language-independent)."""
    quality_cmp = results["comparison"]["quality"]
    values_by_arm = {
        arm: [float(results["per_arm"][arm]["quality"][key]) for _, key, _ in _FIG2_METRICS]
        for arm in ARM_ORDER
    }
    pvalues = [None if pk is None else float(quality_cmp[pk]) for *_, pk in _FIG2_METRICS]
    return {
        "label_keys": [lk for lk, _, _ in _FIG2_METRICS],
        "values_by_arm": values_by_arm,
        "pvalues": pvalues,
    }


def make_figure2(data: dict[str, Any], out_dir: Path, lang: str) -> tuple[Path, Path]:
    """Render Figure 2 (answer quality) for one language."""
    L = LABELS[lang]
    values_by_arm = data["values_by_arm"]
    x_positions = [float(i) for i in range(len(_FIG2_METRICS))]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    arm_labels = {"enum": L["fig2_legend_enum"], "free": L["fig2_legend_free"]}
    _grouped_bars(ax, values_by_arm, x_positions, arm_labels, lang)

    for x, p in zip(x_positions, data["pvalues"], strict=True):
        pair_top = max(values_by_arm["enum"][int(x)], values_by_arm["free"][int(x)])
        text = (
            L["fig2_annot_descriptive"]
            if p is None
            else f"{L['fig2_annot_p_prefix']}{format_pvalue(p, lang)}"
        )
        _annotate_pair(ax, x, pair_top, text)

    ax.set_ylim(0.0, 0.8)
    ax.set_ylabel(L["fig2_y_axis"])
    ax.set_xticks(x_positions)
    ax.set_xticklabels([L[lk] for lk in data["label_keys"]])
    _style_axes(ax)
    ax.legend(loc="upper left", framealpha=0.95)

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    _add_footnote(fig, L["fig2_footnote"])
    return _save(fig, out_dir, "fig2", lang)


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
    fig1_data = _fig1_data(results)
    fig2_data = _fig2_data(results)

    written: list[Path] = []
    for lang in LANGS:
        written.extend(make_figure1(fig1_data, args.out_dir, lang))
        written.extend(make_figure2(fig2_data, args.out_dir, lang))

    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()