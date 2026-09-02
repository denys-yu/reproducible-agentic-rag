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
        # Pinned explicitly rather than inherited: the font decides glyph metrics, so leaving it to
        # the matplotlib default would make output depend on the machine's font configuration. This
        # face ships with matplotlib and covers Cyrillic, so both variants render from one file.
        "font.family": "DejaVu Sans",
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


#: Metadata suppressed so two runs of this module produce byte-identical files. Without this,
#: matplotlib stamps `/CreationDate` into every PDF and a `Software` chunk into every PNG, and the
#: output differs run to run even though the figure does not. A figure that cannot be rebuilt
#: byte-for-byte is not reproducible evidence.
_PDF_METADATA: dict[str, None] = {"Creator": None, "Producer": None, "CreationDate": None}
_PNG_METADATA: dict[str, None] = {"Software": None}


def _save(fig: plt.Figure, out_dir: Path, figure_key: str, lang: str) -> tuple[Path, Path]:
    """Save `fig` as `<figure_id>_<lang>` in both PNG (600 dpi) and vector PDF; return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{FIGURE_IDS[figure_key]}_{lang}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", metadata=_PNG_METADATA)
    # PDF is vector regardless of dpi.
    fig.savefig(pdf_path, bbox_inches="tight", metadata=_PDF_METADATA)
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



# ==============================================================================================
# Revised-article figures (RF1-RF6)
#
# Data comes from `src.figure_data`, which reads the aggregator reports and recomputes the one
# missing endpoint through the aggregator's own functions, asserting every value against the
# article brief before it is drawn. Nothing below invents, rounds into, or hardcodes a number.
#
# Grayscale rules: no colour carries meaning anywhere. Series separate by HATCH and MARKER SHAPE
# with black edges on white fills, plus direct labelling; a legend appears only where direct
# labelling genuinely cannot work (it does not, in any figure here). Every value is printed on or
# beside its mark, so the figures survive photocopying at journal column width.
#
# Layout constants are chosen so that no label, note or arrow overlaps a mark in EITHER language:
# the Ukrainian strings are the longer of the two, so they set the spacing.
# ==============================================================================================

import textwrap  # noqa: E402

from src.figure_data import CELLS, MODEL_ORDER, STATE_ORDER, cell_for  # noqa: E402
from src.labels import REVISED_FIGURE_KEYS  # noqa: E402

#: One hatch per format state, plus a distinct hatch for the degenerate cell. White fills
#: throughout: lightness never encodes anything, so a photocopy loses no information.
_STATE_HATCH: dict[str, str] = {"no_format": "...", "appendix": "///", "schema": "\\\\\\"}
_DEGENERATE_HATCH = "xxx"
_FACE = "white"
#: Opaque white backing so a label stays legible where it sits over a hatch.
_LABEL_BBOX: dict[str, Any] = {"boxstyle": "square,pad=0.18", "facecolor": "white", "edgecolor": "none"}
_NOTE_BBOX: dict[str, Any] = {
    "boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "black", "linewidth": 0.7
}
_ARROW: dict[str, Any] = {"arrowstyle": "->", "color": "black", "linewidth": 0.9}


def format_number(value: float, lang: str, decimals: int = 4) -> str:
    """Format a value to `decimals` places in the variant's locale."""
    return f"{value:.{decimals}f}".replace(".", LABELS[lang]["decimal_sep"])


def format_count(value: float) -> str:
    """Whole counts carry no decimal separator, so they are locale-independent."""
    return str(int(round(value)))


def _fraction(part: float, whole: float, lang: str) -> str:
    """'52 of 150' / '52 зі 150' — the connector is a label, never punctuation in the code."""
    return f"{format_count(part)}{LABELS[lang]['of_connector']}{format_count(whole)}"


def _wrap(text: str, width: int) -> str:
    """Wrap a label to a column width. Layout only: it never alters the words themselves."""
    return "\n".join(textwrap.wrap(text, width)) or text


def _cell_tick(cell: Any, lang: str) -> str:
    """Two-line tick label: the model name (untranslated) over its format state (translated)."""
    return f"{cell.model}\n{_wrap(LABELS[lang][f'state_{cell.state}'], 18)}"


def _cell_inline(cell: Any, lang: str) -> str:
    """One-line 'model, state' label.

    Built from the raw state string, never by joining the wrapped tick label: replacing the WRAP
    newline with ', ' inserts a comma inside the state's own wording, turning
    'формат не проголошено' into 'формат не, проголошено'.
    """
    return f"{cell.model}, {LABELS[lang][f'state_{cell.state}']}"


def _label_bars(
    ax: plt.Axes,
    bars: Any,
    values: list[float],
    lang: str,
    decimals: int = 4,
    fontsize: float = 8.5,
    dx: float = 0.0,
) -> None:
    """Print each bar's value just above its top, in the variant's locale."""
    for rect, value in zip(bars, values, strict=True):
        ax.annotate(
            format_number(value, lang, decimals),
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(dx, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# --- RF1: route stability across three format states ------------------------------------------
def make_rf1(data: dict[str, Any], out_dir: Path, lang: str) -> tuple[Path, Path]:
    """Route stability, two models x three format states, rewrite rate printed under each bar."""
    L = LABELS[lang]
    route = data["route"]
    fig, ax = plt.subplots(figsize=(9.4, 6.2))

    positions: list[float] = []
    tick_labels: list[str] = []
    group_centres: dict[str, float] = {}
    for group_index, model in enumerate(MODEL_ORDER):
        xs = [group_index * 4.0 + offset for offset in range(len(STATE_ORDER))]
        group_centres[model] = sum(xs) / len(xs)
        for x, state in zip(xs, STATE_ORDER, strict=True):
            cell = cell_for(model, state)
            entry = route[cell.key]
            bars = ax.bar(
                [x], [entry["route_stability"]], width=0.82,
                facecolor=_FACE,
                hatch=_DEGENERATE_HATCH if entry["degenerate"] else _STATE_HATCH[state],
                edgecolor=_EDGE_COLOR, linewidth=_EDGE_WIDTH,
            )
            _label_bars(ax, bars, [entry["route_stability"]], lang)
            positions.append(x)
            # The rewrite rate rides under its own bar, as the second line of the tick label.
            tick_labels.append(
                f"{_wrap(L[f'state_{state}'], 12)}\n{format_number(entry['rewrite_rate'], lang)}"
            )

    for model, centre in group_centres.items():
        ax.text(centre, 1.30, model, ha="center", va="bottom", fontsize=11.5)

    # The degeneracy note lives in the empty gap between the two model groups, so the arrow can
    # reach its bar without the box ever covering a mark.
    degenerate_cell = cell_for("gpt-5.6-luna", "no_format")
    degenerate_x = positions[len(STATE_ORDER) + STATE_ORDER.index("no_format")]
    ax.annotate(
        L["rf1_degenerate_note"],
        xy=(degenerate_x - 0.41, route[degenerate_cell.key]["route_stability"]),
        xytext=(3.0, 1.19),
        ha="center", va="top", fontsize=8.5,
        arrowprops=_ARROW, bbox=_NOTE_BBOX,
    )

    ax.set_ylim(0.0, 1.42)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([format_number(v, lang, 1) for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)])
    ax.set_ylabel(L["rf1_y_axis"])
    ax.set_title(L["rf1_title"], pad=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_xlim(-0.85, positions[-1] + 0.85)
    _style_axes(ax)

    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    _add_footnote(fig, L["rf1_baseline_note"])
    return _save(fig, out_dir, "rf1", lang)


# --- RF2: format-guess stability ---------------------------------------------------------------
def make_rf2(data: dict[str, Any], out_dir: Path, lang: str) -> tuple[Path, Path]:
    """How many of 5 runs returned json_object, per question, with the 4o-mini comparison."""
    L = LABELS[lang]
    fmt = data["format_guess"]
    hist = fmt["hist"]
    fig, ax = plt.subplots(figsize=(9.4, 5.9))

    xs = sorted(hist)
    values = [hist[x] for x in xs]
    for x, value in zip(xs, values, strict=True):
        interior = 1 <= x <= 4  # the unstable region: the format changed between runs
        bars = ax.bar(
            [float(x)], [value], width=0.82, facecolor=_FACE,
            hatch="xxx" if interior else "...",
            edgecolor=_EDGE_COLOR, linewidth=_EDGE_WIDTH,
        )
        _label_bars(ax, bars, [value], lang, decimals=0)

    # The 4o-mini bar is a separate comparison, not a sixth bin: a rule separates it from the axis
    # the luna histogram is counted on.
    omini_x = float(max(xs)) + 2.0
    ax.axvline(omini_x - 1.0, color="black", linewidth=0.8, linestyle=":")
    omini_bars = ax.bar(
        [omini_x], [fmt["omini_stable"]], width=0.82, facecolor=_FACE, hatch="///",
        edgecolor=_EDGE_COLOR, linewidth=_EDGE_WIDTH,
    )
    _label_bars(ax, omini_bars, [float(fmt["omini_stable"])], lang, decimals=0)

    # Bracket spanning the interior bins, with the split share stated on it.
    span_y = max(values[1:5]) + 16
    ax.plot([0.6, 4.4], [span_y, span_y], color="black", linewidth=0.9)
    for edge in (0.6, 4.4):
        ax.plot([edge, edge], [span_y - 4, span_y], color="black", linewidth=0.9)
    ax.text(
        2.5, span_y + 3,
        f"{L['rf2_unstable_label']}: {_fraction(fmt['split_n'], fmt['total'], lang)} = "
        f"{format_number(fmt['split_share'], lang)}",
        ha="center", va="bottom", fontsize=9,
    )
    ax.text(2.5, 172, _wrap(L["rf2_luna_label"], 34), ha="center", va="top", fontsize=9.5)
    ax.text(omini_x, 172, L["rf2_omini_label"], ha="center", va="top", fontsize=8.5)

    ax.set_ylim(0, 196)
    ax.set_xlim(-0.75, omini_x + 0.95)
    ax.set_xticks([float(x) for x in xs] + [omini_x])
    ax.set_xticklabels([format_count(x) for x in xs] + [""], fontsize=10)
    ax.set_xlabel(L["rf2_x_axis"])
    ax.set_ylabel(L["rf2_y_axis"])
    ax.set_title(L["rf2_title"], pad=12)
    _style_axes(ax)

    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    _add_footnote(fig, L["rf2_footnote"])
    return _save(fig, out_dir, "rf2", lang)


# --- RF3: silent failure mechanism --------------------------------------------------------------
def make_rf3(data: dict[str, Any], out_dir: Path, lang: str) -> tuple[Path, Path]:
    """Two aligned panels over one cell order: unparsed rate above, rewrite rate below."""
    L = LABELS[lang]
    parse, route = data["parse"], data["route"]
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9.8, 7.6), sharex=True)

    xs = [float(i) for i in range(len(CELLS))]
    unparsed = [parse[c.key]["unparsed_rate"] for c in CELLS]
    rewrite = [route[c.key]["rewrite_rate"] for c in CELLS]
    partial_cell = cell_for("gpt-4o-mini", "appendix")
    partial_index = CELLS.index(partial_cell)

    panels = (
        (ax_top, unparsed, 0.98, (0.0, 0.2, 0.4, 0.6, 0.8)),
        (ax_bottom, rewrite, 0.46, (0.0, 0.1, 0.2, 0.3, 0.4)),
    )
    for ax, values, ylim, ticks in panels:
        for x, cell, value in zip(xs, CELLS, values, strict=True):
            bars = ax.bar(
                [x], [value], width=0.66, facecolor=_FACE, hatch=_STATE_HATCH[cell.state],
                edgecolor=_EDGE_COLOR, linewidth=_EDGE_WIDTH,
            )
            # In the top panel this cell's bar is zero-height and its value label would land
            # inside the dashed partial marker, reading as that marker's value. Shift it clear.
            dx = -30.0 if (ax is ax_top and cell is partial_cell) else 0.0
            _label_bars(ax, bars, [value], lang, dx=dx)
        ax.set_ylim(0.0, ylim)
        # Tick labels are set explicitly: matplotlib's default formatter writes '0.2' with a
        # period in BOTH variants, which would leave the Ukrainian axis in the wrong locale.
        ax.set_yticks(list(ticks))
        ax.set_yticklabels([format_number(v, lang, 1) for v in ticks])
        _style_axes(ax)

    ax_top.set_ylabel(L["rf3_y_axis_top"])
    ax_bottom.set_ylabel(L["rf3_y_axis_bottom"])
    ax_top.set_title(L["rf3_title"], pad=12)

    # The 20 partial parses are a DIFFERENT phenomenon from an unparsed call and are deliberately
    # not merged into the rate: an open dashed outline at the height they would reach, never a
    # filled bar, so the eye cannot read it as part of the measured quantity.
    entry = parse[partial_cell.key]
    partial_height = entry["partial"] / entry["n"]
    ax_top.bar(
        [xs[partial_index]], [partial_height], width=0.66, facecolor="none",
        edgecolor=_EDGE_COLOR, linewidth=1.0, linestyle="--",
    )
    ax_top.annotate(
        format_count(entry["partial"]),
        xy=(xs[partial_index] + 0.33, partial_height), xytext=(3, 0),
        textcoords="offset points", ha="left", va="center", fontsize=8.5,
    )
    ax_top.annotate(
        L["rf3_partial_note"],
        xy=(xs[partial_index] - 0.33, partial_height),
        xytext=(xs[partial_index] - 0.55, 0.58),
        ha="center", va="top", fontsize=8.5,
        arrowprops=_ARROW, bbox=_NOTE_BBOX,
    )

    ax_bottom.set_xticks(xs)
    ax_bottom.set_xticklabels([_cell_tick(c, lang) for c in CELLS], fontsize=8.5)
    ax_bottom.set_xlim(-0.7, xs[-1] + 0.7)

    # Panel letters, keyed like every other string rather than baked in: the DOCX caption sets
    # them in italic, so these must match. Anchored to the axes' top-left corner and offset
    # outside it, so the letter never sits over a mark in either language.
    for ax, key in ((ax_top, "rf3_panel_a"), (ax_bottom, "rf3_panel_b")):
        ax.annotate(
            L[key],
            xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-38, 6), textcoords="offset points",
            ha="left", va="bottom",
            fontsize=plt.rcParams["axes.labelsize"], fontstyle="italic",
        )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return _save(fig, out_dir, "rf3", lang)


# --- RF4: why coverage must be reported ---------------------------------------------------------
def make_rf4(data: dict[str, Any], out_dir: Path, lang: str) -> tuple[Path, Path]:
    """EMA beside usable agreement, with coverage printed over each pair."""
    L = LABELS[lang]
    family = data["family"]
    fig, ax = plt.subplots(figsize=(10.4, 6.2))

    width = 0.34
    xs = [float(i) for i in range(len(CELLS))]
    for x, cell in zip(xs, CELLS, strict=True):
        entry = family[cell.key]
        for position, value, hatch in (
            (x - width / 2, entry["ema"], "..."),
            (x + width / 2, entry["usable"], "///"),
        ):
            bars = ax.bar(
                [position], [value], width=width, facecolor=_FACE, hatch=hatch,
                edgecolor=_EDGE_COLOR, linewidth=_EDGE_WIDTH,
            )
            _label_bars(ax, bars, [value], lang, fontsize=8)
        # Coverage sits at one constant height across all cells, so the collapsed cell is found by
        # reading along a single line rather than hunting above bars of different heights.
        ax.text(
            x, 1.10,
            f"{L['rf4_coverage_prefix']}{format_number(entry['coverage'], lang)}",
            ha="center", va="bottom", fontsize=8.5,
        )

    # Direct labelling instead of a legend: the two series are named inside the first pair, on an
    # opaque backing so the hatch does not run through the text, and the hatch carries the
    # identification across the remaining cells.
    first = family[CELLS[0].key]
    ax.text(xs[0] - width / 2, first["ema"] / 2, L["rf4_series_ema"],
            ha="center", va="center", rotation=90, fontsize=8.5, bbox=_LABEL_BBOX)
    ax.text(xs[0] + width / 2, first["usable"] / 2, _wrap(L["rf4_series_usable"], 9),
            ha="center", va="center", rotation=90, fontsize=8.5, bbox=_LABEL_BBOX)

    collapse_cell = cell_for("gpt-5.6-luna", "no_format")
    collapse_index = CELLS.index(collapse_cell)
    ax.annotate(
        L["rf4_collapse_note"],
        # Aim at the right flank of the collapsed bar, mid-height: its top carries the value label.
        xy=(xs[collapse_index] + width, family[collapse_cell.key]["usable"] / 2),
        xytext=(xs[collapse_index] - 1.55, 0.74),
        ha="center", va="top", fontsize=8.5,
        arrowprops=_ARROW, bbox=_NOTE_BBOX,
    )

    ax.set_ylim(0.0, 1.24)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([format_number(v, lang, 1) for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)])
    ax.set_ylabel(L["rf4_y_axis"])
    ax.set_title(L["rf4_title"], pad=14)
    ax.set_xticks(xs)
    ax.set_xticklabels([_cell_tick(c, lang) for c in CELLS], fontsize=8.5)
    ax.set_xlim(-0.7, xs[-1] + 0.7)
    _style_axes(ax)

    fig.tight_layout()
    return _save(fig, out_dir, "rf4", lang)


# --- RF5: field collapse -------------------------------------------------------------------------
def make_rf5(data: dict[str, Any], out_dir: Path, lang: str) -> tuple[Path, Path]:
    """Bijection breaks between synthesize.scope and synthesize.confidence, out of 750."""
    L = LABELS[lang]
    bijection = data["bijection"]
    cells = [c for c in CELLS if c.key in bijection]
    fig, ax = plt.subplots(figsize=(7.8, 5.6))

    xs = [float(i) for i in range(len(cells))]
    for x, cell in zip(xs, cells, strict=True):
        value = float(bijection[cell.key]["breaks"])
        bars = ax.bar(
            [x], [value], width=0.62, facecolor=_FACE, hatch=_STATE_HATCH[cell.state],
            edgecolor=_EDGE_COLOR, linewidth=_EDGE_WIDTH,
        )
        _label_bars(ax, bars, [value], lang, decimals=0, fontsize=9)

    ax.set_ylim(0, 80)
    ax.set_ylabel(L["rf5_y_axis"])
    ax.set_title(L["rf5_title"], pad=12)
    ax.set_xticks(xs)
    ax.set_xticklabels([_cell_tick(c, lang) for c in cells], fontsize=8.5)
    ax.set_xlim(-0.7, xs[-1] + 0.7)
    _style_axes(ax)

    fig.tight_layout()
    return _save(fig, out_dir, "rf5", lang)


# --- RF6: length artifact --------------------------------------------------------------------------
#: Marker shape, not shade, separates the two quality metrics.
_RF6_MARKERS: dict[str, str] = {"em": "s", "f1": "^"}
#: Cell labels are parked on these heights, cycling in ascending order of answer length. Two cells
#: sit 0.01 tokens apart (luna free published vs ablation) and two more 0.21 apart, so labels
#: anchored at the same x MUST be separated vertically or they overprint each other. Four levels,
#: not three: with three, ranks 0 and 3 share a level while sitting 2.27 tokens apart, which is
#: narrower than the Ukrainian label itself.
_RF6_LABEL_LEVELS: tuple[float, ...] = (0.755, 0.70, 0.645, 0.59)


def make_rf6(data: dict[str, Any], out_dir: Path, lang: str) -> tuple[Path, Path]:
    """Mean answer length against Exact Match and token-level F1, per cell."""
    L = LABELS[lang]
    length = data["length"]
    fig, ax = plt.subplots(figsize=(9.6, 6.4))

    # Ascending answer length fixes both the label levels and the side each value is printed on;
    # the ordering is data-driven but deterministic, so the layout is stable across runs.
    ordered = sorted(CELLS, key=lambda c: (length[c.key]["tokens"], c.key))
    token_xs = [length[c.key]["tokens"] for c in ordered]
    for rank, cell in enumerate(ordered):
        entry = length[cell.key]
        x = entry["tokens"]
        # Print each value on the side AWAY from its nearest neighbour. Two cells sit 0.01 tokens
        # apart and two more 0.21 apart, so a fixed side (or a parity rule) drives those labels
        # into each other; the gap comparison always sends them apart.
        gap_left = x - token_xs[rank - 1] if rank > 0 else float("inf")
        gap_right = token_xs[rank + 1] - x if rank + 1 < len(token_xs) else float("inf")
        side = 1 if gap_right > gap_left else -1
        # A thin connector ties the two metrics of one cell together without implying a trend.
        ax.plot([x, x], [entry["em"], entry["f1"]], color="black", linewidth=0.6, linestyle=":")
        for metric in ("em", "f1"):
            ax.plot(
                [x], [entry[metric]], marker=_RF6_MARKERS[metric], markersize=8,
                markerfacecolor=_FACE, markeredgecolor=_EDGE_COLOR, markeredgewidth=1.1,
                linestyle="none",
            )
            ax.annotate(
                format_number(entry[metric], lang),
                xy=(x, entry[metric]), xytext=(7 * side, -3), textcoords="offset points",
                ha="left" if side > 0 else "right", va="center", fontsize=8,
            )
        level = _RF6_LABEL_LEVELS[rank % len(_RF6_LABEL_LEVELS)]
        ax.annotate(
            _cell_inline(cell, lang),
            xy=(x, entry["f1"]), xytext=(x, level),
            ha="center", va="bottom", fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 0.5, "linestyle": ":"},
        )

    # Direct series labelling instead of a legend, keyed to the RIGHTMOST cell: it is the only one
    # with empty space beside it, and its own values print to the right at a different height.
    right_cell = ordered[-1]
    right_entry = length[right_cell.key]
    for metric, label_key, offset in (
        ("f1", "rf6_series_f1", 0.072),
        ("em", "rf6_series_em", 0.072),
    ):
        ax.annotate(
            L[label_key],
            xy=(right_entry["tokens"] + 0.06, right_entry[metric]),
            xytext=(right_entry["tokens"] + 0.72, right_entry[metric] + offset),
            ha="left", va="bottom", fontsize=9, arrowprops=_ARROW,
        )

    ax.set_xlim(8.1, 16.6)
    ax.set_ylim(0.0, 0.83)
    ax.set_xlabel(L["rf6_x_axis"])
    ax.set_ylabel(L["rf6_y_axis"])
    ax.set_title(L["rf6_title"], pad=12)
    ax.set_xticks([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    ax.set_xticklabels([format_number(v, lang, 0) for v in (9, 10, 11, 12, 13, 14, 15, 16)])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([format_number(v, lang, 1) for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)])
    _style_axes(ax)
    ax.xaxis.grid(True, color=_GRID_COLOR, linewidth=0.6)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    _add_footnote(fig, L["rf6_footnote"])
    return _save(fig, out_dir, "rf6", lang)


#: Figure key -> renderer, in build order.
REVISED_BUILDERS: dict[str, Any] = {
    "rf1": make_rf1, "rf2": make_rf2, "rf3": make_rf3,
    "rf4": make_rf4, "rf5": make_rf5, "rf6": make_rf6,
}


def build_revised(out_dir: Path, data: dict[str, Any] | None = None) -> list[Path]:
    """Render all six revised figures in both languages; return the paths in a stable order."""
    if data is None:
        from src.figure_data import load_all

        data = load_all()
    written: list[Path] = []
    for key in REVISED_FIGURE_KEYS:
        for lang in LANGS:
            written.extend(REVISED_BUILDERS[key](data, out_dir, lang))
    return written


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
    parser.add_argument(
        "--revised",
        action="store_true",
        help=(
            "Build the revised article's six figures (RF1-RF6) instead of the original two. "
            "These read the aggregator reports and runs/, not results.json."
        ),
    )
    args = parser.parse_args()

    if args.revised:
        written = build_revised(args.out_dir)
    else:
        results = load_results(args.results)
        fig1_data = _fig1_data(results)
        fig2_data = _fig2_data(results)

        written = []
        for lang in LANGS:
            written.extend(make_figure1(fig1_data, args.out_dir, lang))
            written.extend(make_figure2(fig2_data, args.out_dir, lang))

    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()