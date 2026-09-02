"""Read every number the revised-article figures draw, and assert it against the brief.

Two rules govern this module:

1. **Nothing is hardcoded as an input.** Every value is either READ from an aggregator report
   (`report_series2_agg.txt`, `report_final_analysis.txt`) or RECOMPUTED from `runs/` and `runs2/`
   through the aggregator's own functions in `src.metrics` — never a second implementation.
2. **The numbers quoted in the article brief are ASSERTIONS, not inputs.** `check()` compares every
   value this module produces against `EXPECTED` and raises `FigureDataMismatch` naming both sides.
   A figure is never drawn from a number that failed its check.

`unparsed_rate` is the one quantity absent from both reports. It is not re-derived here: it comes
from `src.metrics.unparsed_rate`, the same function that produced the family table, applied to the
same manifests through the same `load_manifests` loader.

Read-only. No API calls, no writes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.metrics import load_manifests, rewrite_rate, unparsed_rate

REPORT_SERIES2 = Path("report_series2_agg.txt")
REPORT_FINAL = Path("report_final_analysis.txt")

#: Tolerances: the reports print agreement to 4 decimals, answer length to 2.
_TOL_4DP = 5e-5
_TOL_2DP = 5e-3


class FigureDataMismatch(RuntimeError):
    """A value read from the corpus disagrees with the article brief."""


@dataclass(frozen=True)
class Cell:
    """One (series, arm) experimental cell and the format state it represents."""

    series: str
    arm: str
    model: str  # display name, kept identical in both languages
    state: str  # "no_format" | "appendix" | "schema"
    runs_dir: str

    @property
    def key(self) -> str:
        return f"{self.series}|{self.arm}"


#: Fixed left-to-right ordering used by every per-cell figure (3, 4, 6).
CELLS: tuple[Cell, ...] = (
    Cell("4omini_published", "enum", "gpt-4o-mini", "schema", "runs"),
    Cell("4omini_published", "free", "gpt-4o-mini", "no_format", "runs"),
    Cell("4omini_ablation", "free", "gpt-4o-mini", "appendix", "runs2/4omini_ablation"),
    Cell("luna_published", "enum", "gpt-5.6-luna", "schema", "runs2/luna_published"),
    Cell("luna_published", "free", "gpt-5.6-luna", "no_format", "runs2/luna_published"),
    Cell("luna_ablation", "free", "gpt-5.6-luna", "appendix", "runs2/luna_ablation"),
)

#: Figure 1 reads the three states per model in this order (the chain the article argues).
STATE_ORDER: tuple[str, ...] = ("no_format", "appendix", "schema")
MODEL_ORDER: tuple[str, ...] = ("gpt-4o-mini", "gpt-5.6-luna")

CELL_BY_KEY: dict[str, Cell] = {c.key: c for c in CELLS}


def cell_for(model: str, state: str) -> Cell:
    """The single cell realising one (model, format state)."""
    matches = [c for c in CELLS if c.model == model and c.state == state]
    if len(matches) != 1:
        raise FigureDataMismatch(f"expected exactly one cell for ({model}, {state}), got {matches}")
    return matches[0]


# --- assertions from the article brief ---------------------------------------------------------
#: Every number the brief states, as {check name: expected value}. Checked against, never drawn from.
EXPECTED: dict[str, float] = {
    # Figure 1 / 3 — route stability, flips, rewrite_rate
    "route_stab|4omini_published|free": 0.8333, "flips|4omini_published|free": 25,
    "rewrite|4omini_published|free": 0.2947,
    "route_stab|4omini_ablation|free": 0.9333, "flips|4omini_ablation|free": 10,
    "rewrite|4omini_ablation|free": 0.2827,
    "route_stab|4omini_published|enum": 0.9733, "flips|4omini_published|enum": 4,
    "rewrite|4omini_published|enum": 0.2373,
    "route_stab|luna_published|free": 0.8867, "flips|luna_published|free": 17,
    "rewrite|luna_published|free": 0.0720,
    "route_stab|luna_ablation|free": 0.9267, "flips|luna_ablation|free": 11,
    "rewrite|luna_ablation|free": 0.3520,
    "route_stab|luna_published|enum": 0.9200, "flips|luna_published|enum": 12,
    "rewrite|luna_published|enum": 0.3373,
    # Figure 2 — format-guess stability
    "fmt_hist|0": 9, "fmt_hist|1": 7, "fmt_hist|2": 8, "fmt_hist|3": 11, "fmt_hist|4": 26,
    "fmt_hist|5": 89, "fmt_split_n": 52, "fmt_split_share": 0.3467, "fmt_4omini_stable": 150,
    "fmt_total": 150,
    # Figure 3 — unparsed_rate at the grade node
    "unparsed|4omini_published|enum": 0.0000, "unparsed|4omini_published|free": 0.0013,
    "unparsed|4omini_ablation|free": 0.0000, "unparsed|luna_published|enum": 0.0000,
    "unparsed|luna_published|free": 0.8067, "unparsed|luna_ablation|free": 0.0000,
    "partials|4omini_ablation|free": 20,
    # Figure 4 — grade.needs_more_context EMA / coverage / usable_agreement
    "ema|4omini_published|enum": 0.9893, "cov|4omini_published|enum": 1.0000,
    "usable|4omini_published|enum": 0.9893,
    "ema|4omini_published|free": 0.9213, "cov|4omini_published|free": 0.9973,
    "usable|4omini_published|free": 0.9189,
    "ema|4omini_ablation|free": 0.9627, "cov|4omini_ablation|free": 1.0000,
    "usable|4omini_ablation|free": 0.9627,
    "ema|luna_published|enum": 0.9600, "cov|luna_published|enum": 1.0000,
    "usable|luna_published|enum": 0.9600,
    "ema|luna_published|free": 0.9571, "cov|luna_published|free": 0.1113,
    "usable|luna_published|free": 0.1066,
    "ema|luna_ablation|free": 0.9627, "cov|luna_ablation|free": 1.0000,
    "usable|luna_ablation|free": 0.9627,
    # Figure 5 — bijection breaks out of 750
    "breaks|4omini_published|enum": 0, "breaks|4omini_ablation|free": 5,
    "breaks|luna_published|enum": 44, "breaks|luna_ablation|free": 65, "breaks_n": 750,
    # Figure 6 — answer length against EM and token-F1
    "tokens|4omini_published|enum": 14.59, "em|4omini_published|enum": 0.0280,
    "f1|4omini_published|enum": 0.2580,
    "tokens|4omini_published|free": 11.86, "em|4omini_published|free": 0.1360,
    "f1|4omini_published|free": 0.3656,
    "tokens|4omini_ablation|free": 11.65, "em|4omini_ablation|free": 0.1360,
    "f1|4omini_ablation|free": 0.3659,
    "tokens|luna_published|enum": 13.93, "em|luna_published|enum": 0.0680,
    "f1|luna_published|enum": 0.3035,
    "tokens|luna_published|free": 9.59, "em|luna_published|free": 0.2067,
    "f1|luna_published|free": 0.4390,
    "tokens|luna_ablation|free": 9.60, "em|luna_ablation|free": 0.2240,
    "f1|luna_ablation|free": 0.4577,
    # Figure 2 caption — decoding provenance
    "decode_temperature": 0.0, "decode_top_p": 1.0, "decode_seed": 42,
    "decode_prompt_identical_questions": 150,
}


def check(name: str, value: float, *, tol: float = _TOL_4DP) -> float:
    """Assert a read value against the brief and return it unchanged."""
    if name not in EXPECTED:
        raise FigureDataMismatch(f"no expected value registered for {name!r}")
    want = EXPECTED[name]
    if abs(float(value) - float(want)) > tol:
        raise FigureDataMismatch(
            f"ASSERTION FAILED for {name}: read {value!r} from the corpus, "
            f"brief states {want!r} (tolerance {tol})"
        )
    return value


# --- report parsing ----------------------------------------------------------------------------
_ROUTE_RE = re.compile(r"^\s+(\w+)\s+(enum|free)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*(DEGENERATE)?\s*$")
_FAMILY_RE = re.compile(
    r"^\s+(\w+)\s+(enum|free)\s+(\S+\.\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)
_HIST_RE = re.compile(r"^\s+([0-5])\s+(\d+)\s+[\d.]+\s*$")
_SPLIT_RE = re.compile(r"shape-SPLIT within a question \(counts 1-4\)\s*:\s*(\d+) of (\d+) = ([\d.]+)")
_STABLE_RE = re.compile(r"shape-stable across all 5 runs:\s*(\d+) of (\d+) = ([\d.]+)")
_BIJ_HEAD_RE = re.compile(r"^\s+(\w+) / (enum|free)\s+\(n=(\d+)\)")
_BIJ_BREAK_RE = re.compile(r"^\s+breaks:\s*(\d+) of (\d+)")
_LENGTH_RE = re.compile(
    r"^\s+(\w+)\s+(enum|free)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
)


def _lines(path: Path) -> list[str]:
    if not path.exists():
        raise FigureDataMismatch(f"aggregator report not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def read_route_and_rewrite() -> dict[str, dict[str, Any]]:
    """Section [2]: route stability, flip count, rewrite_rate, degeneracy flag, per cell."""
    out: dict[str, dict[str, Any]] = {}
    for line in _lines(REPORT_SERIES2):
        m = _ROUTE_RE.match(line)
        if not m or f"{m.group(1)}|{m.group(2)}" not in CELL_BY_KEY:
            continue
        key = f"{m.group(1)}|{m.group(2)}"
        out[key] = {
            "route_stability": check(f"route_stab|{key}", float(m.group(3))),
            "flips": int(check(f"flips|{key}", int(m.group(4)), tol=0)),
            "rewrite_rate": check(f"rewrite|{key}", float(m.group(5))),
            "degenerate": m.group(6) is not None,
        }
    if len(out) != len(CELLS):
        raise FigureDataMismatch(f"section [2] gave {len(out)} cells, expected {len(CELLS)}")
    return out


def read_family(node_key: str = "grade.needs_more_context") -> dict[str, dict[str, float]]:
    """Section [3]: kappa, EMA, coverage, usable_agreement for one endpoint, per cell."""
    out: dict[str, dict[str, float]] = {}
    for line in _lines(REPORT_SERIES2):
        m = _FAMILY_RE.match(line)
        if not m or m.group(3) != node_key:
            continue
        key = f"{m.group(1)}|{m.group(2)}"
        if key not in CELL_BY_KEY:
            continue
        out[key] = {
            "kappa": float(m.group(4)),
            "ema": check(f"ema|{key}", float(m.group(5))),
            "coverage": check(f"cov|{key}", float(m.group(6))),
            "usable": check(f"usable|{key}", float(m.group(7))),
        }
    if len(out) != len(CELLS):
        raise FigureDataMismatch(f"section [3] gave {len(out)} cells for {node_key}")
    return out


def read_format_guess() -> dict[str, Any]:
    """Section [1]: the luna histogram over 5 runs, the split share, and the 4o-mini comparison."""
    lines = _lines(REPORT_SERIES2)
    try:
        start = next(i for i, ln in enumerate(lines) if "luna / published / free / grade" in ln)
        stop = next(i for i, ln in enumerate(lines[start:], start) if "TOTAL" in ln)
    except StopIteration as exc:
        raise FigureDataMismatch("section [1] luna block not found") from exc

    hist: dict[int, int] = {}
    for line in lines[start:stop]:
        m = _HIST_RE.match(line)
        if m:
            hist[int(m.group(1))] = int(check(f"fmt_hist|{m.group(1)}", int(m.group(2)), tol=0))
    if sorted(hist) != [0, 1, 2, 3, 4, 5]:
        raise FigureDataMismatch(f"section [1] histogram incomplete: {sorted(hist)}")

    split = next((m for ln in lines if (m := _SPLIT_RE.search(ln))), None)
    if split is None:
        raise FigureDataMismatch("section [1] shape-SPLIT line not found")
    stable = next((m for ln in lines[stop:] if (m := _STABLE_RE.search(ln))), None)
    if stable is None:
        raise FigureDataMismatch("section [1] 4o-mini shape-stable line not found")

    total = int(check("fmt_total", int(split.group(2)), tol=0))
    if sum(hist.values()) != total:
        raise FigureDataMismatch(f"histogram sums to {sum(hist.values())}, not {total}")
    if hist[1] + hist[2] + hist[3] + hist[4] != int(split.group(1)):
        raise FigureDataMismatch("interior bins do not sum to the reported split count")
    return {
        "hist": hist,
        "split_n": int(check("fmt_split_n", int(split.group(1)), tol=0)),
        "split_share": check("fmt_split_share", float(split.group(3))),
        "total": total,
        "omini_stable": int(check("fmt_4omini_stable", int(stable.group(1)), tol=0)),
    }


def read_bijection() -> dict[str, dict[str, int]]:
    """Section [5]: bijection breaks between synthesize.scope and synthesize.confidence."""
    out: dict[str, dict[str, int]] = {}
    current: str | None = None
    for line in _lines(REPORT_SERIES2):
        head = _BIJ_HEAD_RE.match(line)
        if head:
            current = f"{head.group(1)}|{head.group(2)}"
            continue
        brk = _BIJ_BREAK_RE.match(line)
        if brk and current in CELL_BY_KEY:
            out[current] = {
                "breaks": int(check(f"breaks|{current}", int(brk.group(1)), tol=0)),
                "n": int(check("breaks_n", int(brk.group(2)), tol=0)),
            }
            current = None
    if len(out) != 4:
        raise FigureDataMismatch(f"section [5] gave {len(out)} cells, expected 4")
    return out


def read_length_quality() -> dict[str, dict[str, float]]:
    """`report_final_analysis.txt` section [3]: mean answer tokens, EM, token-F1, per cell."""
    out: dict[str, dict[str, float]] = {}
    for line in _lines(REPORT_FINAL):
        m = _LENGTH_RE.match(line)
        if not m:
            continue
        key = f"{m.group(1)}|{m.group(2)}"
        if key not in CELL_BY_KEY:
            continue
        out[key] = {
            "tokens": check(f"tokens|{key}", float(m.group(3)), tol=_TOL_2DP),
            "em": check(f"em|{key}", float(m.group(5))),
            "f1": check(f"f1|{key}", float(m.group(6))),
        }
    if len(out) != len(CELLS):
        raise FigureDataMismatch(f"final-analysis section [3] gave {len(out)} cells")
    return out


# --- recomputation through the aggregator's own functions --------------------------------------
def recompute_grade_parse() -> dict[str, dict[str, Any]]:
    """unparsed_rate and the partial count at the grade node, via `src.metrics.unparsed_rate`.

    Absent from both reports, so it is recomputed — but by the aggregator's own function over the
    aggregator's own loader, not by a second implementation of the same idea.
    """
    out: dict[str, dict[str, Any]] = {}
    for runs_dir in sorted({c.runs_dir for c in CELLS}):
        ds = load_manifests(Path(runs_dir))
        recomputed_rewrite = {arm: rewrite_rate(ds, arm) for arm in ds.arms}
        for entry in unparsed_rate(ds).values():
            if entry["node"] != "grade":
                continue
            series = next(
                (c.series for c in CELLS if c.runs_dir == runs_dir and c.arm == entry["arm"]), None
            )
            if series is None:
                continue
            key = f"{series}|{entry['arm']}"
            out[key] = {
                "unparsed_rate": check(f"unparsed|{key}", entry["unparsed_rate"]),
                "unparsed": entry["unparsed"],
                "n": entry["n"],
                "partial": entry["status_counts"].get("partial", 0),
                # Cross-check: the report's rewrite_rate must equal a fresh recomputation.
                "rewrite_rate_recomputed": recomputed_rewrite[entry["arm"]],
            }
    if len(out) != len(CELLS):
        raise FigureDataMismatch(f"grade parse recompute gave {len(out)} cells")
    check("partials|4omini_ablation|free", out["4omini_ablation|free"]["partial"], tol=0)
    return out


def recompute_decoding_provenance() -> dict[str, Any]:
    """Figure 2's caption facts, read from the luna published free manifests."""
    cell = cell_for("gpt-5.6-luna", "no_format")
    params: set[tuple[float, float, int]] = set()
    prompts: dict[str, set[str]] = {}
    for index in range(1, 6):
        path = Path(cell.runs_dir) / f"{cell.arm}_run{index}" / "run_manifest.jsonl"
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record["node"] != "grade":
                    continue
                params.add((record["temperature"], record["top_p"], record["seed"]))
                prompts.setdefault(record["question_id"], set()).add(record["prompt_sha256"])
    if len(params) != 1:
        raise FigureDataMismatch(f"decoding parameters are not uniform: {sorted(params)}")
    temperature, top_p, seed = params.pop()
    identical = sum(1 for shas in prompts.values() if len(shas) == 1)
    return {
        "temperature": check("decode_temperature", temperature),
        "top_p": check("decode_top_p", top_p),
        "seed": int(check("decode_seed", seed, tol=0)),
        "prompt_identical_questions": int(
            check("decode_prompt_identical_questions", identical, tol=0)
        ),
        "questions": len(prompts),
    }


def load_all() -> dict[str, Any]:
    """Every number the six figures draw, each already checked against the brief."""
    route = read_route_and_rewrite()
    parse = recompute_grade_parse()
    for key, entry in parse.items():  # report vs fresh recomputation must agree
        if abs(entry["rewrite_rate_recomputed"] - route[key]["rewrite_rate"]) > _TOL_4DP:
            raise FigureDataMismatch(
                f"rewrite_rate for {key}: report {route[key]['rewrite_rate']!r} but recomputation "
                f"gives {entry['rewrite_rate_recomputed']!r}"
            )
    return {
        "route": route,
        "parse": parse,
        "family": read_family(),
        "format_guess": read_format_guess(),
        "bijection": read_bijection(),
        "length": read_length_quality(),
        "decoding": recompute_decoding_provenance(),
    }
