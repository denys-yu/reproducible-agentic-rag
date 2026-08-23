"""Second-stratum feasibility pilot: the frozen experiment, one variable changed (the model).

This is a PROBE, not a publishable stratum. It runs the existing design at reduced size — the
FIRST 15 questions of the frozen 150-question sample, k=5 runs, both arms — to answer one
question before a full run is funded: does the enum-vs-free agreement gap that the published
`gpt-4o-mini` stratum measured still exist on a 5th-generation model, or has the model's own
determinism closed it?

Nothing about the experiment is redefined here. Prompts, the graph, retrieval, chunking, the
free-text parser, the schemas and every metric come from the same modules the published stratum
used; this module only selects a model profile, points the manifests at a separate output tree,
and summarizes. The 15 questions are `load_sampled_questions(...)[:15]` — the same seed, the same
sort, the same first 15 — never a fresh sample.

Everything written lands under `results/pilot_<model>/`; the published `runs/` tree is never
touched.

    python -m src.pilot                     # run, then aggregate
    python -m src.pilot --analyze-only      # re-aggregate an existing pilot, no API calls
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Arm, Config, ModelName

# ---- pilot design (fixed) --------------------------------------------------------------------

PILOT_QUESTIONS = 15  # the FIRST 15 of the frozen 150 — a prefix, never a re-sample
PILOT_RUNS = 5
PILOT_ARMS = [Arm.FREE, Arm.ENUM]

# Fields whose inter-run stability is the actual object of study. The two `grade` fields are the
# control fields the ceiling verdict is decided on.
CONTROL_FIELDS = [("grade", "needs_more_context"), ("grade", "confidence")]
REPORTED_FIELDS = CONTROL_FIELDS + [("synthesize", "confidence")]

# Free-arm fields that the deterministic parser can fail to recover (None = "not confidently
# present"). This unparseability is the mechanism the enum arm is supposed to remove, so its rate
# bounds how large an enum-vs-free gap is even possible.
FREE_PARSE_FIELDS = {
    "grade": ["scope", "confidence", "needs_more_context"],
    "synthesize": ["confidence", "scope"],
}


def pilot_dir(config: Config) -> Path:
    return Path("results") / f"pilot_{config.model_profile.name}"


def pilot_runs_dir(config: Config) -> Path:
    return pilot_dir(config) / "runs"


# ---- provenance rollup -----------------------------------------------------------------------


def _read_records(runs_dir: Path) -> list[dict[str, Any]]:
    """Read every manifest under `runs_dir` in a deterministic (run_id, file order) order."""
    records: list[dict[str, Any]] = []
    for manifest in sorted(runs_dir.glob("*/run_manifest.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def write_provenance(records: list[dict[str, Any]], out: Path) -> Path:
    """Concatenate the per-run manifests into one pilot-wide provenance log."""
    path = out / "provenance.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return path


@dataclass(frozen=True)
class CallAudit:
    """Whole-pilot audit of the request/usage facts that the run's validity rests on."""

    n_calls: int
    run_window_utc: tuple[str | None, str | None]
    models: dict[str, int]
    reasoning_effort: dict[str, int]
    system_fingerprints: dict[str, int]
    distinct_payload_hashes: int
    reasoning_tokens_total: int
    calls_with_nonzero_reasoning_tokens: int
    tokens_in: int
    tokens_out: int
    cached_tokens: int
    cache_write_tokens: int
    cache_hits: int


def audit_calls(records: list[dict[str, Any]]) -> CallAudit:
    """Summarize the per-call provenance: what was sent, what the server reported back."""

    def total(key: str) -> int:
        return sum(r.get(key) or 0 for r in records)

    timestamps = sorted(r["timestamp"] for r in records)
    return CallAudit(
        n_calls=len(records),
        run_window_utc=(timestamps[0] if timestamps else None, timestamps[-1] if timestamps else None),
        models=dict(Counter(r["model"] for r in records)),
        # `str(None)` keeps a null effort visible as a key rather than silently absent.
        reasoning_effort=dict(Counter(str(r.get("reasoning_effort")) for r in records)),
        system_fingerprints=dict(Counter(str(r.get("system_fingerprint")) for r in records)),
        distinct_payload_hashes=len({r.get("request_payload_sha256") for r in records}),
        reasoning_tokens_total=total("reasoning_tokens"),
        calls_with_nonzero_reasoning_tokens=sum(1 for r in records if (r.get("reasoning_tokens") or 0) > 0),
        tokens_in=total("tokens_in"),
        tokens_out=total("tokens_out"),
        cached_tokens=total("cached_tokens"),
        cache_write_tokens=total("cache_write_tokens"),
        cache_hits=sum(1 for r in records if r.get("cache_hit")),
    )


def unparseable_rates(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Per arm and node.field, the fraction of calls whose parsed value came back None.

    In the enum arm this is structurally zero (strict schemas cannot omit a field); in the free
    arm it is the parser's failure rate — the mechanism the treatment removes.
    """
    counts: dict[str, dict[str, list[int]]] = {}
    for record in records:
        node = record["node"]
        if node not in FREE_PARSE_FIELDS:
            continue
        parsed = record.get("parsed") or {}
        arm_counts = counts.setdefault(record["arm"], {})
        for key in FREE_PARSE_FIELDS[node]:
            total_and_none = arm_counts.setdefault(f"{node}.{key}", [0, 0])
            total_and_none[0] += 1
            if parsed.get(key) is None:
                total_and_none[1] += 1
    return {
        arm: {
            label: {"n": n, "none": none, "rate": (none / n if n else 0.0)}
            for label, (n, none) in sorted(fields.items())
        }
        for arm, fields in sorted(counts.items())
    }


def free_format_shape(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per node, how often the FREE arm answered with a JSON object instead of prose.

    The free arm's prompt asks for labelled prose and its parser reads `label <delim> value`
    lines. A model that volunteers `{"scope":"fully",...}` instead is not disagreeing with
    itself — it is answering in a shape the (deliberately unchanged) parser does not accept.
    Tracking the shape separates that from real judgement instability.
    """
    shape: dict[str, dict[str, float]] = {}
    for node in FREE_PARSE_FIELDS:
        node_records = [r for r in records if r["arm"] == "free" and r["node"] == node]
        json_shaped = sum(1 for r in node_records if r["raw_response"].lstrip().startswith("{"))
        n = len(node_records)
        shape[node] = {"n": n, "json_shaped": json_shaped, "rate": (json_shaped / n if n else 0.0)}
    return shape


def disagreement_source(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Split the FREE arm's per-question inter-run disagreement into its two possible causes.

    A run-to-run difference in a free-arm field is either:
      - `none_value_flap` — the same judgement, recovered on some runs and not on others,
         because the model changed OUTPUT SHAPE between runs; or
      - `value_disagreement` — the model actually reported different judgements.

    Only the second is the instability the experiment set out to measure. Where the first
    dominates, an enum-vs-free gap is a parser-coverage artifact and is NOT comparable to a
    stratum whose free arm parsed cleanly.
    """
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["arm"] != "free" or record["node"] not in FREE_PARSE_FIELDS:
            continue
        key = f"{record['node']}.{record['question_id']}"
        by_question.setdefault(key, {})[record["run_id"]] = record.get("parsed") or {}

    out: dict[str, dict[str, int]] = {}
    for node, keys in FREE_PARSE_FIELDS.items():
        for field in keys:
            counts = {"all_none": 0, "all_same_value": 0, "none_value_flap": 0, "value_disagreement": 0}
            for key, runs in by_question.items():
                if not key.startswith(f"{node}."):
                    continue
                tokens = {str(parsed.get(field)) for parsed in runs.values()}
                values = tokens - {"None"}
                if tokens == {"None"}:
                    counts["all_none"] += 1
                elif len(tokens) == 1:
                    counts["all_same_value"] += 1
                elif len(values) <= 1:  # None on some runs, one consistent value on the others
                    counts["none_value_flap"] += 1
                else:
                    counts["value_disagreement"] += 1
            out[f"{node}.{field}"] = counts
    return out


# ---- verdict ---------------------------------------------------------------------------------

CEILING = "ceiling"
GAP_PRESERVED = "gap_preserved"
GAP_NARROWED = "gap_narrowed"

_CEILING_KAPPA = 0.99


def _control_stat(metrics: dict[str, Any], arm: str, node: str, key: str, stat: str) -> float | None:
    return metrics["per_arm"][arm]["fields"][f"{node}.{key}"][stat]


def classify(metrics: dict[str, Any], sources: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
    """Decide between ceiling / gap preserved / gap narrowed on the control fields.

    Ceiling means the measurement instrument has no room left: BOTH arms agree with themselves
    almost perfectly on the control fields, so no enum-vs-free difference could be detected at
    any sample size and a full run would buy nothing.

    Cohen's kappa is undefined when a field never varies across runs — which is itself perfect
    agreement, so an undefined kappa on a field whose EMA is 1.0 counts as ceiling, not as
    missing data. Reporting it any other way would let a total ceiling masquerade as a gap.
    """
    per_field = {}
    for node, key in CONTROL_FIELDS:
        label = f"{node}.{key}"
        entry = {}
        for arm in ("enum", "free"):
            kappa = _control_stat(metrics, arm, node, key, "cohen_kappa")
            ema = _control_stat(metrics, arm, node, key, "ema")
            entry[arm] = {
                "cohen_kappa": kappa,
                "ema": ema,
                # "At ceiling" = perfect (or near-perfect) inter-run agreement, whether that shows
                # up as kappa >= 0.99 or as an undefined kappa over a constant, fully-agreeing field.
                "at_ceiling": (kappa is not None and kappa >= _CEILING_KAPPA)
                or (kappa is None and ema is not None and ema >= _CEILING_KAPPA),
            }
        entry["delta_ema"] = (
            None
            if entry["enum"]["ema"] is None or entry["free"]["ema"] is None
            else entry["enum"]["ema"] - entry["free"]["ema"]
        )
        per_field[label] = entry

    both_at_ceiling = all(
        entry["enum"]["at_ceiling"] and entry["free"]["at_ceiling"] for entry in per_field.values()
    )
    max_delta = max(
        (abs(e["delta_ema"]) for e in per_field.values() if e["delta_ema"] is not None), default=0.0
    )

    if both_at_ceiling:
        verdict = CEILING
    elif max_delta >= 0.05:
        verdict = GAP_PRESERVED
    else:
        verdict = GAP_NARROWED

    result = {"verdict": verdict, "control_fields": per_field, "max_abs_delta_ema": max_delta}

    # A gap built entirely out of parse failures is not the effect the first stratum measured.
    # Say so on the verdict itself, so the number cannot be quoted without the caveat.
    if sources is not None:
        control_sources = [sources[f"{n}.{k}"] for n, k in CONTROL_FIELDS if f"{n}.{k}" in sources]
        flap = sum(s["none_value_flap"] for s in control_sources)
        real = sum(s["value_disagreement"] for s in control_sources)
        result["free_disagreement_is_parse_only"] = real == 0 and flap > 0
        result["free_control_disagreement"] = {"none_value_flap": flap, "value_disagreement": real}
    return result


# ---- aggregation -----------------------------------------------------------------------------


def published_baseline(config: Config, question_ids: list[str]) -> dict[str, Any] | None:
    """The published stratum's own numbers on THESE SAME questions — read-only, for comparison.

    A pilot metric means little in isolation; what matters is whether it moved relative to the
    stratum it is being compared against, on the same questions. `runs/` is opened read-only and
    never written. Returns None when the published manifests are not present.
    """
    from src.metrics import _answer_metrics, _field_metrics, load_manifests, rewrite_rate

    published = Path("runs")
    if not any(published.glob("*/run_manifest.jsonl")):
        return None

    ds = load_manifests(published)
    wanted = set(question_ids)
    # Restrict to the pilot's questions so the two strata are compared like for like.
    ds.complete_qids = [qid for qid in ds.complete_qids if qid in wanted]
    if not ds.complete_qids:
        return None

    fields = {
        f"{node}.{key}": {arm: _field_metrics(ds, arm, node, key) for arm in ds.arms}
        for node, key in REPORTED_FIELDS
    }
    return {
        "runs_dir": str(published),
        "n_questions": len(ds.complete_qids),
        "k": ds.k,
        "fields": fields,
        "answer": {arm: _answer_metrics(ds, arm) for arm in ds.arms},
        "rewrite_rate": {arm: rewrite_rate(ds, arm) for arm in ds.arms},
    }


def aggregate(config: Config) -> dict[str, Any]:
    """Compute the pilot's aggregate result from its manifests (no API calls)."""
    from src.data import load_sampled_questions
    from src.metrics import compute_metrics

    runs_dir = pilot_runs_dir(config)
    records = _read_records(runs_dir)
    questions = load_sampled_questions(config)[:PILOT_QUESTIONS]
    metrics = compute_metrics(config, runs_dir=runs_dir, questions=questions)

    audit = audit_calls(records)
    sources = disagreement_source(records)
    return {
        "pilot": {
            "model_profile": config.model_profile.name,
            "model_id": config.model_profile.model_id,
            "dated_snapshot": config.model_profile.dated_snapshot,
            "extra_params": config.model_profile.extra_params,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.llm_seed,
            "n_questions": PILOT_QUESTIONS,
            "k_runs": PILOT_RUNS,
            "arms": [a.value for a in PILOT_ARMS],
            "question_ids": [q["question_id"] for q in questions],
            "llm_cache_enabled": config.llm_cache_enabled,
            "embedding_cache_enabled": config.embedding_cache_enabled,
            "expected_pipeline_executions": len(PILOT_ARMS) * PILOT_RUNS * PILOT_QUESTIONS,
        },
        "call_audit": audit.__dict__,
        "unparseable_rates": unparseable_rates(records),
        "published_baseline": published_baseline(config, [q["question_id"] for q in questions]),
        "free_format_shape": free_format_shape(records),
        "free_disagreement_source": sources,
        "metrics": metrics,
        "classification": classify(metrics, sources),
    }


# ---- summary ---------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


_VERDICT_TEXT = {
    CEILING: (
        "**Ceiling.** Both arms sit at (near-)perfect inter-run agreement on the control fields. "
        "The enum-vs-free gap cannot be measured on this model as configured — a full "
        "150-question run would measure nothing."
    ),
    GAP_PRESERVED: (
        "**Gap preserved.** The enum arm is clearly ahead on the control fields, comparable to "
        "the published first stratum. A full run is warranted."
    ),
    GAP_NARROWED: (
        "**Gap narrowed but present.** The enum arm still leads, but by less than the first "
        "stratum. A full run would need the larger sample to resolve it."
    ),
}


def render_summary(result: dict[str, Any]) -> str:
    pilot, audit = result["pilot"], result["call_audit"]
    metrics, classification = result["metrics"], result["classification"]
    lines: list[str] = []
    add = lines.append

    add(f"# Pilot — {pilot['model_id']} ({pilot['n_questions']} questions)")
    add("")
    add("Feasibility probe, not a publishable result. Exactly one thing differs from the")
    add("published `gpt-4o-mini-2024-07-18` stratum: the model.")
    add("")
    add("## Configuration")
    add("")
    add(f"- model: `{pilot['model_id']}` (dated snapshot: {pilot['dated_snapshot']})")
    add(f"- decoding: temperature={pilot['temperature']}, top_p={pilot['top_p']}, seed={pilot['seed']}")
    add(f"- extra params: `{pilot['extra_params']}`")
    add(f"- design: {pilot['k_runs']} runs x {len(pilot['arms'])} arms x {pilot['n_questions']} questions "
        f"= {pilot['expected_pipeline_executions']} pipeline executions")
    add(f"- LLM response cache: {'ENABLED' if pilot['llm_cache_enabled'] else 'disabled'}; "
        f"embedding cache: {'enabled' if pilot['embedding_cache_enabled'] else 'disabled'}")
    add(f"- questions: first {pilot['n_questions']} of the frozen 150-question sample (same seed, same order)")
    add("")
    add("## Run audit")
    add("")
    add(f"- LLM calls: {audit['n_calls']}  (cache hits: {audit['cache_hits']})")
    add(f"- run window (UTC): {audit['run_window_utc'][0]} -> {audit['run_window_utc'][1]}")
    add(f"- models seen: {audit['models']}")
    add(f"- reasoning_effort seen: {audit['reasoning_effort']}")
    add(f"- system_fingerprint seen: {audit['system_fingerprints']}")
    add(f"- distinct resolved request payloads: {audit['distinct_payload_hashes']}")
    add(f"- reasoning tokens: {audit['reasoning_tokens_total']} total, "
        f"{audit['calls_with_nonzero_reasoning_tokens']} calls non-zero")
    add(f"- tokens: {audit['tokens_in']} in / {audit['tokens_out']} out "
        f"(server-cached prompt tokens: {audit['cached_tokens']}, cache writes: {audit['cache_write_tokens']})")
    add("")
    add("## Agreement by arm")
    add("")
    add("| field | arm | TARa@5 | EMA@5 | Cohen kappa |")
    add("|---|---|---|---|---|")
    for node, key in REPORTED_FIELDS:
        label = f"{node}.{key}"
        for arm in sorted(metrics["per_arm"]):
            m = metrics["per_arm"][arm]["fields"][label]
            add(f"| {label} | {arm} | {_fmt(m['tar_a'])} | {_fmt(m['ema'])} | {_fmt(m['cohen_kappa'])} |")
    for arm in sorted(metrics["per_arm"]):
        a = metrics["per_arm"][arm]["answer"]
        add(f"| answer (normalized) | {arm} | {_fmt(a['tar_a_normalized'])} | "
            f"{_fmt(a['ema_normalized'])} | {_fmt(a['cohen_kappa_normalized'])} |")
    add("")
    add("Kappa is undefined where a field never varies across runs — that is perfect agreement,")
    add("not missing data; read it together with EMA in the same row.")
    add("")
    baseline = result.get("published_baseline")
    if baseline:
        add(f"## Same numbers on the published `gpt-4o-mini` stratum (same {baseline['n_questions']} questions)")
        add("")
        add("| field | arm | TARa@5 | EMA@5 | Cohen kappa |")
        add("|---|---|---|---|---|")
        for node, key in REPORTED_FIELDS:
            label = f"{node}.{key}"
            for arm in sorted(baseline["fields"][label]):
                m = baseline["fields"][label][arm]
                add(f"| {label} | {arm} | {_fmt(m['tar_a'])} | {_fmt(m['ema'])} | {_fmt(m['cohen_kappa'])} |")
        for arm in sorted(baseline["answer"]):
            a = baseline["answer"][arm]
            add(f"| answer (normalized) | {arm} | {_fmt(a['tar_a_normalized'])} | "
                f"{_fmt(a['ema_normalized'])} | {_fmt(a['cohen_kappa_normalized'])} |")
        add("")
    add("## Unparseable (None) fields — free arm mechanism")
    add("")
    add("| arm | field | n | None | rate |")
    add("|---|---|---|---|---|")
    for arm, fields in result["unparseable_rates"].items():
        for label, stats in fields.items():
            add(f"| {arm} | {label} | {stats['n']} | {stats['none']} | {_fmt(stats['rate'])} |")
    add("")
    add("## Free-arm output shape")
    add("")
    add("| node | JSON-shaped responses | rate |")
    add("|---|---|---|")
    for node, shape in result["free_format_shape"].items():
        add(f"| {node} | {shape['json_shaped']}/{shape['n']} | {_fmt(shape['rate'])} |")
    add("")
    add("The free arm's prompt asks for labelled prose; its parser reads `label: value` lines and")
    add("does not accept quoted JSON keys. Where the model volunteers JSON, the parser correctly")
    add("returns None. The parser was NOT modified — that would change what the free arm measures.")
    add("")
    add("## What the free arm actually disagrees about")
    add("")
    add("| field | all None | all same value | None/value flap | genuine value disagreement |")
    add("|---|---|---|---|---|")
    for label, counts in result["free_disagreement_source"].items():
        add(f"| {label} | {counts['all_none']} | {counts['all_same_value']} | "
            f"{counts['none_value_flap']} | {counts['value_disagreement']} |")
    add("")
    add("## Rewrite frequency")
    add("")
    for arm in sorted(metrics["per_arm"]):
        line = f"- {arm}: {_fmt(metrics['per_arm'][arm]['rewrite_rate'])}"
        if baseline:
            line += f"  (published stratum: {_fmt(baseline['rewrite_rate'].get(arm))})"
        add(line)
    add("")
    add("## Verdict")
    add("")
    add(_VERDICT_TEXT[classification["verdict"]])
    add("")
    if classification.get("free_disagreement_is_parse_only"):
        counts = classification["free_control_disagreement"]
        add("> **Caveat — the gap is not the same gap.** On the control fields the free arm shows")
        add(f"> {counts['value_disagreement']} questions of genuine value disagreement and")
        add(f"> {counts['none_value_flap']} of None/value flapping. Every run-to-run difference in")
        add("> the free arm therefore comes from the model switching OUTPUT SHAPE between runs")
        add("> (JSON vs prose), not from the model changing its judgement. The published")
        add("> `gpt-4o-mini` stratum had a 0.000 unparseable rate, so its gap was semantic. These")
        add("> two numbers measure different things and should not be compared directly.")
        add("")
    add("| control field | enum kappa | enum EMA | free kappa | free EMA | delta EMA |")
    add("|---|---|---|---|---|---|")
    for label, entry in classification["control_fields"].items():
        add(f"| {label} | {_fmt(entry['enum']['cohen_kappa'])} | {_fmt(entry['enum']['ema'])} | "
            f"{_fmt(entry['free']['cohen_kappa'])} | {_fmt(entry['free']['ema'])} | "
            f"{_fmt(entry['delta_ema'])} |")
    add("")
    return "\n".join(lines)


# ---- driver ----------------------------------------------------------------------------------


def run_pilot(config: Config, *, resume: bool = True) -> None:
    """Execute the pilot runs into the pilot's own output tree."""
    from src.run_experiment import run_experiment

    run_experiment(
        config,
        arms=PILOT_ARMS,
        runs=PILOT_RUNS,
        limit=PILOT_QUESTIONS,
        resume=resume,
    )


def analyze(config: Config) -> dict[str, Any]:
    """Aggregate the pilot's manifests and write provenance.jsonl, aggregate.json, summary.md."""
    from src.metrics import _sanitize

    out = pilot_dir(config)
    out.mkdir(parents=True, exist_ok=True)

    records = _read_records(pilot_runs_dir(config))
    write_provenance(records, out)

    result = aggregate(config)
    (out / "aggregate.json").write_text(
        json.dumps(_sanitize(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "summary.md").write_text(render_summary(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.pilot",
        description="Run the reduced-size second-stratum pilot and summarize it.",
    )
    parser.add_argument(
        "--model",
        choices=[m.value for m in ModelName],
        default=ModelName.GPT_5_6_LUNA.value,
        help="model profile to pilot (default: gpt-5.6-luna)",
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="re-aggregate existing manifests; no API calls"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="re-run every (arm, run), ignoring .done markers"
    )
    args = parser.parse_args(argv)

    model = ModelName(args.model)
    config = Config(model=model)
    config = Config(model=model, runs_dir=pilot_runs_dir(config))

    if not args.analyze_only:
        run_pilot(config, resume=not args.no_resume)

    result = analyze(config)
    print()
    print(render_summary(result))
    print(f"\nWrote {pilot_dir(config)}/: provenance.jsonl, aggregate.json, summary.md")


if __name__ == "__main__":
    main()
