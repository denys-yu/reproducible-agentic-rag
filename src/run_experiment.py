"""Experiment driver: for each arm, for each run, for each question -> one pipeline invocation.

Runs the frozen design (arms x k runs x N questions) and writes one provenance manifest per
(arm, run). It does NOT compute metrics — that is `src.metrics`, which consumes these manifests.

Execution is strictly SEQUENTIAL — one `run_question` at a time, no threads/processes/async. This
is required (the enum arm's response-serialization warning filter is process-global and only safe
without overlapping invokes) and better for reproducibility (concurrent requests get co-batched
server-side, a known non-determinism source).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from src.agent import build_agent, run_question
from src.config import Arm, Condition, Config, ModelName
from src.preflight import (
    assert_condition_contract,
    assert_format_appendix_wiring,
    assert_output_dir_not_ignored,
    assert_pinned_ids_resolve,
    assert_provenance_durable,
    load_pinned_question_ids,
    question_set_sha256,
    write_smoke_question_ids,
)
from src.provenance import (
    ProvenanceLogger,
    build_run_manifest,
    run_manifest_path,
    write_run_manifest,
)

_DONE_MARKER = ".done"


@dataclass(frozen=True)
class ExperimentSummary:
    """Summary of a driver run (the provenance manifests are the canonical artifacts)."""

    arms: list[str]
    runs: int
    n_questions: int
    invocations: int
    executed_run_ids: list[str]
    skipped_run_ids: list[str]
    manifest_paths: list[str]
    dry_run: bool


def _resolve_arms(arms: list[Arm] | None) -> list[Arm]:
    """Default to both arms (free first), else dedupe the given arms preserving order."""
    if not arms:
        return [Arm.FREE, Arm.ENUM]
    seen: dict[Arm, None] = {}
    for arm in arms:
        seen.setdefault(arm, None)
    return list(seen)


def _planned_question_count(config: Config, questions: list[Any] | None, limit: int | None) -> int:
    base = len(questions) if questions is not None else config.n_questions
    return base if limit is None else min(base, limit)


def _open_built_collection(config: Config) -> Any:
    from src.index import open_collection

    try:
        return open_collection(config)
    except Exception as exc:  # collection missing / not built yet
        raise RuntimeError(
            f"Vector index not found under {config.chroma_dir!s} ({exc}). "
            "Build it first: python -m src.index --build"
        ) from exc


def _print_plan(config: Config, arms: list[Arm], runs: int, n_questions: int) -> None:
    invocations = len(arms) * runs * n_questions
    print("DRY RUN - plan (no API calls, nothing written):")
    print(f"  arms          : {[a.value for a in arms]}")
    print(f"  runs (k)      : {runs}")
    print(f"  questions     : {n_questions}")
    print(f"  invocations   : {invocations}  ({len(arms)} arms x {runs} runs x {n_questions} q)")
    print(f"  approx LLM calls : {2 * invocations}-{3 * invocations} (2 per pipeline + up to 1 rewrite)")
    print(f"  query embeddings : ~{invocations}-{2 * invocations} (1 per retrieval round, cached)")
    print(f"  manifests would go to: {config.runs_dir!s}/<arm>_run<i>/run_manifest.jsonl")


def _print_param_guard(report: Any) -> None:
    """Print the verified outbound request params for the run's first call."""
    print(f"Model: {report.model_id}  (params delivered via: {report.mechanism})")
    if report.rerouted:
        print(f"  rerouted through model_kwargs (client stripped them): {report.rerouted}")
    print(f"  required params verified present : {report.required}")
    print(f"  resolved outbound payload        : {report.payload}")


def _print_summary(config: Config, summary: ExperimentSummary) -> None:
    print("Done.")
    print(f"  arms x runs x questions : {len(summary.arms)} x {summary.runs} x {summary.n_questions}")
    print(f"  pipeline invocations    : {summary.invocations}")
    if summary.skipped_run_ids:
        print(f"  skipped (resume)        : {summary.skipped_run_ids}")
    print(f"  manifests under         : {config.runs_dir!s}/<arm>_run<i>/run_manifest.jsonl")


def run_experiment(
    config: Config,
    *,
    condition: Condition | str,
    arms: list[Arm] | None = None,
    runs: int | None = None,
    limit: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    collection: Any | None = None,
    embedder: Any | None = None,
    model: Any | None = None,
    questions: list[Any] | None = None,
) -> ExperimentSummary:
    """Run the experiment sequentially, writing one provenance manifest per (arm, run).

    Setup is done once: the persisted collection, a caching embedder, and the compiled graph are
    reused across every arm/run/question. `collection`/`embedder`/`model`/`questions` are
    injectable so the driver runs fully offline in tests.

    `condition` is a required keyword: the prompt wiring in force is never implicit. Before any
    call is made, the condition's prompt contract is asserted against the recovered published
    reference and the provenance path is checked for durability — both fail hard.
    """
    condition = Condition(condition)
    arms = _resolve_arms(arms)
    runs = config.k_runs if runs is None else runs

    # ---- run-start gates (offline, fail hard) ----
    print(f"condition: {condition.value}")
    assert_provenance_durable(config)
    assert_output_dir_not_ignored(config.runs_dir)
    for description in assert_condition_contract(condition):
        print(f"  PASS  {description}")
    for description in assert_format_appendix_wiring(condition):
        print(f"  PASS  {description}")

    if dry_run:
        n_questions = _planned_question_count(config, questions, limit)
        _print_plan(config, arms, runs, n_questions)
        return ExperimentSummary(
            arms=[a.value for a in arms],
            runs=runs,
            n_questions=n_questions,
            invocations=0,
            executed_run_ids=[],
            skipped_run_ids=[],
            manifest_paths=[],
            dry_run=True,
        )

    # ---- setup (once) ----
    if questions is None:
        # The pinned 150, read from pins/published_question_ids.json and hash-checked. The sampler
        # is deliberately NOT used here: it reproduces this set only while numpy's bit stream and
        # n_questions both hold still, and a silent drift there would swap the question set under
        # a run that still looked deterministic.
        from src.data import load_questions_by_ids

        pinned_ids = load_pinned_question_ids()
        print(f"pinned question set: n={len(pinned_ids)} sha256={question_set_sha256(pinned_ids)}")
        questions = load_questions_by_ids(config, pinned_ids)
        assert_pinned_ids_resolve(pinned_ids, {q["question_id"] for q in questions})
        print(f"  PASS  all {len(pinned_ids)} pinned ids resolve in the loaded split")
        smoke_path = write_smoke_question_ids(pinned_ids)
        print(f"  smoke subset written: {smoke_path} ({len(pinned_ids[:15])} ids)")
    if limit is not None:
        questions = questions[:limit]

    question_ids = sorted(q["question_id"] for q in questions)
    question_set_hash = question_set_sha256(question_ids)

    if collection is None:
        collection = _open_built_collection(config)
    if embedder is None:
        from src.cache import CachingEmbedder
        from src.index import OpenAIEmbedder

        embedder = CachingEmbedder(OpenAIEmbedder(config), config)

    if model is None:
        # Build the client here (rather than inside build_agent) so the outbound-parameter guard
        # runs — and can abort — before a single question is executed.
        from src.llm import ParamGuardReport, make_llm

        guard: list[ParamGuardReport] = []
        model = make_llm(config, report=guard)
        _print_param_guard(guard[0])

    graph = build_agent(config, collection=collection, embedder=embedder, model=model)

    # ---- sequential iteration ----
    executed: list[str] = []
    skipped: list[str] = []
    manifests: list[str] = []
    invocations = 0

    for arm in arms:
        for i in range(1, runs + 1):
            run_id = f"{arm.value}_run{i}"  # encodes the arm so the two arms never collide
            done_marker = config.runs_dir / run_id / _DONE_MARKER
            if resume and done_marker.exists():
                skipped.append(run_id)
                print(f"[skip] {run_id} (.done present)")
                continue

            run_dir = config.runs_dir / run_id
            manifest = build_run_manifest(
                config,
                run_id=run_id,
                condition=condition.value,
                arms=[a.value for a in arms],
                models=[config.llm_model],
                question_set_sha256=question_set_hash,
                n_questions=len(questions),
                collection=collection,
            )
            manifest_path = write_run_manifest(manifest, run_dir)
            print(
                f"[{run_id}] run manifest: {manifest_path} "
                f"(commit {manifest['git']['commit'][:12]}"
                f"{f", {manifest['git']['tracked_modifications']} tracked mods — diff dumped" if manifest['git']['tracked_modifications'] else ''}"
                f"{f", {manifest['git']['untracked_files']} untracked" if manifest['git']['untracked_files'] else ''}, "
                f"corpus {manifest['corpus']['document_count']} docs)"
            )

            logger = ProvenanceLogger.for_run(run_id, config)  # truncates any prior manifest
            try:
                for qi, question in enumerate(questions, start=1):
                    result = run_question(
                        graph,
                        question["question"],
                        question["question_id"],
                        arm=arm,
                        condition=condition,
                        variant=config.schema_variant,
                        run_id=run_id,
                        logger=logger,
                    )
                    calls = 2 + result.rewrite_count  # grade + synthesize (+ rewrite)
                    print(
                        f"[{arm.value} run {i}/{runs}] question {qi}/{len(questions)} "
                        f"(qid {question['question_id']}) -> {calls} calls, "
                        f"{result.rewrite_count} rewrites"
                    )
                    invocations += 1
            finally:
                logger.close()

            done_marker.write_text("")  # completion marker for --resume
            executed.append(run_id)
            manifests.append(str(run_manifest_path(run_id, config)))

    summary = ExperimentSummary(
        arms=[a.value for a in arms],
        runs=runs,
        n_questions=len(questions),
        invocations=invocations,
        executed_run_ids=executed,
        skipped_run_ids=skipped,
        manifest_paths=manifests,
        dry_run=False,
    )
    _print_summary(config, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Resolve config + CLI flags and launch the experiment sequentially."""
    parser = argparse.ArgumentParser(
        prog="python -m src.run_experiment",
        description="Run the agentic RAG experiment (sequential) and write provenance manifests.",
    )
    parser.add_argument(
        "--arms",
        action="append",
        choices=[Arm.FREE.value, Arm.ENUM.value],
        help="arm to run; repeatable (default: both)",
    )
    parser.add_argument(
        "--model",
        choices=[m.value for m in ModelName],
        help="model profile / stratum to run (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--condition",
        required=True,  # never defaulted: the prompt wiring must always be stated
        choices=[c.value for c in Condition],
        help="prompt-wiring condition (published | ablation)",
    )
    parser.add_argument("--runs", type=int, help="override k (runs per arm)")
    parser.add_argument("--limit", type=int, help="cap the number of questions")
    parser.add_argument(
        "--resume", action="store_true", help="skip any (arm, run) whose .done marker exists"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; make no API calls and write nothing"
    )
    args = parser.parse_args(argv)

    config = Config(model=ModelName(args.model)) if args.model else Config()
    arms = [Arm(value) for value in args.arms] if args.arms else None
    try:
        run_experiment(
            config,
            condition=Condition(args.condition),
            arms=arms,
            runs=args.runs,
            limit=args.limit,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
