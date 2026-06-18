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
from src.config import Arm, Config
from src.provenance import ProvenanceLogger, run_manifest_path

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
    """
    arms = _resolve_arms(arms)
    runs = config.k_runs if runs is None else runs

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
        from src.data import load_sampled_questions

        questions = load_sampled_questions(config)
    if limit is not None:
        questions = questions[:limit]

    if collection is None:
        collection = _open_built_collection(config)
    if embedder is None:
        from src.cache import CachingEmbedder
        from src.index import OpenAIEmbedder

        embedder = CachingEmbedder(OpenAIEmbedder(config), config)
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

            logger = ProvenanceLogger.for_run(run_id, config)  # truncates any prior manifest
            try:
                for qi, question in enumerate(questions, start=1):
                    result = run_question(
                        graph,
                        question["question"],
                        question["question_id"],
                        arm=arm,
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
    parser.add_argument("--runs", type=int, help="override k (runs per arm)")
    parser.add_argument("--limit", type=int, help="cap the number of questions")
    parser.add_argument(
        "--resume", action="store_true", help="skip any (arm, run) whose .done marker exists"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; make no API calls and write nothing"
    )
    args = parser.parse_args(argv)

    config = Config()
    arms = [Arm(value) for value in args.arms] if args.arms else None
    try:
        run_experiment(
            config,
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
