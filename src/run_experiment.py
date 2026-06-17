"""Experiment driver: for each sampled question, for each arm, for each run -> pipeline.

Orchestrates the frozen design (N questions x arms x k runs), invoking the agentic pipeline and
writing one provenance record per LLM call. CLI flags (parsed via `src.config`) override any
frozen parameter for ablations and smoke tests.
"""

from __future__ import annotations

from src.config import Config, load_config


def run_experiment(config: Config) -> None:
    """Execute the full N x arms x k run matrix, logging provenance for every LLM call."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Resolve config from CLI/env and launch the experiment."""
    config = load_config(argv)
    run_experiment(config)


if __name__ == "__main__":
    main()
