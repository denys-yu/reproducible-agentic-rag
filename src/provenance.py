"""JSONL provenance logger — one record appended per LLM call.

Each record captures the full reproducibility context required by CLAUDE.md: run/question ids,
arm, timestamp, git commit, python + library versions, model, `system_fingerprint`, decoding
params, prompt/schema hashes, cache-hit flag, retrieved ids/scores, raw + parsed response, token
counts, and latency. A `system_fingerprint` change must be logged loudly, never swallowed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Config


def run_manifest_path(run_id: str, config: Config) -> Path:
    """Return the path to a run's `run_manifest.jsonl` under `config.runs_dir`."""
    raise NotImplementedError


def make_record(
    run_id: str,
    question_id: str,
    config: Config,
    **call_fields: Any,
) -> dict[str, Any]:
    """Assemble one provenance record (static reproducibility context + per-call fields)."""
    raise NotImplementedError


def append_record(record: dict[str, Any], path: Path) -> None:
    """Append one JSON record as a single line to the run manifest."""
    raise NotImplementedError
