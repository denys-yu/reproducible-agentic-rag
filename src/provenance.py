"""Per-LLM-call provenance logger — append-only JSONL, one record per call.

Captures everything needed to audit and reproduce a run: who emitted the call (run/question/arm/
node), the environment (git commit, python + library versions), the pinned model and decoding
params, the prompt/schema hashes, cache-hit flag, retrieval results, the raw + parsed response,
token counts, and latency.

Records are appended one JSON object per line and flushed after every write, so a crash
mid-experiment never loses an already-logged call. `timestamp` and `latency_ms` are observational
metadata only — they must never feed back into any computation or metric.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from src.config import Arm, Config

NodeName = Literal["grade", "rewrite", "synthesize"]
_TRACKED_LIBS = ("langchain", "langgraph", "chromadb")
_REPO_ROOT = Path(__file__).resolve().parent.parent


class ProvenanceRecord(TypedDict):
    """One provenance record. Field set is exact — `validate_record` enforces completeness."""

    run_id: str
    question_id: str
    arm: str
    node: NodeName
    timestamp: str
    git_commit: str
    python_version: str
    lib_versions: dict[str, str | None]
    model: str
    system_fingerprint: str | None
    seed: int
    temperature: float
    top_p: float
    prompt_sha256: str
    schema_sha256: str | None
    cache_hit: bool
    retrieved_ids: list[str]
    retrieved_scores: list[float]
    raw_response: str
    parsed: dict[str, Any] | None
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: float


REQUIRED_FIELDS: frozenset[str] = frozenset(ProvenanceRecord.__annotations__)


# --- environment capture (computed once) ------------------------------------------------------


def _git_commit() -> str:
    """Return the HEAD commit (with a `-dirty` suffix on a modified tree), or 'unknown'."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"  # not a git repo, or git unavailable

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return f"{head}-dirty" if status.strip() else head


def _lib_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_LIBS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


@functools.lru_cache(maxsize=1)
def capture_environment() -> dict[str, Any]:
    """Capture git commit, python version, and tracked library versions (cached for the process)."""
    return {
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "lib_versions": _lib_versions(),
    }


# --- prompt hashing ---------------------------------------------------------------------------


def prompt_sha256(messages: list[dict[str, Any]]) -> str:
    """SHA-256 of the canonical serialization of the messages (key order / whitespace invariant)."""
    blob = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --- record assembly + validation -------------------------------------------------------------


def validate_record(record: Mapping[str, Any]) -> None:
    """Raise ValueError if any required field is absent (fail loud; never write a partial record)."""
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Provenance record missing required fields: {sorted(missing)}")


def make_record(
    config: Config,
    *,
    run_id: str,
    question_id: str,
    arm: Arm | str,
    node: NodeName,
    prompt_sha256: str,
    schema_sha256: str | None,
    cache_hit: bool,
    retrieved_ids: list[str],
    retrieved_scores: list[float],
    raw_response: str,
    parsed: dict[str, Any] | None,
    system_fingerprint: str | None,
    tokens_in: int | None,
    tokens_out: int | None,
    latency_ms: float,
) -> ProvenanceRecord:
    """Assemble a complete, validated provenance record from config + per-call fields.

    Static reproducibility context (env, model, seed, decoding params) is filled from config;
    the caller supplies the per-call observations. The result is validated before return.
    """
    env = capture_environment()
    record: ProvenanceRecord = {
        "run_id": run_id,
        "question_id": question_id,
        "arm": arm.value if isinstance(arm, Arm) else arm,
        "node": node,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": env["git_commit"],
        "python_version": env["python_version"],
        "lib_versions": dict(env["lib_versions"]),  # copy so callers can't mutate the cache
        "model": config.llm_model,
        "system_fingerprint": system_fingerprint,
        "seed": config.llm_seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "cache_hit": cache_hit,
        "retrieved_ids": list(retrieved_ids),
        "retrieved_scores": [float(score) for score in retrieved_scores],
        "raw_response": raw_response,
        "parsed": parsed,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
    }
    validate_record(record)
    return record


# --- logger -----------------------------------------------------------------------------------


def run_manifest_path(run_id: str, config: Config) -> Path:
    """Return the path to a run's `run_manifest.jsonl` under `config.runs_dir`."""
    return config.runs_dir / run_id / "run_manifest.jsonl"


class ProvenanceLogger:
    """JSONL writer: starts a FRESH manifest per run, one validated record per line, flushed.

    Constructing a logger for a run_id truncates any existing manifest for that run_id, so
    re-running a run_id yields a clean manifest and never accumulates stale records across runs.
    Within a run, records are appended sequentially and flushed after each write (crash-safe).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # "w" truncates any prior manifest at construction; subsequent writes append in-run.
        self._handle = self._path.open("w", encoding="utf-8")

    @classmethod
    def for_run(cls, run_id: str, config: Config) -> ProvenanceLogger:
        """Open the logger at the conventional manifest path for a run."""
        return cls(run_manifest_path(run_id, config))

    @property
    def path(self) -> Path:
        return self._path

    def log(self, record: Mapping[str, Any]) -> None:
        """Validate then append one record as a single JSON line, flushing immediately."""
        validate_record(record)
        self._handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> ProvenanceLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
