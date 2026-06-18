"""Offline tests for src.provenance: env capture, prompt hashing, logging, validation.

No API calls — records are assembled from synthetic per-call fields and written to a temp path.
"""

from __future__ import annotations

import json

import pytest

from src.config import Config
from src.provenance import (
    REQUIRED_FIELDS,
    ProvenanceLogger,
    capture_environment,
    make_record,
    prompt_sha256,
    run_manifest_path,
    validate_record,
)


def _record(config: Config, **overrides):
    fields = dict(
        run_id="run-1",
        question_id="q-1",
        arm="enum",
        node="synthesize",
        prompt_sha256="deadbeef",
        schema_sha256="cafef00d",
        cache_hit=False,
        retrieved_ids=["a", "b"],
        retrieved_scores=[0.9, 0.1],
        raw_response='{"answer": "x"}',
        parsed={"answer": "x"},
        system_fingerprint="fp_123",
        tokens_in=10,
        tokens_out=5,
        latency_ms=12.5,
    )
    fields.update(overrides)
    return make_record(config, **fields)


# --- environment capture ----------------------------------------------------------------------


def test_capture_environment_keys_and_stability():
    env = capture_environment()
    assert set(env) == {"git_commit", "python_version", "lib_versions"}
    assert isinstance(env["git_commit"], str) and env["git_commit"]
    assert isinstance(env["python_version"], str) and env["python_version"]
    assert set(env["lib_versions"]) == {"langchain", "langgraph", "chromadb"}
    # Stable across calls (cached).
    assert capture_environment() == env


# --- prompt hashing ---------------------------------------------------------------------------


def test_prompt_sha256_is_deterministic_and_key_order_invariant():
    a = [{"role": "user", "content": "hi"}]
    b = [{"content": "hi", "role": "user"}]  # same content, different key order
    assert prompt_sha256(a) == prompt_sha256(b)
    assert len(prompt_sha256(a)) == 64


def test_prompt_sha256_is_content_sensitive():
    assert prompt_sha256([{"role": "user", "content": "hi"}]) != prompt_sha256(
        [{"role": "user", "content": "hello"}]
    )
    # Message order is semantic, so it changes the hash.
    two = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    swapped = [{"role": "user", "content": "u"}, {"role": "system", "content": "s"}]
    assert prompt_sha256(two) != prompt_sha256(swapped)


# --- record assembly / validation -------------------------------------------------------------


def test_make_record_has_exactly_the_required_fields(tmp_path):
    record = _record(Config(runs_dir=tmp_path))
    assert set(record) == REQUIRED_FIELDS
    # Config-sourced fields are filled from the frozen defaults.
    assert record["model"] == "gpt-4o-mini-2024-07-18"
    assert record["seed"] == 42
    assert record["temperature"] == 0.0
    assert record["top_p"] == 1.0


def test_validate_record_raises_on_missing_field(tmp_path):
    record = dict(_record(Config(runs_dir=tmp_path)))
    del record["model"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_record(record)


# --- logging ----------------------------------------------------------------------------------


def test_logger_round_trips_a_record(tmp_path):
    config = Config(runs_dir=tmp_path)
    record = _record(config)
    logger = ProvenanceLogger.for_run("run-1", config)
    logger.log(record)
    logger.close()

    path = run_manifest_path("run-1", config)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_logger_appends_one_line_per_call(tmp_path):
    config = Config(runs_dir=tmp_path)
    with ProvenanceLogger.for_run("run-2", config) as logger:
        logger.log(_record(config, question_id="q-1"))
        logger.log(_record(config, question_id="q-2"))
        logger.log(_record(config, node="grade"))

    lines = run_manifest_path("run-2", config).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["question_id"] for line in lines] == ["q-1", "q-2", "q-1"]


def test_logger_flushes_so_records_are_readable_immediately(tmp_path):
    config = Config(runs_dir=tmp_path)
    logger = ProvenanceLogger.for_run("run-3", config)
    try:
        logger.log(_record(config))
        # Read via a separate handle while the logger is still open — flush must have happened.
        content = logger.path.read_text(encoding="utf-8")
        assert content.count("\n") == 1
        assert json.loads(content.strip())["run_id"] == "run-1"
    finally:
        logger.close()


def test_logger_rejects_partial_record_without_writing(tmp_path):
    config = Config(runs_dir=tmp_path)
    record = dict(_record(config))
    del record["latency_ms"]
    logger = ProvenanceLogger.for_run("run-4", config)
    try:
        with pytest.raises(ValueError, match="missing required fields"):
            logger.log(record)
        # Nothing should have been written.
        assert logger.path.read_text(encoding="utf-8") == ""
    finally:
        logger.close()


def test_logger_starts_fresh_manifest_for_run_id(tmp_path):
    config = Config(runs_dir=tmp_path)

    # A prior run leaves stale records under this run_id.
    first = ProvenanceLogger.for_run("run-redo", config)
    first.log(_record(config, question_id="old-1"))
    first.log(_record(config, question_id="old-2"))
    first.close()
    assert len(run_manifest_path("run-redo", config).read_text(encoding="utf-8").splitlines()) == 2

    # Re-running the same run_id must truncate and yield exactly the new N records.
    second = ProvenanceLogger.for_run("run-redo", config)
    second.log(_record(config, question_id="new-1"))
    second.log(_record(config, question_id="new-2"))
    second.close()

    lines = run_manifest_path("run-redo", config).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # not appended to the old 2
    assert [json.loads(line)["question_id"] for line in lines] == ["new-1", "new-2"]
