"""Offline tests for the condition switch, the LF helper, and the run-start gates.

Nothing here touches the network or any model provider. The payload-capture test builds the real
request body via langchain_openai's own request builder and never sends it — that is the whole
point of the mechanism under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Arm, Condition, Config
from src.hashing import sha256_file, sha256_text, to_lf
from src.preflight import (
    PILOT_15_QIDS,
    PreflightError,
    assert_condition_contract,
    assert_dataset_pin,
    assert_output_dir_not_ignored,
    assert_provenance_durable,
    is_git_ignored,
    load_reference,
)
from src.prompts import (
    BASE_PROMPTS,
    FREE_FORMAT_SUFFIX,
    GRADE_PROMPT,
    NODES,
    REWRITE_PROMPT,
    SYNTHESIZE_PROMPT,
    compose_system_prompt,
    composed_hash_table,
)


# --- §1 line-ending normalisation --------------------------------------------------------------


def test_to_lf_converts_crlf_fixture():
    crlf = "line one\r\nline two\r\nline three"
    assert to_lf(crlf) == "line one\nline two\nline three"
    assert "\r" not in to_lf(crlf)


def test_to_lf_converts_lone_cr_without_doubling_crlf():
    assert to_lf("a\rb") == "a\nb"  # classic-Mac CR
    assert to_lf("a\r\nb") == "a\nb"  # CRLF must not become two newlines


def test_to_lf_is_idempotent_and_leaves_lf_untouched():
    lf = "already\nnormalised\n"
    assert to_lf(lf) == lf
    assert to_lf(to_lf(lf)) == to_lf(lf)


def test_sha256_text_is_line_ending_invariant():
    """The bug this whole helper exists to prevent: same text, different endings, same digest."""
    body = "Write one line per field, in this exact form:\nfield_name: value\ndone."
    assert sha256_text(body) == sha256_text(body.replace("\n", "\r\n"))


def test_sha256_text_is_still_content_sensitive():
    assert sha256_text("alpha") != sha256_text("beta")


def test_sha256_file_hashes_raw_bytes(tmp_path):
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"abc\r\ndef")
    import hashlib

    assert sha256_file(path) == hashlib.sha256(b"abc\r\ndef").hexdigest()


# --- §2 restored published wording -------------------------------------------------------------


def test_prompt_constants_match_the_published_reference():
    reference = load_reference()
    for name, text in (
        ("GRADE_PROMPT", GRADE_PROMPT),
        ("REWRITE_PROMPT", REWRITE_PROMPT),
        ("SYNTHESIZE_PROMPT", SYNTHESIZE_PROMPT),
    ):
        assert sha256_text(text) == reference["prompts"][name]["sha256"], name


def test_published_wording_is_not_modernised():
    """The old phrasing is the point; the revised wording must not creep back in."""
    assert "fully, partially, or not at all (none)" in GRADE_PROMPT
    assert "(high, medium, low)" in GRADE_PROMPT
    assert "(high, medium, low)" in SYNTHESIZE_PROMPT
    assert "fully, partially, or none" in SYNTHESIZE_PROMPT
    for text in BASE_PROMPTS.values():
        assert "full, partial, or none" not in text
        assert "high, medium, or low" not in text


def test_phrase_exactly_one_of_appears_nowhere():
    for text in (*BASE_PROMPTS.values(), FREE_FORMAT_SUFFIX):
        assert "exactly one of" not in text.lower()


# --- §3 condition is required ------------------------------------------------------------------


def test_compose_system_prompt_requires_a_condition():
    with pytest.raises(TypeError):
        compose_system_prompt("grade", Arm.FREE)  # type: ignore[call-arg]


def test_config_leaves_condition_unset_by_default():
    assert Config().condition is None


def test_unknown_condition_and_node_fail_loud():
    with pytest.raises(ValueError):
        compose_system_prompt("grade", Arm.FREE, "not-a-condition")
    with pytest.raises(ValueError):
        compose_system_prompt("nonesuch", Arm.FREE, Condition.PUBLISHED)


# --- §4 condition contract ---------------------------------------------------------------------


def test_published_condition_contract_holds():
    passed = assert_condition_contract(Condition.PUBLISHED)
    assert passed, "contract returned no assertions"


def test_ablation_condition_contract_holds():
    passed = assert_condition_contract(Condition.ABLATION)
    assert passed


def test_published_arms_are_byte_identical_at_every_node():
    for node in NODES:
        assert compose_system_prompt(node, Arm.ENUM, Condition.PUBLISHED) == compose_system_prompt(
            node, Arm.FREE, Condition.PUBLISHED
        )


def test_published_has_no_suffix_anywhere():
    for arm in (Arm.ENUM, Arm.FREE):
        for node in NODES:
            assert FREE_FORMAT_SUFFIX not in compose_system_prompt(node, arm, Condition.PUBLISHED)


def test_ablation_suffixes_only_the_free_grade_node():
    free_grade = compose_system_prompt("grade", Arm.FREE, Condition.ABLATION)
    enum_grade = compose_system_prompt("grade", Arm.ENUM, Condition.ABLATION)
    assert free_grade == enum_grade + FREE_FORMAT_SUFFIX
    assert free_grade != enum_grade


def test_ablation_suffix_never_reaches_synthesize():
    """The single most important assertion in this module."""
    free = compose_system_prompt("synthesize", Arm.FREE, Condition.ABLATION)
    enum = compose_system_prompt("synthesize", Arm.ENUM, Condition.ABLATION)
    assert free == enum, "suffix leaked into the synthesize node"
    assert FREE_FORMAT_SUFFIX not in free
    assert FREE_FORMAT_SUFFIX not in enum


def test_ablation_rewrite_is_unsuffixed_in_both_arms():
    assert compose_system_prompt("rewrite", Arm.FREE, Condition.ABLATION) == compose_system_prompt(
        "rewrite", Arm.ENUM, Condition.ABLATION
    )
    assert FREE_FORMAT_SUFFIX not in compose_system_prompt("rewrite", Arm.FREE, Condition.ABLATION)


def test_ablation_leaves_every_unsuffixed_node_at_the_published_hash():
    published = composed_hash_table(Condition.PUBLISHED)
    ablation = composed_hash_table(Condition.ABLATION)
    changed = [key for key in published if published[key] != ablation[key]]
    assert changed == ["free|grade"], f"ablation changed more than the free grade node: {changed}"


def test_contract_detects_a_leaked_suffix(monkeypatch):
    """Break the containment property on purpose; the gate must catch it."""
    import src.prompts as prompts

    monkeypatch.setattr(
        prompts,
        "_SUFFIXED",
        frozenset({(Condition.ABLATION, Arm.FREE.value, "grade"),
                   (Condition.ABLATION, Arm.FREE.value, "synthesize")}),
    )
    with pytest.raises(PreflightError, match="synthesize"):
        assert_condition_contract(Condition.ABLATION)


def test_contract_detects_reworded_prompt(monkeypatch):
    import src.prompts as prompts

    monkeypatch.setitem(prompts.BASE_PROMPTS, "grade", "reworded prompt")
    with pytest.raises(PreflightError):
        assert_condition_contract(Condition.PUBLISHED)


# --- §6 dataset pin ----------------------------------------------------------------------------


def test_pilot_qids_constant_is_fifteen_unique_ids():
    assert len(PILOT_15_QIDS) == 15
    assert len(set(PILOT_15_QIDS)) == 15


def test_dataset_pin_rejects_a_wrong_hash(tmp_path):
    shard = tmp_path / "shard.arrow"
    shard.write_bytes(b"not the pinned dataset")
    config = Config(dataset_file=shard, dataset_file_sha256="0" * 64)
    with pytest.raises(PreflightError, match="hash mismatch"):
        assert_dataset_pin(config)


def test_dataset_pin_rejects_a_missing_file(tmp_path):
    config = Config(dataset_file=tmp_path / "absent.arrow")
    with pytest.raises(PreflightError, match="missing"):
        assert_dataset_pin(config)


def test_dataset_pin_rejects_unresolved_pilot_ids(tmp_path):
    shard = tmp_path / "shard.arrow"
    shard.write_bytes(b"fixture")
    config = Config(
        dataset_file=shard,
        dataset_file_sha256=sha256_file(shard),
        dataset_record_count=3,
    )
    with pytest.raises(PreflightError, match="pilot question ids"):
        assert_dataset_pin(config, question_ids=["a", "b", "c"])


def test_dataset_pin_accepts_a_matching_fixture(tmp_path):
    shard = tmp_path / "shard.arrow"
    shard.write_bytes(b"fixture")
    ids = list(PILOT_15_QIDS) + ["extra"]
    config = Config(
        dataset_file=shard,
        dataset_file_sha256=sha256_file(shard),
        dataset_record_count=len(ids),
    )
    report = assert_dataset_pin(config, question_ids=ids)
    assert report.pilot_resolved == 15
    assert report.record_count == len(ids)


# --- §7 provenance durability ------------------------------------------------------------------


def test_default_runs_dir_is_not_git_ignored():
    assert not is_git_ignored(Config().runs_dir)


def test_runs_and_results_are_no_longer_git_ignored():
    """Module 2 reversed the `runs/` ignore rule; both evidence dirs must now be writable."""
    assert not is_git_ignored("runs")
    assert not is_git_ignored("results")
    assert assert_output_dir_not_ignored("runs") == Path("runs")
    assert assert_output_dir_not_ignored("results") == Path("results")


def test_ignored_output_dir_is_refused_and_names_the_responsible_rule():
    """An ignored path must abort AND say which .gitignore line is responsible."""
    with pytest.raises(PreflightError) as excinfo:
        assert_output_dir_not_ignored("chroma")
    message = str(excinfo.value)
    assert "responsible rule" in message
    assert ".gitignore:" in message and "chroma" in message


def test_output_dir_check_fails_closed_outside_a_work_tree(tmp_path):
    """Outside any repo git cannot answer, and an unanswerable path is rejected, not allowed."""
    with pytest.raises(PreflightError, match="fails CLOSED"):
        assert_output_dir_not_ignored(tmp_path / "nowhere")


def test_provenance_path_is_printed_at_start(capsys, git_tmp_path):
    assert_provenance_durable(Config(runs_dir=git_tmp_path))
    assert "provenance output path:" in capsys.readouterr().out


# --- §5 payload capture (no network) -----------------------------------------------------------


def test_payload_capture_writes_bodies_with_pinned_params(tmp_path):
    from src.preflight import capture_payloads

    written = capture_payloads(condition=Condition.PUBLISHED, out_dir=tmp_path)
    assert len(written) == 2 * 2 * 3  # profiles x arms x nodes

    for path in written:
        record = json.loads(path.read_text(encoding="utf-8"))
        params = record["captured_params"]
        assert record["sent"] is False
        assert params["temperature"] == 0.0
        assert params["top_p"] == 1.0
        assert params["seed"] == 42
        if record["model_profile"].startswith("gpt-5"):
            assert params["reasoning_effort"] == "none"
