"""Offline tests for the Module 2 additions.

Covers the per-record fields, the output-shape observer, the format-appendix wiring assertion, the
pinned question set, the fail-closed output-dir gate, and the run-level manifest.

ZERO network calls. Every model here is a fake, and the one real artifact touched — `runs/` — is
only ever read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Arm, Condition, Config
from src.llm import (
    intercepted_prompt_sha256,
    intercepted_request_params,
    parse_free_response,
)
from src.preflight import (
    PINNED_QUESTION_COUNT,
    PINNED_QUESTION_IDS_SHA256,
    PreflightError,
    assert_format_appendix_wiring,
    assert_pinned_ids_resolve,
    load_pinned_question_ids,
    question_set_sha256,
    write_smoke_question_ids,
)
from src.prompts import (
    FREE_FORMAT_SUFFIX,
    NODES,
    compose_system_prompt,
    suffixed_nodes,
    suffixed_nodes_for_condition,
)
from src.provenance import (
    OUTPUT_SHAPES,
    REQUIRED_FIELDS,
    build_run_manifest,
    classify_output_shape,
    classify_parse_status,
    corpus_digest,
    make_record,
    run_index_from_id,
    validate_record,
    write_run_manifest,
)

RUNS_DIR = Path("runs")


# --- section 2: output_shape, a pure first-match-wins observer ---------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "empty"),
        ("   \n\t ", "empty"),
        ('{"scope": "full"}', "json_object"),
        ("[1, 2, 3]", "json_object"),
        ('```json\n{"scope": "full"}\n```', "json_fenced"),
        ('Here you go:\n```\n{"a": 1}\n```\n', "json_fenced"),
        ("scope: full\nconfidence: high", "labeled_lines"),
        ("- **Scope**: the context fully covers the question.", "labeled_lines"),
        ("The context covers this question completely.", "prose"),
        ("```\nnot json at all\n```", "prose"),
    ],
)
def test_output_shape_classification(text, expected):
    assert classify_output_shape(text) == expected


def test_output_shape_is_deterministic_and_total():
    """Every input lands in exactly one declared shape, and repeat calls agree."""
    samples = ["", "{}", '```json\n{}\n```', "a: b", "words", "— dash", "null"]
    for text in samples:
        first = classify_output_shape(text)
        assert first in OUTPUT_SHAPES
        assert classify_output_shape(text) == first


def test_observer_delimiters_derive_from_the_parser(monkeypatch):
    """The observer must READ `llm._DELIMITERS`, never restate it.

    Regression guard for a real defect: the observer hard-coded `:` while the parser also accepted
    em- and en-dashes, so 750 em-dash-delimited synthesize records parsed as labelled fields while
    being reported as `prose`. Both halves below must hold — the structural half alone would pass
    against a copy-pasted duplicate of the class, and the behavioural half is what proves there is
    a single source of truth.
    """
    import src.provenance as prov
    from src.llm import _DELIMITERS

    # Structural: the parser's class appears verbatim in the compiled observer pattern.
    assert _DELIMITERS in prov._labelled_line_re().pattern

    # Behavioural: change the parser's delimiters and the observer must follow.
    prov._labelled_line_re.cache_clear()
    try:
        monkeypatch.setattr("src.llm._DELIMITERS", r"[=]")
        rebuilt = prov._labelled_line_re()
        assert rebuilt.search("answer = x")          # the new delimiter is honoured
        assert not rebuilt.search("answer — x")      # the old one is no longer special
    finally:
        # Order matters: monkeypatch only undoes at teardown, which is AFTER this block, so the
        # patch is reverted by hand and only then is the memoised pattern dropped. Clearing first
        # would rebuild from the still-patched value and leak it into every later test.
        monkeypatch.undo()
        prov._labelled_line_re.cache_clear()

    # Restored: the real class is back in force.
    assert prov.classify_output_shape("answer — x") == "labeled_lines"


def test_labeled_lines_accepts_every_parser_delimiter():
    """Each delimiter the parser recognises must also make the observer say labeled_lines."""
    for delimiter in (":", "—", "–", "-"):
        assert classify_output_shape(f"answer{delimiter} 42") == "labeled_lines", delimiter


def test_hyphenated_first_word_is_not_a_false_positive():
    """`-` is a parser delimiter, so the required whitespace is what keeps prose out."""
    assert classify_output_shape("Multi-word answers are common in this corpus.") == "prose"
    assert classify_output_shape("Mensch-argere dich nicht is a board game.") == "prose"


def test_output_shape_precedence_fenced_beats_labeled():
    """First match wins: fenced JSON stays json_fenced even with labelled lines around it."""
    text = 'scope: full\n```json\n{"scope": "full"}\n```'
    assert classify_output_shape(text) == "json_fenced"


def test_output_shape_never_influences_parsing():
    """The observer is inert: parsing the same text yields the same fields either way."""
    text = "scope: full\nconfidence: high\nneeds_more_context: no"
    before = parse_free_response("grade", text)
    classify_output_shape(text)
    assert parse_free_response("grade", text) == before


# --- section 1: parse_status -------------------------------------------------------------------


def test_parse_status_levels():
    full = {"scope": "full", "confidence": "high", "needs_more_context": False}
    some = {"scope": "full", "confidence": None, "needs_more_context": None}
    none = {"scope": None, "confidence": None, "needs_more_context": None}
    assert classify_parse_status("grade", full) == "ok"
    assert classify_parse_status("grade", some) == "partial"
    assert classify_parse_status("grade", none) == "unparsed"
    assert classify_parse_status("grade", None) == "unparsed"


def test_run_index_from_id():
    assert run_index_from_id("free_run3") == 3
    assert run_index_from_id("enum_run12") == 12
    assert run_index_from_id("smoke-free-5a71") is None


# --- section 1: the record carries every Module 2 field ----------------------------------------


def _record(config, **overrides):
    fields = dict(
        run_id="free_run2",
        question_id="q-1",
        arm="free",
        node="grade",
        prompt_sha256="deadbeef",
        schema_sha256=None,
        cache_hit=False,
        retrieved_ids=["a"],
        retrieved_scores=[0.9],
        raw_response="scope: full\nconfidence: high\nneeds_more_context: no",
        parsed={"scope": "full", "confidence": "high", "needs_more_context": False},
        system_fingerprint=None,
        tokens_in=10,
        tokens_out=5,
        latency_ms=1.0,
        condition="ablation",
        request_params={"temperature": 0.0, "top_p": 1.0, "seed": 42, "reasoning_effort": None},
        format_suffix_nodes=["grade"],
    )
    fields.update(overrides)
    return make_record(config, **fields)


def test_record_carries_every_module2_field(tmp_path):
    record = _record(Config(runs_dir=tmp_path))
    for field in (
        "condition", "arm", "model", "node", "run_index", "question_id",
        "output_shape", "parse_status", "raw_response", "prompt_sha256",
        "request_params", "system_fingerprint", "format_suffix_nodes",
    ):
        assert field in record, field
    assert record["condition"] == "ablation"
    assert record["run_index"] == 2
    assert record["output_shape"] == "labeled_lines"
    assert record["parse_status"] == "ok"
    assert record["format_suffix_nodes"] == ["grade"]
    assert set(record["request_params"]) >= {"temperature", "top_p", "seed", "reasoning_effort"}


def test_record_keeps_the_pre_existing_fields(tmp_path):
    """Module 2 is additive: no original field was dropped or renamed."""
    record = _record(Config(runs_dir=tmp_path))
    for field in (
        "run_id", "timestamp", "git_commit", "python_version", "lib_versions", "model",
        "model_profile", "seed", "temperature", "top_p", "reasoning_effort", "schema_sha256",
        "cache_hit", "retrieved_ids", "retrieved_scores", "parsed", "tokens_in", "tokens_out",
        "latency_ms",
    ):
        assert field in record, field
    assert set(record) == REQUIRED_FIELDS


def test_system_fingerprint_is_null_not_omitted(tmp_path):
    """A model publishing no fingerprint yields an explicit null, never a missing key."""
    record = _record(Config(runs_dir=tmp_path), system_fingerprint=None)
    assert "system_fingerprint" in record
    assert record["system_fingerprint"] is None
    assert json.loads(json.dumps(record))["system_fingerprint"] is None


def test_record_without_a_condition_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="condition"):
        _record(Config(runs_dir=tmp_path), condition=None)
    tampered = {**_record(Config(runs_dir=tmp_path)), "condition": "made-up"}
    with pytest.raises(ValueError, match="condition"):
        validate_record(tampered)


def test_raw_response_is_stored_verbatim(tmp_path):
    """Byte for byte, including the CRLF and markdown the model actually emitted."""
    raw = "- **Scope**: full\r\n- **Confidence**: high\r\n"
    record = _record(Config(runs_dir=tmp_path), raw_response=raw)
    assert record["raw_response"] == raw


# --- section 1: request params come from the intercepted body, not from config -----------------


class _FakePayloadModel:
    """A client whose rendered body DROPS temperature — the gpt-5.x failure mode, in miniature."""

    def _get_request_payload(self, messages, stop=None):
        return {
            "model": "gpt-5.6-luna",
            "messages": [dict(message) for message in messages],
            "top_p": 1.0,
            "seed": 42,
            "reasoning_effort": "none",
        }


def test_request_params_report_a_dropped_param_as_null():
    params = intercepted_request_params(_FakePayloadModel(), [{"role": "user", "content": "q"}])
    assert params["temperature"] is None  # dropped by the client: the finding, not hidden
    assert params["top_p"] == 1.0
    assert params["seed"] == 42
    assert params["reasoning_effort"] == "none"
    assert params["source"] == "intercepted_request_body"


def test_request_params_do_not_fall_back_to_config(tmp_path):
    """config says temperature=0.0; the body says nothing. The log must say nothing."""
    config = Config(runs_dir=tmp_path)
    assert config.temperature == 0.0
    params = intercepted_request_params(_FakePayloadModel(), [{"role": "user", "content": "q"}])
    record = _record(config, request_params=params)
    assert record["request_params"]["temperature"] is None
    assert record["temperature"] == 0.0  # config's intent kept alongside, never merged in


def test_prompt_sha256_normalises_crlf_to_lf():
    """The same prompt on a CRLF and an LF checkout must hash identically."""
    model = _FakePayloadModel()
    crlf = [{"role": "system", "content": "line one\r\nline two"}]
    lf = [{"role": "system", "content": "line one\nline two"}]
    assert intercepted_prompt_sha256(model, crlf) == intercepted_prompt_sha256(model, lf)


def test_prompt_sha256_reads_the_intercepted_body():
    """Hashing the body, not the input list: a client that rewrites messages changes the hash."""

    class Rewriting(_FakePayloadModel):
        def _get_request_payload(self, messages, stop=None):
            payload = super()._get_request_payload(messages)
            payload["messages"] = [{"role": "system", "content": "REWRITTEN"}]
            return payload

    messages = [{"role": "system", "content": "original"}]
    rewritten = intercepted_prompt_sha256(Rewriting(), messages)
    plain = intercepted_prompt_sha256(_FakePayloadModel(), messages)
    assert rewritten != plain


# --- section 3: format-appendix wiring assertion -----------------------------------------------


def test_published_attaches_the_appendix_to_no_node():
    assert suffixed_nodes_for_condition(Condition.PUBLISHED) == []
    for arm in (Arm.ENUM, Arm.FREE):
        assert suffixed_nodes(arm, Condition.PUBLISHED) == []
        for node in NODES:
            composed = compose_system_prompt(node, arm, Condition.PUBLISHED)
            assert FREE_FORMAT_SUFFIX not in composed
    assert assert_format_appendix_wiring(Condition.PUBLISHED)


def test_ablation_attaches_the_appendix_to_the_grade_node_only():
    assert suffixed_nodes_for_condition(Condition.ABLATION) == ["grade"]
    assert suffixed_nodes(Arm.FREE, Condition.ABLATION) == ["grade"]
    assert suffixed_nodes(Arm.ENUM, Condition.ABLATION) == []
    assert assert_format_appendix_wiring(Condition.ABLATION)


def test_wiring_assertion_aborts_when_the_appendix_leaks(monkeypatch):
    """A suffix reaching synthesize must abort the run, not warn."""
    import src.prompts as prompts

    leaked = frozenset(
        {
            (Condition.ABLATION, Arm.FREE.value, "grade"),
            (Condition.ABLATION, Arm.FREE.value, "synthesize"),
        }
    )
    monkeypatch.setattr(prompts, "_SUFFIXED", leaked)
    with pytest.raises(PreflightError, match="wiring violated"):
        assert_format_appendix_wiring(Condition.ABLATION)


def test_wiring_assertion_aborts_when_published_gains_a_suffix(monkeypatch):
    import src.prompts as prompts

    monkeypatch.setattr(
        prompts,
        "_SUFFIXED",
        frozenset({(Condition.PUBLISHED, Arm.FREE.value, "grade")}),
    )
    with pytest.raises(PreflightError, match="wiring violated"):
        assert_format_appendix_wiring(Condition.PUBLISHED)


# --- section 5: pinned question set ------------------------------------------------------------


def test_pinned_ids_load_with_the_expected_count_and_hash():
    ids = load_pinned_question_ids()
    assert len(ids) == PINNED_QUESTION_COUNT == 150
    assert ids == sorted(ids)
    assert len(set(ids)) == 150
    assert question_set_sha256(ids) == PINNED_QUESTION_IDS_SHA256


def test_pinned_ids_match_the_published_manifests():
    """The pin is not an independent claim: it must equal what the shipped runs actually used."""
    observed = set()
    for manifest in sorted(RUNS_DIR.glob("*/*.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                observed.add(json.loads(line)["question_id"])
    assert sorted(observed) == load_pinned_question_ids()


def test_pinned_ids_reject_a_tampered_file(tmp_path):
    ids = load_pinned_question_ids()
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps({"question_ids": ["x" * 24, *ids[1:]]}), encoding="utf-8")
    with pytest.raises(PreflightError, match="hash mismatch"):
        load_pinned_question_ids(tampered)


def test_pinned_ids_reject_a_short_file(tmp_path):
    ids = load_pinned_question_ids()
    short = tmp_path / "short.json"
    short.write_text(json.dumps({"question_ids": ids[:149]}), encoding="utf-8")
    with pytest.raises(PreflightError, match="expected 150"):
        load_pinned_question_ids(short)


def test_missing_pinned_ids_are_listed_not_skipped():
    ids = load_pinned_question_ids()
    available = set(ids[:148])
    with pytest.raises(PreflightError) as excinfo:
        assert_pinned_ids_resolve(ids, available)
    message = str(excinfo.value)
    assert "2 of the 150" in message
    for missing in ids[148:]:
        assert missing in message


def test_smoke_file_holds_the_first_fifteen_sorted_ids(tmp_path):
    ids = load_pinned_question_ids()
    path = write_smoke_question_ids(ids, tmp_path / "smoke.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["question_ids"] == ids[:15]
    assert payload["n"] == 15


def test_shipped_smoke_file_is_current():
    payload = json.loads(Path("pins/smoke_question_ids.json").read_text(encoding="utf-8"))
    assert payload["question_ids"] == load_pinned_question_ids()[:15]


def test_pipeline_entry_point_does_not_use_the_sampler():
    """The driver must reach for the pinned list; the sampler must not appear at all."""
    source = Path("src/run_experiment.py").read_text(encoding="utf-8")
    assert "load_pinned_question_ids" in source
    assert "sample_question_ids" not in source
    assert "load_sampled_questions" not in source


# --- section 6: run manifest -------------------------------------------------------------------


class _FakeCollection:
    name = "hotpotqa_distractor"

    def get(self, **kwargs):
        return {
            "ids": ["q1:aaa", "q1:bbb"],
            "documents": ["first document", "second document"],
            "metadatas": [{"doc_id": "aaa"}, {"doc_id": "bbb"}],
        }


def test_corpus_digest_is_stable_and_order_independent():
    class Reordered(_FakeCollection):
        def get(self, **kwargs):
            base = super().get()
            return {key: list(reversed(value)) for key, value in base.items()}

    first, count = corpus_digest(_FakeCollection())
    second, recount = corpus_digest(Reordered())
    assert first == second  # sorted pairs: storage order cannot move the digest
    assert count == recount == 2


def test_corpus_digest_changes_when_a_document_changes():
    class Edited(_FakeCollection):
        def get(self, **kwargs):
            base = super().get()
            base["documents"] = ["first document", "second document EDITED"]
            return base

    assert corpus_digest(_FakeCollection())[0] != corpus_digest(Edited())[0]


def test_corpus_digest_normalises_line_endings():
    class Crlf(_FakeCollection):
        def get(self, **kwargs):
            return {
                "ids": ["q1:aaa"],
                "documents": ["line one\r\nline two"],
                "metadatas": [{"doc_id": "aaa"}],
            }

    class Lf(_FakeCollection):
        def get(self, **kwargs):
            return {
                "ids": ["q1:aaa"],
                "documents": ["line one\nline two"],
                "metadatas": [{"doc_id": "aaa"}],
            }

    assert corpus_digest(Crlf())[0] == corpus_digest(Lf())[0]


def _manifest(config, **overrides):
    fields = dict(
        run_id="free_run1",
        condition="published",
        arms=["free", "enum"],
        models=["gpt-4o-mini-2024-07-18"],
        question_set_sha256=PINNED_QUESTION_IDS_SHA256,
        n_questions=150,
        collection=_FakeCollection(),
    )
    fields.update(overrides)
    return build_run_manifest(config, **fields)


def test_run_manifest_carries_every_required_field(tmp_path):
    manifest = _manifest(Config(runs_dir=tmp_path))
    assert set(manifest["git"]) == {"commit", "dirty", "diff_file"}
    assert manifest["condition"] == "published"
    assert manifest["arms"] == ["free", "enum"]
    assert manifest["models"] == ["gpt-4o-mini-2024-07-18"]
    assert manifest["question_set"]["sha256"] == PINNED_QUESTION_IDS_SHA256
    assert manifest["question_set"]["n_questions"] == 150

    corpus = manifest["corpus"]
    assert corpus["chroma_collection"] == "hotpotqa_distractor"
    assert corpus["document_count"] == 2
    assert len(corpus["corpus_digest"]) == 64
    assert corpus["embedding_model"] == Config().embedding_model
    assert corpus["top_k"] == Config().top_k

    for library in ("langchain", "langchain_openai", "chromadb", "pydantic", "openai"):
        assert library in manifest["lib_versions"], library


def test_dirty_tree_dumps_the_diff_beside_the_manifest(tmp_path):
    manifest = _manifest(Config(runs_dir=tmp_path))
    run_dir = tmp_path / "free_run1"
    path = write_run_manifest(manifest, run_dir)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["git"]["dirty"] is manifest["git"]["dirty"]
    if manifest["git"]["dirty"]:
        assert written["git"]["diff_file"] == "git_diff_HEAD.patch"
        assert (run_dir / "git_diff_HEAD.patch").exists()
    else:
        assert written["git"]["diff_file"] is None


def test_run_manifest_is_json_serialisable_and_lf_on_disk(tmp_path):
    path = write_run_manifest(_manifest(Config(runs_dir=tmp_path)), tmp_path / "r")
    raw = path.read_bytes()
    assert json.loads(raw.decode("utf-8"))
    assert b"\r\n" not in raw  # LF on disk, matching .gitattributes
