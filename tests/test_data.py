"""Offline tests for src.data pure functions.

Covers normalization, content-addressed doc_ids, deterministic chunking, and seeded sampling.
The network-bound loaders (load_dataset_split / load_sampled_questions) are intentionally not
exercised here — these tests must run without HuggingFace access.
"""

from __future__ import annotations

import hashlib
import random

import pytest

from src.config import Config
from src.data import (
    Paragraph,
    Question,
    compute_doc_id,
    normalize_text,
    sample_question_ids,
    to_chunks,
)


def make_question(question_id: str, paragraphs: list[Paragraph], gold: list[str]) -> Question:
    return Question(
        question_id=question_id,
        question="who?",
        answer="x",
        type="comparison",
        level="hard",
        supporting_titles=gold,
        paragraphs=paragraphs,
    )


# --- normalize_text ---------------------------------------------------------------------------


def test_normalize_collapses_whitespace_and_strips():
    assert normalize_text("  Alpha   is\ta\n\nthing.  ") == "Alpha is a thing."


def test_normalize_applies_nfc():
    # "é" as e + combining acute (NFD) must normalize to the single NFC codepoint.
    nfd = "é"
    assert normalize_text(nfd) == "é"


def test_normalize_is_idempotent():
    once = normalize_text("  multiple   spaces here ")
    assert normalize_text(once) == once


# --- compute_doc_id ---------------------------------------------------------------------------


def test_doc_id_is_sha256_hex_of_utf8():
    text = "Alpha is a thing."
    assert compute_doc_id(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_doc_id_is_64_hex_chars():
    digest = compute_doc_id("anything")
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_doc_id_is_deterministic_and_distinguishes_text():
    assert compute_doc_id("same") == compute_doc_id("same")
    assert compute_doc_id("a") != compute_doc_id("b")


# --- to_chunks --------------------------------------------------------------------------------


def test_to_chunks_basic_fields_and_gold_flag():
    cfg = Config()
    question = make_question(
        "q1",
        paragraphs=[
            Paragraph(title="Alpha", sentences=["Alpha is a   thing. ", "It does stuff."]),
            Paragraph(title="Beta", sentences=["Beta is noise."]),
        ],
        gold=["Alpha"],
    )
    chunks = to_chunks(question, cfg)

    assert len(chunks) == 2
    assert all(c["question_id"] == "q1" for c in chunks)
    by_title = {c["title"]: c for c in chunks}
    assert by_title["Alpha"]["is_gold"] is True
    assert by_title["Beta"]["is_gold"] is False
    # Whitespace-normalized body, and doc_id is the content address of that exact text.
    assert by_title["Alpha"]["text"] == "Alpha is a thing. It does stuff."
    assert by_title["Alpha"]["doc_id"] == compute_doc_id(by_title["Alpha"]["text"])


def test_to_chunks_skips_empty_paragraphs():
    cfg = Config()
    question = make_question(
        "q2",
        paragraphs=[
            Paragraph(title="Empty", sentences=["   ", "\n"]),
            Paragraph(title="Real", sentences=["Has content."]),
        ],
        gold=[],
    )
    chunks = to_chunks(question, cfg)
    assert [c["title"] for c in chunks] == ["Real"]


def test_to_chunks_content_addressing_ignores_title():
    # Identical body text under different titles yields the same doc_id (content-addressed).
    cfg = Config()
    body = ["Shared body sentence."]
    question = make_question(
        "q3",
        paragraphs=[
            Paragraph(title="TitleA", sentences=body),
            Paragraph(title="TitleB", sentences=body),
        ],
        gold=["TitleA"],
    )
    chunks = to_chunks(question, cfg)
    assert chunks[0]["doc_id"] == chunks[1]["doc_id"]
    # ...but the gold flag still differs, since it depends on title membership.
    assert chunks[0]["is_gold"] is True
    assert chunks[1]["is_gold"] is False


def test_to_chunks_splits_long_paragraph_with_increasing_index():
    cfg = Config(chunk_size=16, chunk_overlap=0)
    long_para = Paragraph(
        title="Long",
        sentences=[f"Token{i} alpha beta gamma delta epsilon." for i in range(40)],
    )
    question = make_question("q4", paragraphs=[long_para], gold=[])
    chunks = to_chunks(question, cfg)

    assert len(chunks) > 1
    assert all(c["para_index"] == 0 for c in chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c["text"].strip() for c in chunks)


# --- sample_question_ids ----------------------------------------------------------------------


def test_sample_is_deterministic_and_order_independent():
    cfg = Config(n_questions=10)
    ids = [f"qid_{i:03d}" for i in range(50)]
    shuffled = ids[:]
    random.Random(7).shuffle(shuffled)

    first = sample_question_ids(ids, cfg)
    assert first == sample_question_ids(ids, cfg)
    assert first == sample_question_ids(shuffled, cfg)


def test_sample_returns_sorted_unique_subset_of_correct_size():
    cfg = Config(n_questions=10)
    ids = [f"qid_{i:03d}" for i in range(50)]
    chosen = sample_question_ids(ids, cfg)

    assert len(chosen) == 10
    assert chosen == sorted(chosen)
    assert len(set(chosen)) == 10  # no replacement
    assert set(chosen) <= set(ids)


def test_sample_dedupes_input_before_drawing():
    cfg = Config(n_questions=2)
    chosen = sample_question_ids(["a", "a", "b", "b", "c"], cfg)
    assert len(chosen) == 2
    assert set(chosen) <= {"a", "b", "c"}


def test_sample_raises_when_n_exceeds_available():
    cfg = Config(n_questions=5)
    with pytest.raises(ValueError, match="only 1 are available"):
        sample_question_ids(["only-one"], cfg)
