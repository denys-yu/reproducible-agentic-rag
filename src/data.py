"""HotpotQA loader, SHA-256 doc_ids, and seeded question sampling.

Loads the HotpotQA *distractor* dev set, normalizes each question's bundled paragraphs into
chunks with content-addressed `doc_id`s, and draws the frozen N-question subset deterministically
(numpy seed from config). Pure data in / plain data out — no network or model calls beyond the
dataset download.

Determinism notes:
- Text is unicode-normalized (NFC) with collapsed whitespace once, then both embedded and hashed,
  so a chunk's `doc_id` always matches the exact string the model sees.
- Chunking uses a pinned `RecursiveCharacterTextSplitter` over the `cl100k_base` tiktoken encoder
  (the `text-embedding-3-small` encoding), sized in tokens from `config.chunk_size/overlap`.
- Sampling uses an isolated `numpy.random.default_rng(config.numpy_seed)` over the sorted unique
  ids — no global RNG state is mutated, so the same N questions are drawn every run.
"""

from __future__ import annotations

import functools
import hashlib
import re
import unicodedata
from collections.abc import Sequence
from typing import TypedDict

import numpy as np

from src.config import Config


class Paragraph(TypedDict):
    """One context paragraph: a title and its ordered sentences (as shipped by HotpotQA)."""

    title: str
    sentences: list[str]


class Question(TypedDict):
    """A normalized HotpotQA distractor question with its 10 bundled paragraphs."""

    question_id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_titles: list[str]
    paragraphs: list[Paragraph]


class Chunk(TypedDict):
    """A retrievable, content-addressed chunk scoped to its source question."""

    doc_id: str
    question_id: str
    title: str
    text: str
    is_gold: bool
    para_index: int
    chunk_index: int


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return text in a canonical form: NFC unicode, whitespace runs collapsed, stripped.

    Applied once before both embedding and hashing so the embedded string and its `doc_id`
    are derived from the exact same bytes.
    """
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def compute_doc_id(normalized_text: str) -> str:
    """Return the SHA-256 hex digest of a normalized chunk — the content-addressed `doc_id`.

    The caller is responsible for passing already-normalized text (see `normalize_text`).
    """
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=None)
def _get_splitter(chunk_size: int, chunk_overlap: int):  # noqa: ANN202 (lazy heavy import)
    """Build (and memoize) the pinned token-based recursive splitter for the given sizes."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def load_dataset_split(config: Config) -> list[Question]:
    """Load the raw HotpotQA distractor dev split as a list of normalized question records."""
    from datasets import load_dataset

    raw = load_dataset(config.dataset_name, config.dataset_config, split=config.dataset_split)

    questions: list[Question] = []
    for row in raw:
        context = row["context"]
        paragraphs: list[Paragraph] = [
            Paragraph(title=title, sentences=list(sentences))
            for title, sentences in zip(context["title"], context["sentences"], strict=True)
        ]
        # supporting_facts lists one (title, sent_id) per supporting sentence; dedupe to titles.
        supporting_titles = list(dict.fromkeys(row["supporting_facts"]["title"]))
        questions.append(
            Question(
                question_id=str(row["id"]),
                question=row["question"],
                answer=row["answer"],
                type=row.get("type", ""),
                level=row.get("level", ""),
                supporting_titles=supporting_titles,
                paragraphs=paragraphs,
            )
        )
    return questions


def to_chunks(question: Question, config: Config) -> list[Chunk]:
    """Split a question's bundled paragraphs into deterministic, content-addressed chunks."""
    splitter = _get_splitter(config.chunk_size, config.chunk_overlap)
    gold_titles = set(question["supporting_titles"])

    chunks: list[Chunk] = []
    for para_index, paragraph in enumerate(question["paragraphs"]):
        body = normalize_text("".join(paragraph["sentences"]))
        if not body:
            continue
        for chunk_index, piece in enumerate(splitter.split_text(body)):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    doc_id=compute_doc_id(text),
                    question_id=question["question_id"],
                    title=paragraph["title"],
                    text=text,
                    is_gold=paragraph["title"] in gold_titles,
                    para_index=para_index,
                    chunk_index=chunk_index,
                )
            )
    return chunks


def sample_question_ids(question_ids: Sequence[str], config: Config) -> list[str]:
    """Draw the frozen N-question subset deterministically (seed from `config.numpy_seed`).

    Selects without replacement from the sorted unique ids using an isolated RNG, so the same
    N questions are chosen on every run regardless of input ordering. The result is returned
    sorted for stable downstream iteration.
    """
    unique_ids = sorted(set(question_ids))
    if config.n_questions > len(unique_ids):
        raise ValueError(
            f"Requested n_questions={config.n_questions} but only {len(unique_ids)} are available."
        )
    rng = np.random.default_rng(config.numpy_seed)
    chosen = rng.choice(np.array(unique_ids, dtype=object), size=config.n_questions, replace=False)
    return sorted(chosen.tolist())


def load_sampled_questions(config: Config) -> list[Question]:
    """Load the dev split and return the frozen N-question subset (sorted by question_id)."""
    questions = load_dataset_split(config)
    chosen = set(sample_question_ids([q["question_id"] for q in questions], config))
    subset = [q for q in questions if q["question_id"] in chosen]
    subset.sort(key=lambda q: q["question_id"])
    return subset
