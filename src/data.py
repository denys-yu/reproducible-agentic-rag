"""HotpotQA loader, SHA-256 doc_ids, and seeded question sampling.

Loads the HotpotQA *distractor* dev set, normalizes each question's bundled paragraphs into
chunks with content-addressed `doc_id`s, and draws the frozen N-question subset deterministically
(numpy seed from config). Pure data in / plain data out — no network or model calls beyond the
dataset download.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.config import Config


def load_dataset_split(config: Config) -> list[dict[str, Any]]:
    """Load the raw HotpotQA distractor dev split as a list of question records."""
    raise NotImplementedError


def compute_doc_id(normalized_text: str) -> str:
    """Return the SHA-256 hex digest of a normalized chunk — the content-addressed `doc_id`."""
    raise NotImplementedError


def to_chunks(question_record: dict[str, Any], config: Config) -> list[dict[str, Any]]:
    """Split a question's bundled paragraphs into deterministic chunks with `doc_id`s."""
    raise NotImplementedError


def sample_question_ids(question_ids: Sequence[str], config: Config) -> list[str]:
    """Draw the frozen N-question subset with `numpy.random.seed(config.numpy_seed)`."""
    raise NotImplementedError
