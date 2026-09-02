"""The missing-value sentinel must not live in the same value space as the data (section 6.1).

`_token()` once rendered an absent field as the string `"None"`. `grade.scope` legitimately takes
the value `none` — one capital letter away — so the sentinel was distinguishable from real data
only by casing, and any node that emitted the literal text `None` would have been silently read as
"no answer". These tests assert the separation is structural, and that it holds against the actual
corpus in `runs/` and `runs2/` rather than against a hand-written example.

Read-only: parses the committed manifests, writes nothing, makes no API calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.metrics import MISSING, _token

#: Every parsed field the sentinel could ever stand in for, per node.
_FIELDS: dict[str, tuple[str, ...]] = {
    "grade": ("scope", "confidence", "needs_more_context"),
    "rewrite": ("query",),
    "synthesize": ("answer", "confidence", "scope", "supporting_doc_ids"),
}

_ROOTS = (Path("runs"), Path("runs2"))


def _manifests() -> list[Path]:
    """Every run manifest under both series (the `*` matches the published double-underscore)."""
    return sorted(p for root in _ROOTS for p in root.glob("**/run_manifest*.jsonl"))


def _corpus_values() -> list[tuple[str, str, object]]:
    """(node, field, value) for every parsed field of every call in both series."""
    out: list[tuple[str, str, object]] = []
    for path in _manifests():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                node = record["node"]
                parsed = record.get("parsed") or {}
                for field in _FIELDS.get(node, ()):
                    out.append((node, field, parsed.get(field)))
    return out


def test_sentinel_is_not_equal_to_any_string_bool_or_number() -> None:
    """No value a model can emit is equal to the sentinel — identity, not spelling."""
    for candidate in ("None", "none", "NONE", "null", "", "<MISSING>", "MISSING", 0, 1, False, True, []):
        assert MISSING != candidate
        assert candidate != MISSING
    assert MISSING == MISSING
    assert _token(None) is MISSING


def test_no_real_corpus_value_collides_with_the_sentinel() -> None:
    """Across runs/ and runs2/, no present field value tokenizes to the sentinel."""
    values = _corpus_values()
    assert values, "no manifests found; the corpus this test guards is missing"
    collisions = [
        (node, field, value)
        for node, field, value in values
        if value is not None and _token(value) is MISSING
    ]
    assert not collisions, f"field values collide with the missing sentinel: {collisions[:5]}"
    for node, field, value in values:
        assert (value is None) == (_token(value) is MISSING), (node, field, value)


def test_the_casing_collision_this_guards_is_real_not_hypothetical() -> None:
    """`none` really does occur as a value, so a string sentinel really was ambiguous."""
    scopes = {value for node, field, value in _corpus_values() if field == "scope" and value is not None}
    assert "none" in scopes, f"expected a literal 'none' scope value in the corpus, saw {sorted(scopes)}"
    assert _token("none") is not MISSING


@pytest.mark.parametrize("root", _ROOTS, ids=lambda p: str(p))
def test_both_series_are_present(root: Path) -> None:
    assert list(root.glob("**/run_manifest*.jsonl")), f"{root} has no manifests"
