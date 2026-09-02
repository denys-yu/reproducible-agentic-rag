"""The revised-article figures must rebuild byte-for-byte and stay bilingual-consistent.

A figure that cannot be regenerated identically is not reproducible evidence: a reader who reruns
the build and gets different bytes cannot tell a layout change from a data change. matplotlib
stamps a `/CreationDate` into every PDF and a `Software` chunk into every PNG by default, so this
property has to be asserted, not assumed.

Read-only on `runs/`, `runs2/` and the aggregator reports; writes only into pytest's tmp_path.
"""

from __future__ import annotations

import re

import matplotlib.text
import pytest

import src.figures as figures
from src.figure_data import load_all
from src.labels import IDENTIFIER_ALLOWLIST, LANGS, MODEL_NAMES, REVISED_FIGURE_KEYS

#: Two languages x two formats for each figure.
_EXPECTED_FILES = len(REVISED_FIGURE_KEYS) * len(LANGS) * 2

#: ASCII runs inside a Ukrainian string that would be untranslated English if not allowlisted.
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*")


@pytest.fixture(scope="module")
def data() -> dict:
    """Every figure number, already asserted against the article brief by `load_all`."""
    return load_all()


def test_load_all_asserts_every_briefed_number(data: dict) -> None:
    """`load_all` raises FigureDataMismatch on any disagreement, so reaching here is the check."""
    assert set(data) == {
        "route", "parse", "family", "format_guess", "bijection", "length", "decoding"
    }


def test_rebuild_is_byte_identical(tmp_path, data: dict) -> None:
    """Running the build twice must produce identical bytes in every file."""
    first, second = tmp_path / "first", tmp_path / "second"
    written = figures.build_revised(first, data)
    figures.build_revised(second, data)

    assert len(written) == _EXPECTED_FILES
    differing = [
        path.name
        for path in sorted(first.iterdir())
        if path.read_bytes() != (second / path.name).read_bytes()
    ]
    assert not differing, f"non-deterministic output: {differing}"


def test_both_languages_and_formats_are_written(tmp_path, data: dict) -> None:
    written = {path.name for path in figures.build_revised(tmp_path, data)}
    assert len(written) == _EXPECTED_FILES
    for name in written:
        stem, _, suffix = name.rpartition(".")
        assert suffix in {"png", "pdf"}
        assert stem.rsplit("_", 1)[1] in LANGS


def _texts(figure_key: str, lang: str, out_dir, data: dict) -> list[str]:
    """Every non-empty string drawn on one figure, in drawing order."""
    collected: list[str] = []
    real_save = figures._save

    def capture(fig, directory, key, language):
        collected.extend(
            text
            for artist in fig.findobj(matplotlib.text.Text)
            if (text := artist.get_text()) and text.strip()
        )
        return real_save(fig, directory, key, language)

    figures._save = capture
    try:
        figures.REVISED_BUILDERS[figure_key](data, out_dir, lang)
    finally:
        figures._save = real_save
    return collected


@pytest.mark.parametrize("figure_key", REVISED_FIGURE_KEYS)
def test_language_variants_have_the_same_number_of_text_elements(tmp_path, data, figure_key) -> None:
    """Layout is identical between variants; only the words differ."""
    uk = _texts(figure_key, "uk", tmp_path, data)
    en = _texts(figure_key, "en", tmp_path, data)
    assert len(uk) == len(en), f"{figure_key}: uk has {len(uk)} strings, en has {len(en)}"


@pytest.mark.parametrize("figure_key", REVISED_FIGURE_KEYS)
def test_ukrainian_carries_no_untranslated_english(tmp_path, data, figure_key) -> None:
    """Latin-script words in a Ukrainian figure must be code identifiers or model names."""
    allowed = {word.lower() for word in IDENTIFIER_ALLOWLIST | MODEL_NAMES}
    offenders = [
        (word, string)
        for string in _texts(figure_key, "uk", tmp_path, data)
        for word in _WORD.findall(string)
        if word.lower() not in allowed
    ]
    assert not offenders, f"{figure_key}: untranslated English {offenders}"


#: A bare numeric label, i.e. a drawn value rather than a number inside a sentence.
_BARE_VALUE = re.compile(r"^-?\d+[.,]\d+$")
#: A measured value embedded in prose: three or more decimals. Deliberately NOT `\d[.,]\d`, which
#: also matches the model name `gpt-5.6-luna` and the pre-registration clause `6.8` — neither is a
#: measurement and neither takes a locale separator.
_EMBEDDED_VALUE = re.compile(r"\d[.,]\d{3,}")


@pytest.mark.parametrize("figure_key", REVISED_FIGURE_KEYS)
def test_decimal_separator_follows_the_locale(tmp_path, data, figure_key) -> None:
    """Ukrainian prints a decimal comma, English a decimal point — on every drawn value."""
    for lang, wrong in (("uk", "."), ("en", ",")):
        strings = _texts(figure_key, lang, tmp_path, data)
        for string in strings:
            if _BARE_VALUE.match(string.strip()):
                assert wrong not in string, f"{figure_key}/{lang}: value {string!r} uses {wrong!r}"
            for match in _EMBEDDED_VALUE.findall(string):
                assert wrong not in match, f"{figure_key}/{lang}: {string!r} uses {wrong!r}"


def test_the_locale_separator_is_actually_exercised(tmp_path, data) -> None:
    """Guard against the separator rule passing vacuously.

    Not asserted per figure: `format_guess_stability` and `field_collapse` draw whole counts only
    (questions, bijection breaks), so they legitimately contain no decimal at all.
    """
    uk = [s for key in REVISED_FIGURE_KEYS for s in _texts(key, "uk", tmp_path, data)]
    en = [s for key in REVISED_FIGURE_KEYS for s in _texts(key, "en", tmp_path, data)]
    assert any(re.search(r"\d,\d", s) for s in uk), "no decimal comma anywhere in the uk figures"
    assert any(re.search(r"\d\.\d\d", s) for s in en), "no decimal point anywhere in the en figures"
