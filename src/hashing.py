"""Line-ending normalisation for every hash comparison in the condition module.

`core.autocrlf=true` in this repository: text on disk is CRLF while the git blob is LF. The
published-series recovery hit this directly — `results.json` hashes to `14a0f42e…` off disk and
`bfd84e73…` as a blob, purely from line endings. A hash comparison that skips normalisation will
therefore report a spurious mismatch on Windows and a match on Linux, which is the worst possible
failure mode for a reproducibility check.

So: ONE helper, `to_lf`, applied before every digest taken in this module. The digest itself is
always computed by the harness's own functions (`provenance.prompt_sha256`,
`schemas.schema_sha256`) — nothing here reimplements SHA-256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.provenance import prompt_sha256

# 1 MiB: large enough that hashing a 45 MB dataset shard is one pass, small enough to stay cheap.
_FILE_CHUNK = 1 << 20


def to_lf(text: str) -> str:
    """Normalise all line endings to LF.

    Handles CRLF and lone CR (classic-Mac) alike; CRLF is converted first so a CRLF pair never
    degrades into a double newline. Idempotent: `to_lf(to_lf(x)) == to_lf(x)`.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_text(text: str) -> str:
    """SHA-256 of a single text, LF-normalised, via the harness's own `prompt_sha256`.

    `prompt_sha256` is a MESSAGES-level function — the harness never hashes a bare string — so the
    text is wrapped as one system message and handed to it unmodified. The wrapping convention is
    ours and is applied identically on both sides of every comparison; the digest algorithm is the
    harness's. This is the same convention `docs/published_series_reference.json` was built with,
    which is what makes the reference hashes directly comparable.
    """
    return prompt_sha256([{"role": "system", "content": to_lf(text)}])


def sha256_messages(messages: list[dict[str, Any]]) -> str:
    """SHA-256 of a full messages list with every string content LF-normalised first."""
    normalised = [
        {k: (to_lf(v) if isinstance(v, str) else v) for k, v in message.items()}
        for message in messages
    ]
    return prompt_sha256(normalised)


def sha256_file(path: Path | str) -> str:
    """SHA-256 of a file's RAW bytes, streamed.

    Deliberately NOT LF-normalised: this is used to pin binary artifacts (Arrow shards), where
    the bytes are the identity and there are no line endings to normalise. Text comparisons go
    through `sha256_text` / `sha256_messages` instead.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_FILE_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()
