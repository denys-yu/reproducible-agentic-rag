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
import re
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
    model_profile: str
    system_fingerprint: str | None
    seed: int
    temperature: float
    top_p: float
    reasoning_effort: str | None
    prompt_sha256: str
    schema_sha256: str | None
    request_payload_sha256: str | None
    cache_hit: bool
    retrieved_ids: list[str]
    retrieved_scores: list[float]
    raw_response: str
    parsed: dict[str, Any] | None
    tokens_in: int | None
    tokens_out: int | None
    cached_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    run_started_at: str | None
    latency_ms: float
    # --- Module 2 additions (additive only; every field above is unchanged) ---
    condition: str
    run_index: int | None
    output_shape: str
    parse_status: str
    request_params: dict[str, Any]
    format_suffix_nodes: list[str]


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


# --- output shape (pure observer) --------------------------------------------------------------
# NOTHING here feeds parsing, routing, or any value written into graph state. `output_shape` exists
# only so we can SEE what a model put in the place the format contract used to occupy: gpt-4o-mini
# guessed labelled lines, gpt-5.6-luna returns JSON. Classifying that is a measurement, and a
# measurement that changed the thing it measures would be worthless.

#: A fenced code block, optionally tagged (```json). Group 1 is the block body.
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)

@functools.lru_cache(maxsize=1)
def _labelled_line_re() -> re.Pattern[str]:
    """A line that opens with a short field label followed by one of the PARSER's delimiters.

    The delimiter class is taken from `llm._DELIMITERS` — never restated here. An earlier version
    of this observer hard-coded `:` while the parser also accepted em- and en-dashes, and the two
    silently disagreed on 750 records: gpt-4o-mini answered `answer — ...` at the synthesize node,
    which the parser read as labelled fields and the observer reported as `prose`. A measurement
    that contradicts the thing it measures is worse than no measurement, so the constant now has
    exactly one home and this pattern is built from it.

    Imported lazily and memoised because `llm` imports this module; a module-level import would
    close the cycle.

    Whitespace after the delimiter is REQUIRED. `-` is one of the parser's delimiters, so without
    it any hyphenated first word ("Multi-word answer") would read as a label named `Multi`.
    """
    from src.llm import _DELIMITERS

    return re.compile(
        r"^[ \t]*(?:[-*+>][ \t]*)*(?:\*\*|__|`)?[ \t]*"
        r"[A-Za-z_][A-Za-z0-9 _]{0,40}"
        rf"(?:\*\*|__|`)?[ \t]*{_DELIMITERS}[ \t]",
        re.MULTILINE,
    )

#: Ordered classifier table. FIRST MATCH WINS, so the order here is the specification.
OUTPUT_SHAPES: tuple[str, ...] = ("empty", "json_fenced", "json_object", "labeled_lines", "prose")


def _is_json_container(text: str) -> bool:
    """True if `text` parses as a JSON object or array (scalars do not count as a container)."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        return isinstance(json.loads(stripped), (dict, list))
    except (ValueError, TypeError):
        return False


def classify_output_shape(raw_text: str) -> str:
    """Classify raw model text into exactly one of `OUTPUT_SHAPES`. Pure and deterministic.

    First match wins, in declaration order:

    - `empty`         — nothing but whitespace.
    - `json_fenced`   — a fenced code block whose body is a JSON object/array.
    - `json_object`   — the whole text is a JSON object/array.
    - `labeled_lines` — at least one line opening with a label followed by one of the parser's
      delimiters (`:`, em-dash, en-dash, `-`) and whitespace.
    - `prose`         — anything else.

    Same input always yields the same label; no config, clock, or model is consulted.
    """
    text = raw_text or ""
    if not text.strip():
        return "empty"
    for body in _FENCE_RE.findall(text):
        if _is_json_container(body):
            return "json_fenced"
    if _is_json_container(text):
        return "json_object"
    if _labelled_line_re().search(text):
        return "labeled_lines"
    return "prose"


# --- parse status -----------------------------------------------------------------------------

#: The fields each node's parse is judged on. A node is `unparsed` when NONE of its fields came
#: back, `partial` when some did, `ok` when all did. This reads the parser's OUTPUT; it never
#: reimplements or influences the parser's rules.
_PARSE_FIELDS: dict[str, tuple[str, ...]] = {
    "grade": ("scope", "confidence", "needs_more_context"),
    "rewrite": ("query",),
    "synthesize": ("answer", "confidence", "scope"),
}


def classify_parse_status(node: str, parsed: Mapping[str, Any] | None) -> str:
    """Report how completely a node's fields were recovered: `ok` | `partial` | `unparsed`.

    Observational, like `output_shape`: it summarises what the parser already returned and is
    never consulted when routing or writing state.
    """
    if parsed is None:
        return "unparsed"
    fields = _PARSE_FIELDS.get(node)
    if fields is None:
        raise ValueError(f"Unknown agent node {node!r}")
    present = sum(1 for name in fields if parsed.get(name) not in (None, "", []))
    if present == 0:
        return "unparsed"
    return "ok" if present == len(fields) else "partial"


# --- run index --------------------------------------------------------------------------------

_RUN_INDEX_RE = re.compile(r"_run(\d+)$")


def run_index_from_id(run_id: str) -> int | None:
    """Extract the 1-based run index from a `"{arm}_run{i}"` run_id; None when it carries none.

    Smoke and ad-hoc run ids ("smoke-free-<qid>") legitimately have no index, and None records
    that faithfully rather than inventing a 0.
    """
    match = _RUN_INDEX_RE.search(run_id)
    return int(match.group(1)) if match else None


# --- record assembly + validation -------------------------------------------------------------


#: The two prompt-wiring conditions a record may declare. Kept as bare strings so `provenance`
#: stays importable by `hashing` without dragging in the prompt layer.
_VALID_CONDITIONS: frozenset[str] = frozenset({"published", "ablation"})


def validate_record(record: Mapping[str, Any]) -> None:
    """Raise ValueError if the record is incomplete (fail loud; never write a partial record).

    Beyond field presence, two fields are checked for CONTENT, because a null in either would
    make the record unusable for the analysis it exists to support:

    - `condition` must name a real condition. A record that cannot say which prompt wiring
      produced it cannot be assigned to an arm of the experiment at all.
    - `system_fingerprint` must be PRESENT, though it may legitimately be null (gpt-5.6-luna
      publishes none). Presence-with-null distinguishes "the provider offered nothing" from
      "we forgot to read it"; omitting the key erases that distinction.
    """
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Provenance record missing required fields: {sorted(missing)}")
    condition = record.get("condition")
    if condition not in _VALID_CONDITIONS:
        raise ValueError(
            f"Provenance record has condition={condition!r}; expected one of "
            f"{sorted(_VALID_CONDITIONS)}. Every record must state the prompt wiring it ran under."
        )


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
    cached_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    request_payload_sha256: str | None = None,
    condition: Any = None,
    request_params: Mapping[str, Any] | None = None,
    format_suffix_nodes: list[str] | None = None,
) -> ProvenanceRecord:
    """Assemble a complete, validated provenance record from config + per-call fields.

    Static reproducibility context (env, model, seed, decoding params) is filled from config;
    the caller supplies the per-call observations. The result is validated before return.

    `run_started_at` is left None here and stamped by the `ProvenanceLogger` that writes the
    record — the run window belongs to the run, not to any single call.

    `request_params` MUST be the params read back off the intercepted request body, not the values
    config asked for: langchain_openai silently drops `temperature` for gpt-5.x, so the two can
    disagree, and only the intercepted one describes the call that actually happened. The
    config-sourced `temperature`/`top_p`/`seed`/`reasoning_effort` fields are kept alongside it, so
    a divergence between intent and outbound request stays visible in the log rather than being
    resolved silently in either direction. When the caller supplies none (offline fakes that render
    no payload), the intent values are recorded and flagged with `"source": "config"`.
    """
    env = capture_environment()
    profile = config.model_profile
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
        "model_profile": profile.name,
        "system_fingerprint": system_fingerprint,
        "seed": config.llm_seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "reasoning_effort": profile.extra_params.get("reasoning_effort"),
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "request_payload_sha256": request_payload_sha256,
        "cache_hit": cache_hit,
        "retrieved_ids": list(retrieved_ids),
        "retrieved_scores": [float(score) for score in retrieved_scores],
        "raw_response": raw_response,
        "parsed": parsed,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "reasoning_tokens": reasoning_tokens,
        "run_started_at": None,  # stamped by the logger that writes this record
        "latency_ms": latency_ms,
        # --- Module 2 additions ---
        "condition": condition.value if hasattr(condition, "value") else condition,
        "run_index": run_index_from_id(run_id),
        "output_shape": classify_output_shape(raw_response),
        "parse_status": classify_parse_status(node, parsed),
        "request_params": (
            dict(request_params)
            if request_params is not None
            else {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "seed": config.llm_seed,
                "reasoning_effort": profile.extra_params.get("reasoning_effort"),
                "source": "config",  # no request body was available to intercept
            }
        ),
        "format_suffix_nodes": list(format_suffix_nodes or []),
    }
    validate_record(record)
    return record


# --- run-level manifest -------------------------------------------------------------------------
# Distinct from the per-call JSONL: this describes the RUN as a whole — the code, the corpus, the
# question set and the library stack it executed against. The JSONL says what each call did; this
# says what world it did it in. Both are needed to reproduce a result.

#: Libraries whose versions are recorded in the run manifest. Wider than `_TRACKED_LIBS` (which is
#: kept as-is so per-call records stay comparable with the published series): a run manifest is
#: written once, so it can afford to name everything that can move a result.
_MANIFEST_LIBS = (
    "langchain",
    "langchain_openai",
    "langgraph",
    "chromadb",
    "pydantic",
    "openai",
)

#: Filename of the working-tree diff dumped beside a manifest when the tree is dirty.
DIRTY_DIFF_FILENAME = "git_diff_HEAD.patch"


def _git_output(args: list[str]) -> str | None:
    """Run a git command in the repo root, returning stdout or None if git cannot answer."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=False
        )
    except (OSError, FileNotFoundError):
        return None
    return result.stdout if result.returncode == 0 else None


def _manifest_lib_versions() -> dict[str, str | None]:
    """Installed version of each manifest-tracked library; None when it is not installed."""
    versions: dict[str, str | None] = {}
    for name in _MANIFEST_LIBS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def corpus_digest(collection: Any) -> tuple[str, int]:
    """Digest the corpus AS STORED: sha256 over sorted `(doc_id, sha256(LF text))` pairs.

    Read out of the existing collection — nothing is re-embedded and no index is rebuilt. The
    point is to fingerprint the documents the run will actually retrieve over, which is a property
    of what is on disk right now, not of what a rebuild would produce.

    Text is LF-normalised before hashing so the digest is identical on a CRLF checkout, matching
    every other text hash in this repository. Returns `(digest, document_count)`.
    """
    from src.hashing import to_lf

    stored = collection.get(include=["documents", "metadatas"])
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    record_ids = stored.get("ids") or []

    pairs: list[tuple[str, str]] = []
    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        # Prefer the stored doc_id; fall back to the composite record id's doc_id half.
        doc_id = (metadata or {}).get("doc_id")
        if not doc_id and index < len(record_ids):
            doc_id = str(record_ids[index]).partition(":")[2] or str(record_ids[index])
        text_digest = hashlib.sha256(to_lf(text or "").encode("utf-8")).hexdigest()
        pairs.append((str(doc_id), text_digest))

    blob = "\n".join(f"{doc_id}\t{text_digest}" for doc_id, text_digest in sorted(pairs))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), len(pairs)


def build_run_manifest(
    config: Config,
    *,
    run_id: str,
    condition: str,
    arms: list[str],
    models: list[str],
    question_set_sha256: str,
    n_questions: int,
    collection: Any | None = None,
) -> dict[str, Any]:
    """Assemble the run-level manifest. Pure description — it never mutates anything it reads."""
    head = _git_output(["rev-parse", "HEAD"])
    status = _git_output(["status", "--porcelain"])
    dirty = bool((status or "").strip())

    collection_name: str | None = None
    document_count: int | None = None
    digest: str | None = None
    if collection is not None:
        collection_name = getattr(collection, "name", None)
        digest, document_count = corpus_digest(collection)

    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": (head or "unknown").strip(),
            "dirty": dirty,
            # Named unconditionally so a reader never has to infer whether a diff exists.
            "diff_file": DIRTY_DIFF_FILENAME if dirty else None,
        },
        "condition": condition,
        "arms": list(arms),
        "models": list(models),
        "question_set": {
            "sha256": question_set_sha256,
            "n_questions": n_questions,
        },
        "corpus": {
            "chroma_collection": collection_name,
            "document_count": document_count,
            # sha256 over sorted (doc_id, sha256(LF text)) pairs, read from the STORED documents.
            "corpus_digest": digest,
            "embedding_model": config.embedding_model,
            "embedding_dimensions": config.embedding_dimensions,
            "top_k": config.top_k,
        },
        "python_version": platform.python_version(),
        "lib_versions": _manifest_lib_versions(),
    }


def write_run_manifest(manifest: dict[str, Any], run_dir: Path) -> Path:
    """Write the run manifest, dumping `git diff HEAD` alongside it when the tree is dirty.

    A dirty tree means the commit hash alone does not identify the code that ran, so the diff is
    captured as part of the run's evidence rather than left to be reconstructed later — by which
    time the working tree will have moved on.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if manifest.get("git", {}).get("dirty"):
        diff = _git_output(["diff", "HEAD"])
        if diff is None:
            raise RuntimeError(
                "Working tree is dirty but `git diff HEAD` could not be captured; refusing to "
                "write a manifest that claims a diff file it does not have."
            )
        (run_dir / DIRTY_DIFF_FILENAME).write_text(diff, encoding="utf-8", newline="\n")

    path = run_dir / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


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
        # Run window. For models that publish no `system_fingerprint`, the window is the only
        # handle on which server-side build served the run, so it is recorded unconditionally.
        self._run_started_at = datetime.now(timezone.utc).isoformat()
        self._run_ended_at: str | None = None

    @property
    def run_started_at(self) -> str:
        return self._run_started_at

    @property
    def run_ended_at(self) -> str | None:
        """UTC timestamp of `close()`; None while the run is still open."""
        return self._run_ended_at

    @classmethod
    def for_run(cls, run_id: str, config: Config) -> ProvenanceLogger:
        """Open the logger at the conventional manifest path for a run."""
        return cls(run_manifest_path(run_id, config))

    @property
    def path(self) -> Path:
        return self._path

    def log(self, record: Mapping[str, Any]) -> None:
        """Validate then append one record as a single JSON line, flushing immediately.

        Stamps `run_started_at` when the record does not already carry one, so every call in the
        manifest points back to the run window it belongs to.
        """
        validate_record(record)
        stamped = dict(record)
        if stamped.get("run_started_at") is None:
            stamped["run_started_at"] = self._run_started_at
        self._handle.write(json.dumps(stamped, sort_keys=True, ensure_ascii=False))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._run_ended_at = datetime.now(timezone.utc).isoformat()
            self._handle.close()

    def __enter__(self) -> ProvenanceLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
