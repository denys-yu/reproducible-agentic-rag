"""Run-start checks that must pass before any paid run: prompts, dataset, payload, provenance.

Everything here is offline. No module in this file makes a network call to any model provider —
the payload capture deliberately builds the outbound request body and writes it to disk WITHOUT
sending it, which is the entire point: `langchain_openai` is known to silently drop `temperature`
for gpt-5.x, and that must be caught before a run is paid for, not inferred from the log after.

Four independent gates, each failing hard:

1. `assert_condition_contract` — the composed prompts for a condition are exactly what that
   condition claims, checked by hash against `docs/published_series_reference.json`.
2. `assert_dataset_pin` — the dataset file on disk is the one config declares, by sha256 and
   record count, and all 15 pilot question ids resolve against it.
3. `capture_payloads` — the real request body for every (profile, arm, node), asserted to carry
   the pinned decoding params, written to `docs/payload_capture/`.
4. `assert_provenance_durable` — the run's output path is not inside a git-ignored directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Arm, Config, ModelName, get_model_profile
from src.hashing import sha256_file, sha256_text, to_lf
from src.prompts import (
    NODES,
    Condition,
    FREE_FORMAT_SUFFIX,
    compose_system_prompt,
    composed_hash_table,
    suffixed_nodes,
    suffixed_nodes_for_condition,
)

REFERENCE_PATH = Path("docs/published_series_reference.json")
PAYLOAD_CAPTURE_DIR = Path("docs/payload_capture")
PINNED_QUESTION_IDS_PATH = Path("pins/published_question_ids.json")
SMOKE_QUESTION_IDS_PATH = Path("pins/smoke_question_ids.json")

#: sha256 over the 150 published question ids, sorted then joined with LF, no trailing newline.
#: This is the identity of the question set; the file is only its carrier.
PINNED_QUESTION_IDS_SHA256 = "498ec7b8b03177bf6ec2ef7804977289589b0448eed4f3cfec11784fbd7fee47"
PINNED_QUESTION_COUNT = 150
SMOKE_QUESTION_COUNT = 15

# The 15 pilot question ids, drawn from the published 150. Pinned as a named constant so a run
# can never be pointed at a silently different subset.
PILOT_15_QIDS: tuple[str, ...] = (
    "5a713f395542994082a3e6ea",
    "5a7313865542994cef4bc442",
    "5a7336bd5542991f9a20c68c",
    "5a734dad5542994cef4bc522",
    "5a738bd0554299623ed4abee",
    "5a74f3f55542993748c8974b",
    "5a753c8c55429916b01642ab",
    "5a7570a65542996c70cfaefd",
    "5a76139b5542994ccc9186be",
    "5a7636fb55429976ec32bd79",
    "5a76fda05542994aec3b71b0",
    "5a772c925542993735360208",
    "5a773ec355429966f1a36ccb",
    "5a77414155429972597f14d7",
    "5a7750a155429966f1a36cef",
)


class PreflightError(RuntimeError):
    """A run-start gate failed. Always fatal; never caught and downgraded to a warning."""


# --- 1. condition contract --------------------------------------------------------------------


def load_reference(path: Path = REFERENCE_PATH) -> dict[str, Any]:
    """Load the recovered published-series reference, or fail loud."""
    if not path.exists():
        raise PreflightError(f"Missing published-series reference: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_hashes(reference: dict[str, Any]) -> dict[str, str]:
    """`"{arm}|{node}" -> sha256` as recorded for the published commit."""
    return {key: entry["system_sha256"] for key, entry in reference["composed_prompts"].items()}


def assert_condition_contract(
    condition: Condition | str, *, reference: dict[str, Any] | None = None
) -> list[str]:
    """Assert every property the given condition promises. Returns the checks that passed.

    Raises `PreflightError` naming the first violated property. Both conditions are checked
    against the same recovered reference, so a drift in the base wording fails BOTH — the
    ablation is defined as published-plus-suffix, not as a free-floating variant.
    """
    condition = condition if isinstance(condition, Condition) else Condition(condition)
    reference = reference if reference is not None else load_reference()
    ref_hashes = _reference_hashes(reference)
    got = composed_hash_table(condition)
    passed: list[str] = []

    def require(ok: bool, description: str) -> None:
        if not ok:
            raise PreflightError(f"[{condition.value}] FAILED: {description}")
        passed.append(description)

    if condition is Condition.PUBLISHED:
        for node in NODES:
            require(
                got[f"enum|{node}"] == got[f"free|{node}"],
                f"(enum,{node}) sha256 == (free,{node}) sha256",
            )
        for arm in ("enum", "free"):
            for node in NODES:
                key = f"{arm}|{node}"
                require(
                    got[key] == ref_hashes[key],
                    f"({arm},{node}) sha256 == published_series_reference {ref_hashes[key][:12]}…",
                )
        for arm in ("enum", "free"):
            for node in NODES:
                text = compose_system_prompt(node, arm, condition)
                require(
                    to_lf(FREE_FORMAT_SUFFIX) not in to_lf(text),
                    f"FREE_FORMAT_SUFFIX absent from ({arm},{node})",
                )

    elif condition is Condition.ABLATION:
        require(
            got["free|grade"] != got["enum|grade"],
            "(free,grade) differs from (enum,grade)",
        )
        require(
            compose_system_prompt("grade", Arm.FREE, condition)
            == compose_system_prompt("grade", Arm.ENUM, condition) + FREE_FORMAT_SUFFIX,
            "(free,grade) == (enum,grade) + FREE_FORMAT_SUFFIX, byte-exact",
        )
        # The load-bearing one: the suffix must not leak past the grade node.
        require(
            compose_system_prompt("synthesize", Arm.FREE, condition)
            == compose_system_prompt("synthesize", Arm.ENUM, condition),
            "(free,synthesize) == (enum,synthesize), BYTE-IDENTICAL — suffix must not reach "
            "the synthesize node",
        )
        for arm in ("enum", "free"):
            require(
                to_lf(FREE_FORMAT_SUFFIX)
                not in to_lf(compose_system_prompt("synthesize", arm, condition)),
                f"FREE_FORMAT_SUFFIX absent from ({arm},synthesize)",
            )
        require(
            got["free|rewrite"] == got["enum|rewrite"],
            "(free,rewrite) == (enum,rewrite)",
        )
        # The unsuffixed nodes must still be the published text, or the ablation is confounded.
        for key in ("enum|grade", "enum|rewrite", "enum|synthesize", "free|rewrite", "free|synthesize"):
            require(
                got[key] == ref_hashes[key],
                f"({key.replace('|', ',')}) sha256 unchanged from published reference",
            )
    else:  # pragma: no cover - Condition has exactly two members
        raise PreflightError(f"Unhandled condition {condition!r}")

    return passed


# --- 1b. format-appendix wiring ---------------------------------------------------------------


def assert_format_appendix_wiring(condition: Condition | str) -> list[str]:
    """Assert WHICH nodes receive FREE_FORMAT_SUFFIX under a condition. Aborts on violation.

    The contract, per condition:

    - `published` — the appendix reaches NO node, in either arm.
    - `ablation`  — the appendix reaches the GRADE node and nothing else.

    `assert_condition_contract` checks prompt hashes against the recovered reference; this checks
    the wiring itself, stated as node sets. The two overlap deliberately: a bug that satisfied one
    formulation while breaking the other is exactly the kind that produces a confounded ablation.
    Any violation raises — this never degrades to a warning, because a warning on a 3400-call paid
    run is a warning nobody reads until the data is already spent.
    """
    condition = condition if isinstance(condition, Condition) else Condition(condition)
    expected: tuple[str, ...] = () if condition is Condition.PUBLISHED else ("grade",)

    actual = tuple(suffixed_nodes_for_condition(condition))
    if actual != expected:
        raise PreflightError(
            f"[{condition.value}] format-appendix wiring violated: appendix reaches nodes "
            f"{list(actual)}, expected {list(expected)}. Aborting the run."
        )

    # Node sets agreeing is necessary but not sufficient — confirm against the composed TEXT, per
    # (arm, node), so the wiring table and the prompts cannot silently disagree.
    for arm in (Arm.ENUM, Arm.FREE):
        arm_nodes = suffixed_nodes(arm, condition)
        for node in NODES:
            text = compose_system_prompt(node, arm, condition)
            carried = to_lf(FREE_FORMAT_SUFFIX) in to_lf(text)
            should_carry = node in arm_nodes
            if carried != should_carry:
                raise PreflightError(
                    f"[{condition.value}] format-appendix wiring violated at "
                    f"({arm.value},{node}): the composed prompt "
                    f"{'carries' if carried else 'lacks'} FREE_FORMAT_SUFFIX but the wiring table "
                    f"says {'it should' if should_carry else 'it should not'}. Aborting the run."
                )

    passed = [
        f"format appendix attached to nodes {list(expected) or 'NONE'} under "
        f"condition={condition.value}"
    ]
    passed.extend(
        f"({arm.value}) appendix nodes == {suffixed_nodes(arm, condition)}"
        for arm in (Arm.ENUM, Arm.FREE)
    )
    return passed


# --- 1c. pinned question set ------------------------------------------------------------------


def question_set_sha256(question_ids: list[str]) -> str:
    """SHA-256 over ids joined by LF with no trailing newline — the question set's identity."""
    return hashlib.sha256(to_lf("\n".join(question_ids)).encode("utf-8")).hexdigest()


def load_pinned_question_ids(path: Path = PINNED_QUESTION_IDS_PATH) -> list[str]:
    """Load the 150 published question ids, asserting count and content hash. Never samples.

    The published series' question set is a FACT recovered from the shipped manifests, not
    something to be re-derived. `data.sample_question_ids` reproduces it only so long as numpy's
    `default_rng` bit stream, the sorted-id input, and `n_questions` all stay fixed — three
    dependencies a library upgrade can break silently, yielding a different 150 that still looks
    perfectly deterministic. Reading a pinned list and hashing it removes all three.

    Returns the ids SORTED, which is the order the hash is defined over.
    """
    if not path.exists():
        raise PreflightError(
            f"Missing pinned question-id file: {path}\n"
            "The pipeline must read the published 150 from this file, never re-sample them."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [str(qid) for qid in payload["question_ids"]]

    if len(ids) != PINNED_QUESTION_COUNT:
        raise PreflightError(
            f"Pinned question set has n={len(ids)}, expected {PINNED_QUESTION_COUNT} ({path})."
        )
    duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
    if duplicates:
        raise PreflightError(f"Pinned question set contains duplicate ids: {duplicates} ({path}).")

    ordered = sorted(ids)
    digest = question_set_sha256(ordered)
    if digest != PINNED_QUESTION_IDS_SHA256:
        raise PreflightError(
            f"Pinned question-set hash mismatch for {path}\n"
            f"  expected : {PINNED_QUESTION_IDS_SHA256}\n"
            f"  computed : {digest}\n"
            "This is not the published 150. Refusing to run against a different question set."
        )
    return ordered


def assert_pinned_ids_resolve(question_ids: list[str], available: set[str]) -> None:
    """STOP and list every pinned id absent from the loaded split. Never runs a partial set."""
    missing = [qid for qid in question_ids if qid not in available]
    if missing:
        raise PreflightError(
            f"{len(missing)} of the {len(question_ids)} pinned question ids do not resolve in "
            f"the loaded HotpotQA distractor split:\n"
            + "\n".join(f"    {qid}" for qid in missing)
            + "\nRefusing to run on a partial question set."
        )


def write_smoke_question_ids(
    question_ids: list[str], path: Path = SMOKE_QUESTION_IDS_PATH
) -> Path:
    """Write the first 15 ids of the sorted pinned list as the smoke subset."""
    subset = question_ids[:SMOKE_QUESTION_COUNT]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": len(subset),
        "source": PINNED_QUESTION_IDS_PATH.as_posix(),
        "note": f"First {SMOKE_QUESTION_COUNT} ids of the sorted published list.",
        "question_ids": subset,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


# --- 2. dataset pin -------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetPinReport:
    """Outcome of the dataset-identity check."""

    path: str
    sha256: str
    record_count: int
    pilot_resolved: int


def assert_dataset_pin(config: Config, *, question_ids: list[str] | None = None) -> DatasetPinReport:
    """Assert the on-disk dataset file is the one config pins, and that all 15 pilot ids resolve.

    `question_ids` is injectable so tests can exercise the resolution check without loading a
    45 MB Arrow shard. In a real run it is None and the ids come from the pinned file.
    """
    path = Path(config.dataset_file).expanduser()
    if not path.exists():
        raise PreflightError(
            f"Pinned dataset file is missing: {path}\n"
            "Config declares this exact shard; a run must not silently fall back to a re-download."
        )

    actual = sha256_file(path)
    if actual != config.dataset_file_sha256:
        raise PreflightError(
            f"Dataset file hash mismatch for {path}\n"
            f"  config declares : {config.dataset_file_sha256}\n"
            f"  file on disk is : {actual}\n"
            "Refusing to run against a dataset that is not the pinned one."
        )

    if question_ids is None:
        question_ids = _load_pinned_question_ids(config)

    if len(question_ids) != config.dataset_record_count:
        raise PreflightError(
            f"Dataset record count mismatch: config declares {config.dataset_record_count}, "
            f"loaded {len(question_ids)}."
        )

    available = set(question_ids)
    missing = [qid for qid in PILOT_15_QIDS if qid not in available]
    if missing:
        raise PreflightError(
            f"{len(missing)} of the 15 pilot question ids do not resolve against the pinned "
            f"dataset: {missing}"
        )

    return DatasetPinReport(
        path=str(path),
        sha256=actual,
        record_count=len(question_ids),
        pilot_resolved=len(PILOT_15_QIDS),
    )


def _load_pinned_question_ids(config: Config) -> list[str]:
    """Read the question ids from the pinned split, forcing HuggingFace fully offline.

    The offline env vars are set before `datasets` is imported so no hub request is even
    attempted — this function must not touch the network.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    split = load_dataset(config.dataset_name, config.dataset_config, split=config.dataset_split)
    return [str(row) for row in split["id"]]


# --- 3. outgoing payload capture (builds the body; never sends it) -----------------------------


def capture_payloads(
    *,
    condition: Condition | str,
    out_dir: Path = PAYLOAD_CAPTURE_DIR,
    profiles: tuple[ModelName, ...] = (ModelName.GPT_4O_MINI, ModelName.GPT_5_6_LUNA),
) -> list[Path]:
    """Build and persist the real outbound request body for every (profile, arm, node).

    NO REQUEST IS SENT. `llm.resolve_request_payload` runs langchain_openai's own request builder
    and hands back the dict the SDK would post, which is exactly what makes this check meaningful:
    a param the client strips is absent here too.

    Asserts each captured body carries, at top level, the profile's pinned `temperature`, `top_p`
    and `seed`, plus `reasoning_effort` for gpt-5.x. Writes one JSON per combination and returns
    the paths.
    """
    from src.llm import make_llm, required_request_params, resolve_request_payload

    condition = condition if isinstance(condition, Condition) else Condition(condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for profile_name in profiles:
        config = Config(model=profile_name)
        profile = get_model_profile(profile_name)
        required = required_request_params(config)
        model = make_llm(config)

        for arm in (Arm.ENUM, Arm.FREE):
            for node in NODES:
                system = compose_system_prompt(node, arm, condition)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "<payload-capture probe: body is built, never sent>"},
                ]
                payload = dict(resolve_request_payload(model, messages))

                missing = {
                    name: expected
                    for name, expected in required.items()
                    if name not in payload or payload[name] != expected
                }
                if missing:
                    raise PreflightError(
                        f"Captured payload for ({profile.model_id}, {arm.value}, {node}) is "
                        f"missing/contradicting required params {sorted(missing)} "
                        f"(expected {missing}). This is the silent-drop failure mode; refusing "
                        f"to proceed to a paid run."
                    )
                if not profile.dated_snapshot and "reasoning_effort" not in payload:
                    raise PreflightError(
                        f"Captured payload for {profile.model_id!r} lacks `reasoning_effort`; "
                        "without it the API rejects temperature=0 and the stratum silently "
                        "leaves the pinned decoding params."
                    )

                record = {
                    "condition": condition.value,
                    "model_profile": profile.name,
                    "model_id": profile.model_id,
                    "arm": arm.value,
                    "node": node,
                    "required_params": required,
                    "captured_params": {
                        k: v for k, v in payload.items() if k != "messages"
                    },
                    "system_prompt": system,
                    "system_prompt_sha256": sha256_text(system),
                    "sent": False,
                    "note": "Body built via langchain_openai's own request builder. NEVER sent.",
                }
                path = out_dir / f"{profile.name}_{arm.value}_{node}.json"
                path.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                written.append(path)

    return written


# --- 4. provenance durability -----------------------------------------------------------------


def _existing_ancestor(path: Path) -> Path:
    """The nearest existing directory at or above `path` — where git must be asked from.

    `git check-ignore` answers relative to the work tree containing its CWD, not to the argument,
    so asking from the repo root about a path in a different repository yields "outside repository"
    rather than an answer. The output directory itself usually does not exist yet (that is the
    normal case before a run), hence the walk upwards.
    """
    path = path.resolve()
    for candidate in (path, *path.parents):
        if candidate.is_dir():
            return candidate
    return Path.cwd()


def _git_check_ignore(path: Path) -> subprocess.CompletedProcess[str]:
    """Run `git check-ignore -v` on a path. Raises PreflightError if git cannot be run at all."""
    try:
        return subprocess.run(
            ["git", "check-ignore", "-v", "--no-index", str(Path(path).resolve())],
            cwd=_existing_ancestor(Path(path)),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError) as exc:
        raise PreflightError(
            f"Cannot verify that {path} is writable-and-tracked: git is unavailable ({exc}).\n"
            "This check fails CLOSED. An unverified output path is how the published series' "
            "per-call log was nearly lost; there is no override."
        ) from exc



def assert_output_dir_not_ignored(path: Path | str) -> Path:
    """Refuse to start unless git demonstrably does NOT ignore `path`. Fails closed, no override.

    `git check-ignore` exit codes: 0 = ignored, 1 = not ignored, 128 = git cannot answer (no repo,
    corrupt index, path outside the work tree). Only exit 1 is a pass. Both 0 and 128 raise, and
    so does a missing git binary — an output path we cannot prove is durable is treated exactly
    like one we have proved is not.

    On exit 0 the `-v` output names the source of the rule (file:line:pattern), so the error says
    which .gitignore line has to change rather than leaving the reader to hunt for it.
    """
    path = Path(path)
    result = _git_check_ignore(path)

    if result.returncode == 0:
        rule = (result.stdout or "").strip().splitlines()
        detail = rule[0] if rule else "(git named no rule)"
        raise PreflightError(
            f"Refusing to start: git IGNORES the output path {path}.\n"
            f"  responsible rule : {detail}\n"
            "That rule must be removed before a run writes here. A run whose provenance cannot "
            "survive a clean checkout is not reproducible; this check has no override."
        )
    if result.returncode != 1:
        raise PreflightError(
            f"Refusing to start: git could not determine whether {path} is ignored "
            f"(exit {result.returncode}). Is this inside a git work tree?\n"
            f"  stderr: {(result.stderr or '').strip() or '(none)'}\n"
            "This check fails CLOSED — an unverifiable output path is rejected."
        )
    return path


def is_git_ignored(path: Path) -> bool:
    """True if git would ignore `path`. Falls back to False outside a work tree."""
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def assert_provenance_durable(config: Config) -> Path:
    """Print the run's provenance path and refuse to start if git would ignore it.

    The published series' per-call log was believed lost precisely because `runs/` is git-ignored
    and untracked. A run whose output cannot survive a clean checkout is not reproducible, so this
    aborts rather than warns.
    """
    path = Path(config.runs_dir)
    print(f"provenance output path: {path.resolve()}")
    # Delegated to the fail-closed check: `is_git_ignored` returns False when git cannot answer,
    # which would let an unverifiable path through. Durability must not depend on git succeeding.
    return assert_output_dir_not_ignored(path)


# --- CLI ---------------------------------------------------------------------------------------


def run_preflight(
    condition: Condition | str, *, skip_dataset: bool = False, skip_payload: bool = False
) -> int:
    """Run every gate for one condition, printing results. Returns a process exit code."""
    condition = condition if isinstance(condition, Condition) else Condition(condition)
    print(f"=== preflight: condition={condition.value} ===")

    reference = load_reference()
    print(f"reference published_commit: {reference['published_commit']}")

    passed = assert_condition_contract(condition, reference=reference)
    print(f"\n[1] condition contract — {len(passed)} assertions passed:")
    for description in passed:
        print(f"    PASS  {description}")

    print("\n[2] composed prompt hashes (LF-normalised):")
    for key, digest in composed_hash_table(condition).items():
        print(f"    {key:16s} {digest}")

    if not skip_dataset:
        config = Config()
        report = assert_dataset_pin(config)
        print(
            f"\n[3] dataset pin — OK\n"
            f"    path         : {report.path}\n"
            f"    sha256       : {report.sha256}\n"
            f"    records      : {report.record_count}\n"
            f"    pilot ids    : {report.pilot_resolved}/15 resolved"
        )

    if not skip_payload:
        written = capture_payloads(condition=condition)
        print(f"\n[4] payload capture — {len(written)} bodies built and written, ZERO sent")
        for path in written:
            print(f"    {path}")

    print("\npreflight OK")
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI: run the offline preflight gates for one condition."""
    parser = argparse.ArgumentParser(
        prog="python -m src.preflight",
        description="Offline run-start gates: prompts, dataset pin, payload capture, provenance.",
    )
    parser.add_argument(
        "--condition",
        required=True,  # never defaulted: the condition must always be stated
        choices=[c.value for c in Condition],
    )
    parser.add_argument("--skip-dataset", action="store_true", help="skip the 45MB shard check")
    parser.add_argument("--skip-payload", action="store_true", help="skip building request bodies")
    args = parser.parse_args(argv)
    raise SystemExit(
        run_preflight(
            args.condition, skip_dataset=args.skip_dataset, skip_payload=args.skip_payload
        )
    )


if __name__ == "__main__":
    main()
