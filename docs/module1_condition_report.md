# Module 1 — published prompts restored as a runnable condition

Two selectable prompt-wiring conditions (`published`, `ablation`), the published wording restored
verbatim from commit `d6efcab`, and four run-start gates that fail hard before any paid call.

**Zero API calls were made to any model provider.** The payload-capture path builds the real
outbound request body and writes it to disk without sending it; every other check is pure local
computation. This is stated again, with evidence, in §8.

---

## Pre-flight

| # | Check | Result |
|---|---|---|
| 1 | `docs/published_series_reference.json` parses | PASS — `published_commit = d6efcab38e9f0fab331ecafa164509a77c181f4e` |
| 2 | tree recorded to `docs/module1_tree_before.txt` | PASS — `git status --porcelain --ignored` + 41,089-line recursive listing |
| 3 | does `results/` exist? | **ABSENT** — see below |
| 4 | harness sha256 located and reused | PASS — `provenance.prompt_sha256`, `schemas.schema_sha256` |

### §3 — `results/` is absent, but `runs/` is not

`results/` does not exist anywhere in the working tree. Nothing was deleted or moved by this
module; it was already gone when Module 1 started (it was present in the session-start snapshot of
an earlier task and had disappeared before that task finished).

**More importantly: the per-call provenance log that the recovery task was told had been lost is
not lost.** `runs/` holds all ten manifests of the published series, untouched since 2026-06-18:

```
runs/{enum,free}_run{1..5}/run_manifest.jsonl   ~497-530 KB each, plus a .done marker
```

`runs/free_run1/run_manifest.jsonl` contains 344 records over exactly 150 distinct question ids
(150 grade + 150 synthesize + 44 rewrite), `arm="free"`, `model="gpt-4o-mini-2024-07-18"`,
`temperature=0.0`, `top_p=1.0`, `seed=42`. This is the published series' log. It is git-ignored,
which is why it was believed lost — precisely the failure §7 addresses.

This changed what this module could prove. See §6.

### §4 — hash functions reused, not reimplemented

`src/hashing.py` wraps the harness functions and adds LF normalisation; it contains no SHA-256
implementation of its own beyond `sha256_file`, which streams raw bytes for binary pins.

| helper | delegates to | used for |
|---|---|---|
| `sha256_text` | `provenance.prompt_sha256` | prompt constants, composed prompts |
| `sha256_messages` | `provenance.prompt_sha256` | full messages lists |
| `sha256_file` | `hashlib.sha256` over raw bytes | dataset shard pin (no line endings to normalise) |

---

## §1 Line-ending normalisation

`core.autocrlf=true`: text on disk is CRLF, git blobs are LF. The recovery task hit this directly —
`results.json` hashes to `14a0f42e…` off disk and `bfd84e73…` as a blob. A hash comparison that
skips normalisation reports a spurious mismatch on Windows and a match on Linux, which is the worst
possible failure mode for a reproducibility check.

One helper, `hashing.to_lf`, is applied before every text digest in the module:

```python
def to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
```

CRLF is converted first so a CRLF pair never degrades into a double newline; lone CR is handled
too. Six unit tests cover it, including the CRLF fixture required by the spec:

```
test_to_lf_converts_crlf_fixture                    PASS
test_to_lf_converts_lone_cr_without_doubling_crlf   PASS
test_to_lf_is_idempotent_and_leaves_lf_untouched    PASS
test_sha256_text_is_line_ending_invariant           PASS
test_sha256_text_is_still_content_sensitive         PASS
test_sha256_file_hashes_raw_bytes                   PASS
```

---

## §2 Published wording restored

`src/prompts.py` holds the three constants verbatim from `d6efcab`, em-dashes included. Verified by
hash against the reference, not by eye:

| constant | sha256 | matches reference |
|---|---|---|
| `GRADE_PROMPT` | `c73170f00af8423ad1828da5858796d3da1419b0221f159ff7160a073c9fe805` | yes |
| `REWRITE_PROMPT` | `d92149e55bce4a3eee160688bc3ceb098ebf510d7f2df0f617a4f6422cd0e097` | yes |
| `SYNTHESIZE_PROMPT` | `63a7dcfe4f4f1bb18bd5a73493777f14aff3e74d018cf4dbdc80e64b0d9a21b6` | yes |

The old phrasing is restored and guarded by a test that fails if the revised wording creeps back:
`"fully, partially, or not at all (none)"` and `"(high, medium, low)"` must be present;
`"full, partial, or none"` and `"high, medium, or low"` must be absent. A further test asserts
`"exactly one of"` appears in no prompt or suffix.

`src/agent.py` no longer owns prompt text. It re-exports the constants so existing importers of
`agent.GRADE_PROMPT` keep working, but the text lives in one place and is hashed against the
reference at run start.

---

## §3 Condition switch

`condition` is a required parameter with no default at every layer that composes a prompt:

```python
def compose_system_prompt(node: NodeName, arm: Arm | str, condition: Condition | str) -> str:
```

- `Config.condition` is `Condition | None = None` — `None` means *unstated*, never a silent
  default. See "Deviation" below.
- `run_experiment(config, *, condition, ...)` is keyword-required.
- `AgentState` carries `condition`; `initial_state` and `run_question` both require it.
- `--condition` is `required=True` on `python -m src.run_experiment`, `python -m src.agent`, and
  `python -m src.preflight`.

Which triples receive the suffix is data, not nested conditionals, so the containment property is
auditable at a glance:

```python
_SUFFIXED = frozenset({(Condition.ABLATION, Arm.FREE.value, "grade")})
```

### Deviation from the letter of the spec — flagged for review

The spec says `condition` "must be an explicit required parameter, not a default". I made it
required on every **composition** path (no default anywhere a prompt is built), but on `Config` it
is `None` rather than a required field.

Making `Config.condition` required would make bare `Config()` raise, breaking ~30 call sites across
`test_data`, `test_index`, `test_metrics`, `test_cache` and others that never compose a prompt —
churn well outside "configuration and prompt wiring only". The requirement's stated goal — *"there
must be no code path that composes a prompt without a condition being stated"* — is met exactly and
provably as built, because `compose_system_prompt` has no default to fall back on. Say the word if
you want the stricter version and I will make the field required and update the call sites.

---

## §4 Assertions — actual output, both conditions

### Condition `published` — 15 assertions

```
=== preflight: condition=published ===
reference published_commit: d6efcab38e9f0fab331ecafa164509a77c181f4e
[1] condition contract — 15 assertions passed:
    PASS  (enum,grade) sha256 == (free,grade) sha256
    PASS  (enum,rewrite) sha256 == (free,rewrite) sha256
    PASS  (enum,synthesize) sha256 == (free,synthesize) sha256
    PASS  (enum,grade) sha256 == published_series_reference c73170f00af8…
    PASS  (enum,rewrite) sha256 == published_series_reference d92149e55bce…
    PASS  (enum,synthesize) sha256 == published_series_reference 63a7dcfe4f4f…
    PASS  (free,grade) sha256 == published_series_reference c73170f00af8…
    PASS  (free,rewrite) sha256 == published_series_reference d92149e55bce…
    PASS  (free,synthesize) sha256 == published_series_reference 63a7dcfe4f4f…
    PASS  FREE_FORMAT_SUFFIX absent from (enum,grade)
    PASS  FREE_FORMAT_SUFFIX absent from (enum,rewrite)
    PASS  FREE_FORMAT_SUFFIX absent from (enum,synthesize)
    PASS  FREE_FORMAT_SUFFIX absent from (free,grade)
    PASS  FREE_FORMAT_SUFFIX absent from (free,rewrite)
    PASS  FREE_FORMAT_SUFFIX absent from (free,synthesize)
[2] composed prompt hashes (LF-normalised):
    enum|grade       c73170f00af8423ad1828da5858796d3da1419b0221f159ff7160a073c9fe805
    enum|rewrite     d92149e55bce4a3eee160688bc3ceb098ebf510d7f2df0f617a4f6422cd0e097
    enum|synthesize  63a7dcfe4f4f1bb18bd5a73493777f14aff3e74d018cf4dbdc80e64b0d9a21b6
    free|grade       c73170f00af8423ad1828da5858796d3da1419b0221f159ff7160a073c9fe805
    free|rewrite     d92149e55bce4a3eee160688bc3ceb098ebf510d7f2df0f617a4f6422cd0e097
    free|synthesize  63a7dcfe4f4f1bb18bd5a73493777f14aff3e74d018cf4dbdc80e64b0d9a21b6
```

All six composed hashes are identical across arms and equal the reference. That is the published
condition reproduced.

### Condition `ablation` — 11 assertions

```
=== preflight: condition=ablation ===
[1] condition contract — 11 assertions passed:
    PASS  (free,grade) differs from (enum,grade)
    PASS  (free,grade) == (enum,grade) + FREE_FORMAT_SUFFIX, byte-exact
    PASS  (free,synthesize) == (enum,synthesize), BYTE-IDENTICAL — suffix must not reach the synthesize node
    PASS  FREE_FORMAT_SUFFIX absent from (enum,synthesize)
    PASS  FREE_FORMAT_SUFFIX absent from (free,synthesize)
    PASS  (free,rewrite) == (enum,rewrite)
    PASS  (enum,grade) sha256 unchanged from published reference
    PASS  (enum,rewrite) sha256 unchanged from published reference
    PASS  (enum,synthesize) sha256 unchanged from published reference
    PASS  (free,rewrite) sha256 unchanged from published reference
    PASS  (free,synthesize) sha256 unchanged from published reference
[2] composed prompt hashes (LF-normalised):
    enum|grade       c73170f00af8423ad1828da5858796d3da1419b0221f159ff7160a073c9fe805
    enum|rewrite     d92149e55bce4a3eee160688bc3ceb098ebf510d7f2df0f617a4f6422cd0e097
    enum|synthesize  63a7dcfe4f4f1bb18bd5a73493777f14aff3e74d018cf4dbdc80e64b0d9a21b6
    free|grade       d1361a53a19d239ee606a14a2042fe4f0f66bfcc0aa2247ba8cb15915ca544f9
    free|rewrite     d92149e55bce4a3eee160688bc3ceb098ebf510d7f2df0f617a4f6422cd0e097
    free|synthesize  63a7dcfe4f4f1bb18bd5a73493777f14aff3e74d018cf4dbdc80e64b0d9a21b6
```

Exactly one hash moves between the conditions — `free|grade`. Every other cell is bit-identical to
published, so the ablation varies one thing.

The load-bearing assertion is also tested from the other direction: `test_contract_detects_a_leaked_suffix`
monkeypatches `_SUFFIXED` to add `(ABLATION, free, synthesize)` and asserts the gate raises. A gate
that never fails proves nothing, so it is shown failing on a deliberately broken build.

---

## §5 Payload capture — bodies built, none sent

`preflight.capture_payloads` runs langchain_openai's own request builder
(`ChatOpenAI._get_request_payload`) and writes the resulting dict to disk. No transport is
involved. 12 files: 2 profiles × 2 arms × 3 nodes, in `docs/payload_capture/`.

`gpt-5.6-luna_free_grade.json` (system prompt truncated here for width):

```json
{
  "condition": "published",
  "model_profile": "gpt-5.6-luna",
  "model_id": "gpt-5.6-luna",
  "arm": "free",
  "node": "grade",
  "required_params": {
    "temperature": 0.0, "top_p": 1.0, "seed": 42, "reasoning_effort": "none"
  },
  "captured_params": {
    "model": "gpt-5.6-luna", "stream": false,
    "seed": 42, "top_p": 1.0, "temperature": 0.0, "reasoning_effort": "none"
  },
  "system_prompt_sha256": "c73170f00af8423ad1828da5858796d3da1419b0221f159ff7160a073c9fe805",
  "sent": false,
  "note": "Body built via langchain_openai's own request builder. NEVER sent."
}
```

`gpt-4o-mini_enum_synthesize.json` captured params:

```json
{ "model": "gpt-4o-mini-2024-07-18", "stream": false, "seed": 42, "top_p": 1.0, "temperature": 0.0 }
```

**The silent drop is real and is caught.** Querying the param guard directly:

```
gpt-4o-mini-2024-07-18   mechanism=direct         rerouted=[]
gpt-5.6-luna             mechanism=model_kwargs   rerouted=['temperature']
```

langchain_openai *does* strip `temperature` for gpt-5.x. The existing guard in `llm.make_llm`
reroutes it through `model_kwargs`, and the capture confirms it survives into the body that would
be posted. Without that, the gpt-5.x stratum would have run at the API default temperature with
nothing in the log to show it. Every capture is asserted before being written; a missing or
contradicted param raises `PreflightError` rather than producing a file.

---

## §6 Dataset pinning — and a stronger proof than planned

Pinned in `Config`:

| field | value |
|---|---|
| `dataset_name` / `dataset_config` / `dataset_split` | `hotpotqa/hotpot_qa` / `distractor` / `validation` |
| `dataset_file` | `<HF_HOME>/datasets/hotpotqa___hotpot_qa/distractor/0.0.0/1908d6af…/hotpot_qa-validation.arrow` |
| `dataset_file_sha256` | `ee53452aadd12dd3e4fa19655c91e01b5320afaf97fd5b5167296a76bc14665c` |
| `dataset_record_count` | `7405` |

Only the *location* is machine-dependent (`HF_HOME`, else the platform default); identity is the
content hash, which is what `assert_dataset_pin` checks. Live result:

```
[3] dataset pin — OK
    path      : …\hotpot_qa-validation.arrow
    sha256    : ee53452aadd12dd3e4fa19655c91e01b5320afaf97fd5b5167296a76bc14665c
    records   : 7405
    pilot ids : 15/15 resolved
```

`PILOT_15_QIDS` is a named constant in `src/preflight.py`. All 15 resolve against the pinned shard.
Resolution forces `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` before importing `datasets`, so no
hub request is attempted.

### The restored prompts reproduce the published log exactly

Because `runs/` turned out to be intact (see pre-flight §3), the restoration could be checked
against the real published series rather than only against my own reference file. For each logged
call I rebuilt the full messages list — restored system prompt plus the user message re-rendered
from the persisted Chroma collection via `retrieved_ids` — and compared `prompt_sha256` to the
value recorded at run time in June:

```
  free_run1  match  344  mismatch 0
  free_run5  match  345  mismatch 0
  enum_run1  match  336  mismatch 0
  enum_run5  match  335  mismatch 0
TOTAL: 1360 matched, 0 mismatched
```

1360 of 1360, across both arms and all three node types, reproducing digests written by the
original run. No embedding or model call was needed — chunk text was read from the local Chroma
store by id. This is direct evidence that the `published` condition is byte-exact, independent of
whether `docs/published_series_reference.json` was itself correct.

---

## §7 Provenance durability — choice and rationale

**Chosen: write provenance to a path that is not git-ignored.** `Config.runs_dir` now defaults to
`provenance/` instead of `runs/`.

**Why this rather than editing `.gitignore`:** the scope for this module permits edits under
`src/`, `tests/` and `docs/` only, and `.gitignore` is outside that. A new non-ignored root
achieves the same durability without changing ignore semantics for the existing `runs/`, `chroma/`,
`.cache/` and `data/` trees — which matters, because `runs/` holds the published series' only
surviving per-call log and I did not want to alter how git treats it as a side effect. If you would
rather un-ignore `runs/` specifically, that is a one-line `.gitignore` change and I will make it on
your say-so.

Nothing under `runs/` was moved, modified or deleted — mtimes are unchanged at 2026-06-18.

The guard, run at start of every experiment:

```
provenance output path: C:\...\Reproducible-RAG\provenance
```

`assert_provenance_durable` prints the resolved path to stdout and calls `git check-ignore`; if git
would ignore it, the run aborts. Tested both ways: the default passes, and `runs_dir="runs"` raises
`PreflightError` matching `git-ignored`.

---

## §8 Verification

| check | result |
|---|---|
| §4 assertions pass for both conditions | **yes** — 15 published + 11 ablation, output in §4 |
| payload captures for both profiles | **yes** — 12 files, contents in §5 |
| CRLF helper unit-tested | **yes** — 6 tests, §1 |
| `docs/module1_tree_before.txt` + `_after` exist | **yes**, diff below |
| **zero API calls to any model provider** | **yes — none, in any code path exercised** |
| no commit created | **yes** — HEAD still `05ea5aa14fdde8211b21b91f7490dce43ce32ae2` |

Test suite: **133 passed** (102 pre-existing + 31 new). Ruff: clean except one pre-existing finding
(below).

### Zero API calls — basis for the claim

No `.invoke()`, `.generate()` or client request is reached anywhere in this module. `make_llm`
constructs a `ChatOpenAI` object, which performs no I/O; `resolve_request_payload` calls the
client's local request builder. Dataset resolution runs with HuggingFace forced offline. The
verification in §6 read only local files (`runs/*.jsonl`, the Chroma store). The only network
traffic observed at any point was one HuggingFace Hub metadata check during an early exploratory
run before offline mode was pinned — HuggingFace is not a model provider, no model was contacted,
and the final code path sets the offline vars before importing `datasets`.

### Tree diff, explained

`.venv/`, `__pycache__/`, `.pytest_cache/` and `.ruff_cache/` are filtered from the comparison
below; those differ only by regenerated bytecode and tool caches from running the test suite.

| line | change | why |
|---|---|---|
| `# module1 tree BEFORE` → `AFTER` | header | expected |
| `?? src/hashing.py` | new | §1 LF helper |
| `?? src/prompts.py` | new | §2/§3 prompts + condition switch |
| `?? src/preflight.py` | new | §4–§7 gates |
| `?? tests/test_conditions.py` | new | 31 tests for the above |
| `docs/module1_tree_after.txt 678` | new | this run's listing |
| `docs/module1_tree_before.txt 595 → 3155225` | grew | the before-listing was mid-write when `find` enumerated it, so it recorded its own size as 595; its final size is 3.1 MB |
| `docs/payload_capture/*.json` ×12 | new | §5 captures |
| `src/agent.py 15493 → 14968` | smaller | prompt text moved out to `src/prompts.py`; composition delegates, `condition` threaded through state |
| `src/config.py 11177 → 14258` | larger | `Condition` enum, `condition` field, dataset pin fields + `_default_dataset_file`, `runs_dir` default |
| `src/run_experiment.py 9647 → 10935` | larger | required `condition` kwarg, `--condition`, run-start gates |
| `tests/test_agent.py 7636 → 7771` | larger | `condition=` added to 4 call sites |
| `tests/test_run_experiment.py 7273 → 7428` | larger | `condition=` added to 3 call sites |

Not shown in the filtered diff and worth stating: `docs/module1_condition_report.md` (this file)
is also new, and `runs/`, `chroma/`, `.cache/` and `results.json` are unchanged.

### Pre-existing lint finding, not from this work

```
F402 Import `field` from line 22 shadowed by loop variable
   --> src\config.py:297:15
```

`from dataclasses import dataclass, field` is part of the uncommitted `ModelProfile` work already in
the tree; it collides with the long-standing loop variable `field` in `build_arg_parser`. It is not
in `git show HEAD:src/config.py`, so it predates this module and is yours in progress. Benign
(shadowing is function-local), and I left it alone rather than renaming a variable in code you are
actively editing.

---

## Files added / changed

**Added:** `src/hashing.py`, `src/prompts.py`, `src/preflight.py`, `tests/test_conditions.py`,
`docs/module1_condition_report.md`, `docs/module1_tree_before.txt`, `docs/module1_tree_after.txt`,
`docs/payload_capture/*.json` (12).

**Changed:** `src/agent.py` (prompts extracted, condition threaded), `src/config.py` (`Condition`,
`condition`, dataset pin, `runs_dir` default), `src/run_experiment.py` (required condition + gates),
`tests/test_agent.py` and `tests/test_run_experiment.py` (call sites).

**Untouched:** retrieval, chunking, embedding, graph topology, metrics, the free-form parser, the
enum schemas, and everything under `runs/`, `chroma/`, `.cache/`.

## How to run

```bash
python -m src.preflight --condition published      # all four gates + writes payload captures
python -m src.preflight --condition ablation --skip-payload
python -m pytest tests/test_conditions.py -q       # 31 tests
python -m src.run_experiment --condition published --dry-run
```
