# Published-series recovery report

Recovery of the exact prompt texts and run configuration that produced the repository-root
`results.json` (the first, published experimental series), from git history alone.

**Method constraint.** Read-only git inspection (`log`, `show`, `diff`, `grep`, `ls-tree`,
`archive`). No commit, branch, checkout, reset, or working-tree modification. No experiment
run and no API call of any kind. Two new files were created, both under `docs/`.

**Definition of "HEAD" used in this report.** The working tree was dirty at start (see §1).
Per the operator's explicit instruction, work proceeded with **HEAD defined as the committed
blobs** (`git show HEAD:<path>`), not the on-disk files. Where the uncommitted working tree
diverges materially, it is reported as a clearly-labelled *third* state, never merged into the
HEAD column.

---

## 1. `git status --porcelain` — taken before any work

```
 M src/agent.py
 M src/config.py
 M src/llm.py
 M src/provenance.py
 M src/run_experiment.py
 M tests/test_agent.py
 M tests/test_llm.py
 M tests/test_provenance.py
 M tests/test_run_experiment.py
?? src/pilot.py
```

`git rev-parse HEAD` before work: `05ea5aa14fdde8211b21b91f7490dce43ce32ae2`

**The tree was dirty, which is a fail condition under pre-flight assertion #1.** Work was
halted and reported. The operator then directed that it proceed anyway under the
committed-blob definition of HEAD. This caveat is recorded here rather than resolved: the
uncommitted diff is +509/−56 across nine tracked files, and it touches `agent.py` (prompts),
`llm.py` (parser), `config.py` (run config) and `provenance.py` (hashing) — precisely the
files this recovery reads. Every conclusion below therefore states which of the three states
it refers to.

The uncommitted work is an in-progress **second** series: `config.py` gains a `ModelName`
enum and `MODEL_PROFILES` table introducing a `gpt-5.6-luna` stratum, matching the current
branch name `experiment/gpt-5.6-luna`. It postdates and does not describe the published run.

### Pre-flight assertions

| # | Assertion | Result |
|---|---|---|
| 1 | working tree clean | **FAILED** — dirty; overridden by operator instruction (above) |
| 2 | `results.json` parses, has `comparison`/`meta`/`per_arm` | PASS |
| 3 | `meta.n_complete == 150`, `meta.k == 5` | PASS |
| 4 | harness hash function located | PASS — see below |

**Hash functions (assertion #4).** Two, both reused verbatim for every digest in this task;
neither was reimplemented.

| | prompt hashing | schema hashing |
|---|---|---|
| file | `src/provenance.py` | `src/schemas.py` |
| name | `prompt_sha256` | `schema_sha256` |
| signature | `(messages: list[dict]) -> str` | `(schema: type[BaseModel]) -> str` |
| serialization | `json.dumps(messages, sort_keys=True, separators=(",",":"), ensure_ascii=True)` | `json.dumps(to_openai_strict_schema(m), sort_keys=True, separators=(",",":"), ensure_ascii=True)` |
| encoding | `.encode("utf-8")` | `.encode("utf-8")` |
| strip | **no strip** at any stage | n/a |
| newlines | preserved, escaped as `\n` inside the JSON string | n/a |
| non-ASCII | escaped to `\uXXXX` (`ensure_ascii=True`) | same |
| invariants | key order and inter-key whitespace invariant; message **order** and content significant | property order invariant; `required` sorted |

`prompt_sha256` is a **messages-level** function — the harness never hashes a bare prompt
string. To hash a prompt constant, this report wraps it as
`[{"role": "system", "content": <constant>}]` and passes that to the harness function
unmodified. The wrapping convention is mine and is stated explicitly in the reference JSON
(`hash_function.prompt.application_note`); the digest algorithm is the harness's.

To guarantee the published-commit versions of these functions were used, `src/` at the
published commit was extracted with `git archive` into a scratchpad outside the repository
and imported from there. The working-tree `provenance.py` was not imported. (Its
`prompt_sha256` body is in fact unchanged — the only `json.dumps` edit in that file is in the
logger write path — but it was avoided on principle.)

---

## 2. Full commit list touching `results.json`

```
d6efcab38e9f0fab331ecafa164509a77c181f4e  2026-07-01 17:32:26 +0300  Public release: reproducible agentic RAG experiment
```

Exactly **one** commit, confirmed three ways: `git log --follow -- results.json`,
`git log --all -- results.json`, and a scan of all 25 commits in the repository. The file was
introduced once and never subsequently modified.

---

## 3. Choice of `published_commit`, with byte-identity check

**`published_commit = d6efcab38e9f0fab331ecafa164509a77c181f4e`**

The selection rule ("earliest commit whose content matches the current file byte-for-byte")
is trivially satisfied — there is only one candidate — but the byte-identity check was run
anyway, and it surfaced a wrinkle worth recording.

| object | sha256 |
|---|---|
| `git show d6efcab:results.json` | `bfd84e73…09bb95c` |
| `git show HEAD:results.json` | `bfd84e73…09bb95c` |
| working-tree file `results.json` (raw bytes) | `14a0f42e…502a3df1ff` |
| working-tree file, after CRLF→LF | `bfd84e73…09bb95c` |

The raw working-tree digest **differs** from the blob. This is not a content difference:
`core.autocrlf=true`, the on-disk file is CRLF (8653 bytes) and the blob is LF (8374 bytes).
Normalised, they are identical, and `git status` correctly reports the file as unmodified.
Byte-identity therefore **holds at the blob level**, which is the level that matters. Any
future verifier that hashes `results.json` off disk on Windows must normalise line endings
first or it will get `14a0f42e…` and wrongly conclude the file was tampered with.

### Is the *code* at `d6efcab` the code that produced `results.json`?

This is the question the selection rule does not by itself answer, and it needed checking:
`d6efcab` is dated 2026-07-01, whereas `results.json` has an on-disk mtime of 2026-06-19,
twelve days earlier. The commit that *introduced* an artifact is not necessarily the commit
whose code *produced* it.

Two checks establish that `d6efcab` is nonetheless a sound read point:

1. **`d6efcab` touched no experiment source.** Against its parent `5d6298f` it changed only
   `results.json`, `figures/*`, `src/figures.py`, `README.md`, `LICENSE`, `.env.example`,
   `pyproject.toml`, `uv.lock`. No prompt, schema, agent, config, or parser file is in the
   diff.
2. **The experiment code was frozen well before the run.** Last-modifying commit per file at
   `d6efcab`: `agent.py` → `f245238` (06-18), `llm.py` → `788db9e` (06-18),
   `run_experiment.py` → `0489a2a` (06-18), `provenance.py` → `5b1543f` (06-18),
   `config.py`/`cache.py` → `eee15fd` (06-17), `schemas.py` → `26c2975` (06-17),
   `data.py` → `a81f01d` (06-17), `index.py` → `2018e7c` (06-17). Only `metrics.py` changed
   on 06-19 (`5d6298f`), and metrics are computed *from* the runs, not used to produce them.

So the pipeline configuration at `d6efcab` is identical to the configuration in force
throughout the 06-19 run window. Recovering config from `d6efcab` is safe.

---

## 4. Diff table: `published_commit` vs HEAD

HEAD = committed blobs. A third column reports the uncommitted working tree wherever it
diverges, since that is where all the substantive drift lives.

### 4a. Source files, whole-file identity

| file | d6efcab vs HEAD (committed) | vs working tree |
|---|---|---|
| `src/agent.py` | **identical** | **DIFFERS** (prompts rewritten, suffix added) |
| `src/llm.py` | **identical** | DIFFERS (+251/−? lines; parser region — see 4d) |
| `src/schemas.py` | **identical** | **identical** |
| `src/data.py` | **identical** | identical |
| `src/index.py` | **identical** | identical |
| `src/provenance.py` | **identical** | DIFFERS (logger write path; `prompt_sha256` unchanged) |
| `src/cache.py` | **identical** | identical |
| `src/config.py` | DIFFERS — **docstring only** | DIFFERS (new model-stratum machinery) |
| `src/run_experiment.py` | **identical** | DIFFERS |

The single committed change between `d6efcab` and HEAD in any experiment-critical file is a
`config.py` module docstring edit removing a reference to a deleted `CLAUDE.md`:

```
-allowed to hardcode these values — they import `Config` instead (CLAUDE.md: "Every randomness
-source seeded from one place in config.py").
+allowed to hardcode these values — they import `Config` instead, so every randomness source
+is seeded from one place.
```

No executable behaviour changes. **Every prompt, schema, and hash below is therefore identical
at `d6efcab` and at committed HEAD.**

### 4b. Prompt constants (hashed with `prompt_sha256`)

| item | d6efcab | HEAD (committed) | same? | working tree |
|---|---|---|---|---|
| `GRADE_PROMPT` | `c73170f0…3c9fe805` | `c73170f0…3c9fe805` | **yes** | `a52aff41…096ba782` ✗ |
| `REWRITE_PROMPT` | `d92149e5…2cd0e097` | `d92149e5…2cd0e097` | **yes** | `d92149e5…2cd0e097` ✓ |
| `SYNTHESIZE_PROMPT` | `63a7dcfe…0d9a21b6` | `63a7dcfe…0d9a21b6` | **yes** | `12abcee8…5959b9ea` ✗ |
| free-format suffix constant | **does not exist** | **does not exist** | **yes** | `FREE_FORMAT_SUFFIX` = `ff62246a…f2707517` ✗ |

### 4c. Composed system prompts per (arm, node)

At `d6efcab` the composition logic is a bare constant — `_grade_messages`, `_rewrite_messages`
and `_synthesize_messages` take `(question, chunks)` only, have no `arm` parameter, and emit
`{"role": "system", "content": <CONSTANT>}` verbatim. Both arms are byte-identical.

| (arm, node) | d6efcab & HEAD | working tree | same? |
|---|---|---|---|
| enum, grade | `c73170f0…3c9fe805` | `a52aff41…096ba782` | ✗ |
| enum, rewrite | `d92149e5…2cd0e097` | `d92149e5…2cd0e097` | ✓ |
| enum, synthesize | `63a7dcfe…0d9a21b6` | `12abcee8…5959b9ea` | ✗ |
| free, grade | `c73170f0…3c9fe805` | `44a7fb86…5e9a4dbe` | ✗ |
| free, rewrite | `d92149e5…2cd0e097` | `d92149e5…2cd0e097` | ✓ |
| free, synthesize | `63a7dcfe…0d9a21b6` | `a5d31c4e…98e807373` | ✗ |

### 4d. Specifically-requested checks

**The phrase "exactly one of".** Absent at `d6efcab`, absent at committed HEAD, and absent in
the working tree. `git grep -i 'exactly one of'` returns nothing in `src/` at any of the three
states. It plays no part in this experiment at any point in its history.

**Enum value domains for `scope` and `confidence`.** Identical at all three states —
`schemas.py` is byte-identical everywhere:

| enum | values |
|---|---|
| `AnswerScope` | `full`, `partial`, `none` |
| `ConfidenceLevel` | `high`, `medium`, `low` |

**Does the format suffix reach the synthesize node?**

| state | suffix exists? | reaches synthesize? |
|---|---|---|
| `d6efcab` (published) | **no — no such constant anywhere in git history** | no |
| HEAD (committed) | **no** | no |
| working tree (uncommitted) | yes, `FREE_FORMAT_SUFFIX` | **yes, free arm only** |

This is the single most consequential finding for interpreting `results.json`.
`FREE_FORMAT_SUFFIX` exists **only in the uncommitted working tree**. `git log --all -S` and
`git grep` across every commit find no free-format suffix constant at any point in the
repository's history, and `src/prompts.py` has never existed. In the working tree the suffix
is applied through a new helper:

```python
def _system_prompt(base: str, arm: Arm | str) -> str:
    arm_value = arm.value if isinstance(arm, Arm) else str(arm)
    return base + FREE_FORMAT_SUFFIX if arm_value == Arm.FREE.value else base
```

called at exactly two sites, both gaining a new `arm` parameter —
`_grade_messages` → `_system_prompt(GRADE_PROMPT, arm)` and
`_synthesize_messages` → `_system_prompt(SYNTHESIZE_PROMPT, arm)`. `_rewrite_messages` is
deliberately not suffixed. **None of this was in force for the published series.**

**Working-tree base-prompt rewording** (beyond the suffix — the enum-arm hashes move too):

```
grade:      "scope — whether the context covers the question fully, partially, or not at all (none); "
        →   "scope — whether the context covers the question: full, partial, or none; "
            "confidence — your confidence (high, medium, low) in that judgement; "
        →   "confidence — your confidence in that judgement: high, medium, or low; "
            "needs_more_context — whether the system should search again with a reformulated query. "
        →   "needs_more_context — whether the system should search again with a reformulated query: yes or no. "

synthesize: "confidence — your confidence (high, medium, low); "
        →   "confidence — your confidence in the answer: high, medium, or low; "
            "scope — whether the context covered the question fully, partially, or none; "
        →   "scope — whether the context covered the question: full, partial, or none; "
```

### 4e. Enum-arm JSON schemas (hashed with `schema_sha256`)

Identical at all three states.

| node | model | sha256 |
|---|---|---|
| grade | `ContextGrade` | `24aff099…e5892587` |
| rewrite | `RewriteQuery` | `3eef9a8d…4786a2ea` |
| synthesize | `AnswerV1` | `c1c4015f…8012f48a` |

Free arm sends no schema; the provenance record logs `schema_sha256: null`.

### 4f. Run and dataset configuration

Identical at `d6efcab` and committed HEAD for every value below.

| item | value |
|---|---|
| model | `gpt-4o-mini-2024-07-18` |
| temperature / top_p / seed | `0.0` / `1.0` / `42` |
| reasoning params | none — `make_llm` sets only model, temperature, top_p, seed |
| embedding model | `text-embedding-3-small`, `dimensions=1536` |
| chunk size / overlap | `512` / `64` (tokens) |
| splitter | `RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", …)` |
| top_k | `4` |
| max retrieval rounds | `2` |
| tie-break | `(similarity DESC, doc_id ASC)`; `similarity = 1.0 − chroma_cosine_distance` |
| retrieval scoping | `where={"question_id": …}` — never crosses questions |
| LLM cache | **off** (`llm_cache_enabled=False`) |
| embedding cache | **on** (`embedding_cache_enabled=True`) |
| dataset | `hotpotqa/hotpot_qa`, config `distractor`, split `validation` |
| sampling seed | `42` (`Config.numpy_seed`) |
| id selection | `sorted(set(ids))` → `np.random.default_rng(42).choice(…, size=150, replace=False)` → `sorted()`; isolated RNG |
| n_questions / k_runs / arms | `150` / `5` / `["free", "enum"]` |
| execution | strictly sequential — no threads, processes, or async |

The LLM cache being **off** is load-bearing and the config comments say why: the headline runs
measure inter-run agreement across k independent calls of the same prompt, and a warm cache
would replay run 1 and force a trivial 100 % agreement.

---

## 5. Plausibility cross-check against `results.json`

**Design.** `meta.arms == ["enum", "free"]` (2), `meta.k == 5`, `meta.n_complete == 150`,
`meta.n_excluded == 0`, `meta.excluded_qids == []`. `config.py` at `d6efcab` declares
`n_questions=150`, `k_runs=5`; `run_experiment._resolve_arms` defaults to `[FREE, ENUM]`; and
the driver's own plan line computes `len(arms) * runs * n_questions`. The design is
**2 arms × 5 runs × 150 questions = 1500 pipeline invocations**, consistent in both directions.

**`rewrite_rate`.** Reported values: enum `0.23733333333333334`, free `0.2946666666666667`.
`metrics.rewrite_rate` is documented as "fraction of pipeline runs (arm × question × run) that
triggered a rewrite", so the denominator per arm is 150 × 5 = 750. Multiplying back:

- enum: `0.23733333333333334 × 750 = 178.0` — exact integer
- free: `0.2946666666666667 × 750 = 221.0` — exact integer

Both land on whole numbers, which independently confirms the 750-run-per-arm denominator and
hence the 150 × 5 design.

**Rewrite node exists and is capped at one.** In `agent.py` at `d6efcab`, `rewrite_node` is
registered in the graph, and the cap is enforced at three points:

- `_route_after_grade` routes to `rewrite` only when `grade.needs_more_context is True` **and**
  `state["rewrite_count"] == 0`;
- `rewrite_node` returns `{"rewrite_count": 1}` — a set, not an increment;
- `_route_after_retrieve` sends the post-rewrite retrieval (`rewrite_count == 1`) straight to
  `synthesize`, so grading never happens twice.

The module docstring states the same bound: 2 LLM calls (grade + synthesize) or 3
(grade + rewrite + synthesize). A rewrite rate in `[0, 1]` is therefore the only possible
range, and both observed values sit inside it.

One note on the free arm: `_route_after_grade` treats a `None` `needs_more_context` — the free
parser's "not confidently present" signal — as `False`, routing to synthesize. The free arm's
higher rewrite rate (0.2947 vs 0.2373) is thus a floor, not an artefact of `None` handling.

---

## 6. Unresolved

Items that could **not** be determined from git history. Nothing here has been filled in from
the current code and presented as recovered.

1. **The per-call `prompt_sha256` values actually recorded in the published runs.**
   `llm.call_llm` logs `prompt_sha256(messages)` over the **full** messages list — system
   *and* user. The user message embeds the runtime question text and the rendered retrieved
   context (`_format_context`), which depend on the HotpotQA rows and on retrieval output.
   Neither is in git. Only the **system half** of each composed prompt is recovered; the
   reference JSON stores the user-message *templates* verbatim but cannot instantiate them.
   A restoration module can verify prompt *constants* by hash, but cannot reproduce the lost
   log's per-call digests.

2. **The concrete list of 150 sampled `question_id`s.** The selection procedure is fully
   recovered and deterministic (seed 42, `sorted(set(...))`, `default_rng.choice`, `replace=False`),
   but resolving it to actual ids requires downloading the HotpotQA `distractor` validation
   split and executing the sampler. That is a network call and an execution step, both out of
   scope here. Stored as `resolved_question_ids: null` with the reason attached.

3. **Library versions actually in force during the run.** `provenance.capture_environment()`
   records `git_commit`, `python_version` and `lib_versions` into every provenance record —
   but that is exactly the log that was lost. `uv.lock` at `d6efcab` pins a resolvable set, but
   whether the 06-19 run used that resolution is not verifiable from history, and the two are
   twelve days apart. Not asserted.

4. **`system_fingerprint` values returned by the API.** Captured per call into the provenance
   log and nowhere else. Irrecoverable.

5. **Which `results.json` fields came from which run directory.** `meta.runs_dir` is `"runs"`,
   but `runs/` is git-ignored and absent. The mapping from manifests to aggregates cannot be
   re-derived.

6. **Whether the published run used the frozen defaults or CLI/env overrides.** `Config`
   resolves CLI flags > `REPRORAG_*` env / `.env` > frozen defaults. Only the defaults are in
   git; `.env` is git-ignored and `.env.example` was deleted in `d6efcab` itself. The recovered
   values are the **frozen defaults**, and they are consistent with `results.json` on every
   quantity `results.json` exposes (`n_complete=150`, `k=5`, both arms, `runs_dir="runs"`) —
   but an override of a parameter that `results.json` does not surface (e.g. `top_k`,
   `chunk_size`, `temperature`) would be invisible to this recovery. Flagged rather than
   assumed away.

7. **The exact wall-clock date of the run.** Inferred as on/around 2026-06-19 from the
   `results.json` on-disk mtime (14:25) and the adjacent `metrics.py` commit `5d6298f`
   (14:26). An mtime is not authoritative — it survives neither a fresh clone nor a checkout —
   so this is offered as an indication, not a recovered fact.

---

## Verification

| check | result |
|---|---|
| `git rev-parse HEAD` before | `05ea5aa14fdde8211b21b91f7490dce43ce32ae2` |
| `git rev-parse HEAD` after | `05ea5aa14fdde8211b21b91f7490dce43ce32ae2` — **unchanged** |
| new files created | exactly two, both under `docs/` |
| `docs/published_series_reference.json` parses as JSON | yes |
| all sha256 digests produced by the harness's own functions | yes — `prompt_sha256` and `schema_sha256` imported from the `d6efcab` source; no ad-hoc hashing |
| "Unresolved" section present | yes — §6, seven items |
| commits created | none |
| historical commits checked out / modified | none |

The pre-existing dirty working-tree entries from §1 remain untouched and are unrelated to this
recovery.
