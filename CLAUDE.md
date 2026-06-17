# CLAUDE.md

## Project

Research experiment for an ICSFTI 2026 conference paper on **reproducibility in agentic RAG**. Goal: measure whether **enum-constrained structured outputs** (Pydantic schemas with `Enum` fields) increase **inter-run agreement** versus **free-form text** in an agentic RAG pipeline, while preserving answer quality. This is a **local research artifact**, not a production system. Keep it minimal, reproducible, and journal-extensible.

## Non-negotiable reproducibility invariants

These rules must hold in every module. Never relax them for convenience.

- **Pinned model snapshots only.** Use `gpt-4o-mini-2024-07-18` — never the floating alias `gpt-4o-mini`. Use `text-embedding-3-small` for embeddings (pin `dimensions`).
- **Fixed decoding params on every LLM call:** `temperature=0`, `top_p=1`, `seed=42`.
- **Log `system_fingerprint`** from every OpenAI call. If it changes mid-run, treat affected runs as a separate stratum — do not silently mix.
- **Deterministic retrieval tie-breaking.** Sort hits by `(score DESC, doc_id ASC)`. Never rely on vector-store insertion order for ties.
- **Exact-match caching ONLY.** Cache key = SHA-256 of canonical JSON `{model, model_kwargs, system_prompt, user_prompt, schema_hash}` via `diskcache`. **Never** implement semantic / cosine-similarity caching — it introduces non-determinism and directly contradicts the paper's central claim.
- **Fixed sampling seed.** Question subset is drawn with `numpy.random.seed(42)`. Same 150 questions every run.
- **Deterministic chunking.** Pin splitter, `chunk_size`, `chunk_overlap`, and tokenizer. `doc_id` = SHA-256 of the normalized chunk text.
- **Provenance for every LLM call** written to JSONL (format below).
- **Framing:** this system is *reproducible-by-design under best-effort determinism*. Bit-exact output through the OpenAI API is unattainable — do not claim "deterministic" anywhere in code, comments, or logs.

## Tech stack (fixed)

- Python **3.12**, dependency management with **`uv`** + committed `uv.lock`.
- **LangChain 0.3.x**, **LangGraph 0.2.x** (agentic state machine).
- **ChromaDB** `PersistentClient`, cosine space (`hnsw:space = cosine`).
- **Pydantic v2** (structured-output schemas with `Enum` fields).
- **OpenAI API** (chat + embeddings).
- Metrics: **scikit-learn** (Cohen's κ), **scipy** (Wilcoxon), **bert-score** pinned to **CPU** with checkpoint `microsoft/deberta-xlarge-mnli` (GPU BERTScore is non-deterministic).
- Secrets via `.env` (`OPENAI_API_KEY`). Never hardcode keys.

## Experiment design (frozen)

- **Dataset:** HotpotQA **distractor** dev set. Each question ships its own 10 paragraphs (2 gold + 8 distractors) — use them; do not substitute Wikipedia dumps.
- **N = 150** questions (seed 42), **k = 5** runs per question per arm.
- **Two arms:**
  - `free` (baseline): final answer + intermediate judgements as plain strings.
  - `enum` (treatment): every LLM call returns a Pydantic model via `with_structured_output(..., strict=True)`.
- **Agentic loop (LangGraph):** `retrieve → grade_context → [rewrite_query (max 1 retry)] → synthesize`. Must make **≥2 LLM calls** per question so the "agentic" label is justified. Max **2** retrieval rounds to bound cost.
- **Retrieval:** `top_k = 4`, cosine, deterministic tie-break.
- **Headline runs use a COLD cache** (measure agreement of the LLM call, not the cache). Cache-warm is a separate ablation.

## Treatment-arm schema (target shape)

Every agent step in the `enum` arm returns a Pydantic v2 model with at least:
- `answer: str`
- `confidence: ConfidenceLevel` — enum `{high, medium, low}`
- `scope: AnswerScope` — enum `{full, partial, none}`
- `supporting_doc_ids: list[str]`

OpenAI strict mode requires `additionalProperties: false`, all fields `required` (use nullable for optional), no root `oneOf`, keep schema small (< 30 fields).

## Metrics (pre-registered — fixed before running)

- **Agreement (primary):** Cohen's κ per enum field (pairwise between runs, averaged); `TARa@5` and `EMA@5` on the final answer.
- **Semantic agreement:** pairwise BERTScore-F1 and cosine similarity (`text-embedding-3-small`) between runs.
- **Quality (control):** Exact Match and token-F1 vs HotpotQA gold — confirm enum constraints don't hurt accuracy.
- **Statistical:** paired Wilcoxon signed-rank on per-question deltas + 95% bootstrap CI. Report `TARr@5` (raw string) for honesty.

## Provenance log (JSONL, one record per LLM call)

Required fields: `run_id`, `question_id`, `arm`, `timestamp`, `git_commit`, `python_version`, lib versions (`langchain`, `langgraph`, `chromadb`), `model`, `system_fingerprint`, `seed`, `temperature`, `top_p`, `prompt_sha256`, `schema_sha256`, `cache_hit`, `retrieved_ids`, `retrieved_scores`, `raw_response`, `parsed`, `tokens_in`, `tokens_out`, `latency_ms`.

## Project structure (target)

- `pyproject.toml`, `uv.lock`, `.env.example`, `CLAUDE.md`, `README.md`
- `src/config.py` — central config; all of {dataset, llm, schema variant, arm, k, n, top_k} overridable via **CLI flags**.
- `src/data.py` — HotpotQA loader, SHA-256 doc_ids, seeded sampling.
- `src/index.py` — ChromaDB builder (persistent, cosine).
- `src/cache.py` — SHA-256 exact-match cache (diskcache).
- `src/schemas.py` — Pydantic enum schemas.
- `src/agent.py` — LangGraph loop (retrieve / grade / rewrite / synthesize).
- `src/provenance.py` — JSONL logger.
- `src/run_experiment.py` — driver: for q in subset, for arm, for run → pipeline.
- `src/metrics.py` — κ, TARa/TARr, EMA, BERTScore, EM/F1, Wilcoxon, bootstrap.
- `data/`, `runs/` (JSONL), `chroma/` (persisted index) — git-ignored except small fixtures.

Keep modules independently testable; each accepts config and returns plain data.

## DO NOT

- No UI, no web server, no Docker / containerization.
- No semantic / similarity caching.
- No new code written inside research-report files.
- No non-pinned model strings.
- No hidden randomness (unseeded shuffles, set ordering, dict-order assumptions).
- No closed or paid datasets — public free datasets only.
- Do not commit `OPENAI_API_KEY`, raw `data/`, `chroma/`, or large `runs/` artifacts.

## Commands

- Install: `uv sync`
- Build index: `uv run python -m src.index`
- Run one arm: `uv run python -m src.run_experiment --arm enum --n 150 --k 5`
- Compute metrics: `uv run python -m src.metrics --runs runs/`

## Conventions

- Type hints everywhere; prefer pure functions that take config and return data.
- Fail loud: a malformed structured output or a `system_fingerprint` change must be logged, not swallowed.
- Every randomness source seeded from one place in `config.py`.
- 