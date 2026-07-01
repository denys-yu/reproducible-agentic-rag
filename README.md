# Reproducible-by-Design Agentic RAG

Attribution and Contact

This repository is provided for research and educational purposes. If you use this codebase, experiments, or ideas in your own work, please acknowledge the author.

For questions, collaboration, or consulting related to RAG systems, chunking strategies, or experimental research pipelines, you are welcome to contact the author or engage them as a consultant.

🔗 LinkedIn: https://www.linkedin.com/in/dyuvzhenko

Research artifact for a paper on **reproducibility of enum-constrained structured
outputs in agentic RAG**. It measures whether forcing every LLM step to return a Pydantic model
with `Enum` fields (the `enum` arm) yields higher **inter-run agreement** than free-form text (the
`free` arm) in an agentic retrieval-augmented pipeline over the HotpotQA distractor dev set — while
preserving answer correctness. Both arms run through the *same* LangGraph loop
(`retrieve → grade → [rewrite ×1] → synthesize`) with the *same* prompts and the *same* pinned
decoding parameters; the only difference is whether the model's output is schema-constrained. The
headline finding: **enum constraints raise inter-run agreement on the agent's internal routing
decisions** (e.g. the re-retrieval decision Cohen's κ rises from 0.81 to 0.97, grading confidence
from 0.64 to 0.91; both significant under a paired Wilcoxon test) **without harming correctness** —
containment of the gold answer is statistically indistinguishable between arms (n.s.).

> Note on quality metrics: enum answers are more verbose, which lowers exact-match and token-F1 even
> though the gold span is still present. The paper reports **containment** (gold tokens present in
> the answer) as the correctness control precisely to separate this formatting/verbosity effect from
> a real accuracy loss. See `src/figures.py` and the generated figures.

This is a local, reproducible-by-design research artifact — not a production system. The full set of
frozen reproducibility invariants and the experiment design are documented in the sections below and
in the accompanying paper.

## Requirements

- **Python 3.12** (the project pins `>=3.12,<3.13`).
- **[uv](https://docs.astral.sh/uv/)** for dependency management (a committed `uv.lock` pins every
  transitive dependency).
- An **OpenAI API key** supplied via the `OPENAI_API_KEY` environment variable. The key is read from
  the environment (or a local, git-ignored `.env` file) at runtime and is **never stored in this
  repository** — `.env` is git-ignored and no key is committed.

## Setup

```bash
uv sync                              # create the venv and install pinned dependencies

# Provide your key (either export it, or put it in a local .env — .env is git-ignored):
export OPENAI_API_KEY=sk-...         # bash/zsh
#   or:  echo 'OPENAI_API_KEY=sk-...' > .env

uv run python -m src.config          # print the fully resolved configuration to verify setup
```

## Reproduce

The full pipeline is three commands, plus a fourth to render the figures. Each `python` invocation
below is run through uv (`uv run python -m ...`).

```bash
uv run python -m src.index --build          # 1. build & persist the cosine vector index (chroma/)
uv run python -m src.run_experiment         # 2. run both arms (N=150, k=5) -> runs/ manifests
uv run python -m src.metrics                # 3. compute agreement + quality metrics -> results.json
uv run python -m src.figures                # 4. render publication figures -> figures/
```

Defaults reproduce the frozen design: both arms, **N = 150** questions, **k = 5** runs per arm.
Useful flags for a smaller/cheaper trial: `--limit 10` and `--runs 2` on `run_experiment`, and
`--dry-run` to print the plan (question count, arms, and approximate call counts) without making any
API calls. `run_experiment --resume` skips any `(arm, run)` already marked complete.

**Cost & API use.** Steps 1–2 call the OpenAI API (chat + embeddings, pinned snapshots). A full
`N=150 × k=5 × 2 arms` run makes on the order of a few thousand small `gpt-4o-mini` calls and costs
roughly **a few USD**; the exact figure depends on OpenAI pricing at run time. Steps 3–4 are offline
(`src.metrics --cosine` and `--bertscore` are the only optional flags that touch the API or download
a model). `src.figures` only reads `results.json` — it never re-runs the experiment.

## Determinism & reproducibility notes

This system is *reproducible-by-design under best-effort determinism* — bit-exact output through a
hosted API is unattainable, so we do not claim "deterministic." The safeguards in place:

- **Pinned model snapshot** `gpt-4o-mini-2024-07-18` (never the floating `gpt-4o-mini` alias) and
  `text-embedding-3-small` at fixed `dimensions=1536`.
- **Fixed decoding** on every LLM call: `temperature=0`, `top_p=1`, `seed=42`.
- **Fixed sampling seed** (`numpy` seed 42) selects the same 150-question subset every run;
  **deterministic chunking** (pinned splitter, `chunk_size`/`chunk_overlap`, `cl100k_base` tokenizer)
  gives each chunk a content-addressed SHA-256 `doc_id`.
- **Deterministic retrieval:** candidates are sorted by `(similarity DESC, doc_id ASC)` in our own
  code — Chroma's internal tie ordering is never trusted.
- **Exact-match caching only** (SHA-256 of canonical JSON; `diskcache`). There is deliberately no
  semantic / cosine / nearest-neighbour cache anywhere — that would inject non-determinism and
  contradict the paper's claim. The **embedding cache is ON** (identical text → identical vector);
  the **LLM-response cache is OFF** for headline runs, so the k runs measure agreement across
  *independent* calls rather than replaying run 1.
- **Sequential execution:** the driver runs one pipeline invocation at a time (no threads/async), so
  concurrent requests are never co-batched server-side.
- **Append-only provenance:** every LLM call writes one JSONL record to
  `runs/<arm>_run<i>/run_manifest.jsonl`, capturing git commit, Python + library versions, model,
  decoding params, prompt/schema SHA-256 hashes, cache-hit flag, retrieved ids/scores, raw + parsed
  response, token counts, and latency. `system_fingerprint` is logged on every call; if it changes
  mid-run the affected runs should be treated as a separate stratum, not silently mixed.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/config.py` | Single source of truth for every frozen parameter; all fields overridable via CLI flags / env. |
| `src/data.py` | HotpotQA distractor loader, SHA-256 `doc_id`s, seeded 150-question sampling. |
| `src/index.py` | ChromaDB persistent cosine index builder + deterministic per-question retrieval. |
| `src/schemas.py` | Pydantic v2 enum schemas for the strict structured-output (`enum`) arm. |
| `src/llm.py` | Single logged/cached LLM-call layer (structured for `enum`, parsed text for `free`). |
| `src/agent.py` | LangGraph agentic loop: `retrieve → grade → [rewrite ×1] → synthesize`. |
| `src/cache.py` | SHA-256 exact-match disk cache (embeddings ON, LLM responses OFF); never semantic. |
| `src/provenance.py` | Append-only JSONL provenance logger, one record per LLM call. |
| `src/run_experiment.py` | Sequential driver: for each arm × run × question → one pipeline invocation. |
| `src/metrics.py` | Cohen's/Fleiss' κ, TARa/TARr, EMA, EM/token-F1/containment, Wilcoxon + bootstrap → `results.json`. |
| `src/figures.py` | Grayscale-safe publication figures (agreement κ, quality) from `results.json` → `figures/`. |
| `runs/` | Per-`(arm, run)` provenance manifests (git-ignored). |
| `figures/` | Generated PNG (300 dpi) + PDF figures. |
| `chroma/`, `data/`, `.cache/` | Persisted index, downloaded dataset, and exact-match cache (all git-ignored). |

## Dataset

**HotpotQA** distractor dev set (`hotpotqa/hotpot_qa`, `distractor` config, `validation` split),
licensed **CC BY-SA 4.0**. It is **downloaded on first run** by `src/data.py` via the Hugging Face
`datasets` library and is **not redistributed in this repository**. Each question ships its own 10
paragraphs (2 gold + 8 distractors), which are used as-is; a fixed `numpy` seed (42) selects the same
150-question sample every run.

## License

Released under the **MIT License**.

