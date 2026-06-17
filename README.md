# Reproducible-by-Design Agentic RAG

Research artifact for an ICSFTI 2026 paper measuring whether enum-constrained structured outputs
(Pydantic schemas with `Enum` fields) increase inter-run agreement versus free-form text in an
agentic RAG pipeline over the HotpotQA distractor dev set, while preserving answer quality. It is a
local, reproducible-by-design research artifact — not production. See `CLAUDE.md` for the frozen
reproducibility invariants and experiment design.

## Setup

```bash
uv sync                              # install pinned dependencies
cp .env.example .env                 # then add your OPENAI_API_KEY
uv run python -m src.config          # print the fully resolved configuration
```
