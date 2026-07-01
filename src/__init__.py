"""Reproducible-by-Design Agentic RAG — research artifact for the accompanying paper.

Package layout:
    config         frozen reproducibility parameters + CLI (the single source of truth)
    data           HotpotQA loader, SHA-256 doc_ids, seeded sampling
    index          ChromaDB persistent index builder + deterministic retrieval
    cache          SHA-256 exact-match cache (diskcache)
    schemas        Pydantic v2 enum schemas for the structured-output arm
    agent          LangGraph agentic loop (retrieve / grade / rewrite / synthesize)
    provenance     JSONL provenance logger (one record per LLM call)
    run_experiment experiment driver
    metrics        agreement / quality / statistical metrics
"""
