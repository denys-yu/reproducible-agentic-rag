"""Single source of truth for every frozen reproducibility parameter.

Each model snapshot, decoding param, randomness seed, dataset identifier, and filesystem
path used anywhere in the experiment is declared here exactly once. No other module is
allowed to hardcode these values — they import `Config` instead (CLAUDE.md: "Every randomness
source seeded from one place in config.py").

Resolution precedence (highest first):
    1. CLI flags     (e.g. --n-questions 10 --arm enum)
    2. Environment   (REPRORAG_* vars, plus the unprefixed OPENAI_API_KEY) and a local .env
    3. Frozen defaults declared below

Run `python -m src.config` to print the fully resolved configuration.
"""

from __future__ import annotations

import argparse
import enum
import json
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Arm(str, enum.Enum):
    """Experimental arm. Both arms share IDENTICAL prompts; only the response schema differs."""

    FREE = "free"  # baseline: answers + intermediate judgements as plain strings
    ENUM = "enum"  # treatment: every LLM call returns a Pydantic model (strict structured output)


class SchemaVariant(str, enum.Enum):
    """Structured-output schema variant used by the `enum` arm.

    Kept as an enum so the schema shape is journal-extensible without touching call sites.
    """

    ANSWER_V1 = "answer_v1"  # answer + confidence + scope + supporting_doc_ids


# Fields that must never be exposed on the CLI or printed in cleartext.
_SECRET_FIELDS: frozenset[str] = frozenset({"openai_api_key"})


class Config(BaseSettings):
    """Frozen, fully-resolved experiment configuration.

    Immutable after construction (`frozen=True`) so a single instance can be threaded through
    the whole pipeline without any module mutating shared state.
    """

    model_config = SettingsConfigDict(
        env_prefix="REPRORAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        protected_namespaces=(),
    )

    # ---- LLM / embeddings (pinned snapshots only) ----
    llm_model: str = "gpt-4o-mini-2024-07-18"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    temperature: float = 0.0
    top_p: float = 1.0
    llm_seed: int = 42

    # ---- Global randomness ----
    numpy_seed: int = 42

    # ---- Experiment design (frozen) ----
    n_questions: int = 150
    k_runs: int = 5
    top_k: int = 4
    max_retrieval_rounds: int = 2
    arm: Arm = Arm.FREE
    schema_variant: SchemaVariant = SchemaVariant.ANSWER_V1

    # ---- Dataset (HotpotQA distractor dev set) ----
    dataset_name: str = "hotpot_qa"
    dataset_config: str = "distractor"
    dataset_split: str = "validation"

    # ---- Deterministic chunking ----
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ---- BERTScore (CPU only, pinned checkpoint) ----
    bert_score_model: str = "microsoft/deberta-xlarge-mnli"
    bert_score_device: str = "cpu"

    # ---- Filesystem paths (git-ignored at runtime) ----
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    chroma_dir: Path = Path("chroma")
    cache_dir: Path = Path(".cache")

    # ---- Secrets (loaded from env/.env, never logged or printed) ----
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")


def _add_field_argument(parser: argparse.ArgumentParser, name: str, annotation: Any) -> None:
    """Register a single CLI flag derived from a Config field's name and type."""
    flag = f"--{name.replace('_', '-')}"
    # default=SUPPRESS so that an omitted flag leaves no key in the namespace; this preserves
    # the env > defaults precedence (only explicitly-passed flags become overrides).
    common: dict[str, Any] = {"dest": name, "default": argparse.SUPPRESS}

    if annotation is bool:
        parser.add_argument(flag, action=argparse.BooleanOptionalAction, **common)
    elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        parser.add_argument(flag, choices=[e.value for e in annotation], **common)
    elif annotation is int:
        parser.add_argument(flag, type=int, **common)
    elif annotation is float:
        parser.add_argument(flag, type=float, **common)
    elif annotation is Path:
        parser.add_argument(flag, type=Path, **common)
    else:
        parser.add_argument(flag, type=str, **common)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build an argparse parser exposing every (non-secret) Config field as an optional flag."""
    parser = argparse.ArgumentParser(
        prog="python -m src.config",
        description="Resolve and print the frozen experiment configuration.",
    )
    for name, field in Config.model_fields.items():
        if name in _SECRET_FIELDS:
            continue  # secrets come from env/.env only, never the command line
        _add_field_argument(parser, name, field.annotation)
    return parser


def load_config(argv: list[str] | None = None) -> Config:
    """Resolve the configuration from CLI flags, environment, and frozen defaults."""
    args = build_arg_parser().parse_args(argv)
    overrides = vars(args)  # only explicitly-passed flags are present (default=SUPPRESS)
    return Config(**overrides)


def as_printable_dict(config: Config) -> dict[str, Any]:
    """Render the config as JSON-serializable data, masking secrets to set/unset only."""
    data = config.model_dump(mode="json")
    for name in _SECRET_FIELDS:
        if name in data:
            data[name] = "<set>" if getattr(config, name) is not None else None
    return data


def main(argv: list[str] | None = None) -> None:
    """Print the fully resolved configuration as indented JSON."""
    config = load_config(argv)
    print(json.dumps(as_printable_dict(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
