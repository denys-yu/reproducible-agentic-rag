"""Single source of truth for every frozen reproducibility parameter.

Each model snapshot, decoding param, randomness seed, dataset identifier, and filesystem
path used anywhere in the experiment is declared here exactly once. No other module is
allowed to hardcode these values — they import `Config` instead, so every randomness source
is seeded from one place.

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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, model_validator
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


class Condition(str, enum.Enum):
    """Prompt-wiring condition. See `src.prompts` for the composition rules.

    Declared here rather than in `src.prompts` so that `prompts` can import from `config` without
    a cycle — the condition is a configuration fact, and the prompt module is its consumer.
    """

    PUBLISHED = "published"  # no format suffix anywhere; both arms byte-identical (d6efcab)
    ABLATION = "ablation"  # free-arm GRADE gains the suffix; SYNTHESIZE never does


# Fields that must never be exposed on the CLI or printed in cleartext.
_SECRET_FIELDS: frozenset[str] = frozenset({"openai_api_key"})


# --- model profiles ---------------------------------------------------------------------------


class ModelName(str, enum.Enum):
    """Selectable model profile (`--model`). One stratum of the experiment per profile."""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_5_6_LUNA = "gpt-5.6-luna"


@dataclass(frozen=True)
class ModelProfile:
    """Everything that differs between model strata — and nothing else.

    The experiment's whole claim rests on exactly ONE thing varying across strata: the model.
    A profile therefore carries only the model identity and the request-shape facts forced by
    that model's API (which sampling params it accepts, which extra params it *requires*).
    Prompts, schemas, the graph, retrieval and every metric are profile-independent.

    `supports` lists the sampling params that may be sent. `extra_params` are non-sampling
    params the API requires for the pinned decoding to be accepted at all — they are mandatory,
    not optional: `llm.make_llm` verifies each one survives into the outbound request and aborts
    the run if it does not.
    """

    name: str
    model_id: str
    supports: frozenset[str]
    extra_params: dict[str, Any] = field(default_factory=dict)
    dated_snapshot: bool = True


MODEL_PROFILES: dict[ModelName, ModelProfile] = {
    # Stratum 1 (published). Dated snapshot; plain sampling params, no extras.
    ModelName.GPT_4O_MINI: ModelProfile(
        name=ModelName.GPT_4O_MINI.value,
        model_id="gpt-4o-mini-2024-07-18",
        supports=frozenset({"temperature", "top_p", "seed"}),
        extra_params={},
        dated_snapshot=True,
    ),
    # Stratum 2 (pilot). No dated snapshot is published for this model, so the alias is pinned
    # as-is and `system_fingerprint` comes back null — the run window in the provenance log is
    # the only handle on which server-side build answered.
    #
    # `reasoning_effort="none"` is NOT a tuning choice: with any other effort the API rejects
    # temperature=0 outright ("Unsupported value: 'temperature' does not support 0 with this
    # model. Only the default (1) value is supported."), which would silently move this stratum
    # off the pinned decoding params the first stratum used.
    ModelName.GPT_5_6_LUNA: ModelProfile(
        name=ModelName.GPT_5_6_LUNA.value,
        model_id="gpt-5.6-luna",
        supports=frozenset({"temperature", "top_p", "seed"}),
        extra_params={"reasoning_effort": "none"},
        dated_snapshot=False,
    ),
}


def get_model_profile(name: ModelName | str) -> ModelProfile:
    """Return the profile for a model name; fail loud on an unregistered one."""
    key = name if isinstance(name, ModelName) else ModelName(name)
    return MODEL_PROFILES[key]


# --- dataset pin ------------------------------------------------------------------------------

# The HuggingFace cache layout for the pinned HotpotQA build. The revision hash is part of the
# path, so this resolves to one specific dataset build rather than "whatever is cached".
_HF_DATASET_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
_HF_DATASET_RELPATH = (
    f"datasets/hotpotqa___hotpot_qa/distractor/0.0.0/{_HF_DATASET_REVISION}/"
    "hotpot_qa-validation.arrow"
)


def _default_dataset_file() -> Path:
    """Resolve the pinned validation shard inside the HuggingFace cache.

    Honours `HF_HOME` when set, else the platform default `~/.cache/huggingface`. Only the
    *location* is machine-dependent; the shard's identity is pinned by `dataset_file_sha256`,
    which is what the preflight actually checks.
    """
    hf_home = os.environ.get("HF_HOME")
    root = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return root / _HF_DATASET_RELPATH


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
    # `model` selects the profile (the stratum); `llm_model` is DERIVED from it and must not be
    # set independently — see `_resolve_model_id`.
    model: ModelName = ModelName.GPT_4O_MINI
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

    # ---- Prompt-wiring condition ----
    # Intentionally UNSET by default. `None` means "no condition stated", and every prompt
    # composition path requires an explicit condition (`prompts.compose_system_prompt` has no
    # default for it), so an unstated condition fails loud instead of silently picking one.
    condition: Condition | None = None

    # ---- Dataset (HotpotQA distractor dev set) ----
    dataset_name: str = "hotpotqa/hotpot_qa"
    dataset_config: str = "distractor"
    dataset_split: str = "validation"

    # ---- Dataset pinning (identity of the exact shard this run reads) ----
    # The published series was recovered without any record of which dataset build produced it
    # (see docs/published_series_recovery_report.md §6). Pinning the shard by content hash closes
    # that gap: `preflight.assert_dataset_pin` refuses to run if the bytes on disk differ.
    dataset_file: Path = Field(default_factory=lambda: _default_dataset_file())
    dataset_file_sha256: str = "ee53452aadd12dd3e4fa19655c91e01b5320afaf97fd5b5167296a76bc14665c"
    dataset_record_count: int = 7405

    # ---- Deterministic chunking ----
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ---- Caching (exact-match SHA-256 only; never semantic) ----
    # Embedding cache ON: identical text -> identical vector across runs, avoids re-embedding.
    embedding_cache_enabled: bool = True
    # LLM-response cache OFF by default: headline runs measure inter-run agreement of k INDEPENDENT
    # calls of the same prompt; a warm cache would replay run 1 and force a trivial 100% agreement.
    # Enable only for the replay / cache-warm ablation.
    llm_cache_enabled: bool = False

    # ---- BERTScore (CPU only, pinned checkpoint) ----
    bert_score_model: str = "microsoft/deberta-xlarge-mnli"
    bert_score_device: str = "cpu"

    # ---- Filesystem paths ----
    # `runs_dir` defaults to `provenance/`, which is NOT git-ignored — see
    # docs/module1_condition_report.md §7. The legacy `runs/` tree is git-ignored and is what
    # nearly cost us the published series' per-call log; nothing there is moved or deleted, but
    # new runs no longer write into an ignored directory by default.
    data_dir: Path = Path("data")
    runs_dir: Path = Path("provenance")
    chroma_dir: Path = Path("chroma")
    cache_dir: Path = Path(".cache")

    # ---- Secrets (loaded from env/.env, never logged or printed) ----
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    @model_validator(mode="before")
    @classmethod
    def _resolve_model_id(cls, values: Any) -> Any:
        """Derive `llm_model` from the selected profile; reject a conflicting explicit override.

        Keeping the model id in exactly one place (the profile) is what makes `--model` a true
        one-variable switch. An explicit `llm_model` that agrees with the profile is tolerated
        (env files in the wild pin it); one that disagrees is a silent stratum mix-up, so it
        fails loud.
        """
        if not isinstance(values, dict):
            return values
        profile = get_model_profile(values.get("model") or ModelName.GPT_4O_MINI)
        declared = values.get("llm_model")
        if declared is not None and declared != profile.model_id:
            message = (
                f"llm_model={declared!r} conflicts with model profile {profile.name!r} "
                f"(model_id={profile.model_id!r}). Select the model with `model` / --model only."
            )
            # Pydantic renders the validator's *input* alongside the message, and that input is
            # this very dict — which carries the API key straight from the environment. Mask it
            # in place first: the same object is what the rendered error will read.
            for secret in _SECRET_FIELDS | {"OPENAI_API_KEY"}:
                if secret in values:
                    values[secret] = "<redacted>"
            raise ValueError(message)
        values["llm_model"] = profile.model_id
        return values

    @property
    def model_profile(self) -> ModelProfile:
        """The resolved profile for the selected model."""
        return get_model_profile(self.model)


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
