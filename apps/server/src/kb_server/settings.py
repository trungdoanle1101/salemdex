from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The same values as the framework's huggingface Device. Defined here because importing
# that one loads torch, and every process reads Settings, not only the worker.
type Device = Literal["auto", "cuda", "mps", "cpu"]
type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

type Positive = Annotated[int, Field(gt=0)]


class Settings(BaseSettings):
    """Configuration from KB_* environment variables, or a .env file next to where it runs.

    Every process reads the same Settings. Worker-only fields have defaults, or are None
    until set, so the API server and create-admin start without them; the worker checks
    the ones it needs when it builds its tools.
    """

    # hide_input_in_errors: a whole-model error would print every input, and
    # KB_DATABASE_URL holds the password
    model_config = SettingsConfigDict(
        env_prefix="KB_", env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    # every process
    database_url: str  # holds the password: .env only, never committed
    log_level: LogLevel = "INFO"

    # the API: sessions and requests
    session_ttl_hours: Positive = 12
    act_for_headers: tuple[str, ...] = ("x-act-for-user",)
    max_upload_mb: Positive = 50

    # file storage: the API writes originals, the worker reads them, so both need the same
    storage_kind: Literal["local", "s3"] = "local"
    storage_path: Path

    # worker, preparation: candidate profiles, most specific first
    profiles: tuple[str, ...] = ("vn_regulation",)
    fallback_profile: str = "plain"
    # counted in words, so no model's tokenizer is needed; 450 words is about 512
    # bge-m3 tokens for Vietnamese (1.10 tokens per word)
    chunk_words: Positive = 450

    # worker, models
    models_folder: Path | None = None  # required by the worker

    # worker, embedder: a name from the framework's HUGGINGFACE_MODEL_SPECS
    embedder_spec: str = "bge-m3"
    embedder_device: Device = "auto"
    # memory and speed only; the vectors don't change
    embedder_batch_size: Positive = 32

    reranker_spec: str = "bge-reranker-v2-m3"
    reranker_device: Device = "auto"
    reranker_batch_size: Positive = 32
    rerank_depth: Positive = 10

    search_max_limit: Positive = 5

    # worker, index
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kb_chunks"
    qdrant_api_key: SecretStr | None = None

    # worker, the job loop
    worker_poll_seconds: Annotated[float, Field(gt=0)] = 2.0
    job_max_attempts: Annotated[int, Field(ge=1)] = 3
    job_timeout_minutes: Positive = 60  # longer than the longest real job

    @model_validator(mode="after")
    def _check_rerank_depth_covers_search_max_limit(self) -> Self:
        """Refuse a search limit the reranker can't fill.

        Raises:
            ValueError: `search_max_limit` is above `rerank_depth`.
        """
        # Checked at load, not per request: a search would otherwise return fewer
        # passages than asked for, and the caller couldn't tell why
        if self.search_max_limit > self.rerank_depth:
            raise ValueError(
                f"KB_SEARCH_MAX_LIMIT ({self.search_max_limit}) is above "
                f"KB_RERANK_DEPTH ({self.rerank_depth}): search can't return more "
                "passages than it reranks. Lower the limit or raise the depth."
            )
        return self
