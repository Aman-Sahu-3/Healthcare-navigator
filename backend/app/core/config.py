"""Central configuration.

Every tunable value lives here and is read from environment variables (or a .env file).
Nothing else in the codebase should call os.getenv directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "Healthcare Knowledge Navigator"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    # --- LLM (model is configurable; GPT-3.5-Turbo is legacy) ---
    # Any OpenAI-compatible provider works: leave LLM_BASE_URL empty for OpenAI, or set e.g.
    # https://api.groq.com/openai/v1 (Groq, free tier) and put that provider's key below.
    openai_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=700, gt=0)
    llm_timeout_s: float = Field(default=30.0, gt=0)

    # --- Embeddings & vector store ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    chroma_collection: str = "clinical_docs"

    # --- Ingestion ---
    raw_docs_dir: Path = PROJECT_ROOT / "data" / "raw"
    chunk_size: int = Field(default=1000, gt=100)
    chunk_overlap: int = Field(default=150, ge=0)
    min_chunk_chars: int = Field(default=80, ge=0)  # drop tiny fragments (page numbers, headers)

    # --- Evaluation ---
    eval_dir: Path = PROJECT_ROOT / "eval"

    # --- Retrieval & confidence ---
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)
    high_conf_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    medium_conf_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    # --- Security ---
    admin_api_key: SecretStr | None = None  # required for /ingest endpoints
    rate_limit: str = Field(default="20/minute", pattern=r"^\d+/(second|minute|hour|day)$")
    trust_proxy_headers: bool = False  # use X-Forwarded-For for client IP (behind a proxy)
    max_upload_mb: int = Field(default=25, ge=1, le=200)
    # Index data/raw in the background at startup when the index is empty (fresh deploys).
    auto_ingest_on_startup: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        # Allow CORS_ORIGINS="http://a.com,http://b.com" in env files.
        if isinstance(v, str) and not v.strip().startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_lt_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 1000)
        if v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @field_validator("medium_conf_threshold")
    @classmethod
    def _medium_below_high(cls, v: float, info) -> float:
        high = info.data.get("high_conf_threshold", 0.75)
        if v >= high:
            raise ValueError("medium_conf_threshold must be below high_conf_threshold")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — use as a FastAPI dependency."""
    return Settings()
