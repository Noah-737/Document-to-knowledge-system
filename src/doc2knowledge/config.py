from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DOC2KNOWLEDGE_",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    embedding_dimensions: int = 1024
    llm_model: str = "gemma-4-31b-it"
    top_k: int = Field(default=6, ge=1, le=50)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GEMINI_API_KEY",
            "DOC2KNOWLEDGE_GEMINI_API_KEY",
        ),
    )
