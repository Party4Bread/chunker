"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNKER_", env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path("./data"))
    vllm_base_url: str = Field(default="http://127.0.0.1:8001")
    vllm_model: str = Field(default="p4b/qwen3.5-4b-chunky-FP8")
    prompt_path: Path = Field(default=Path(__file__).resolve().parent.parent.parent / "prompt.txt")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    enable_splitter: bool = Field(default=True)
    max_tokens: int = Field(default=512)
    request_timeout_seconds: float = Field(default=180.0)
    # None = auto-detect from vLLM /v1/models on first call.
    vllm_max_model_len: int | None = Field(default=None)
    context_safety_margin: int = Field(default=16)
    min_output_tokens: int = Field(default=32)

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "_registry.sqlite"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    return settings
