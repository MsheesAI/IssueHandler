"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (Groq)
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # GitHub
    github_token: str
    default_repo: Optional[str] = None
    allowed_repos: Optional[str] = None  # comma-separated list of allowed repos

    # Backend & security (leave api_key empty to disable the API-key check)
    api_key: Optional[str] = None
    database_url: str = f"sqlite+aiosqlite:///{(BASE_DIR / 'issue_agent.db').as_posix()}"
    checkpoint_db_path: str = str(BASE_DIR / "checkpoints.db")
    log_level: str = "INFO"
    require_close_approval: bool = True

    def is_repo_allowed(self, repo: str) -> bool:
        if not self.allowed_repos:
            return True
        allowed = [r.strip().lower() for r in self.allowed_repos.split(",") if r.strip()]
        return repo.lower() in allowed if allowed else True


@lru_cache
def get_settings() -> Settings:
    return Settings()