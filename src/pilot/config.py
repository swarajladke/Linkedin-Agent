"""Configuration settings for Pilot loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for Pilot."""

    # Database
    database_url: SecretStr = Field(
        default=SecretStr("postgresql+psycopg://pilot:pilot_password@localhost:5432/pilot"),
        description="Database connection URL",
    )

    # LLM Provider Configuration
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for structured claim extraction and goal compilation",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for structured output extraction",
    )
    critic_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for critic structured decomposition",
    )

    # External APIs
    github_token: SecretStr | None = Field(
        default=None,
        description="Optional personal GitHub token to avoid public rate-limiting",
    )

    # Application Environment
    pilot_env: str = Field(
        default="development",
        description="Runtime environment: development | test | production",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()
