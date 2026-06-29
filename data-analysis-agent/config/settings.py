"""Application settings loaded from environment variables and optional .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    llm_timeout_seconds: int = 60

    max_file_size_mb: float = 50.0
    max_row_count: int = 500_000
    max_column_count: int = 200
    sample_size: int = 50_000
    sample_threshold: int = 100_000

    code_execution_timeout: int = 30
    max_stdout_bytes: int = 51_200
    max_critique_iterations: int = 2
    critique_approval_threshold: float = 7.0

    log_level: str = "INFO"
    log_file: str = "agent_run.log"


settings = Settings()
