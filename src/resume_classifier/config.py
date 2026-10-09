from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True

    postgres_host: str
    postgres_port: int = 5432
    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str
    postgres_pool_max_size: int = Field(default=5, ge=1)

    health_check_timeout_seconds: float = Field(default=2.0, gt=0)

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "resume-classifier-hw3"
    mlflow_model_name: str = "resume-classifier"
    mlflow_model_alias: str = "champion"
    inference_max_concurrency: int = Field(default=2, ge=1, le=32)


@lru_cache
def get_settings() -> Settings:
    return Settings()
