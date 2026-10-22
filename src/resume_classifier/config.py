import math
from functools import lru_cache
from pathlib import Path
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

    mlflow_tracking_uri: str = "http://127.0.0.1:5050"
    mlflow_model_name: str = "resume-classifier"
    mlflow_model_alias: str = "champion"
    evaluation_data_dir: Path = Path("data/evaluation")

    rabbitmq_url: SecretStr
    tasks_queue: str = "tasks"
    queue_max_length: int = Field(default=100, ge=0)
    delivery_limit: int = Field(default=3, ge=1)
    worker_threads: int = Field(default=4, ge=1)
    worker_prefetch: int = Field(default=4, ge=0)
    inference_seconds: float = Field(default=0.5, ge=0)

    @property
    def retry_after_seconds(self) -> int:
        return math.ceil(self.queue_max_length * self.inference_seconds / self.worker_threads)


@lru_cache
def get_settings() -> Settings:
    return Settings()
