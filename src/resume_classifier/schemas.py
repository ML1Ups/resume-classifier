from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Status = Literal["ok", "error"]
MetricName = Literal["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]


class HealthzResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str


class ComponentHealth(BaseModel):
    name: str
    status: Status
    version: str | None
    response_time_ms: float


class HealthResponse(BaseModel):
    status: Status
    components: list[ComponentHealth]


ResumeText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)
]


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: ResumeText
    model: str = Field(default="champion", min_length=1, max_length=64)


class TaskMessage(BaseModel):
    id: UUID
    text: str
    model: str
    created_at: datetime


class TaskResponse(BaseModel):
    id: UUID
    status: Literal["PENDING", "DONE"]
    model: str
    category: str | None
    model_version: str | None
    created_at: datetime
    finished_at: datetime | None


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="champion", min_length=1, max_length=64)
    dataset: str = Field(pattern=r"^v[0-9]+$")
    metrics: list[MetricName] = Field(
        default_factory=lambda: ["accuracy", "f1_macro"],
        min_length=1,
        max_length=2,
    )
    sample_size: int = Field(default=5, ge=0, le=20)

    @model_validator(mode="after")
    def validate_metrics(self) -> "EvaluationRequest":
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Metrics must be unique")
        return self


class Prediction(BaseModel):
    index: int
    category: str
    expected: str


class ModelInfo(BaseModel):
    name: str
    alias: str | None
    version: str
    run_id: str | None
    model_uri: str
    candidate: str | None = None
    dataset_version: str | None = None
    train_rows: int | None = None
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, float] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    name: str
    models: list[ModelInfo]


class DatasetInfo(BaseModel):
    version: str
    rows: int


class EvaluationResponse(BaseModel):
    model: ModelInfo
    dataset: DatasetInfo
    metrics: dict[str, float]
    predictions: list[Prediction]
