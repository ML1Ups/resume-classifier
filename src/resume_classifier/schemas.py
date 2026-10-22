from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Status = Literal["ok", "error"]


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


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="champion", min_length=1, max_length=64)
    texts: list[ResumeText] = Field(min_length=1, max_length=32)


class Prediction(BaseModel):
    index: int
    category: str


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


class ProcessResponse(BaseModel):
    predictions: list[Prediction]
    model: ModelInfo
