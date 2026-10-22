from typing import Annotated, Literal

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


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="champion", min_length=1, max_length=64)
    texts: list[ResumeText] | None = Field(default=None, min_length=1, max_length=32)
    dataset: str | None = Field(default=None, min_length=2, max_length=32)
    metrics: list[MetricName] = Field(
        default_factory=lambda: ["accuracy", "f1_macro"],
        min_length=1,
        max_length=2,
    )
    sample_size: int = Field(default=5, ge=0, le=20)

    @model_validator(mode="after")
    def validate_input_source(self) -> "ProcessRequest":
        if (self.texts is None) == (self.dataset is None):
            raise ValueError("Provide exactly one of 'texts' or 'dataset'")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Metrics must be unique")
        return self


class Prediction(BaseModel):
    index: int
    category: str
    expected: str | None = None


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


class ProcessResponse(BaseModel):
    predictions: list[Prediction]
    model: ModelInfo
    dataset: DatasetInfo | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
