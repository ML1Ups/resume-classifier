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

    texts: list[ResumeText] = Field(min_length=1, max_length=32)


class Prediction(BaseModel):
    index: int
    category: str


class ModelInfo(BaseModel):
    name: str
    alias: str
    version: str
    run_id: str | None
    model_uri: str


class ProcessResponse(BaseModel):
    predictions: list[Prediction]
    model: ModelInfo
