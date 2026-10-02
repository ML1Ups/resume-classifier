from typing import Literal

from pydantic import BaseModel

Status = Literal["ok", "error"]


class HealthzResponse(BaseModel):
    status: Literal["hello world"]


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
