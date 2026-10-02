from fastapi import APIRouter

from resume_classifier.schemas import HealthzResponse

router = APIRouter(tags=["infrastructure"])


@router.get("/healthz")
async def healthz() -> HealthzResponse:
    return HealthzResponse(status="hello world")
