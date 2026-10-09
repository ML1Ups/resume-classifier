import structlog
from fastapi import APIRouter, HTTPException, Request, status

from resume_classifier.schemas import ModelInfo, ProcessRequest, ProcessResponse

router = APIRouter(tags=["inference"])
logger = structlog.get_logger(__name__)


@router.post("/process")
async def process(payload: ProcessRequest, request: Request) -> ProcessResponse:
    try:
        return await request.app.state.model_service.predict(payload.texts)
    except Exception as exc:
        logger.exception("inference_failed", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed",
        ) from exc


@router.get("/api/v1/model")
async def model_info(request: Request) -> ModelInfo:
    return request.app.state.model_service.info
