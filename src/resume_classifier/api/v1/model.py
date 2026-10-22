from fastapi import APIRouter, HTTPException, Request, status
from mlflow.exceptions import MlflowException
from starlette.concurrency import run_in_threadpool

from resume_classifier.model import list_models
from resume_classifier.schemas import ModelListResponse

router = APIRouter(tags=["inference"])


@router.get(
    "/model",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Model Registry is unavailable"
        }
    },
)
async def model_info(request: Request) -> ModelListResponse:
    try:
        models = await run_in_threadpool(
            list_models,
            request.app.state.settings,
        )
    except (MlflowException, OSError) as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model Registry is unavailable",
        ) from exc

    if not models:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No registered models",
        )
    return ModelListResponse(
        name=request.app.state.settings.mlflow_model_name,
        models=models,
    )
