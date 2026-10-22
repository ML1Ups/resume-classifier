from fastapi import APIRouter, HTTPException, Request, status
from mlflow.exceptions import MlflowException
from starlette.concurrency import run_in_threadpool

from resume_classifier.model import list_models
from resume_classifier.schemas import ModelListResponse

router = APIRouter(tags=["inference"])


REGISTRY_UNAVAILABLE = "Model Registry is unavailable"


@router.get(
    "/model",
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": REGISTRY_UNAVAILABLE}},
)
async def registered_models(request: Request) -> ModelListResponse:
    settings = request.app.state.settings
    try:
        models = await run_in_threadpool(list_models, settings)
    except (MlflowException, OSError) as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, REGISTRY_UNAVAILABLE) from exc
    return ModelListResponse(name=settings.mlflow_model_name, models=models)
