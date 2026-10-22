from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from resume_classifier.model import LoadedModel, predict
from resume_classifier.schemas import ProcessRequest, ProcessResponse

router = APIRouter(tags=["inference"])


def loaded_models(request: Request) -> dict[str, LoadedModel]:
    models: dict[str, LoadedModel] = request.app.state.models
    if not models:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="No models are loaded")
    return models


@router.post(
    "/process",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Model alias is not loaded"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "No models are loaded"},
    },
)
async def process(
    payload: ProcessRequest,
    models: Annotated[dict[str, LoadedModel], Depends(loaded_models)],
) -> ProcessResponse:
    if payload.model not in models:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"Model alias '{payload.model}' is not loaded. "
            f"Available aliases: {', '.join(sorted(models))}",
        )
    model = models[payload.model]
    predictions = await run_in_threadpool(predict, model, payload.texts)
    return ProcessResponse(predictions=predictions, model=model.info)
