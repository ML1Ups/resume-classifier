from typing import Annotated

from fastapi import APIRouter, Depends

from resume_classifier.api.process import MODEL_NOT_LOADED, loaded_model
from resume_classifier.model import LoadedModel
from resume_classifier.schemas import ModelInfo

router = APIRouter(tags=["inference"])


@router.get("/model", responses=MODEL_NOT_LOADED)
async def model_info(model: Annotated[LoadedModel, Depends(loaded_model)]) -> ModelInfo:
    return model.info
