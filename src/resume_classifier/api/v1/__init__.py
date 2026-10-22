from fastapi import APIRouter

from resume_classifier.api.v1 import evaluations, health, model, tasks, version

router = APIRouter(prefix="/api/v1")
router.include_router(version.router)
router.include_router(health.router)
router.include_router(model.router)
router.include_router(tasks.router)
router.include_router(evaluations.router)
