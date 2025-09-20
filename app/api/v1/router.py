"""
API v1 Router
Main router that aggregates all API v1 endpoints
"""

from fastapi import APIRouter

from app.api.v1 import auth, students, doubts, feedback, analytics, enterprise, health


# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    students.router,
    prefix="/students",
    tags=["Students"]
)

api_router.include_router(
    doubts.router,
    prefix="/doubts",
    tags=["Doubts"]
)

api_router.include_router(
    feedback.router,
    prefix="/feedback",
    tags=["Feedback"]
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["Analytics"]
)

api_router.include_router(
    enterprise.router,
    prefix="/enterprise",
    tags=["Enterprise"]
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)
