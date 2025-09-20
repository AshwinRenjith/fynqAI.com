"""
Feedback Endpoints
User feedback collection and processing
"""

from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_database, get_current_user
from app.models.user import User
from app.schemas.feedback import FeedbackSubmission, FeedbackResponse
from app.services.feedback_service import FeedbackService


router = APIRouter()


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    feedback_data: FeedbackSubmission,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Submit feedback for a doubt response"""
    feedback_service = FeedbackService(db)
    feedback = await feedback_service.create_feedback(current_user.id, feedback_data)
    return feedback


@router.get("/", response_model=List[FeedbackResponse])
async def get_user_feedback(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Get user's feedback history"""
    feedback_service = FeedbackService(db)
    feedback_list = await feedback_service.get_user_feedback(current_user.id)
    return feedback_list
