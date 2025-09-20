"""
Students Endpoints
Student profile management, progress tracking, and learning analytics
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_database, get_current_user
from app.models.user import User
from app.schemas.student import (
    StudentProfileResponse,
    StudentProfileUpdate,
    LearningAnalytics,
    ProgressSummary
)
from app.services.student_service import StudentService


router = APIRouter()


@router.get("/profile", response_model=StudentProfileResponse)
async def get_student_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Get student profile with learning analytics"""
    student_service = StudentService(db)
    profile = await student_service.get_student_profile(current_user.id)
    return profile


@router.put("/profile", response_model=StudentProfileResponse)
async def update_student_profile(
    profile_update: StudentProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Update student profile and preferences"""
    student_service = StudentService(db)
    updated_profile = await student_service.update_profile(current_user.id, profile_update)
    return updated_profile


@router.get("/analytics", response_model=LearningAnalytics)
async def get_learning_analytics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Get detailed learning analytics and insights"""
    student_service = StudentService(db)
    analytics = await student_service.get_learning_analytics(current_user.id)
    return analytics


@router.get("/progress", response_model=ProgressSummary)
async def get_progress_summary(
    days: Optional[int] = 30,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Get progress summary for specified time period"""
    student_service = StudentService(db)
    progress = await student_service.get_progress_summary(current_user.id, days)
    return progress
