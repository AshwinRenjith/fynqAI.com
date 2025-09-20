"""
Analytics Endpoints
Usage analytics, performance metrics, and insights
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_database, get_current_user
from app.models.user import User
from app.schemas.analytics import (
    UsageAnalytics,
    PerformanceMetrics,
    LearningInsights
)
from app.services.analytics_service import AnalyticsService


router = APIRouter()


@router.get("/usage", response_model=UsageAnalytics)
async def get_usage_analytics(
    days: Optional[int] = 30,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Get usage analytics for specified period"""
    analytics_service = AnalyticsService(db)
    analytics = await analytics_service.get_usage_analytics(current_user.id, days)
    return analytics


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Get performance metrics and learning velocity"""
    analytics_service = AnalyticsService(db)
    metrics = await analytics_service.get_performance_metrics(current_user.id)
    return metrics


@router.get("/insights", response_model=LearningInsights)
async def get_learning_insights(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """Get AI-powered learning insights and recommendations"""
    analytics_service = AnalyticsService(db)
    insights = await analytics_service.get_learning_insights(current_user.id)
    return insights
