"""
Analytics Worker
Background processing for analytics computation and insights
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import uuid

from app.workers.celery_app import celery_app
from app.database import db_manager


logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    queue="analytics",
    max_retries=2
)
def compute_student_analytics(self, student_id: str) -> Dict[str, Any]:
    """
    Compute comprehensive analytics for a student
    
    Args:
        student_id: Student UUID
        
    Returns:
        Analytics data
    """
    try:
        logger.info(f"Computing analytics for student {student_id}")
        
        async def compute_analytics():
            async with db_manager.get_postgres_session() as session:
                from app.services.student_service import get_student_service
                from app.database import get_cache_manager
                
                cache_manager = await get_cache_manager()
                student_service = get_student_service(cache_manager)
                
                analytics = await student_service.compute_analytics(
                    student_id=uuid.UUID(student_id),
                    session=session
                )
                
                return analytics
        
        result = asyncio.run(compute_analytics())
        
        logger.info(f"Analytics computed for student {student_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to compute analytics for student {student_id}: {exc}", exc_info=True)
        
        # Retry on temporary errors
        if isinstance(exc, (ConnectionError, TimeoutError)):
            raise self.retry(exc=exc, countdown=60)
        
        return {
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(queue="analytics")
def compute_system_metrics() -> Dict[str, Any]:
    """
    Compute system-wide metrics and performance indicators
    
    Returns:
        System metrics
    """
    try:
        logger.info("Computing system metrics")
        
        async def compute_metrics():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select, func
                from app.models import Doubt, Student
                
                # Date ranges
                now = datetime.utcnow()
                today = now.replace(hour=0, minute=0, second=0, microsecond=0)
                week_ago = today - timedelta(days=7)
                month_ago = today - timedelta(days=30)
                
                # Daily metrics
                daily_metrics = await session.execute(
                    select(
                        func.count(Doubt.id).label('total_doubts'),
                        func.count(Doubt.id).filter(Doubt.is_resolved).label('resolved_doubts'),
                        func.avg(Doubt.resolution_time_ms).label('avg_resolution_time'),
                        func.count(Student.id.distinct()).label('active_students')
                    )
                    .select_from(Doubt)
                    .outerjoin(Student)
                    .where(Doubt.created_at >= today)
                )
                
                daily_row = daily_metrics.first()
                
                # Weekly metrics
                weekly_metrics = await session.execute(
                    select(
                        func.count(Doubt.id).label('total_doubts'),
                        func.count(Doubt.id).filter(Doubt.is_resolved).label('resolved_doubts'),
                        func.count(Student.id.distinct()).label('active_students')
                    )
                    .select_from(Doubt)
                    .outerjoin(Student)
                    .where(Doubt.created_at >= week_ago)
                )
                
                weekly_row = weekly_metrics.first()
                
                # Monthly metrics  
                monthly_metrics = await session.execute(
                    select(
                        func.count(Doubt.id).label('total_doubts'),
                        func.count(Doubt.id).filter(Doubt.is_resolved).label('resolved_doubts'),
                        func.count(Student.id.distinct()).label('active_students')
                    )
                    .select_from(Doubt)
                    .outerjoin(Student)
                    .where(Doubt.created_at >= month_ago)
                )
                
                monthly_row = monthly_metrics.first()
                
                return {
                    "daily": {
                        "total_doubts": daily_row.total_doubts or 0,
                        "resolved_doubts": daily_row.resolved_doubts or 0,
                        "resolution_rate": round((daily_row.resolved_doubts or 0) / max(daily_row.total_doubts or 1, 1) * 100, 2),
                        "avg_resolution_time_ms": int(daily_row.avg_resolution_time or 0),
                        "active_students": daily_row.active_students or 0
                    },
                    "weekly": {
                        "total_doubts": weekly_row.total_doubts or 0,
                        "resolved_doubts": weekly_row.resolved_doubts or 0,
                        "resolution_rate": round((weekly_row.resolved_doubts or 0) / max(weekly_row.total_doubts or 1, 1) * 100, 2),
                        "active_students": weekly_row.active_students or 0
                    },
                    "monthly": {
                        "total_doubts": monthly_row.total_doubts or 0,
                        "resolved_doubts": monthly_row.resolved_doubts or 0,
                        "resolution_rate": round((monthly_row.resolved_doubts or 0) / max(monthly_row.total_doubts or 1, 1) * 100, 2),
                        "active_students": monthly_row.active_students or 0
                    },
                    "timestamp": now.isoformat()
                }
        
        result = asyncio.run(compute_metrics())
        
        logger.info("System metrics computed successfully")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to compute system metrics: {exc}", exc_info=True)
        return {
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(queue="analytics") 
def generate_performance_report(period_days: int = 7) -> Dict[str, Any]:
    """
    Generate detailed performance report
    
    Args:
        period_days: Report period in days
        
    Returns:
        Performance report
    """
    try:
        logger.info(f"Generating performance report for {period_days} days")
        
        async def generate_report():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import text
                
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=period_days)
                
                # Performance by hour
                hourly_performance = await session.execute(
                    text("""
                        SELECT 
                            EXTRACT(hour FROM created_at) as hour,
                            COUNT(*) as total_doubts,
                            COUNT(CASE WHEN is_resolved THEN 1 END) as resolved_doubts,
                            AVG(resolution_time_ms) as avg_resolution_time
                        FROM doubts 
                        WHERE created_at >= :start_date AND created_at <= :end_date
                        GROUP BY EXTRACT(hour FROM created_at)
                        ORDER BY hour
                    """),
                    {"start_date": start_date, "end_date": end_date}
                )
                
                hourly_data = []
                for row in hourly_performance:
                    resolution_rate = (row.resolved_doubts / row.total_doubts * 100) if row.total_doubts > 0 else 0
                    hourly_data.append({
                        "hour": int(row.hour),
                        "total_doubts": row.total_doubts,
                        "resolved_doubts": row.resolved_doubts,
                        "resolution_rate": round(resolution_rate, 2),
                        "avg_resolution_time_ms": int(row.avg_resolution_time or 0)
                    })
                
                # Daily trends
                daily_trends = await session.execute(
                    text("""
                        SELECT 
                            DATE(created_at) as date,
                            COUNT(*) as total_doubts,
                            COUNT(CASE WHEN is_resolved THEN 1 END) as resolved_doubts
                        FROM doubts 
                        WHERE created_at >= :start_date AND created_at <= :end_date
                        GROUP BY DATE(created_at)
                        ORDER BY date
                    """),
                    {"start_date": start_date, "end_date": end_date}
                )
                
                trends_data = []
                for row in daily_trends:
                    resolution_rate = (row.resolved_doubts / row.total_doubts * 100) if row.total_doubts > 0 else 0
                    trends_data.append({
                        "date": row.date.isoformat(),
                        "total_doubts": row.total_doubts,
                        "resolved_doubts": row.resolved_doubts,
                        "resolution_rate": round(resolution_rate, 2)
                    })
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": period_days
                    },
                    "hourly_performance": hourly_data,
                    "daily_trends": trends_data,
                    "summary": {
                        "peak_hour": max(hourly_data, key=lambda x: x["total_doubts"])["hour"] if hourly_data else None,
                        "total_period_doubts": sum(d["total_doubts"] for d in trends_data),
                        "overall_resolution_rate": round(
                            sum(d["resolved_doubts"] for d in trends_data) / 
                            max(sum(d["total_doubts"] for d in trends_data), 1) * 100, 2
                        ) if trends_data else 0
                    },
                    "generated_at": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(generate_report())
        
        logger.info(f"Performance report generated for {period_days} days")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to generate performance report: {exc}", exc_info=True)
        return {
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(queue="analytics")
def update_student_recommendations(student_id: str) -> Dict[str, Any]:
    """
    Update personalized recommendations for a student
    
    Args:
        student_id: Student UUID
        
    Returns:
        Updated recommendations
    """
    try:
        logger.info(f"Updating recommendations for student {student_id}")
        
        async def update_recommendations():
            async with db_manager.get_postgres_session() as session:
                from app.services.student_service import get_student_service
                from app.database import get_cache_manager
                
                cache_manager = await get_cache_manager()
                student_service = get_student_service(cache_manager)
                
                recommendations = await student_service.generate_recommendations(
                    student_id=uuid.UUID(student_id),
                    session=session
                )
                
                return recommendations
        
        result = asyncio.run(update_recommendations())
        
        logger.info(f"Recommendations updated for student {student_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to update recommendations for student {student_id}: {exc}", exc_info=True)
        return {
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


# Health check
@celery_app.task(queue="analytics")
def analytics_health_check() -> Dict[str, Any]:
    """Health check for analytics worker"""
    
    try:
        async def check_health():
            async with db_manager.get_postgres_session() as session:
                # Test database connectivity
                await session.execute("SELECT 1")
                
                # Test basic analytics computation
                from sqlalchemy import select, func
                from app.models import Doubt
                
                count_result = await session.execute(
                    select(func.count(Doubt.id))
                )
                doubt_count = count_result.scalar()
                
                return {
                    "status": "healthy",
                    "database": "ok",
                    "total_doubts": doubt_count,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(check_health())
        return result
        
    except Exception as exc:
        logger.error(f"Analytics health check failed: {exc}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


# Export tasks
__all__ = [
    "compute_student_analytics",
    "compute_system_metrics", 
    "generate_performance_report",
    "update_student_recommendations",
    "analytics_health_check"
]
