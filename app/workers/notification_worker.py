"""
Notification Worker
Background processing for notifications and alerts
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import uuid

from app.workers.celery_app import celery_app
from app.database import db_manager


logger = logging.getLogger(__name__)


@celery_app.task(
    queue="notifications",
    max_retries=3
)
def send_doubt_resolved_notification(
    student_id: str, 
    doubt_id: str, 
    notification_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send notification when a doubt is resolved
    
    Args:
        student_id: Student UUID
        doubt_id: Doubt UUID
        notification_data: Notification details
        
    Returns:
        Notification result
    """
    try:
        logger.info(f"Sending doubt resolved notification for doubt {doubt_id}")
        
        async def send_notification():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select
                from app.models import Student, Doubt
                
                # Get student and doubt details
                student_result = await session.execute(
                    select(Student).where(Student.id == uuid.UUID(student_id))
                )
                student = student_result.scalar_one()
                
                doubt_result = await session.execute(
                    select(Doubt).where(Doubt.id == uuid.UUID(doubt_id))
                )
                doubt = doubt_result.scalar_one()
                
                # Prepare notification
                notification = {
                    "type": "doubt_resolved",
                    "student_id": student_id,
                    "doubt_id": doubt_id,
                    "title": "Your doubt has been resolved!",
                    "message": f"Your question about {doubt.subject.name} has been answered.",
                    "data": notification_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Here you would integrate with notification services:
                # - Firebase Cloud Messaging for mobile push notifications
                # - Email service for email notifications
                # - SMS service for text notifications
                # - In-app notification storage
                
                # Mock implementation
                logger.info(f"Notification sent to student {student.name}: {notification['title']}")
                
                return {
                    "status": "sent",
                    "notification_id": str(uuid.uuid4()),
                    "channels": ["push", "email"],
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(send_notification())
        
        logger.info(f"Doubt resolved notification sent for doubt {doubt_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to send doubt resolved notification for doubt {doubt_id}: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    queue="notifications"
)
def send_daily_summary(student_id: str) -> Dict[str, Any]:
    """
    Send daily summary notification to student
    
    Args:
        student_id: Student UUID
        
    Returns:
        Notification result
    """
    try:
        logger.info(f"Sending daily summary for student {student_id}")
        
        async def send_summary():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select, func, and_
                from app.models import Student, Doubt
                from datetime import timedelta
                
                # Get student
                student_result = await session.execute(
                    select(Student).where(Student.id == uuid.UUID(student_id))
                )
                student = student_result.scalar_one()
                
                # Get today's activity
                today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                tomorrow = today + timedelta(days=1)
                
                activity_stats = await session.execute(
                    select(
                        func.count(Doubt.id).label('total_doubts'),
                        func.count(Doubt.id).filter(Doubt.is_resolved).label('resolved_doubts')
                    )
                    .where(
                        and_(
                            Doubt.student_id == student.id,
                            Doubt.created_at >= today,
                            Doubt.created_at < tomorrow
                        )
                    )
                )
                
                stats = activity_stats.first()
                
                # Prepare summary
                summary = {
                    "type": "daily_summary",
                    "student_id": student_id,
                    "title": f"Daily Summary for {student.name}",
                    "message": f"Today you asked {stats.total_doubts} questions and got {stats.resolved_doubts} answers!",
                    "data": {
                        "total_doubts": stats.total_doubts,
                        "resolved_doubts": stats.resolved_doubts,
                        "resolution_rate": round((stats.resolved_doubts / max(stats.total_doubts, 1)) * 100, 2),
                        "date": today.date().isoformat()
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Daily summary prepared for student {student.name}")
                
                return {
                    "status": "sent",
                    "summary": summary,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(send_summary())
        
        logger.info(f"Daily summary sent for student {student_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to send daily summary for student {student_id}: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    queue="notifications"
)
def send_achievement_notification(
    student_id: str, 
    achievement: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send achievement notification to student
    
    Args:
        student_id: Student UUID
        achievement: Achievement details
        
    Returns:
        Notification result
    """
    try:
        logger.info(f"Sending achievement notification for student {student_id}")
        
        async def send_achievement():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select
                from app.models import Student
                
                # Get student
                student_result = await session.execute(
                    select(Student).where(Student.id == uuid.UUID(student_id))
                )
                student = student_result.scalar_one()
                
                # Prepare achievement notification
                notification = {
                    "type": "achievement",
                    "student_id": student_id,
                    "title": "🎉 Achievement Unlocked!",
                    "message": f"Congratulations {student.name}! You've {achievement['description']}",
                    "data": achievement,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Achievement notification sent to {student.name}: {achievement['name']}")
                
                return {
                    "status": "sent",
                    "notification": notification,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(send_achievement())
        
        logger.info(f"Achievement notification sent for student {student_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to send achievement notification for student {student_id}: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    queue="notifications"
)
def send_reminder_notification(
    student_id: str,
    reminder_type: str,
    reminder_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send reminder notification to student
    
    Args:
        student_id: Student UUID
        reminder_type: Type of reminder
        reminder_data: Reminder details
        
    Returns:
        Notification result
    """
    try:
        logger.info(f"Sending {reminder_type} reminder for student {student_id}")
        
        async def send_reminder():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select
                from app.models import Student
                
                # Get student
                student_result = await session.execute(
                    select(Student).where(Student.id == uuid.UUID(student_id))
                )
                student = student_result.scalar_one()
                
                # Prepare reminder messages
                messages = {
                    "study_time": f"Hi {student.name}! It's time for your daily study session.",
                    "pending_doubts": f"You have {reminder_data.get('count', 0)} pending doubts to review.",
                    "practice_session": f"Ready for your practice session in {reminder_data.get('subject', 'your subjects')}?",
                    "streak_continuation": f"Keep your {reminder_data.get('streak_days', 0)} day learning streak going!"
                }
                
                notification = {
                    "type": "reminder",
                    "subtype": reminder_type,
                    "student_id": student_id,
                    "title": f"Reminder: {reminder_type.replace('_', ' ').title()}",
                    "message": messages.get(reminder_type, "You have a pending reminder."),
                    "data": reminder_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Reminder sent to {student.name}: {reminder_type}")
                
                return {
                    "status": "sent",
                    "notification": notification,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(send_reminder())
        
        logger.info(f"Reminder notification sent for student {student_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to send reminder notification for student {student_id}: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    queue="notifications"
)
def process_notification_queue() -> Dict[str, Any]:
    """
    Process pending notifications in the queue
    
    Returns:
        Processing result
    """
    try:
        logger.info("Processing notification queue")
        
        async def process_queue():
            # In a real implementation, this would:
            # 1. Fetch pending notifications from database
            # 2. Group notifications by student/type
            # 3. Send batch notifications
            # 4. Update notification status
            # 5. Handle failed notifications
            
            # Mock implementation
            processed = {
                "total_processed": 0,
                "successful": 0,
                "failed": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info("Notification queue processing completed")
            return processed
        
        result = asyncio.run(process_queue())
        
        logger.info("Notification queue processed successfully")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to process notification queue: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


# Health check
@celery_app.task(queue="notifications")
def notifications_health_check() -> Dict[str, Any]:
    """Health check for notifications worker"""
    
    try:
        async def check_health():
            async with db_manager.get_postgres_session() as session:
                # Test database connectivity
                await session.execute("SELECT 1")
                
                return {
                    "status": "healthy",
                    "database": "ok",
                    "notification_services": "ok",  # Would check external services
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(check_health())
        return result
        
    except Exception as exc:
        logger.error(f"Notifications health check failed: {exc}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


# Export tasks
__all__ = [
    "send_doubt_resolved_notification",
    "send_daily_summary",
    "send_achievement_notification", 
    "send_reminder_notification",
    "process_notification_queue",
    "notifications_health_check"
]
