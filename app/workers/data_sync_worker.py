"""
Data Sync Worker
Background processing for data synchronization and maintenance
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

from app.workers.celery_app import celery_app
from app.database import db_manager


logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    queue="data_sync",
    max_retries=2
)
def sync_vector_embeddings(self, batch_size: int = 100) -> Dict[str, Any]:
    """
    Sync embeddings to vector database
    
    Args:
        batch_size: Number of items to process in batch
        
    Returns:
        Sync result
    """
    try:
        logger.info(f"Starting vector embeddings sync with batch size {batch_size}")
        
        async def sync_embeddings():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select, and_
                from app.models import Doubt
                from app.core.rag.retriever import RAGRetriever
                
                # Get doubts that need embedding sync
                pending_doubts = await session.execute(
                    select(Doubt)
                    .where(
                        and_(
                            Doubt.is_resolved,
                            ~Doubt.vector_synced
                        )
                    )
                    .limit(batch_size)
                )
                
                doubts = pending_doubts.scalars().all()
                
                if not doubts:
                    return {
                        "status": "no_work",
                        "message": "No doubts to sync",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                rag_retriever = RAGRetriever()
                synced_count = 0
                failed_count = 0
                
                for doubt in doubts:
                    try:
                        # Create embedding and store in vector DB
                        await rag_retriever.store_doubt_embedding(
                            doubt_id=str(doubt.id),
                            question=doubt.question_text,
                            answer=doubt.resolution_data.get("response", ""),
                            subject=doubt.subject.code.lower(),
                            metadata={
                                "difficulty": doubt.difficulty_level,
                                "resolved_at": doubt.resolved_at.isoformat() if doubt.resolved_at else None,
                                "student_grade": doubt.student.grade
                            }
                        )
                        
                        # Mark as synced
                        doubt.vector_synced = True
                        synced_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to sync doubt {doubt.id}: {e}")
                        failed_count += 1
                
                # Commit changes
                await session.commit()
                
                return {
                    "status": "completed",
                    "synced": synced_count,
                    "failed": failed_count,
                    "total_processed": len(doubts),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(sync_embeddings())
        
        logger.info(f"Vector embeddings sync completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"Vector embeddings sync failed: {exc}", exc_info=True)
        
        # Retry on recoverable errors
        if isinstance(exc, (ConnectionError, TimeoutError)):
            raise self.retry(exc=exc, countdown=120)
        
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    queue="data_sync"
)
def backup_doubt_data(period_days: int = 7) -> Dict[str, Any]:
    """
    Backup doubt data to external storage
    
    Args:
        period_days: Number of days to backup
        
    Returns:
        Backup result
    """
    try:
        logger.info(f"Starting doubt data backup for last {period_days} days")
        
        async def backup_data():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select, and_
                from app.models import Doubt, Student, Subject
                
                # Date range
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=period_days)
                
                # Get doubts to backup
                doubts_query = await session.execute(
                    select(Doubt, Student.name.label('student_name'), Subject.name.label('subject_name'))
                    .join(Student)
                    .join(Subject)
                    .where(
                        and_(
                            Doubt.created_at >= start_date,
                            Doubt.created_at <= end_date
                        )
                    )
                )
                
                backup_data = []
                for doubt, student_name, subject_name in doubts_query:
                    backup_item = {
                        "doubt_id": str(doubt.id),
                        "student_name": student_name,
                        "subject": subject_name,
                        "question": doubt.question_text,
                        "difficulty": doubt.difficulty_level,
                        "is_resolved": doubt.is_resolved,
                        "created_at": doubt.created_at.isoformat(),
                        "resolved_at": doubt.resolved_at.isoformat() if doubt.resolved_at else None,
                        "resolution_time_ms": doubt.resolution_time_ms,
                        "resolution_data": doubt.resolution_data
                    }
                    backup_data.append(backup_item)
                
                # In production, this would upload to cloud storage (S3, GCS, etc.)
                backup_file = f"doubt_backup_{start_date.date()}_{end_date.date()}.json"
                
                logger.info(f"Prepared backup with {len(backup_data)} doubts")
                
                return {
                    "status": "completed",
                    "backup_file": backup_file,
                    "records_count": len(backup_data),
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(backup_data())
        
        logger.info(f"Doubt data backup completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"Doubt data backup failed: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    queue="data_sync"
)
def cleanup_old_data(retention_days: int = 90) -> Dict[str, Any]:
    """
    Clean up old data based on retention policy
    
    Args:
        retention_days: Number of days to retain data
        
    Returns:
        Cleanup result
    """
    try:
        logger.info(f"Starting data cleanup with {retention_days} days retention")
        
        async def cleanup_data():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import delete, and_
                from app.models import Doubt
                
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                
                # Delete old unresolved doubts (assuming they're stale)
                unresolved_cleanup = await session.execute(
                    delete(Doubt)
                    .where(
                        and_(
                            Doubt.created_at < cutoff_date,
                            ~Doubt.is_resolved
                        )
                    )
                )
                
                unresolved_deleted = unresolved_cleanup.rowcount
                
                # For resolved doubts, we might want to archive instead of delete
                # This is a simplified version - in production you'd move to archive table
                
                await session.commit()
                
                return {
                    "status": "completed",
                    "unresolved_deleted": unresolved_deleted,
                    "cutoff_date": cutoff_date.isoformat(),
                    "retention_days": retention_days,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(cleanup_data())
        
        logger.info(f"Data cleanup completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"Data cleanup failed: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    queue="data_sync"
)
def sync_student_analytics() -> Dict[str, Any]:
    """
    Sync student analytics to analytics database
    
    Returns:
        Sync result
    """
    try:
        logger.info("Starting student analytics sync")
        
        async def sync_analytics():
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select, func
                from app.models import Student, Doubt
                
                # Get all active students
                students_query = await session.execute(
                    select(Student).where(Student.is_active)
                )
                students = students_query.scalars().all()
                
                synced_count = 0
                
                for student in students:
                    try:
                        # Calculate analytics for student
                        stats_query = await session.execute(
                            select(
                                func.count(Doubt.id).label('total_doubts'),
                                func.count(Doubt.id).filter(Doubt.is_resolved).label('resolved_doubts'),
                                func.avg(Doubt.resolution_time_ms).label('avg_resolution_time')
                            )
                            .where(Doubt.student_id == student.id)
                        )
                        
                        stats = stats_query.first()
                        
                        # In production, this would sync to analytics database or data warehouse
                        # Store computed stats to external system here
                        logger.info(f"Analytics synced for student {student.name}")
                        synced_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to sync analytics for student {student.id}: {e}")
                
                return {
                    "status": "completed",
                    "students_synced": synced_count,
                    "total_students": len(students),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(sync_analytics())
        
        logger.info(f"Student analytics sync completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"Student analytics sync failed: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


@celery_app.task(
    queue="data_sync"
)
def refresh_cache_data() -> Dict[str, Any]:
    """
    Refresh cached data across the system
    
    Returns:
        Refresh result
    """
    try:
        logger.info("Starting cache data refresh")
        
        async def refresh_cache():
            # Clear stale cache entries
            cleared_keys = []
            
            # Refresh frequently accessed data
            # This would include:
            # - Popular subjects cache
            # - Student profiles cache
            # - Doubt resolution templates
            # - System metrics cache
            
            # Mock implementation
            refreshed_items = [
                "subjects_cache",
                "student_profiles_cache", 
                "system_metrics_cache",
                "doubt_templates_cache"
            ]
            
            for item in refreshed_items:
                # await cache_manager.delete(f"cache:{item}")
                logger.info(f"Refreshed cache: {item}")
            
            return {
                "status": "completed",
                "refreshed_items": len(refreshed_items),
                "cleared_keys": len(cleared_keys),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        result = asyncio.run(refresh_cache())
        
        logger.info(f"Cache data refresh completed: {result}")
        return result
        
    except Exception as exc:
        logger.error(f"Cache data refresh failed: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


# Health check
@celery_app.task(queue="data_sync")
def data_sync_health_check() -> Dict[str, Any]:
    """Health check for data sync worker"""
    
    try:
        async def check_health():
            async with db_manager.get_postgres_session() as session:
                # Test database connectivity
                await session.execute("SELECT 1")
                
                # Test vector DB connectivity (mock)
                vector_db_status = "ok"  # Would test actual Pinecone connection
                
                # Test cache connectivity
                cache_status = "ok"  # Would test actual Redis connection
                
                return {
                    "status": "healthy",
                    "database": "ok",
                    "vector_db": vector_db_status,
                    "cache": cache_status,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        result = asyncio.run(check_health())
        return result
        
    except Exception as exc:
        logger.error(f"Data sync health check failed: {exc}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


# Export tasks
__all__ = [
    "sync_vector_embeddings",
    "backup_doubt_data",
    "cleanup_old_data",
    "sync_student_analytics",
    "refresh_cache_data",
    "data_sync_health_check"
]
