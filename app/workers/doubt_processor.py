"""
Doubt Processing Worker
Background processing for AI-powered doubt resolution
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import uuid

from app.workers.celery_app import celery_app, PRIORITY_HIGH
from app.services.doubt_service import get_doubt_processing_service
from app.database import db_manager


logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    queue="doubt_processing",
    priority=PRIORITY_HIGH,
    max_retries=2,
    default_retry_delay=30
)
def process_doubt_task(self, doubt_id: str) -> Dict[str, Any]:
    """
    Process a doubt using AI pipeline
    
    Args:
        doubt_id: UUID of the doubt to process
        
    Returns:
        Processing result
    """
    try:
        logger.info(f"Starting doubt processing for doubt_id: {doubt_id}")
        
        # Convert string back to UUID
        doubt_uuid = uuid.UUID(doubt_id)
        
        # Get database session
        async def process_doubt():
            async with db_manager.get_postgres_session() as session:
                doubt_service = await get_doubt_processing_service()
                
                # Process the doubt
                result = await doubt_service._process_doubt_async(doubt_uuid, session)
                
                return {
                    "doubt_id": doubt_id,
                    "status": "completed",
                    "processing_time": result.get("processing_time_ms", 0),
                    "provider": result.get("provider"),
                    "tokens_used": result.get("tokens_used", 0),
                    "cost": result.get("cost", 0.0)
                }
        
        # Run the async function
        result = asyncio.run(process_doubt())
        
        logger.info(f"Successfully processed doubt {doubt_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Error processing doubt {doubt_id}: {exc}", exc_info=True)
        
        # Retry on recoverable errors
        if isinstance(exc, (ConnectionError, TimeoutError)):
            logger.info(f"Retrying doubt {doubt_id} due to {type(exc).__name__}")
            raise self.retry(exc=exc, countdown=60)
        
        # Mark as failed for non-recoverable errors
        return {
            "doubt_id": doubt_id,
            "status": "failed",
            "error": str(exc)
        }


@celery_app.task(
    bind=True,
    queue="doubt_processing",
    priority=PRIORITY_HIGH
)
def batch_process_doubts(self, doubt_ids: list) -> Dict[str, Any]:
    """
    Process multiple doubts in batch
    
    Args:
        doubt_ids: List of doubt IDs to process
        
    Returns:
        Batch processing results
    """
    try:
        logger.info(f"Starting batch processing for {len(doubt_ids)} doubts")
        
        results = {
            "total_doubts": len(doubt_ids),
            "processed": 0,
            "failed": 0,
            "results": []
        }
        
        # Process each doubt
        for doubt_id in doubt_ids:
            try:
                # Queue individual processing task
                task_result = process_doubt_task.delay(doubt_id)
                
                # Wait for result with timeout
                result = task_result.get(timeout=300)  # 5 minutes
                
                results["results"].append(result)
                results["processed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to process doubt {doubt_id}: {e}")
                results["failed"] += 1
                results["results"].append({
                    "doubt_id": doubt_id,
                    "status": "failed",
                    "error": str(e)
                })
        
        logger.info(f"Batch processing completed: {results['processed']} processed, {results['failed']} failed")
        return results
        
    except Exception as exc:
        logger.error(f"Batch processing failed: {exc}", exc_info=True)
        raise


@celery_app.task(
    queue="doubt_processing",
    priority=PRIORITY_HIGH
)
def validate_doubt_solution(doubt_id: str, solution: str) -> Dict[str, Any]:
    """
    Validate a doubt solution using PIL reasoning engine
    
    Args:
        doubt_id: Doubt ID
        solution: Solution to validate
        
    Returns:
        Validation result
    """
    try:
        logger.info(f"Validating solution for doubt {doubt_id}")
        
        async def validate_solution():
            async with db_manager.get_postgres_session() as session:
                from app.core.pil.reasoning_engine import ReasoningEngine
                reasoning_engine = ReasoningEngine()
                
                # Get doubt details
                from sqlalchemy import select
                from app.models import Doubt
                
                result = await session.execute(
                    select(Doubt).where(Doubt.id == uuid.UUID(doubt_id))
                )
                doubt = result.scalar_one()
                
                # Validate solution
                validation_result = await reasoning_engine.validate_solution(
                    problem=doubt.question_text,
                    solution=solution,
                    subject=doubt.subject.code.lower()
                )
                
                return validation_result
        
        result = asyncio.run(validate_solution())
        
        logger.info(f"Solution validation completed for doubt {doubt_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Solution validation failed for doubt {doubt_id}: {exc}", exc_info=True)
        return {
            "is_valid": False,
            "errors": [str(exc)],
            "confidence_score": 0.0
        }


@celery_app.task(
    queue="doubt_processing"
)
def personalize_response(doubt_id: str, response: str, student_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Personalize a response using MCP adaptive engine
    
    Args:
        doubt_id: Doubt ID
        response: Original response
        student_profile: Student profile data
        
    Returns:
        Personalized response
    """
    try:
        logger.info(f"Personalizing response for doubt {doubt_id}")
        
        async def personalize():
            from app.core.mcp.adaptive_engine import AdaptiveEngine
            adaptive_engine = AdaptiveEngine()
            
            # Get subject information
            async with db_manager.get_postgres_session() as session:
                from sqlalchemy import select
                from app.models import Doubt
                
                result = await session.execute(
                    select(Doubt).where(Doubt.id == uuid.UUID(doubt_id))
                )
                doubt = result.scalar_one()
                
                # Adapt response
                adapted_response = await adaptive_engine.adapt_response(
                    original_response=response,
                    student_profile=student_profile,
                    subject=doubt.subject.code.lower(),
                    difficulty=doubt.difficulty_level
                )
                
                return adapted_response
        
        result = asyncio.run(personalize())
        
        logger.info(f"Response personalization completed for doubt {doubt_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Response personalization failed for doubt {doubt_id}: {exc}", exc_info=True)
        return {
            "adapted_response": response,  # Fallback to original
            "adjustments": {
                "difficulty_adjusted": False,
                "style_adapted": False
            }
        }


@celery_app.task(
    queue="doubt_processing"
)
def enrich_with_context(question: str, subject: str, topic: str = None) -> Dict[str, Any]:
    """
    Enrich question with contextual information using RAG
    
    Args:
        question: Student's question
        subject: Subject area
        topic: Optional topic
        
    Returns:
        Enriched context
    """
    try:
        logger.info(f"Enriching context for question in {subject}")
        
        async def enrich():
            from app.core.rag.retriever import RAGRetriever
            rag_retriever = RAGRetriever()
            
            # Retrieve similar content
            similar_content = await rag_retriever.retrieve_similar_content(
                query=question,
                subject=subject.lower(),
                top_k=5,
                min_score=0.7
            )
            
            # Retrieve examples
            examples = await rag_retriever.retrieve_examples(
                subject=subject.lower(),
                topic=topic.lower() if topic else None,
                count=3
            )
            
            return {
                "similar_content": similar_content,
                "examples": examples,
                "enriched": True
            }
        
        result = asyncio.run(enrich())
        
        logger.info(f"Context enrichment completed for {subject} question")
        return result
        
    except Exception as exc:
        logger.error(f"Context enrichment failed: {exc}", exc_info=True)
        return {
            "similar_content": [],
            "examples": [],
            "enriched": False,
            "error": str(exc)
        }


@celery_app.task(
    queue="doubt_processing"
)
def update_learning_patterns(student_id: str, interaction_data: Dict[str, Any]) -> bool:
    """
    Update student learning patterns based on interaction
    
    Args:
        student_id: Student ID
        interaction_data: Interaction data
        
    Returns:
        Success status
    """
    try:
        logger.info(f"Updating learning patterns for student {student_id}")
        
        async def update_patterns():
            async with db_manager.get_postgres_session() as session:
                from app.services.student_service import get_student_service
                from app.database import get_cache_manager
                
                cache_manager = await get_cache_manager()
                student_service = get_student_service(cache_manager)
                
                await student_service.update_learning_patterns(
                    student_id=uuid.UUID(student_id),
                    session=session,
                    interaction_data=interaction_data
                )
                
                return True
        
        result = asyncio.run(update_patterns())
        
        logger.info(f"Learning patterns updated for student {student_id}")
        return result
        
    except Exception as exc:
        logger.error(f"Failed to update learning patterns for student {student_id}: {exc}", exc_info=True)
        return False


# Task monitoring and health check
@celery_app.task(queue="doubt_processing")
def health_check() -> Dict[str, Any]:
    """Health check for doubt processing worker"""
    
    try:
        # Check database connectivity
        async def check_db():
            async with db_manager.get_postgres_session() as session:
                await session.execute("SELECT 1")
                return True
        
        db_status = asyncio.run(check_db())
        
        # Check AI services
        ai_status = True  # Placeholder for AI service checks
        
        return {
            "status": "healthy" if db_status and ai_status else "unhealthy",
            "database": "ok" if db_status else "error",
            "ai_services": "ok" if ai_status else "error",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Health check failed: {exc}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }


# Export tasks
__all__ = [
    "process_doubt_task",
    "batch_process_doubts", 
    "validate_doubt_solution",
    "personalize_response",
    "enrich_with_context",
    "update_learning_patterns",
    "health_check"
]
