"""
Supabase Webhooks
Handle database change notifications and real-time events
"""

from fastapi import APIRouter, Request, HTTPException, status
import json
import logging

from app.config import get_settings


router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post("/database-changes")
async def handle_database_changes(request: Request):
    """Handle Supabase database change notifications"""
    
    try:
        # Verify webhook signature (implement based on Supabase docs)
        payload = await request.json()
        
        # Process different types of database changes
        table = payload.get("table")
        event_type = payload.get("type")  # INSERT, UPDATE, DELETE
        record = payload.get("record", {})
        old_record = payload.get("old_record", {})
        
        logger.info(
            f"Database change received: {event_type} on {table}",
            extra={
                "table": table,
                "event_type": event_type,
                "record_id": record.get("id")
            }
        )
        
        # Handle specific table changes
        if table == "users" and event_type == "INSERT":
            await handle_new_user(record)
        elif table == "doubts" and event_type == "INSERT":
            await handle_new_doubt(record)
        elif table == "feedback" and event_type == "INSERT":
            await handle_new_feedback(record)
        
        return {"status": "processed"}
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )


async def handle_new_user(user_record: dict):
    """Handle new user registration"""
    # Trigger welcome email, setup default preferences, etc.
    logger.info(f"Processing new user: {user_record.get('id')}")


async def handle_new_doubt(doubt_record: dict):
    """Handle new doubt submission"""
    # Trigger analytics update, notifications, etc.
    logger.info(f"Processing new doubt: {doubt_record.get('id')}")


async def handle_new_feedback(feedback_record: dict):
    """Handle new feedback submission"""
    # Trigger quality monitoring, model retraining, etc.
    logger.info(f"Processing new feedback: {feedback_record.get('id')}")
