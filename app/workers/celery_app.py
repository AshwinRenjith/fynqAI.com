"""
Celery Application Configuration
Background task processing for fynqAI
"""

from celery import Celery
from kombu import Queue

from app.config import get_settings


settings = get_settings()

# Create Celery application
celery_app = Celery(
    "fynqai_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.doubt_processor",
        "app.workers.analytics_worker", 
        "app.workers.notification_worker",
        "app.workers.data_sync_worker"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "doubt_processing.*": {"queue": "doubt_processing"},
        "analytics.*": {"queue": "analytics"},
        "notifications.*": {"queue": "notifications"},
        "data_sync.*": {"queue": "data_sync"},
        "maintenance.*": {"queue": "maintenance"}
    },
    
    # Queue definitions
    task_queues=(
        Queue("doubt_processing", routing_key="doubt_processing"),
        Queue("analytics", routing_key="analytics"),
        Queue("notifications", routing_key="notifications"),
        Queue("data_sync", routing_key="data_sync"),
        Queue("maintenance", routing_key="maintenance"),
        Queue("celery", routing_key="celery"),  # Default queue
    ),
    
    # Task serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Task execution
    task_always_eager=settings.ENVIRONMENT == "test",  # Execute immediately in tests
    task_eager_propagates=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    
    # Task retry configuration
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Worker settings
    worker_disable_rate_limits=False,
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Monitoring
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
    
    # Beat schedule for periodic tasks
    beat_schedule={
        "reset-daily-limits": {
            "task": "maintenance.reset_daily_limits",
            "schedule": "0 0 * * *",  # Daily at midnight UTC
        },
        "compute-analytics": {
            "task": "analytics.compute_daily_analytics",
            "schedule": "0 1 * * *",  # Daily at 1 AM UTC
        },
        "sync-vector-embeddings": {
            "task": "data_sync.sync_vector_embeddings",
            "schedule": "0 2 * * *",  # Daily at 2 AM UTC
        },
        "cleanup-old-data": {
            "task": "maintenance.cleanup_old_data",
            "schedule": "0 3 * * 0",  # Weekly on Sunday at 3 AM UTC
        },
        "generate-reports": {
            "task": "analytics.generate_weekly_reports",
            "schedule": "0 4 * * 1",  # Weekly on Monday at 4 AM UTC
        }
    },
    beat_schedule_filename="/tmp/celerybeat-schedule"
)

# Task priority levels
PRIORITY_HIGH = 9
PRIORITY_NORMAL = 5  
PRIORITY_LOW = 1

# Queue configurations
QUEUE_CONFIGS = {
    "doubt_processing": {
        "priority": PRIORITY_HIGH,
        "max_retries": 2,
        "retry_delay": 30
    },
    "analytics": {
        "priority": PRIORITY_NORMAL,
        "max_retries": 3,
        "retry_delay": 60
    },
    "notifications": {
        "priority": PRIORITY_NORMAL,
        "max_retries": 5,
        "retry_delay": 30
    },
    "data_sync": {
        "priority": PRIORITY_LOW,
        "max_retries": 3,
        "retry_delay": 120
    },
    "maintenance": {
        "priority": PRIORITY_LOW,
        "max_retries": 1,
        "retry_delay": 300
    }
}


# Helper functions for task management
def get_task_info(task_id: str) -> dict:
    """Get information about a specific task"""
    result = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result,
        "traceback": result.traceback,
        "ready": result.ready(),
        "successful": result.successful(),
        "failed": result.failed()
    }


def cancel_task(task_id: str) -> bool:
    """Cancel a running task"""
    celery_app.control.revoke(task_id, terminate=True)
    return True


def get_worker_stats() -> dict:
    """Get worker statistics"""
    inspect = celery_app.control.inspect()
    
    return {
        "active_tasks": inspect.active(),
        "scheduled_tasks": inspect.scheduled(),
        "reserved_tasks": inspect.reserved(),
        "stats": inspect.stats(),
        "registered_tasks": inspect.registered()
    }


def purge_queue(queue_name: str) -> int:
    """Purge all tasks from a queue"""
    return celery_app.control.purge()


# Export celery app
__all__ = [
    "celery_app",
    "PRIORITY_HIGH",
    "PRIORITY_NORMAL", 
    "PRIORITY_LOW",
    "QUEUE_CONFIGS",
    "get_task_info",
    "cancel_task",
    "get_worker_stats",
    "purge_queue"
]
