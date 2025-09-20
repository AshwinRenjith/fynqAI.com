"""
Health Check Endpoints
System health monitoring and status checks
"""

import time
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_database
from app.database.redis_client import get_redis
from app.config import get_settings


router = APIRouter()
settings = get_settings()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "fynqAI Backend",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@router.get("/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_database)):
    """Detailed health check with dependency verification"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {}
    }
    
    # Check database
    try:
        await db.execute("SELECT 1")
        health_status["services"]["database"] = {"status": "healthy", "response_time_ms": 0}
    except Exception as e:
        health_status["services"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        redis = await get_redis()
        start_time = time.time()
        await redis.ping()
        response_time = (time.time() - start_time) * 1000
        health_status["services"]["redis"] = {"status": "healthy", "response_time_ms": response_time}
    except Exception as e:
        health_status["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    return health_status
