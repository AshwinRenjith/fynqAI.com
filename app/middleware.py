"""
Custom Middleware
Security, logging, monitoring, and performance middleware
"""

import json
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.database.redis_client import get_redis


logger = logging.getLogger(__name__)
settings = get_settings()


async def add_process_time_header(request: Request, call_next: Callable) -> Response:
    """Add processing time header to response"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


async def log_requests(request: Request, call_next: Callable) -> Response:
    """Log all requests with detailed information"""
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Start time
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent"),
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "response_size": response.headers.get("content-length", "unknown")
            }
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        # Calculate duration
        duration = time.time() - start_time
        
        # Log error
        logger.error(
            "Request failed",
            extra={
                "request_id": request_id,
                "duration_ms": round(duration * 1000, 2),
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        
        raise


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Add security headers to all responses"""
    
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    # Remove server header
    response.headers.pop("server", None)
    
    return response


async def rate_limiting_middleware(request: Request, call_next: Callable) -> Response:
    """Global rate limiting middleware"""
    
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
        return await call_next(request)
    
    # Get client identifier
    client_ip = request.client.host
    user_id = getattr(request.state, 'user_id', None)
    
    # Use user ID if available, otherwise IP
    identifier = f"user:{user_id}" if user_id else f"ip:{client_ip}"
    
    try:
        redis = await get_redis()
        
        # Rate limiting key
        rate_limit_key = f"global_rate_limit:{identifier}"
        
        # Check current requests
        current_requests = await redis.get(rate_limit_key)
        current_requests = int(current_requests) if current_requests else 0
        
        # Check if limit exceeded
        if current_requests >= settings.RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {settings.RATE_LIMIT_REQUESTS} requests per minute.",
                headers={"Retry-After": "60"}
            )
        
        # Increment counter
        pipe = redis.pipeline()
        pipe.incr(rate_limit_key)
        pipe.expire(rate_limit_key, settings.RATE_LIMIT_WINDOW)
        await pipe.execute()
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        remaining = max(0, settings.RATE_LIMIT_REQUESTS - current_requests - 1)
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + settings.RATE_LIMIT_WINDOW)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rate limiting error: {e}")
        # Continue without rate limiting if Redis is down
        return await call_next(request)


class CostTrackingMiddleware(BaseHTTPMiddleware):
    """Track API costs for monitoring and billing"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start time and cost tracking
        start_time = time.time()
        request.state.start_time = start_time
        request.state.api_costs = 0.0
        
        response = await call_next(request)
        
        # Calculate final costs
        total_cost = getattr(request.state, 'api_costs', 0.0)
        duration = time.time() - start_time
        
        # Log cost information
        if total_cost > 0:
            logger.info(
                "API cost tracking",
                extra={
                    "request_id": getattr(request.state, 'request_id', 'unknown'),
                    "path": request.url.path,
                    "method": request.method,
                    "total_cost_usd": round(total_cost, 6),
                    "duration_ms": round(duration * 1000, 2),
                    "user_id": getattr(request.state, 'user_id', None)
                }
            )
        
        # Add cost header (for debugging in development)
        if settings.is_development and total_cost > 0:
            response.headers["X-API-Cost"] = f"${total_cost:.6f}"
        
        return response


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Application monitoring and metrics collection"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Metrics collection
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Success metrics
            duration = time.time() - start_time
            await self._record_metrics(
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                duration=duration,
                success=True
            )
            
            return response
            
        except Exception as e:
            # Error metrics
            duration = time.time() - start_time
            await self._record_metrics(
                path=request.url.path,
                method=request.method,
                status_code=500,
                duration=duration,
                success=False,
                error_type=type(e).__name__
            )
            raise
    
    async def _record_metrics(
        self,
        path: str,
        method: str,
        status_code: int,
        duration: float,
        success: bool,
        error_type: str = None
    ):
        """Record metrics to monitoring system"""
        try:
            redis = await get_redis()
            
            # Create metrics payload
            metrics = {
                "timestamp": time.time(),
                "path": path,
                "method": method,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 2),
                "success": success,
                "error_type": error_type
            }
            
            # Store in Redis for batch processing
            await redis.lpush("app_metrics", json.dumps(metrics))
            await redis.expire("app_metrics", 86400)  # 24 hours
            
        except Exception as e:
            logger.warning(f"Failed to record metrics: {e}")


# Health check middleware
async def health_check_middleware(request: Request, call_next: Callable) -> Response:
    """Enhanced health check with dependency verification"""
    
    if request.url.path == "/health":
        # Perform health checks
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT
        }
        
        # Check database
        try:
            from app.database.connection import get_db_session
            async with get_db_session() as session:
                await session.execute("SELECT 1")
            health_status["database"] = "healthy"
        except Exception as e:
            health_status["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "unhealthy"
        
        # Check Redis
        try:
            redis = await get_redis()
            await redis.ping()
            health_status["redis"] = "healthy"
        except Exception as e:
            health_status["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Return health status
        status_code = 200 if health_status["status"] in ["healthy", "degraded"] else 503
        
        return Response(
            content=json.dumps(health_status, indent=2),
            status_code=status_code,
            media_type="application/json"
        )
    
    return await call_next(request)
