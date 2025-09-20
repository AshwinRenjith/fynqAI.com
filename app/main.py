"""
FastAPI Application Entry Point
Production-ready setup with comprehensive middleware, error handling, and monitoring
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.v1.router import api_router
from app.config import get_settings
from app.database.connection import init_databases, close_databases
from app.exceptions import fynqai_exception_handler, FynqAIException
from app.middleware import (
    add_process_time_header,
    log_requests,
    rate_limiting_middleware,
    security_headers_middleware
)
from app.utils.logger import setup_logging


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management"""
    # Startup
    settings = get_settings()
    
    # Initialize logging
    setup_logging(settings.LOG_LEVEL)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting fynqAI Backend...")
    
    # Initialize databases and external services
    await init_databases()
    
    # Verify critical services are available
    await verify_critical_services()
    
    logger.info("✅ fynqAI Backend started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down fynqAI Backend...")
    await close_databases()
    logger.info("✅ fynqAI Backend shutdown complete")


async def verify_critical_services():
    """Verify that critical external services are available"""
    logger = logging.getLogger(__name__)
    
    # Test database connection
    try:
        from app.database.connection import get_db_session
        async with get_db_session() as session:
            await session.execute("SELECT 1")
        logger.info("✅ Database connection verified")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise
    
    # Test Redis connection
    try:
        from app.database.redis_client import get_redis
        redis = await get_redis()
        await redis.ping()
        logger.info("✅ Redis connection verified")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise
    
    # Test LLM providers
    try:
        from app.core.llm.orchestrator import LLMOrchestrator
        orchestrator = LLMOrchestrator()
        await orchestrator.health_check()
        logger.info("✅ LLM providers verified")
    except Exception as e:
        logger.warning(f"⚠️ LLM providers check failed: {e}")


# Create FastAPI application
def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="fynqAI Backend API",
        description="""
        **The most accurate, personalized AI tutoring platform for competitive exam preparation**
        
        ## Features
        
        * **Process Intelligence Layer (PIL)** - Hallucination-free mathematical reasoning
        * **Multi-Context Personalization (MCP)** - Adaptive learning framework  
        * **Advanced RAG** - Exam-specific knowledge retrieval
        * **Multi-LLM Orchestration** - Cost-optimized model selection
        
        ## Core Capabilities
        
        * Mathematical problem solving with step-by-step explanations
        * Personalized learning recommendations
        * JEE/NEET/competitive exam preparation
        * Real-time doubt resolution
        * Progress tracking and analytics
        
        """,
        version="1.0.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.ENVIRONMENT != "production" else None,
        docs_url=f"{settings.API_V1_STR}/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url=f"{settings.API_V1_STR}/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan
    )
    
    # Add middleware in correct order (last added = first executed)
    
    # Security headers (outermost)
    app.middleware("http")(security_headers_middleware)
    
    # Rate limiting
    app.middleware("http")(rate_limiting_middleware)
    
    # Request logging
    app.middleware("http")(log_requests)
    
    # Process time header
    app.middleware("http")(add_process_time_header)
    
    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # CORS (innermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time", "X-Request-ID"]
    )
    
    # Exception handlers
    app.add_exception_handler(FynqAIException, fynqai_exception_handler)
    
    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    # Health check endpoint (outside versioned API)
    @app.get("/health")
    async def health_check():
        """Simple health check endpoint"""
        return {
            "status": "healthy",
            "service": "fynqAI Backend",
            "version": "1.0.0",
            "timestamp": time.time()
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Welcome to fynqAI Backend API",
            "version": "1.0.0",
            "description": "The most accurate, personalized AI tutoring platform",
            "docs_url": f"{settings.API_V1_STR}/docs",
            "health_url": "/health"
        }
    
    return app


# Create the app instance
app = create_application()


if __name__ == "__main__":
    """Development server entry point"""
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        loop="auto"
    )
