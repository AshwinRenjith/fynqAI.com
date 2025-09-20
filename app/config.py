"""
Application Configuration
Centralized configuration management with environment-based settings
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "fynqAI Backend"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "The most accurate, personalized AI tutoring platform"
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Server
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    API_V1_STR: str = Field(default="/api/v1", description="API v1 prefix")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Logging format: json or text")
    
    # Security
    SECRET_KEY: str = Field(..., description="Secret key for JWT signing")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration time")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration time")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        description="Allowed origins for CORS"
    )
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per minute")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limit window in seconds")
    
    # Database - Supabase PostgreSQL
    DATABASE_URL: str = Field(..., description="Supabase PostgreSQL connection URL")
    DATABASE_POOL_SIZE: int = Field(default=20, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=10, description="Database max overflow connections")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    REDIS_TTL: int = Field(default=3600, description="Default Redis TTL in seconds")
    
    # Vector Database - Pinecone
    PINECONE_API_KEY: Optional[str] = Field(default=None, description="Pinecone API key")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, description="Pinecone environment")
    PINECONE_INDEX_NAME: str = Field(default="fynqai-knowledge", description="Pinecone index name")
    PINECONE_DIMENSION: int = Field(default=1536, description="Vector dimension")
    
    # LLM Providers
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4", description="Default OpenAI model")
    OPENAI_MAX_TOKENS: int = Field(default=2000, description="Max tokens for OpenAI")
    OPENAI_TEMPERATURE: float = Field(default=0.1, description="Temperature for OpenAI")
    
    # Google Gemini
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    GEMINI_MODEL: str = Field(default="gemini-1.5-pro", description="Default Gemini model")
    GEMINI_MAX_TOKENS: int = Field(default=2000, description="Max tokens for Gemini")
    GEMINI_TEMPERATURE: float = Field(default=0.1, description="Temperature for Gemini")
    
    # Anthropic Claude
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    ANTHROPIC_MODEL: str = Field(default="claude-3-5-sonnet-20241022", description="Default Claude model")
    ANTHROPIC_MAX_TOKENS: int = Field(default=2000, description="Max tokens for Claude")
    ANTHROPIC_TEMPERATURE: float = Field(default=0.1, description="Temperature for Claude")
    
    # Mistral
    MISTRAL_API_KEY: Optional[str] = Field(default=None, description="Mistral API key")
    MISTRAL_MODEL: str = Field(default="mistral-large-latest", description="Default Mistral model")
    MISTRAL_MAX_TOKENS: int = Field(default=2000, description="Max tokens for Mistral")
    MISTRAL_TEMPERATURE: float = Field(default=0.1, description="Temperature for Mistral")
    
    # Embeddings
    EMBEDDING_MODEL: str = Field(default="text-embedding-ada-002", description="Embedding model")
    EMBEDDING_DIMENSION: int = Field(default=1536, description="Embedding dimension")
    
    # Image Processing
    VISION_API_KEY: Optional[str] = Field(default=None, description="Google Vision API key")
    OCR_ENGINE: str = Field(default="tesseract", description="OCR engine: tesseract or google_vision")
    
    # File Storage
    UPLOAD_MAX_SIZE: int = Field(default=10 * 1024 * 1024, description="Max upload size (10MB)")
    ALLOWED_IMAGE_TYPES: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"],
        description="Allowed image MIME types"
    )
    
    # Caching
    CACHE_TTL: int = Field(default=3600, description="Default cache TTL in seconds")
    CACHE_ENABLED: bool = Field(default=True, description="Enable caching")
    
    # Analytics & Monitoring
    ENABLE_ANALYTICS: bool = Field(default=True, description="Enable analytics collection")
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    DATADOG_API_KEY: Optional[str] = Field(default=None, description="DataDog API key")
    
    # Background Tasks
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", description="Celery broker URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2", description="Celery result backend")
    
    # Feature Flags
    ENABLE_PIL: bool = Field(default=True, description="Enable Process Intelligence Layer")
    ENABLE_MCP: bool = Field(default=True, description="Enable Multi-Context Personalization")
    ENABLE_ENTERPRISE: bool = Field(default=False, description="Enable enterprise features")
    ENABLE_WEBHOOKS: bool = Field(default=False, description="Enable webhooks")
    
    # Cost Management
    MAX_DAILY_COST_USD: float = Field(default=100.0, description="Max daily LLM costs in USD")
    COST_ALERT_THRESHOLD: float = Field(default=0.8, description="Cost alert threshold (80%)")
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, description="Max concurrent requests")
    REQUEST_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")
    
    # Validation
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return v
    
    @validator("ALLOWED_ORIGINS")
    def validate_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # Property helpers
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def is_debug(self) -> bool:
        return self.DEBUG or self.is_development
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
