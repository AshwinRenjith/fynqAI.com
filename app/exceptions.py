"""
Custom Exception Classes
Centralized exception handling for the fynqAI Backend
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import Request, status
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


class FynqAIException(Exception):
    """Base exception class for fynqAI Backend"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "FYNQAI_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


# Authentication and Authorization Exceptions
class AuthenticationError(FynqAIException):
    """Authentication failed"""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="AUTH_FAILED",
            details=details,
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AuthorizationError(FynqAIException):
    """User not authorized for this action"""
    
    def __init__(self, message: str = "Not authorized", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="AUTH_INSUFFICIENT_PERMISSIONS",
            details=details,
            status_code=status.HTTP_403_FORBIDDEN
        )


class TokenExpiredError(FynqAIException):
    """JWT token has expired"""
    
    def __init__(self, message: str = "Token has expired", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="TOKEN_EXPIRED",
            details=details,
            status_code=status.HTTP_401_UNAUTHORIZED
        )


# User and Account Exceptions
class UserNotFoundError(FynqAIException):
    """User not found"""
    
    def __init__(self, user_id: str = None, details: Optional[Dict] = None):
        message = f"User {user_id} not found" if user_id else "User not found"
        super().__init__(
            message=message,
            error_code="USER_NOT_FOUND",
            details=details,
            status_code=status.HTTP_404_NOT_FOUND
        )


class UserAlreadyExistsError(FynqAIException):
    """User already exists"""
    
    def __init__(self, email: str = None, details: Optional[Dict] = None):
        message = f"User with email {email} already exists" if email else "User already exists"
        super().__init__(
            message=message,
            error_code="USER_ALREADY_EXISTS",
            details=details,
            status_code=status.HTTP_409_CONFLICT
        )


class AccountInactiveError(FynqAIException):
    """User account is inactive"""
    
    def __init__(self, message: str = "User account is inactive", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="ACCOUNT_INACTIVE",
            details=details,
            status_code=status.HTTP_401_UNAUTHORIZED
        )


# Subscription and Billing Exceptions
class SubscriptionError(FynqAIException):
    """Subscription-related error"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="SUBSCRIPTION_ERROR",
            details=details,
            status_code=status.HTTP_402_PAYMENT_REQUIRED
        )


class QuotaExceededError(FynqAIException):
    """User has exceeded their quota"""
    
    def __init__(self, quota_type: str, limit: int, details: Optional[Dict] = None):
        message = f"{quota_type} quota exceeded. Limit: {limit}"
        super().__init__(
            message=message,
            error_code="QUOTA_EXCEEDED",
            details=details,
            status_code=status.HTTP_402_PAYMENT_REQUIRED
        )


# AI and Processing Exceptions
class AIProcessingError(FynqAIException):
    """AI processing failed"""
    
    def __init__(self, message: str = "AI processing failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="AI_PROCESSING_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class PILProcessingError(FynqAIException):
    """Process Intelligence Layer processing failed"""
    
    def __init__(self, message: str = "PIL processing failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="PIL_PROCESSING_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class MCPProcessingError(FynqAIException):
    """Multi-Context Personalization processing failed"""
    
    def __init__(self, message: str = "MCP processing failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="MCP_PROCESSING_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class RAGProcessingError(FynqAIException):
    """RAG processing failed"""
    
    def __init__(self, message: str = "RAG processing failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="RAG_PROCESSING_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class LLMProviderError(FynqAIException):
    """LLM provider error"""
    
    def __init__(self, provider: str, message: str, details: Optional[Dict] = None):
        full_message = f"{provider} LLM error: {message}"
        super().__init__(
            message=full_message,
            error_code="LLM_PROVIDER_ERROR",
            details=details,
            status_code=status.HTTP_502_BAD_GATEWAY
        )


# Data and Validation Exceptions
class InvalidInputError(FynqAIException):
    """Invalid input provided"""
    
    def __init__(self, message: str = "Invalid input", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="INVALID_INPUT",
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST
        )


class ValidationError(FynqAIException):
    """Data validation failed"""
    
    def __init__(self, field: str, message: str, details: Optional[Dict] = None):
        full_message = f"Validation error for field '{field}': {message}"
        super().__init__(
            message=full_message,
            error_code="VALIDATION_ERROR",
            details=details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class DataNotFoundError(FynqAIException):
    """Requested data not found"""
    
    def __init__(self, resource: str, identifier: str = None, details: Optional[Dict] = None):
        message = f"{resource}"
        if identifier:
            message += f" with ID {identifier}"
        message += " not found"
        
        super().__init__(
            message=message,
            error_code="DATA_NOT_FOUND",
            details=details,
            status_code=status.HTTP_404_NOT_FOUND
        )


# External Service Exceptions
class ExternalServiceError(FynqAIException):
    """External service error"""
    
    def __init__(self, service: str, message: str, details: Optional[Dict] = None):
        full_message = f"{service} service error: {message}"
        super().__init__(
            message=full_message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=details,
            status_code=status.HTTP_502_BAD_GATEWAY
        )


class DatabaseError(FynqAIException):
    """Database operation failed"""
    
    def __init__(self, message: str = "Database operation failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class CacheError(FynqAIException):
    """Cache operation failed"""
    
    def __init__(self, message: str = "Cache operation failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# Rate Limiting and Performance Exceptions
class RateLimitExceededError(FynqAIException):
    """Rate limit exceeded"""
    
    def __init__(self, limit: int, window: int, details: Optional[Dict] = None):
        message = f"Rate limit exceeded: {limit} requests per {window} seconds"
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )


class TimeoutError(FynqAIException):
    """Request timed out"""
    
    def __init__(self, timeout: int, details: Optional[Dict] = None):
        message = f"Request timed out after {timeout} seconds"
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            details=details,
            status_code=status.HTTP_408_REQUEST_TIMEOUT
        )


# File and Content Exceptions
class FileProcessingError(FynqAIException):
    """File processing failed"""
    
    def __init__(self, message: str = "File processing failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="FILE_PROCESSING_ERROR",
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST
        )


class ImageProcessingError(FynqAIException):
    """Image processing failed"""
    
    def __init__(self, message: str = "Image processing failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="IMAGE_PROCESSING_ERROR",
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST
        )


# Feature and Configuration Exceptions
class FeatureDisabledError(FynqAIException):
    """Feature is currently disabled"""
    
    def __init__(self, feature: str, details: Optional[Dict] = None):
        message = f"Feature '{feature}' is currently disabled"
        super().__init__(
            message=message,
            error_code="FEATURE_DISABLED",
            details=details,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class ConfigurationError(FynqAIException):
    """Configuration error"""
    
    def __init__(self, message: str = "Configuration error", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# Exception Handler
async def fynqai_exception_handler(request: Request, exc: FynqAIException) -> JSONResponse:
    """Global exception handler for FynqAIException"""
    
    # Log the exception
    logger.error(
        f"FynqAI Exception: {exc.error_code}",
        extra={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "request_id": getattr(request.state, 'request_id', 'unknown')
        },
        exc_info=True
    )
    
    # Create error response
    error_response = {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": str(int(time.time() * 1000))
        }
    }
    
    # Add request ID if available
    request_id = getattr(request.state, 'request_id', None)
    if request_id:
        error_response["error"]["request_id"] = request_id
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )
