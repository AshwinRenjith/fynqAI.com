"""
FastAPI Dependencies
Shared dependencies for authentication, database sessions, rate limiting, etc.
"""

import logging
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import jwt

from app.config import get_settings
from app.database.connection import get_db_session
from app.database.redis_client import get_redis
from app.models.user import User
from app.services.auth_service import AuthService
from app.utils.cache_utils import CacheManager


logger = logging.getLogger(__name__)
settings = get_settings()

# Security schemes
security = HTTPBearer(auto_error=False)


# Database dependencies
async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency"""
    async with get_db_session() as session:
        yield session


# Redis dependencies
async def get_redis_client():
    """Get Redis client dependency"""
    return await get_redis()


# Cache dependencies
async def get_cache_manager(redis=Depends(get_redis_client)) -> CacheManager:
    """Get cache manager dependency"""
    return CacheManager(redis)


# Authentication dependencies
class AuthRequired:
    """Authentication dependency class"""
    
    def __init__(self, optional: bool = False):
        self.optional = optional
    
    async def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        db: AsyncSession = Depends(get_database)
    ) -> Optional[User]:
        """Validate JWT token and return user"""
        
        if credentials is None:
            if self.optional:
                return None
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication credentials required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            # Validate JWT token
            payload = jwt.decode(
                credentials.credentials,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            
            user_id: str = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            
            # Get user from database
            auth_service = AuthService(db)
            user = await auth_service.get_user_by_id(user_id)
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is inactive"
                )
            
            # Add user to request state for logging
            request.state.user = user
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            if self.optional:
                return None
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )


# Create dependency instances
get_current_user = AuthRequired(optional=False)
get_current_user_optional = AuthRequired(optional=True)


# Role-based dependencies
async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_premium_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current premium user"""
    if current_user.subscription_tier not in ["premium", "enterprise"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user


async def get_current_enterprise_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current enterprise user"""
    if current_user.subscription_tier != "enterprise":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Enterprise subscription required"
        )
    return current_user


# Rate limiting dependency
class RateLimitChecker:
    """Rate limiting dependency"""
    
    def __init__(self, requests_per_minute: int = None):
        self.requests_per_minute = requests_per_minute or settings.RATE_LIMIT_REQUESTS
    
    async def __call__(
        self,
        request: Request,
        cache: CacheManager = Depends(get_cache_manager)
    ):
        """Check rate limits for the request"""
        
        # Get client identifier
        client_ip = request.client.host
        user_id = getattr(request.state, 'user_id', None)
        
        # Use user ID if authenticated, otherwise IP
        identifier = f"user:{user_id}" if user_id else f"ip:{client_ip}"
        
        # Check rate limit
        key = f"rate_limit:{identifier}:{request.url.path}"
        
        current_requests = await cache.get(key) or 0
        
        if int(current_requests) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.requests_per_minute} requests per minute.",
                headers={"Retry-After": "60"}
            )
        
        # Increment counter
        await cache.set(key, int(current_requests) + 1, ttl=60)


# Create rate limiting instances
standard_rate_limit = RateLimitChecker()
strict_rate_limit = RateLimitChecker(requests_per_minute=20)
api_rate_limit = RateLimitChecker(requests_per_minute=1000)


# Subscription tier dependencies
class SubscriptionChecker:
    """Check subscription tier and limits"""
    
    def __init__(self, required_tier: str = "free", feature: str = None):
        self.required_tier = required_tier
        self.feature = feature
    
    async def __call__(
        self,
        current_user: User = Depends(get_current_active_user),
        cache: CacheManager = Depends(get_cache_manager)
    ) -> User:
        """Check if user has required subscription tier"""
        
        tier_hierarchy = {
            "free": 0,
            "premium": 1,
            "enterprise": 2
        }
        
        required_level = tier_hierarchy.get(self.required_tier, 0)
        user_level = tier_hierarchy.get(current_user.subscription_tier, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"{self.required_tier.capitalize()} subscription required"
            )
        
        # Check feature-specific limits
        if self.feature:
            await self._check_feature_limits(current_user, cache)
        
        return current_user
    
    async def _check_feature_limits(self, user: User, cache: CacheManager):
        """Check feature-specific usage limits"""
        
        daily_limits = {
            "free": {"doubts": 10, "ai_requests": 50},
            "premium": {"doubts": 100, "ai_requests": 500},
            "enterprise": {"doubts": -1, "ai_requests": -1}  # Unlimited
        }
        
        user_limits = daily_limits.get(user.subscription_tier, daily_limits["free"])
        
        if self.feature in user_limits:
            limit = user_limits[self.feature]
            
            if limit == -1:  # Unlimited
                return
            
            # Check daily usage
            today = "2024-01-01"  # Replace with actual date logic
            usage_key = f"usage:{user.id}:{today}:{self.feature}"
            current_usage = await cache.get(usage_key) or 0
            
            if int(current_usage) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=f"Daily {self.feature} limit exceeded. Upgrade your subscription."
                )


# Pagination dependencies
class PaginationParams:
    """Pagination parameters"""
    
    def __init__(
        self,
        page: int = 1,
        size: int = 20,
        max_size: int = 100
    ):
        self.page = max(1, page)
        self.size = min(max(1, size), max_size)
        self.offset = (self.page - 1) * self.size
        self.limit = self.size


def get_pagination_params(
    page: int = 1,
    size: int = 20
) -> PaginationParams:
    """Get pagination parameters"""
    return PaginationParams(page, size)


# Validation dependencies
async def validate_content_type(request: Request):
    """Validate content type for POST/PUT requests"""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        
        if not content_type.startswith("application/json"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Content-Type must be application/json"
            )


# Feature flag dependencies
class FeatureFlag:
    """Feature flag dependency"""
    
    def __init__(self, feature: str):
        self.feature = feature
    
    async def __call__(self):
        """Check if feature is enabled"""
        feature_flags = {
            "pil": settings.ENABLE_PIL,
            "mcp": settings.ENABLE_MCP,
            "enterprise": settings.ENABLE_ENTERPRISE,
            "webhooks": settings.ENABLE_WEBHOOKS
        }
        
        if not feature_flags.get(self.feature, False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Feature '{self.feature}' is currently disabled"
            )


# Service dependencies
async def get_auth_service(db: AsyncSession = Depends(get_database)) -> AuthService:
    """Get authentication service"""
    return AuthService(db)
