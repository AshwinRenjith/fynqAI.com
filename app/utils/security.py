"""
Security Utilities
Authentication, authorization, and security functions
"""

import hashlib
import hmac
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from passlib.context import CryptContext
import base64
import os

from app.config import get_settings
from app.exceptions import AuthenticationError


settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordManager:
    """Password hashing and verification utilities"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_random_password(length: int = 12) -> str:
        """
        Generate a secure random password
        
        Args:
            length: Password length
            
        Returns:
            Random password
        """
        import string
        
        # Define character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*"
        
        # Ensure at least one character from each set
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # Fill the rest randomly
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - 4):
            password.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)


class JWTManager:
    """JWT token management utilities"""
    
    @staticmethod
    def create_access_token(
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create an access token
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        return jwt.encode(
            to_encode, 
            settings.SECRET_KEY, 
            algorithm=settings.JWT_ALGORITHM
        )
    
    @staticmethod
    def create_refresh_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a refresh token
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            JWT refresh token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        return jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        Verify and decode a JWT token
        
        Args:
            token: JWT token to verify
            token_type: Expected token type
            
        Returns:
            Decoded token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                raise AuthenticationError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")
    
    @staticmethod
    def create_verification_token(email: str) -> str:
        """
        Create email verification token
        
        Args:
            email: Email address
            
        Returns:
            Verification token
        """
        data = {
            "email": email,
            "purpose": "email_verification"
        }
        
        expire = datetime.utcnow() + timedelta(hours=24)  # 24 hour expiry
        data["exp"] = expire
        
        return jwt.encode(
            data,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
    
    @staticmethod
    def verify_verification_token(token: str) -> str:
        """
        Verify email verification token
        
        Args:
            token: Verification token
            
        Returns:
            Email address from token
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            if payload.get("purpose") != "email_verification":
                raise AuthenticationError("Invalid verification token")
            
            return payload.get("email")
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Verification token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid verification token")


class SecurityUtils:
    """General security utilities"""
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """
        Generate a secure API key
        
        Args:
            length: Key length in bytes
            
        Returns:
            Base64 encoded API key
        """
        key_bytes = secrets.token_bytes(length)
        return base64.urlsafe_b64encode(key_bytes).decode('utf-8').rstrip('=')
    
    @staticmethod
    def generate_csrf_token() -> str:
        """
        Generate CSRF token
        
        Returns:
            CSRF token
        """
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def verify_csrf_token(token: str, expected: str) -> bool:
        """
        Verify CSRF token
        
        Args:
            token: Provided token
            expected: Expected token
            
        Returns:
            True if tokens match
        """
        return hmac.compare_digest(token, expected)
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """
        Hash an API key for storage
        
        Args:
            api_key: API key to hash
            
        Returns:
            Hashed API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def generate_otp(length: int = 6) -> str:
        """
        Generate numeric OTP
        
        Args:
            length: OTP length
            
        Returns:
            Numeric OTP
        """
        return ''.join([str(secrets.randbelow(10)) for _ in range(length)])
    
    @staticmethod
    def constant_time_compare(a: str, b: str) -> bool:
        """
        Constant time string comparison to prevent timing attacks
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            True if strings are equal
        """
        return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for secure storage
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re
        
        # Remove path traversal attempts
        filename = os.path.basename(filename)
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove null bytes
        filename = filename.replace('\x00', '')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    @staticmethod
    def generate_secure_filename(original_filename: str) -> str:
        """
        Generate secure filename with timestamp and random suffix
        
        Args:
            original_filename: Original filename
            
        Returns:
            Secure filename
        """
        import os
        from datetime import datetime
        
        # Get file extension
        _, ext = os.path.splitext(original_filename)
        
        # Generate timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Generate random suffix
        random_suffix = secrets.token_hex(8)
        
        # Combine parts
        secure_name = f"{timestamp}_{random_suffix}{ext}"
        
        return secure_name


class RateLimiter:
    """Rate limiting utilities"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def is_rate_limited(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> bool:
        """
        Check if request should be rate limited
        
        Args:
            key: Rate limit key (e.g., user_id, ip_address)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if rate limited
        """
        try:
            current_count = await self.redis.incr(f"rate_limit:{key}")
            
            if current_count == 1:
                # Set expiry on first request
                await self.redis.expire(f"rate_limit:{key}", window_seconds)
            
            return current_count > limit
            
        except Exception:
            # If Redis fails, don't rate limit
            return False
    
    async def get_rate_limit_info(
        self,
        key: str
    ) -> Dict[str, Any]:
        """
        Get rate limit information
        
        Args:
            key: Rate limit key
            
        Returns:
            Rate limit info
        """
        try:
            count = await self.redis.get(f"rate_limit:{key}")
            ttl = await self.redis.ttl(f"rate_limit:{key}")
            
            return {
                "current_count": int(count) if count else 0,
                "reset_in_seconds": ttl if ttl > 0 else 0
            }
            
        except Exception:
            return {
                "current_count": 0,
                "reset_in_seconds": 0
            }


class EncryptionUtils:
    """Data encryption utilities"""
    
    @staticmethod
    def encrypt_sensitive_data(data: str, key: Optional[str] = None) -> str:
        """
        Encrypt sensitive data
        
        Args:
            data: Data to encrypt
            key: Encryption key (uses default if not provided)
            
        Returns:
            Encrypted data (base64 encoded)
        """
        from cryptography.fernet import Fernet
        
        if not key:
            key = settings.ENCRYPTION_KEY
        
        if isinstance(key, str):
            key = key.encode()
        
        fernet = Fernet(base64.urlsafe_b64encode(key[:32].ljust(32, b'0')))
        encrypted_data = fernet.encrypt(data.encode())
        
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str, key: Optional[str] = None) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: Encrypted data (base64 encoded)
            key: Decryption key (uses default if not provided)
            
        Returns:
            Decrypted data
        """
        from cryptography.fernet import Fernet
        
        if not key:
            key = settings.ENCRYPTION_KEY
        
        if isinstance(key, str):
            key = key.encode()
        
        fernet = Fernet(base64.urlsafe_b64encode(key[:32].ljust(32, b'0')))
        
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data)
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {str(e)}")


# Security decorators
def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented with actual permission checking
            # For now, it's a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(limit: int, window: int):
    """Decorator for rate limiting"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented with actual rate limiting
            # For now, it's a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Export utilities
__all__ = [
    "PasswordManager",
    "JWTManager", 
    "SecurityUtils",
    "RateLimiter",
    "EncryptionUtils",
    "require_permission",
    "rate_limit"
]
