"""
Authentication Endpoints
User registration, login, token refresh, and account management
"""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_database, get_current_user, standard_rate_limit
from app.models.user import User
from app.schemas.auth import (
    UserCreate,
    UserResponse,
    Token,
    TokenRefresh,
    PasswordReset,
    PasswordResetConfirm
)
from app.services.auth_service import AuthService
from app.config import get_settings


router = APIRouter()
settings = get_settings()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_database),
    _: Any = Depends(standard_rate_limit)
):
    """
    Register a new user account
    
    - **email**: Valid email address
    - **password**: Strong password (8+ characters)
    - **full_name**: User's full name
    - **phone**: Optional phone number
    - **target_exam**: Target exam (JEE, NEET, etc.)
    """
    auth_service = AuthService(db)
    
    try:
        user = await auth_service.create_user(user_data)
        return UserResponse.from_orm(user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_database),
    _: Any = Depends(standard_rate_limit)
):
    """
    Login with email and password
    
    Returns access token and refresh token
    """
    auth_service = AuthService(db)
    
    user = await auth_service.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive"
        )
    
    # Generate tokens
    access_token = auth_service.create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    refresh_token = auth_service.create_refresh_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_database),
    _: Any = Depends(standard_rate_limit)
):
    """
    Refresh access token using refresh token
    """
    auth_service = AuthService(db)
    
    try:
        new_tokens = await auth_service.refresh_access_token(token_data.refresh_token)
        return new_tokens
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Logout user (invalidate tokens)
    """
    auth_service = AuthService(db)
    
    try:
        await auth_service.logout_user(current_user.id)
        return {"message": "Successfully logged out"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information
    """
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Update current user information
    """
    auth_service = AuthService(db)
    
    try:
        updated_user = await auth_service.update_user(current_user.id, user_update)
        return UserResponse.from_orm(updated_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/password-reset")
async def request_password_reset(
    reset_data: PasswordReset,
    db: AsyncSession = Depends(get_database),
    _: Any = Depends(standard_rate_limit)
):
    """
    Request password reset email
    """
    auth_service = AuthService(db)
    
    try:
        await auth_service.request_password_reset(reset_data.email)
        return {"message": "Password reset email sent if account exists"}
    except Exception:
        # Always return success to prevent email enumeration
        return {"message": "Password reset email sent if account exists"}


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_database),
    _: Any = Depends(standard_rate_limit)
):
    """
    Confirm password reset with token
    """
    auth_service = AuthService(db)
    
    try:
        await auth_service.confirm_password_reset(
            reset_data.token,
            reset_data.new_password
        )
        return {"message": "Password reset successfully"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )


@router.delete("/account")
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Delete user account (soft delete)
    """
    auth_service = AuthService(db)
    
    try:
        await auth_service.delete_user(current_user.id)
        return {"message": "Account deleted successfully"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )
