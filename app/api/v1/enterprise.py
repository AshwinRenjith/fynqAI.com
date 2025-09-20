"""
Enterprise Endpoints
B2B API for coaching centers and educational institutions
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import (
    get_database, 
    get_current_enterprise_user,
    api_rate_limit,
    FeatureFlag
)
from app.models.user import User
from app.schemas.enterprise import (
    BatchDoubtRequest,
    BatchDoubtResponse,
    InstitutionStats,
    StudentManagement
)
from app.services.enterprise_service import EnterpriseService


router = APIRouter()

# Feature flag check
check_enterprise_enabled = FeatureFlag("enterprise")


@router.post("/batch-doubts", response_model=BatchDoubtResponse)
async def process_batch_doubts(
    batch_request: BatchDoubtRequest,
    current_user: User = Depends(get_current_enterprise_user),
    db: AsyncSession = Depends(get_database),
    _: bool = Depends(check_enterprise_enabled),
    __: None = Depends(api_rate_limit)
):
    """Process multiple doubts in batch for enterprise clients"""
    enterprise_service = EnterpriseService(db)
    
    try:
        batch_response = await enterprise_service.process_batch_doubts(
            institution_id=current_user.id,
            batch_request=batch_request
        )
        return batch_response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.get("/stats", response_model=InstitutionStats)
async def get_institution_stats(
    days: Optional[int] = 30,
    current_user: User = Depends(get_current_enterprise_user),
    db: AsyncSession = Depends(get_database),
    _: bool = Depends(check_enterprise_enabled)
):
    """Get institution usage statistics and analytics"""
    enterprise_service = EnterpriseService(db)
    stats = await enterprise_service.get_institution_stats(current_user.id, days)
    return stats


@router.get("/students", response_model=List[StudentManagement])
async def get_managed_students(
    current_user: User = Depends(get_current_enterprise_user),
    db: AsyncSession = Depends(get_database),
    _: bool = Depends(check_enterprise_enabled)
):
    """Get list of students managed by the institution"""
    enterprise_service = EnterpriseService(db)
    students = await enterprise_service.get_managed_students(current_user.id)
    return students
