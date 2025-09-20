"""
Doubts Endpoints
Core doubt submission, processing, and retrieval functionality
"""

import logging
from typing import List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import (
    get_database, 
    get_current_user, 
    get_pagination_params,
    PaginationParams,
    SubscriptionChecker,
    FeatureFlag,
    strict_rate_limit
)
from app.models.user import User
from app.schemas.doubt import (
    DoubtSubmission,
    DoubtResponse,
    DoubtListResponse,
    DoubtSearchRequest
)
from app.services.doubt_service import DoubtService
from app.services.student_service import StudentService
from app.core.processing.image_processor import ImageProcessor
from app.exceptions import (
    ValidationError,
    QuotaExceededError,
    AIProcessingError
)


router = APIRouter()
logger = logging.getLogger(__name__)

# Dependencies
require_premium = SubscriptionChecker(required_tier="premium", feature="doubts")
check_pil_enabled = FeatureFlag("pil")
check_mcp_enabled = FeatureFlag("mcp")


@router.post("/submit", response_model=DoubtResponse, status_code=status.HTTP_201_CREATED)
async def submit_doubt(
    doubt_data: DoubtSubmission,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database),
    _: Any = Depends(strict_rate_limit)
):
    """
    Submit a new doubt for AI processing
    
    **Core fynqAI functionality:**
    - Process Intelligence Layer (PIL) for hallucination-free reasoning
    - Multi-Context Personalization (MCP) for adaptive explanations
    - Advanced RAG for exam-specific knowledge retrieval
    
    **Request Format:**
    ```json
    {
        "question": "Solve: ∫(x²+3x+2)dx from 0 to 2",
        "subject": "mathematics",
        "topic": "integration",
        "image_url": "https://...",
        "difficulty_preference": "adaptive",
        "context": {
            "previous_concepts": ["basic_integration"],
            "weak_areas": ["integration_by_parts"],
            "learning_pace": "moderate"
        }
    }
    ```
    """
    doubt_service = DoubtService(db)
    student_service = StudentService(db)
    
    try:
        # Get student profile for personalization
        student_profile = await student_service.get_student_profile(current_user.id)
        
        # Check subscription limits
        await doubt_service.check_daily_limits(current_user.id, current_user.subscription_tier)
        
        # Process the doubt
        doubt_response = await doubt_service.process_doubt(
            user_id=current_user.id,
            doubt_data=doubt_data,
            student_profile=student_profile
        )
        
        logger.info(
            "Doubt processed successfully",
            extra={
                "user_id": str(current_user.id),
                "doubt_id": str(doubt_response.doubt_id),
                "subject": doubt_data.subject,
                "processing_time_ms": doubt_response.processing_time_ms
            }
        )
        
        return doubt_response
        
    except QuotaExceededError:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Daily doubt limit exceeded. Upgrade your subscription for more doubts."
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except AIProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI processing failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Doubt submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process doubt. Please try again."
        )


@router.post("/submit-with-image", response_model=DoubtResponse, status_code=status.HTTP_201_CREATED)
async def submit_doubt_with_image(
    question: str = Form(..., description="Question text"),
    subject: str = Form(..., description="Subject (mathematics, physics, chemistry, biology)"),
    topic: Optional[str] = Form(None, description="Topic within subject"),
    difficulty_preference: str = Form(default="adaptive", description="Difficulty preference"),
    image: UploadFile = File(..., description="Question image (JPG, PNG, WebP)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database),
    _: Any = Depends(strict_rate_limit)
):
    """
    Submit a doubt with an image attachment
    
    **Features:**
    - Advanced OCR with Google Vision API
    - Image preprocessing and enhancement
    - Mathematical notation recognition
    - Handwriting recognition
    
    **Supported formats:** JPG, PNG, WebP (max 10MB)
    """
    doubt_service = DoubtService(db)
    image_processor = ImageProcessor()
    
    try:
        # Validate image
        if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported image format. Use JPG, PNG, or WebP."
            )
        
        # Check file size (10MB limit)
        if image.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image too large. Maximum size is 10MB."
            )
        
        # Process image to extract text
        image_content = await image.read()
        extracted_text = await image_processor.extract_text(image_content)
        
        # Combine extracted text with provided question
        full_question = f"{question}\n\n[Extracted from image: {extracted_text}]" if extracted_text else question
        
        # Create doubt submission
        doubt_data = DoubtSubmission(
            question=full_question,
            subject=subject,
            topic=topic,
            difficulty_preference=difficulty_preference,
            has_image=True,
            context={}
        )
        
        # Get student profile
        student_service = StudentService(db)
        student_profile = await student_service.get_student_profile(current_user.id)
        
        # Process the doubt
        doubt_response = await doubt_service.process_doubt(
            user_id=current_user.id,
            doubt_data=doubt_data,
            student_profile=student_profile,
            image_data=image_content
        )
        
        return doubt_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image doubt submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process image doubt. Please try again."
        )


@router.get("/", response_model=DoubtListResponse)
async def get_user_doubts(
    pagination: PaginationParams = Depends(get_pagination_params),
    subject: Optional[str] = None,
    topic: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Get user's doubt history with pagination and filtering
    
    **Query Parameters:**
    - **page**: Page number (default: 1)
    - **size**: Page size (default: 20, max: 100)
    - **subject**: Filter by subject
    - **topic**: Filter by topic
    """
    doubt_service = DoubtService(db)
    
    try:
        doubts = await doubt_service.get_user_doubts(
            user_id=current_user.id,
            offset=pagination.offset,
            limit=pagination.limit,
            subject=subject,
            topic=topic
        )
        
        total_count = await doubt_service.count_user_doubts(
            user_id=current_user.id,
            subject=subject,
            topic=topic
        )
        
        return DoubtListResponse(
            doubts=doubts,
            total=total_count,
            page=pagination.page,
            size=pagination.size,
            pages=max(1, (total_count + pagination.size - 1) // pagination.size)
        )
        
    except Exception as e:
        logger.error(f"Failed to fetch doubts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch doubts"
        )


@router.get("/{doubt_id}", response_model=DoubtResponse)
async def get_doubt_by_id(
    doubt_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Get a specific doubt by ID
    
    Returns detailed doubt information including:
    - Original question and context
    - AI-generated answer with step-by-step solution
    - Personalization metadata
    - Related concepts and suggestions
    """
    doubt_service = DoubtService(db)
    
    try:
        doubt = await doubt_service.get_doubt_by_id(doubt_id, current_user.id)
        
        if not doubt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Doubt not found"
            )
        
        return doubt
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch doubt {doubt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch doubt"
        )


@router.post("/search", response_model=DoubtListResponse)
async def search_doubts(
    search_request: DoubtSearchRequest,
    pagination: PaginationParams = Depends(get_pagination_params),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Semantic search through user's doubt history
    
    **Features:**
    - Vector-based semantic search
    - Natural language queries
    - Topic and concept matching
    - Difficulty-based filtering
    
    **Example:**
    ```json
    {
        "query": "integration by parts problems",
        "subject": "mathematics",
        "difficulty_level": "advanced"
    }
    ```
    """
    doubt_service = DoubtService(db)
    
    try:
        search_results = await doubt_service.search_doubts(
            user_id=current_user.id,
            search_query=search_request.query,
            subject=search_request.subject,
            difficulty_level=search_request.difficulty_level,
            offset=pagination.offset,
            limit=pagination.limit
        )
        
        return search_results
        
    except Exception as e:
        logger.error(f"Doubt search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed. Please try again."
        )


@router.delete("/{doubt_id}")
async def delete_doubt(
    doubt_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Delete a doubt from user's history
    """
    doubt_service = DoubtService(db)
    
    try:
        success = await doubt_service.delete_doubt(doubt_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Doubt not found"
            )
        
        return {"message": "Doubt deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete doubt {doubt_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete doubt"
        )


@router.get("/{doubt_id}/similar", response_model=List[DoubtResponse])
async def get_similar_doubts(
    doubt_id: UUID,
    limit: int = 5,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database)
):
    """
    Get similar doubts based on semantic similarity
    
    Uses vector embeddings to find conceptually similar questions
    """
    doubt_service = DoubtService(db)
    
    try:
        similar_doubts = await doubt_service.get_similar_doubts(
            doubt_id=doubt_id,
            user_id=current_user.id,
            limit=limit
        )
        
        return similar_doubts
        
    except Exception as e:
        logger.error(f"Failed to find similar doubts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find similar doubts"
        )


@router.post("/{doubt_id}/regenerate", response_model=DoubtResponse)
async def regenerate_answer(
    doubt_id: UUID,
    difficulty_preference: Optional[str] = None,
    explanation_style: Optional[str] = None,
    current_user: User = Depends(require_premium),
    db: AsyncSession = Depends(get_database),
    _: Any = Depends(strict_rate_limit)
):
    """
    Regenerate answer with different parameters (Premium feature)
    
    **Premium Features:**
    - Alternative explanation styles
    - Adjusted difficulty levels
    - Different solution approaches
    - Enhanced step-by-step breakdowns
    """
    doubt_service = DoubtService(db)
    
    try:
        regenerated_doubt = await doubt_service.regenerate_answer(
            doubt_id=doubt_id,
            user_id=current_user.id,
            difficulty_preference=difficulty_preference,
            explanation_style=explanation_style
        )
        
        if not regenerated_doubt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Doubt not found"
            )
        
        return regenerated_doubt
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to regenerate answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to regenerate answer"
        )
