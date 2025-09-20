"""
Pydantic Schemas
Request/Response models for API validation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr, validator, root_validator
from enum import Enum
import uuid


# Enums for consistent value validation
class SubscriptionTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class PreparationLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class LearningStyle(str, Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    MIXED = "mixed"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    HUMAN_REQUIRED = "human_required"


class QuestionType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"


class ExamType(str, Enum):
    JEE_MAIN = "JEE_MAIN"
    JEE_ADVANCED = "JEE_ADVANCED"
    NEET = "NEET"
    BITSAT = "BITSAT"
    BOARD_EXAM = "BOARD_EXAM"


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# Base schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    
    class Config:
        from_attributes = True
        validate_assignment = True
        use_enum_values = True


class TimestampSchema(BaseSchema):
    """Schema with timestamp fields"""
    created_at: datetime
    updated_at: datetime


# User schemas
class UserBase(BaseSchema):
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, pattern=r"^\+?[1-9]\d{1,14}$")
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    
    @root_validator
    def email_or_phone_required(cls, values):
        email, phone = values.get('email'), values.get('phone')
        if not email and not phone:
            raise ValueError('Either email or phone must be provided')
        return values


class UserCreate(UserBase):
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    confirm_password: Optional[str] = None
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v


class UserUpdate(BaseSchema):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = Field(None, pattern=r"^\+?[1-9]\d{1,14}$")


class UserResponse(UserBase, TimestampSchema):
    id: uuid.UUID
    full_name: str
    is_active: bool
    is_verified: bool
    email_verified_at: Optional[datetime]
    phone_verified_at: Optional[datetime]
    last_login_at: Optional[datetime]


class UserLogin(BaseSchema):
    username: str  # Can be email or phone
    password: str


class UserLoginResponse(BaseSchema):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


# Student schemas
class StudentBase(BaseSchema):
    grade: int = Field(..., ge=11, le=12)
    academic_year: str = Field(..., pattern=r"^\d{4}-\d{2}$")
    board: str = Field(..., max_length=50)
    target_exams: List[ExamType]
    preparation_level: PreparationLevel
    learning_style: Optional[LearningStyle] = None
    preferred_language: str = Field(default="en", max_length=10)
    difficulty_preference: str = Field(default="adaptive", max_length=20)
    strong_subjects: Optional[List[str]] = None
    weak_subjects: Optional[List[str]] = None


class StudentCreate(StudentBase):
    pass


class StudentUpdate(BaseSchema):
    grade: Optional[int] = Field(None, ge=11, le=12)
    board: Optional[str] = Field(None, max_length=50)
    target_exams: Optional[List[ExamType]] = None
    preparation_level: Optional[PreparationLevel] = None
    learning_style: Optional[LearningStyle] = None
    preferred_language: Optional[str] = Field(None, max_length=10)
    difficulty_preference: Optional[str] = Field(None, max_length=20)
    strong_subjects: Optional[List[str]] = None
    weak_subjects: Optional[List[str]] = None


class StudentResponse(StudentBase, TimestampSchema):
    id: uuid.UUID
    user_id: uuid.UUID
    total_doubts_asked: int
    doubts_resolved: int
    study_streak_days: int
    last_activity_at: Optional[datetime]
    subscription_tier: SubscriptionTier
    daily_doubt_limit: int
    daily_doubts_used: int
    learning_patterns: Dict[str, Any]
    performance_analytics: Dict[str, Any]


# Subject and Topic schemas
class SubjectBase(BaseSchema):
    name: str = Field(..., max_length=100)
    code: str = Field(..., max_length=20)
    description: Optional[str] = None
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM
    prerequisites: Optional[List[str]] = None


class SubjectResponse(SubjectBase):
    id: int
    is_active: bool


class TopicBase(BaseSchema):
    name: str = Field(..., max_length=200)
    code: str = Field(..., max_length=50)
    description: Optional[str] = None
    difficulty_level: DifficultyLevel = DifficultyLevel.MEDIUM
    estimated_study_hours: Optional[float] = Field(None, ge=0)
    importance_score: float = Field(default=5.0, ge=0, le=10)
    jee_relevance: float = Field(default=0.0, ge=0, le=10)
    neet_relevance: float = Field(default=0.0, ge=0, le=10)
    board_exam_relevance: float = Field(default=0.0, ge=0, le=10)


class TopicResponse(TopicBase):
    id: int
    subject_id: int
    parent_topic_id: Optional[int]
    depth_level: int
    is_active: bool


# Doubt schemas
class DoubtBase(BaseSchema):
    question_text: str = Field(..., min_length=10, max_length=5000)
    question_type: QuestionType = QuestionType.TEXT
    question_images: Optional[List[str]] = None
    subject_id: int
    topic_id: Optional[int] = None
    difficulty_level: Optional[DifficultyLevel] = None
    question_source: Optional[str] = Field(None, max_length=100)


class DoubtCreate(DoubtBase):
    @validator('question_images')
    def validate_images_for_type(cls, v, values):
        question_type = values.get('question_type')
        if question_type == QuestionType.IMAGE and not v:
            raise ValueError('Images required for image type questions')
        if question_type == QuestionType.TEXT and v:
            raise ValueError('Images not allowed for text type questions')
        return v


class DoubtUpdate(BaseSchema):
    question_text: Optional[str] = Field(None, min_length=10, max_length=5000)
    topic_id: Optional[int] = None
    difficulty_level: Optional[DifficultyLevel] = None


class DoubtResponse(DoubtBase, TimestampSchema):
    id: uuid.UUID
    student_id: uuid.UUID
    processing_status: ProcessingStatus
    ai_response: Optional[str]
    ai_explanation: Optional[str]
    ai_confidence_score: Optional[float]
    reasoning_steps: List[Dict[str, Any]]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    tokens_used: Optional[int]
    processing_cost: Optional[float]
    processing_time_ms: Optional[int]
    adapted_explanation: Optional[str]
    learning_level_adjusted: bool
    learning_style_adapted: bool
    is_resolved: bool
    resolved_at: Optional[datetime]
    resolution_method: Optional[str]
    student_satisfaction: Optional[int] = Field(None, ge=1, le=5)
    accuracy_score: Optional[float]
    helpfulness_score: Optional[float]


class DoubtAskRequest(DoubtCreate):
    """Request schema for asking a doubt"""
    pass


class DoubtAskResponse(BaseSchema):
    """Response schema for doubt submission"""
    doubt_id: uuid.UUID
    processing_status: ProcessingStatus
    estimated_processing_time: int  # seconds
    message: str


class DoubtAnswerResponse(BaseSchema):
    """Response schema for doubt answer"""
    doubt_id: uuid.UUID
    ai_response: str
    ai_explanation: str
    confidence_score: float
    reasoning_steps: List[Dict[str, Any]]
    adapted_for_student: bool
    processing_time_ms: int
    suggestions: List[str]


# Feedback schemas
class FeedbackBase(BaseSchema):
    overall_rating: int = Field(..., ge=1, le=5)
    accuracy_rating: Optional[int] = Field(None, ge=1, le=5)
    clarity_rating: Optional[int] = Field(None, ge=1, le=5)
    helpfulness_rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = Field(None, max_length=2000)
    improvement_suggestions: Optional[str] = Field(None, max_length=1000)
    was_explanation_clear: Optional[bool] = None
    was_answer_correct: Optional[bool] = None
    was_solution_complete: Optional[bool] = None
    was_difficulty_appropriate: Optional[bool] = None


class FeedbackCreate(FeedbackBase):
    doubt_id: uuid.UUID


class FeedbackResponse(FeedbackBase, TimestampSchema):
    id: uuid.UUID
    student_id: uuid.UUID
    doubt_id: uuid.UUID
    response_time_seconds: Optional[int]
    feedback_type: str
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    sentiment_label: Optional[str]
    is_processed: bool


# Analytics schemas
class AnalyticsQuery(BaseSchema):
    entity_type: str = Field(..., pattern=r"^(student|subject|topic|global)$")
    entity_id: Optional[str] = None
    period_type: str = Field(..., pattern=r"^(hourly|daily|weekly|monthly)$")
    start_date: datetime
    end_date: datetime
    metrics: Optional[List[str]] = None


class AnalyticsResponse(BaseSchema):
    entity_type: str
    entity_id: Optional[str]
    period_type: str
    period_start: datetime
    period_end: datetime
    metrics: Dict[str, Any]
    data_version: str


class StudentAnalytics(BaseSchema):
    """Student-specific analytics"""
    total_doubts: int
    resolved_doubts: int
    resolution_rate: float
    average_processing_time: float
    satisfaction_score: float
    strong_topics: List[Dict[str, Any]]
    weak_topics: List[Dict[str, Any]]
    learning_progress: Dict[str, Any]
    study_patterns: Dict[str, Any]


# Authentication schemas
class TokenPayload(BaseSchema):
    sub: str  # Subject (user ID)
    exp: int  # Expiration time
    iat: int  # Issued at
    type: str  # Token type ("access" or "refresh")


class RefreshTokenRequest(BaseSchema):
    refresh_token: str


class PasswordResetRequest(BaseSchema):
    email: EmailStr


class PasswordResetConfirm(BaseSchema):
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class EmailVerificationRequest(BaseSchema):
    email: EmailStr


class EmailVerificationConfirm(BaseSchema):
    token: str


# Enterprise schemas
class EnterpriseUserCreate(UserCreate):
    organization_id: uuid.UUID
    role: str = Field(..., max_length=50)
    department: Optional[str] = Field(None, max_length=100)


class OrganizationBase(BaseSchema):
    name: str = Field(..., max_length=200)
    domain: str = Field(..., max_length=100)
    contact_email: EmailStr
    contact_phone: Optional[str] = None
    subscription_plan: str = Field(..., max_length=50)


class OrganizationCreate(OrganizationBase):
    admin_user: UserCreate


class OrganizationResponse(OrganizationBase, TimestampSchema):
    id: uuid.UUID
    is_active: bool
    user_count: int
    subscription_status: str


# Error schemas
class ErrorResponse(BaseSchema):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ValidationErrorResponse(BaseSchema):
    error: str = "validation_error"
    message: str
    details: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Success schemas
class SuccessResponse(BaseSchema):
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Health check schemas
class HealthCheck(BaseSchema):
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    uptime: float
    database: str
    redis: str
    llm_providers: Dict[str, str]


# Export all schemas for easy importing
__all__ = [
    # Enums
    "SubscriptionTier", "PreparationLevel", "LearningStyle", "ProcessingStatus",
    "QuestionType", "ExamType", "DifficultyLevel",
    
    # Base schemas
    "BaseSchema", "TimestampSchema",
    
    # User schemas
    "UserBase", "UserCreate", "UserUpdate", "UserResponse", "UserLogin", "UserLoginResponse",
    
    # Student schemas
    "StudentBase", "StudentCreate", "StudentUpdate", "StudentResponse",
    
    # Subject and Topic schemas
    "SubjectBase", "SubjectResponse", "TopicBase", "TopicResponse",
    
    # Doubt schemas
    "DoubtBase", "DoubtCreate", "DoubtUpdate", "DoubtResponse",
    "DoubtAskRequest", "DoubtAskResponse", "DoubtAnswerResponse",
    
    # Feedback schemas
    "FeedbackBase", "FeedbackCreate", "FeedbackResponse",
    
    # Analytics schemas
    "AnalyticsQuery", "AnalyticsResponse", "StudentAnalytics",
    
    # Authentication schemas
    "TokenPayload", "RefreshTokenRequest", "PasswordResetRequest", 
    "PasswordResetConfirm", "EmailVerificationRequest", "EmailVerificationConfirm",
    
    # Enterprise schemas
    "EnterpriseUserCreate", "OrganizationBase", "OrganizationCreate", "OrganizationResponse",
    
    # Error and Success schemas
    "ErrorResponse", "ValidationErrorResponse", "SuccessResponse",
    
    # Health check
    "HealthCheck"
]
