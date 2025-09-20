"""
SQLAlchemy Models
Database model definitions for fynqAI
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
import uuid


Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class User(Base, TimestampMixin):
    """User model for authentication and basic information"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth users
    
    # Profile information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    full_name = Column(String(200), nullable=False)
    
    # Authentication
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    email_verified_at = Column(DateTime, nullable=True)
    phone_verified_at = Column(DateTime, nullable=True)
    
    # OAuth information
    google_id = Column(String(100), unique=True, nullable=True, index=True)
    oauth_provider = Column(String(50), nullable=True)
    
    # Security
    last_login_at = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    
    # Relationships
    students = relationship("Student", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint('email IS NOT NULL OR phone IS NOT NULL', name='email_or_phone_required'),
        Index('idx_user_auth', 'email', 'is_active', 'is_verified'),
    )


class Student(Base, TimestampMixin):
    """Student profile with academic information"""
    __tablename__ = "students"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Academic information
    grade = Column(Integer, nullable=False)  # 11, 12
    academic_year = Column(String(10), nullable=False)  # "2024-25"
    board = Column(String(50), nullable=False)  # "CBSE", "ICSE", "State Board"
    
    # Exam preparation
    target_exams = Column(ARRAY(String), nullable=False)  # ["JEE_MAIN", "JEE_ADVANCED", "NEET"]
    preparation_level = Column(String(20), nullable=False)  # "beginner", "intermediate", "advanced"
    
    # Learning preferences
    learning_style = Column(String(20), nullable=True)  # "visual", "auditory", "kinesthetic", "mixed"
    preferred_language = Column(String(10), default="en", nullable=False)
    difficulty_preference = Column(String(20), default="adaptive", nullable=False)
    
    # Academic performance
    strong_subjects = Column(ARRAY(String), nullable=True)
    weak_subjects = Column(ARRAY(String), nullable=True)
    
    # Progress tracking
    total_doubts_asked = Column(Integer, default=0, nullable=False)
    doubts_resolved = Column(Integer, default=0, nullable=False)
    study_streak_days = Column(Integer, default=0, nullable=False)
    last_activity_at = Column(DateTime, nullable=True)
    
    # Personalization data
    learning_patterns = Column(JSONB, default=dict, nullable=False)
    performance_analytics = Column(JSONB, default=dict, nullable=False)
    
    # Subscription and limits
    subscription_tier = Column(String(20), default="free", nullable=False)
    daily_doubt_limit = Column(Integer, default=5, nullable=False)
    daily_doubts_used = Column(Integer, default=0, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="students")
    doubts = relationship("Doubt", back_populates="student", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="student", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint('grade IN (11, 12)', name='valid_grade'),
        CheckConstraint('preparation_level IN ("beginner", "intermediate", "advanced")', name='valid_prep_level'),
        CheckConstraint('subscription_tier IN ("free", "premium", "enterprise")', name='valid_subscription'),
        Index('idx_student_academic', 'grade', 'target_exams'),
        Index('idx_student_activity', 'last_activity_at'),
        UniqueConstraint('user_id', 'academic_year', name='one_student_per_year'),
    )


class Subject(Base):
    """Subject definitions for academic content"""
    __tablename__ = "subjects"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    code = Column(String(20), unique=True, nullable=False)  # "MATH", "PHY", "CHEM", "BIO"
    description = Column(Text, nullable=True)
    
    # Subject characteristics
    is_active = Column(Boolean, default=True, nullable=False)
    difficulty_level = Column(String(20), default="medium", nullable=False)
    prerequisites = Column(ARRAY(String), nullable=True)
    
    # Relationships
    topics = relationship("Topic", back_populates="subject", cascade="all, delete-orphan")
    doubts = relationship("Doubt", back_populates="subject")


class Topic(Base):
    """Topic definitions within subjects"""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    
    name = Column(String(200), nullable=False)
    code = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    
    # Hierarchy
    parent_topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True)
    depth_level = Column(Integer, default=0, nullable=False)
    
    # Content metadata
    difficulty_level = Column(String(20), default="medium", nullable=False)
    estimated_study_hours = Column(Float, nullable=True)
    importance_score = Column(Float, default=5.0, nullable=False)
    
    # Exam relevance
    jee_relevance = Column(Float, default=0.0, nullable=False)
    neet_relevance = Column(Float, default=0.0, nullable=False)
    board_exam_relevance = Column(Float, default=0.0, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    subject = relationship("Subject", back_populates="topics")
    parent = relationship("Topic", remote_side=[id], backref="children")
    doubts = relationship("Doubt", back_populates="topic")
    
    __table_args__ = (
        CheckConstraint('difficulty_level IN ("easy", "medium", "hard")', name='valid_difficulty'),
        CheckConstraint('importance_score >= 0 AND importance_score <= 10', name='valid_importance'),
        Index('idx_topic_subject', 'subject_id', 'is_active'),
        Index('idx_topic_hierarchy', 'parent_topic_id', 'depth_level'),
        UniqueConstraint('subject_id', 'code', name='unique_topic_code_per_subject'),
    )


class Doubt(Base, TimestampMixin):
    """Student doubts/questions with AI processing"""
    __tablename__ = "doubts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True)
    
    # Question content
    question_text = Column(Text, nullable=False)
    question_type = Column(String(50), nullable=False)  # "text", "image", "mixed"
    question_images = Column(ARRAY(String), nullable=True)  # Image URLs
    
    # Context and metadata
    difficulty_level = Column(String(20), nullable=True)
    estimated_time_minutes = Column(Integer, nullable=True)
    question_source = Column(String(100), nullable=True)  # "practice", "homework", "test"
    
    # AI Processing
    processing_status = Column(String(20), default="pending", nullable=False)
    ai_response = Column(Text, nullable=True)
    ai_explanation = Column(Text, nullable=True)
    ai_confidence_score = Column(Float, nullable=True)
    reasoning_steps = Column(JSONB, default=list, nullable=False)
    
    # LLM metadata
    llm_provider = Column(String(50), nullable=True)
    llm_model = Column(String(100), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    processing_cost = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Personalization
    adapted_explanation = Column(Text, nullable=True)
    learning_level_adjusted = Column(Boolean, default=False, nullable=False)
    learning_style_adapted = Column(Boolean, default=False, nullable=False)
    
    # Resolution tracking
    is_resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime, nullable=True)
    resolution_method = Column(String(50), nullable=True)  # "ai", "human", "hybrid"
    
    # Quality metrics
    student_satisfaction = Column(Integer, nullable=True)  # 1-5 rating
    accuracy_score = Column(Float, nullable=True)
    helpfulness_score = Column(Float, nullable=True)
    
    # Relationships
    student = relationship("Student", back_populates="doubts")
    subject = relationship("Subject", back_populates="doubts")
    topic = relationship("Topic", back_populates="doubts")
    feedback = relationship("Feedback", back_populates="doubt", uselist=False)
    
    __table_args__ = (
        CheckConstraint('processing_status IN ("pending", "processing", "completed", "failed", "human_required")', 
                       name='valid_processing_status'),
        CheckConstraint('question_type IN ("text", "image", "mixed")', name='valid_question_type'),
        CheckConstraint('student_satisfaction IS NULL OR (student_satisfaction >= 1 AND student_satisfaction <= 5)', 
                       name='valid_satisfaction'),
        Index('idx_doubt_student', 'student_id', 'created_at'),
        Index('idx_doubt_subject_topic', 'subject_id', 'topic_id'),
        Index('idx_doubt_processing', 'processing_status', 'created_at'),
        Index('idx_doubt_resolution', 'is_resolved', 'resolved_at'),
    )


class Feedback(Base, TimestampMixin):
    """Student feedback on AI responses"""
    __tablename__ = "feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=False)
    doubt_id = Column(UUID(as_uuid=True), ForeignKey("doubts.id"), nullable=False)
    
    # Feedback ratings
    overall_rating = Column(Integer, nullable=False)  # 1-5
    accuracy_rating = Column(Integer, nullable=True)  # 1-5
    clarity_rating = Column(Integer, nullable=True)  # 1-5
    helpfulness_rating = Column(Integer, nullable=True)  # 1-5
    
    # Detailed feedback
    feedback_text = Column(Text, nullable=True)
    improvement_suggestions = Column(Text, nullable=True)
    
    # Specific feedback categories
    was_explanation_clear = Column(Boolean, nullable=True)
    was_answer_correct = Column(Boolean, nullable=True)
    was_solution_complete = Column(Boolean, nullable=True)
    was_difficulty_appropriate = Column(Boolean, nullable=True)
    
    # Response metadata
    response_time_seconds = Column(Integer, nullable=True)
    feedback_type = Column(String(50), default="post_resolution", nullable=False)
    
    # Sentiment analysis
    sentiment_score = Column(Float, nullable=True)  # -1 to 1
    sentiment_label = Column(String(20), nullable=True)  # "positive", "neutral", "negative"
    
    # Processing status
    is_processed = Column(Boolean, default=False, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    student = relationship("Student", back_populates="feedback")
    doubt = relationship("Doubt", back_populates="feedback")
    
    __table_args__ = (
        CheckConstraint('overall_rating >= 1 AND overall_rating <= 5', name='valid_overall_rating'),
        CheckConstraint('accuracy_rating IS NULL OR (accuracy_rating >= 1 AND accuracy_rating <= 5)', 
                       name='valid_accuracy_rating'),
        CheckConstraint('clarity_rating IS NULL OR (clarity_rating >= 1 AND clarity_rating <= 5)', 
                       name='valid_clarity_rating'),
        CheckConstraint('helpfulness_rating IS NULL OR (helpfulness_rating >= 1 AND helpfulness_rating <= 5)', 
                       name='valid_helpfulness_rating'),
        CheckConstraint('sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)', 
                       name='valid_sentiment_score'),
        Index('idx_feedback_student', 'student_id', 'created_at'),
        Index('idx_feedback_doubt', 'doubt_id'),
        Index('idx_feedback_ratings', 'overall_rating', 'accuracy_rating'),
        UniqueConstraint('doubt_id', name='one_feedback_per_doubt'),
    )


class Analytics(Base, TimestampMixin):
    """Analytics and metrics storage"""
    __tablename__ = "analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Scope
    entity_type = Column(String(50), nullable=False)  # "student", "subject", "topic", "global"
    entity_id = Column(String(255), nullable=True)
    
    # Time period
    period_type = Column(String(20), nullable=False)  # "hourly", "daily", "weekly", "monthly"
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Metrics data
    metrics = Column(JSONB, nullable=False)
    
    # Aggregation metadata
    data_version = Column(String(20), default="1.0", nullable=False)
    computation_time_ms = Column(Integer, nullable=True)
    
    __table_args__ = (
        CheckConstraint('entity_type IN ("student", "subject", "topic", "global")', name='valid_entity_type'),
        CheckConstraint('period_type IN ("hourly", "daily", "weekly", "monthly")', name='valid_period_type'),
        Index('idx_analytics_entity', 'entity_type', 'entity_id'),
        Index('idx_analytics_period', 'period_type', 'period_start', 'period_end'),
        UniqueConstraint('entity_type', 'entity_id', 'period_type', 'period_start', 
                        name='unique_analytics_record'),
    )


class SubscriptionPlan(Base, TimestampMixin):
    """Subscription plan definitions"""
    __tablename__ = "subscription_plans"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    
    # Plan details
    description = Column(Text, nullable=True)
    price_monthly = Column(Float, nullable=False)
    price_yearly = Column(Float, nullable=True)
    
    # Limits and features
    daily_doubt_limit = Column(Integer, nullable=False)
    features = Column(JSONB, default=dict, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    display_order = Column(Integer, default=0, nullable=False)
    
    # Relationships
    subscriptions = relationship("StudentSubscription", back_populates="plan")


class StudentSubscription(Base, TimestampMixin):
    """Student subscription records"""
    __tablename__ = "student_subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=False)
    plan_id = Column(Integer, ForeignKey("subscription_plans.id"), nullable=False)
    
    # Subscription period
    starts_at = Column(DateTime, nullable=False)
    ends_at = Column(DateTime, nullable=False)
    
    # Status
    status = Column(String(20), default="active", nullable=False)
    auto_renew = Column(Boolean, default=True, nullable=False)
    
    # Payment information
    payment_method = Column(String(50), nullable=True)
    payment_provider = Column(String(50), nullable=True)
    payment_reference = Column(String(255), nullable=True)
    
    # Relationships
    plan = relationship("SubscriptionPlan", back_populates="subscriptions")
    
    __table_args__ = (
        CheckConstraint('status IN ("active", "inactive", "cancelled", "expired")', name='valid_subscription_status'),
        Index('idx_subscription_student', 'student_id', 'status'),
        Index('idx_subscription_period', 'starts_at', 'ends_at'),
    )
