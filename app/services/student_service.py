"""
Student Management Service
Core business logic for student profile and academic data management
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, and_
from sqlalchemy.orm import selectinload

from app.models import Student, User, Doubt, Analytics
from app.schemas import (
    StudentCreate, StudentUpdate, StudentResponse, StudentAnalytics
)
from app.database import CacheManager
from app.exceptions import ServiceError, ValidationError, NotFoundError


logger = logging.getLogger(__name__)


class StudentService:
    """Service for student profile and academic management"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.cache_ttl = 3600  # 1 hour
    
    async def create_student_profile(
        self,
        user_id: uuid.UUID,
        student_data: StudentCreate,
        session: AsyncSession
    ) -> StudentResponse:
        """
        Create a new student profile
        
        Args:
            user_id: ID of the associated user
            student_data: Student profile data
            session: Database session
            
        Returns:
            Created student profile
        """
        try:
            # Verify user exists and is active
            await self._get_user_by_id(user_id, session)
            
            # Check if student profile already exists for this academic year
            existing_student = await self._get_existing_student(
                user_id, student_data.academic_year, session
            )
            
            if existing_student:
                raise ValidationError(
                    f"Student profile already exists for academic year {student_data.academic_year}"
                )
            
            # Create student profile
            student = Student(
                user_id=user_id,
                grade=student_data.grade,
                academic_year=student_data.academic_year,
                board=student_data.board,
                target_exams=student_data.target_exams,
                preparation_level=student_data.preparation_level,
                learning_style=student_data.learning_style,
                preferred_language=student_data.preferred_language,
                difficulty_preference=student_data.difficulty_preference,
                strong_subjects=student_data.strong_subjects or [],
                weak_subjects=student_data.weak_subjects or [],
                subscription_tier="free",  # Default tier
                daily_doubt_limit=5,  # Default limit for free tier
                learning_patterns={},
                performance_analytics={}
            )
            
            session.add(student)
            await session.flush()
            
            # Initialize analytics record
            await self._initialize_student_analytics(student.id, session)
            
            await session.commit()
            
            # Clear cache
            await self._invalidate_student_cache(user_id)
            
            logger.info(f"Created student profile for user {user_id}")
            
            return await self._build_student_response(student, session)
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error creating student profile: {e}", exc_info=True)
            raise ServiceError(f"Failed to create student profile: {str(e)}")
    
    async def get_student_profile(
        self,
        user_id: uuid.UUID,
        session: AsyncSession,
        academic_year: Optional[str] = None
    ) -> StudentResponse:
        """Get student profile by user ID"""
        
        cache_key = f"student_profile:{user_id}:{academic_year or 'current'}"
        
        # Try cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            return StudentResponse.parse_raw(cached_data)
        
        # Get from database
        query = select(Student).options(
            selectinload(Student.user)
        ).where(Student.user_id == user_id)
        
        if academic_year:
            query = query.where(Student.academic_year == academic_year)
        else:
            # Get most recent profile
            query = query.order_by(Student.created_at.desc())
        
        result = await session.execute(query)
        student = result.scalar_one_or_none()
        
        if not student:
            raise NotFoundError("Student profile not found")
        
        response = await self._build_student_response(student, session)
        
        # Cache the result
        await self.cache.set(cache_key, response.json(), self.cache_ttl)
        
        return response
    
    async def update_student_profile(
        self,
        student_id: uuid.UUID,
        update_data: StudentUpdate,
        session: AsyncSession
    ) -> StudentResponse:
        """Update student profile"""
        
        try:
            # Get existing student
            result = await session.execute(
                select(Student)
                .options(selectinload(Student.user))
                .where(Student.id == student_id)
            )
            
            student = result.scalar_one_or_none()
            if not student:
                raise NotFoundError("Student not found")
            
            # Update fields
            update_dict = update_data.dict(exclude_unset=True)
            for field, value in update_dict.items():
                setattr(student, field, value)
            
            student.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Clear cache
            await self._invalidate_student_cache(student.user_id)
            
            logger.info(f"Updated student profile {student_id}")
            
            return await self._build_student_response(student, session)
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error updating student profile: {e}", exc_info=True)
            raise ServiceError(f"Failed to update student profile: {str(e)}")
    
    async def get_student_analytics(
        self,
        student_id: uuid.UUID,
        session: AsyncSession,
        period_days: int = 30
    ) -> StudentAnalytics:
        """Get comprehensive analytics for a student"""
        
        cache_key = f"student_analytics:{student_id}:{period_days}"
        
        # Try cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            return StudentAnalytics.parse_raw(cached_data)
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Get basic stats
        basic_stats = await self._get_basic_statistics(
            student_id, start_date, end_date, session
        )
        
        # Get topic performance
        topic_performance = await self._get_topic_performance(
            student_id, start_date, end_date, session
        )
        
        # Get learning progress
        learning_progress = await self._get_learning_progress(
            student_id, start_date, end_date, session
        )
        
        # Get study patterns
        study_patterns = await self._get_study_patterns(
            student_id, start_date, end_date, session
        )
        
        analytics = StudentAnalytics(
            total_doubts=basic_stats["total_doubts"],
            resolved_doubts=basic_stats["resolved_doubts"],
            resolution_rate=basic_stats["resolution_rate"],
            average_processing_time=basic_stats["avg_processing_time"],
            satisfaction_score=basic_stats["satisfaction_score"],
            strong_topics=topic_performance["strong_topics"],
            weak_topics=topic_performance["weak_topics"],
            learning_progress=learning_progress,
            study_patterns=study_patterns
        )
        
        # Cache for 30 minutes
        await self.cache.set(cache_key, analytics.json(), 1800)
        
        return analytics
    
    async def update_learning_patterns(
        self,
        student_id: uuid.UUID,
        session: AsyncSession,
        interaction_data: Dict[str, Any]
    ) -> None:
        """Update student learning patterns based on interactions"""
        
        try:
            result = await session.execute(
                select(Student).where(Student.id == student_id)
            )
            
            student = result.scalar_one_or_none()
            if not student:
                return
            
            # Update learning patterns
            patterns = student.learning_patterns or {}
            
            # Update interaction count
            patterns["total_interactions"] = patterns.get("total_interactions", 0) + 1
            
            # Update time-based patterns
            current_hour = datetime.utcnow().hour
            hourly_patterns = patterns.get("hourly_activity", {})
            hourly_patterns[str(current_hour)] = hourly_patterns.get(str(current_hour), 0) + 1
            patterns["hourly_activity"] = hourly_patterns
            
            # Update subject preferences
            if "subject" in interaction_data:
                subject_patterns = patterns.get("subject_preferences", {})
                subject = interaction_data["subject"]
                subject_patterns[subject] = subject_patterns.get(subject, 0) + 1
                patterns["subject_preferences"] = subject_patterns
            
            # Update difficulty preferences
            if "difficulty" in interaction_data:
                difficulty_patterns = patterns.get("difficulty_preferences", {})
                difficulty = interaction_data["difficulty"]
                difficulty_patterns[difficulty] = difficulty_patterns.get(difficulty, 0) + 1
                patterns["difficulty_preferences"] = difficulty_patterns
            
            # Update response times
            if "response_time" in interaction_data:
                response_times = patterns.get("response_times", [])
                response_times.append(interaction_data["response_time"])
                # Keep only last 100 response times
                if len(response_times) > 100:
                    response_times = response_times[-100:]
                patterns["response_times"] = response_times
                patterns["avg_response_time"] = sum(response_times) / len(response_times)
            
            # Update the student record
            await session.execute(
                update(Student)
                .where(Student.id == student_id)
                .values(
                    learning_patterns=patterns,
                    last_activity_at=datetime.utcnow()
                )
            )
            
            await session.commit()
            
            # Clear analytics cache
            await self.cache.flush_pattern(f"student_analytics:{student_id}:*")
            
        except Exception as e:
            logger.error(f"Error updating learning patterns: {e}", exc_info=True)
    
    async def get_student_recommendations(
        self,
        student_id: uuid.UUID,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Get personalized recommendations for student"""
        
        cache_key = f"student_recommendations:{student_id}"
        
        # Try cache first
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            return eval(cached_data)  # Safe for our controlled data
        
        # Get student data
        result = await session.execute(
            select(Student)
            .options(selectinload(Student.user))
            .where(Student.id == student_id)
        )
        
        student = result.scalar_one_or_none()
        if not student:
            return {}
        
        # Get recent performance data
        recent_doubts = await self._get_recent_performance(student_id, session)
        
        recommendations = {
            "study_schedule": await self._generate_study_schedule(student, recent_doubts),
            "weak_topics": await self._identify_weak_topics(student, recent_doubts),
            "practice_suggestions": await self._generate_practice_suggestions(student),
            "exam_preparation": await self._generate_exam_prep_plan(student),
            "learning_style_tips": await self._generate_learning_tips(student)
        }
        
        # Cache for 2 hours
        await self.cache.set(cache_key, str(recommendations), 7200)
        
        return recommendations
    
    async def reset_daily_limits(self, session: AsyncSession) -> None:
        """Reset daily doubt limits for all students (called by scheduler)"""
        
        try:
            await session.execute(
                update(Student).values(daily_doubts_used=0)
            )
            
            await session.commit()
            
            logger.info("Reset daily doubt limits for all students")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error resetting daily limits: {e}", exc_info=True)
    
    # Helper methods
    async def _get_user_by_id(self, user_id: uuid.UUID, session: AsyncSession) -> User:
        """Get user by ID with validation"""
        
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        
        user = result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User not found")
        
        if not user.is_active:
            raise ValidationError("User account is inactive")
        
        return user
    
    async def _get_existing_student(
        self,
        user_id: uuid.UUID,
        academic_year: str,
        session: AsyncSession
    ) -> Optional[Student]:
        """Check if student profile exists for academic year"""
        
        result = await session.execute(
            select(Student).where(
                and_(
                    Student.user_id == user_id,
                    Student.academic_year == academic_year
                )
            )
        )
        
        return result.scalar_one_or_none()
    
    async def _initialize_student_analytics(
        self,
        student_id: uuid.UUID,
        session: AsyncSession
    ) -> None:
        """Initialize analytics record for new student"""
        
        analytics = Analytics(
            entity_type="student",
            entity_id=str(student_id),
            period_type="daily",
            period_start=datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0),
            period_end=datetime.utcnow().replace(hour=23, minute=59, second=59, microsecond=999999),
            metrics={
                "doubts_asked": 0,
                "doubts_resolved": 0,
                "study_time_minutes": 0,
                "topics_covered": []
            }
        )
        
        session.add(analytics)
    
    async def _build_student_response(
        self,
        student: Student,
        session: AsyncSession
    ) -> StudentResponse:
        """Build comprehensive student response"""
        
        # Get additional computed fields if needed
        return StudentResponse.from_orm(student)
    
    async def _invalidate_student_cache(self, user_id: uuid.UUID) -> None:
        """Invalidate all cache entries for a student"""
        
        patterns = [
            f"student_profile:{user_id}:*",
            f"student_analytics:{user_id}:*",
            f"student_recommendations:{user_id}"
        ]
        
        for pattern in patterns:
            await self.cache.flush_pattern(pattern)
    
    async def _get_basic_statistics(
        self,
        student_id: uuid.UUID,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Get basic statistics for student"""
        
        # Get doubt statistics
        doubt_stats = await session.execute(
            select(
                func.count(Doubt.id).label("total_doubts"),
                func.count(func.nullif(Doubt.is_resolved, False)).label("resolved_doubts"),
                func.avg(Doubt.processing_time_ms).label("avg_processing_time"),
                func.avg(Doubt.student_satisfaction).label("avg_satisfaction")
            )
            .where(
                and_(
                    Doubt.student_id == student_id,
                    Doubt.created_at.between(start_date, end_date)
                )
            )
        )
        
        stats = doubt_stats.one()
        
        total_doubts = stats.total_doubts or 0
        resolved_doubts = stats.resolved_doubts or 0
        
        return {
            "total_doubts": total_doubts,
            "resolved_doubts": resolved_doubts,
            "resolution_rate": resolved_doubts / total_doubts if total_doubts > 0 else 0.0,
            "avg_processing_time": float(stats.avg_processing_time or 0) / 1000,  # Convert to seconds
            "satisfaction_score": float(stats.avg_satisfaction or 0)
        }
    
    async def _get_topic_performance(
        self,
        student_id: uuid.UUID,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get topic-wise performance analysis"""
        
        # This would involve more complex queries
        # For now, return placeholder data
        return {
            "strong_topics": [
                {"topic": "Algebra", "accuracy": 0.85, "doubts_count": 10},
                {"topic": "Geometry", "accuracy": 0.80, "doubts_count": 8}
            ],
            "weak_topics": [
                {"topic": "Calculus", "accuracy": 0.60, "doubts_count": 15},
                {"topic": "Trigonometry", "accuracy": 0.65, "doubts_count": 12}
            ]
        }
    
    async def _get_learning_progress(
        self,
        student_id: uuid.UUID,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Get learning progress metrics"""
        
        return {
            "concepts_mastered": 25,
            "concepts_in_progress": 8,
            "concepts_to_learn": 45,
            "weekly_progress": [0.1, 0.15, 0.2, 0.18]  # Last 4 weeks
        }
    
    async def _get_study_patterns(
        self,
        student_id: uuid.UUID,
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Get study pattern analysis"""
        
        return {
            "peak_hours": [16, 17, 18, 19],  # 4-7 PM
            "average_session_duration": 45,  # minutes
            "preferred_subjects": ["Mathematics", "Physics"],
            "consistency_score": 0.75
        }
    
    async def _get_recent_performance(
        self,
        student_id: uuid.UUID,
        session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get recent performance data"""
        
        result = await session.execute(
            select(Doubt)
            .where(Student.id == student_id)
            .order_by(Doubt.created_at.desc())
            .limit(20)
        )
        
        doubts = result.scalars().all()
        
        return [
            {
                "subject": doubt.subject_id,
                "topic": doubt.topic_id,
                "difficulty": doubt.difficulty_level,
                "resolved": doubt.is_resolved,
                "satisfaction": doubt.student_satisfaction
            }
            for doubt in doubts
        ]
    
    async def _generate_study_schedule(
        self,
        student: Student,
        recent_performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate personalized study schedule"""
        
        return {
            "daily_study_hours": 3,
            "subjects_allocation": {
                "Mathematics": 1.5,
                "Physics": 1.0,
                "Chemistry": 0.5
            },
            "break_intervals": 15,  # minutes
            "revision_schedule": "weekly"
        }
    
    async def _identify_weak_topics(
        self,
        student: Student,
        recent_performance: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify topics needing more attention"""
        
        return [
            {"topic": "Calculus", "priority": "high", "recommended_hours": 2},
            {"topic": "Optics", "priority": "medium", "recommended_hours": 1}
        ]
    
    async def _generate_practice_suggestions(
        self,
        student: Student
    ) -> List[Dict[str, Any]]:
        """Generate practice suggestions"""
        
        return [
            {
                "type": "practice_problems",
                "subject": "Mathematics",
                "count": 10,
                "difficulty": "medium"
            },
            {
                "type": "mock_test",
                "exam": "JEE_MAIN",
                "duration": 180
            }
        ]
    
    async def _generate_exam_prep_plan(
        self,
        student: Student
    ) -> Dict[str, Any]:
        """Generate exam preparation plan"""
        
        return {
            "target_exam": student.target_exams[0] if student.target_exams else "JEE_MAIN",
            "preparation_phase": "intermediate",
            "weeks_remaining": 12,
            "focus_areas": ["Mathematics", "Physics"],
            "mock_test_frequency": "weekly"
        }
    
    async def _generate_learning_tips(
        self,
        student: Student
    ) -> List[str]:
        """Generate learning style specific tips"""
        
        tips = {
            "visual": [
                "Use diagrams and charts for better understanding",
                "Create mind maps for complex topics",
                "Watch educational videos"
            ],
            "auditory": [
                "Discuss problems with peers",
                "Listen to educational podcasts",
                "Read problems aloud"
            ],
            "kinesthetic": [
                "Use hands-on experiments",
                "Take breaks and move around",
                "Use physical models"
            ]
        }
        
        return tips.get(student.learning_style, tips["visual"])


# Service factory function
def get_student_service(cache_manager: CacheManager) -> StudentService:
    """Factory function to create student service"""
    return StudentService(cache_manager)
