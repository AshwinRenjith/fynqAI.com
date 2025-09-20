"""
Multi-Context Personalization (MCP) Engine
Adaptive learning framework for personalized educational experiences
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import json

from app.config import get_settings
from app.exceptions import MCPProcessingError


logger = logging.getLogger(__name__)
settings = get_settings()


class LearningStyle(Enum):
    """Different learning style preferences"""
    VISUAL = "visual"
    ANALYTICAL = "analytical"
    STEP_BY_STEP = "step_by_step"
    CONCEPTUAL = "conceptual"
    PRACTICAL = "practical"


class DifficultyLevel(Enum):
    """Difficulty adaptation levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ADAPTIVE = "adaptive"


class LearningPace(Enum):
    """Learning pace preferences"""
    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"
    ADAPTIVE = "adaptive"


class StudentProfile:
    """Comprehensive student learning profile"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.learning_style: LearningStyle = LearningStyle.STEP_BY_STEP
        self.difficulty_preference: DifficultyLevel = DifficultyLevel.ADAPTIVE
        self.learning_pace: LearningPace = LearningPace.MODERATE
        
        # Subject proficiency
        self.strong_subjects: List[str] = []
        self.weak_subjects: List[str] = []
        
        # Topic-specific performance
        self.topic_performance: Dict[str, float] = {}
        self.concept_mastery: Dict[str, float] = {}
        
        # Learning patterns
        self.session_patterns: Dict[str, Any] = {}
        self.error_patterns: Dict[str, int] = {}
        self.success_patterns: Dict[str, int] = {}
        
        # Temporal preferences
        self.peak_learning_hours: List[int] = []
        self.session_duration_preference: int = 30  # minutes
        
        # Personalization metrics
        self.engagement_score: float = 0.5
        self.satisfaction_score: float = 0.5
        self.learning_velocity: float = 0.5
        
        # Adaptive parameters
        self.explanation_depth_preference: float = 0.5
        self.example_preference_count: int = 2
        self.formula_preference: bool = True
        self.visual_aid_preference: bool = False


class StudentProfiler:
    """Analyzes student behavior to build comprehensive learning profiles"""
    
    def __init__(self):
        self.profile_cache: Dict[str, StudentProfile] = {}
    
    async def get_student_profile(self, user_id: str) -> StudentProfile:
        """Get or create student profile with latest analytics"""
        
        if user_id in self.profile_cache:
            profile = self.profile_cache[user_id]
            # Update profile with recent data
            await self._update_profile_metrics(profile)
        else:
            profile = await self._create_initial_profile(user_id)
            self.profile_cache[user_id] = profile
        
        return profile
    
    async def _create_initial_profile(self, user_id: str) -> StudentProfile:
        """Create initial student profile based on onboarding data"""
        
        profile = StudentProfile(user_id)
        
        # Load any existing profile data from database
        # This would typically fetch from user preferences, past sessions, etc.
        
        return profile
    
    async def _update_profile_metrics(self, profile: StudentProfile):
        """Update profile metrics based on recent learning data"""
        
        # Analyze recent sessions for pattern updates
        recent_sessions = await self._get_recent_sessions(profile.user_id)
        
        # Update performance metrics
        await self._update_performance_metrics(profile, recent_sessions)
        
        # Update learning patterns
        await self._update_learning_patterns(profile, recent_sessions)
        
        # Update preferences based on feedback
        await self._update_preferences(profile, recent_sessions)
    
    async def _get_recent_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get recent learning sessions for analysis"""
        # This would fetch from database
        return []
    
    async def _update_performance_metrics(
        self, 
        profile: StudentProfile, 
        sessions: List[Dict[str, Any]]
    ):
        """Update performance metrics based on session data"""
        
        if not sessions:
            return
        
        # Calculate topic performance
        topic_scores = {}
        for session in sessions:
            for doubt in session.get("doubts", []):
                topic = doubt.get("topic")
                feedback_rating = doubt.get("feedback", {}).get("rating", 0)
                
                if topic:
                    if topic not in topic_scores:
                        topic_scores[topic] = []
                    topic_scores[topic].append(feedback_rating / 5.0)
        
        # Update profile topic performance
        for topic, scores in topic_scores.items():
            if scores:
                profile.topic_performance[topic] = sum(scores) / len(scores)
        
        # Update subject strength/weakness categorization
        await self._categorize_subject_performance(profile)
    
    async def _categorize_subject_performance(self, profile: StudentProfile):
        """Categorize subjects as strong or weak based on performance"""
        
        subject_scores = {}
        
        # Aggregate topic scores by subject
        for topic, score in profile.topic_performance.items():
            subject = await self._get_subject_for_topic(topic)
            if subject:
                if subject not in subject_scores:
                    subject_scores[subject] = []
                subject_scores[subject].append(score)
        
        # Calculate average scores and categorize
        for subject, scores in subject_scores.items():
            avg_score = sum(scores) / len(scores)
            
            if avg_score >= 0.7:
                if subject not in profile.strong_subjects:
                    profile.strong_subjects.append(subject)
                if subject in profile.weak_subjects:
                    profile.weak_subjects.remove(subject)
            elif avg_score <= 0.4:
                if subject not in profile.weak_subjects:
                    profile.weak_subjects.append(subject)
                if subject in profile.strong_subjects:
                    profile.strong_subjects.remove(subject)
    
    async def _get_subject_for_topic(self, topic: str) -> Optional[str]:
        """Map topic to subject"""
        topic_subject_mapping = {
            # Mathematics topics
            "algebra": "mathematics",
            "calculus": "mathematics",
            "geometry": "mathematics",
            "trigonometry": "mathematics",
            "statistics": "mathematics",
            
            # Physics topics
            "mechanics": "physics",
            "thermodynamics": "physics",
            "electromagnetism": "physics",
            "optics": "physics",
            "modern_physics": "physics",
            
            # Chemistry topics
            "organic_chemistry": "chemistry",
            "inorganic_chemistry": "chemistry",
            "physical_chemistry": "chemistry",
            
            # Biology topics
            "cell_biology": "biology",
            "genetics": "biology",
            "ecology": "biology",
            "human_physiology": "biology"
        }
        
        return topic_subject_mapping.get(topic.lower())
    
    async def _update_learning_patterns(
        self, 
        profile: StudentProfile, 
        sessions: List[Dict[str, Any]]
    ):
        """Update learning patterns based on session behavior"""
        
        # Analyze session timing patterns
        session_hours = []
        session_durations = []
        
        for session in sessions:
            start_time = session.get("start_time")
            end_time = session.get("end_time")
            
            if start_time and end_time:
                start_dt = datetime.fromisoformat(start_time)
                end_dt = datetime.fromisoformat(end_time)
                
                session_hours.append(start_dt.hour)
                duration = (end_dt - start_dt).total_seconds() / 60  # minutes
                session_durations.append(duration)
        
        # Update peak learning hours
        if session_hours:
            hour_frequency = {}
            for hour in session_hours:
                hour_frequency[hour] = hour_frequency.get(hour, 0) + 1
            
            # Find top 3 most frequent hours
            sorted_hours = sorted(hour_frequency.items(), key=lambda x: x[1], reverse=True)
            profile.peak_learning_hours = [hour for hour, _ in sorted_hours[:3]]
        
        # Update session duration preference
        if session_durations:
            avg_duration = sum(session_durations) / len(session_durations)
            profile.session_duration_preference = int(avg_duration)
    
    async def _update_preferences(
        self, 
        profile: StudentProfile, 
        sessions: List[Dict[str, Any]]
    ):
        """Update learning preferences based on feedback and behavior"""
        
        # Analyze explanation preferences from feedback
        explanation_ratings = []
        
        for session in sessions:
            for doubt in session.get("doubts", []):
                feedback = doubt.get("feedback", {})
                explanation_rating = feedback.get("explanation_rating")
                explanation_style = doubt.get("explanation_style")
                
                if explanation_rating and explanation_style:
                    explanation_ratings.append({
                        "rating": explanation_rating,
                        "style": explanation_style
                    })
        
        # Update learning style preference based on highest-rated explanations
        if explanation_ratings:
            style_ratings = {}
            for rating_data in explanation_ratings:
                style = rating_data["style"]
                rating = rating_data["rating"]
                
                if style not in style_ratings:
                    style_ratings[style] = []
                style_ratings[style].append(rating)
            
            # Find best-performing style
            best_style = None
            best_avg_rating = 0
            
            for style, ratings in style_ratings.items():
                avg_rating = sum(ratings) / len(ratings)
                if avg_rating > best_avg_rating:
                    best_avg_rating = avg_rating
                    best_style = style
            
            if best_style and best_avg_rating > 3.5:  # Only update if significantly good
                try:
                    profile.learning_style = LearningStyle(best_style)
                except ValueError:
                    pass  # Keep existing style if mapping fails


class AdaptiveEngine:
    """Core engine for adapting content to student preferences"""
    
    def __init__(self):
        self.profiler = StudentProfiler()
    
    async def personalize_response(
        self, 
        user_id: str, 
        question: str, 
        base_response: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Personalize response based on student profile and context
        
        Args:
            user_id: Student user ID
            question: Original question
            base_response: Base AI response to personalize
            context: Additional context about the learning session
        
        Returns:
            Personalized response adapted to student preferences
        """
        try:
            # Get student profile
            profile = await self.profiler.get_student_profile(user_id)
            
            # Analyze question context
            question_analysis = await self._analyze_question(question, context or {})
            
            # Adapt response based on profile
            personalized_response = await self._adapt_response(
                base_response, profile, question_analysis
            )
            
            # Add personalization metadata
            personalized_response["personalization"] = {
                "learning_style": profile.learning_style.value,
                "difficulty_level": profile.difficulty_preference.value,
                "adaptations_applied": await self._get_adaptations_applied(profile),
                "confidence": await self._calculate_personalization_confidence(profile)
            }
            
            logger.info(
                "Response personalized successfully",
                extra={
                    "user_id": user_id,
                    "learning_style": profile.learning_style.value,
                    "adaptations_count": len(personalized_response["personalization"]["adaptations_applied"])
                }
            )
            
            return personalized_response
            
        except Exception as e:
            logger.error(f"Personalization failed: {e}", exc_info=True)
            # Return base response if personalization fails
            return base_response
    
    async def _analyze_question(
        self, 
        question: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze question to understand personalization requirements"""
        
        analysis = {
            "subject": context.get("subject", "unknown"),
            "topic": context.get("topic", "unknown"),
            "difficulty_indicators": [],
            "question_type": "unknown",
            "concepts_involved": [],
            "mathematical_complexity": "medium"
        }
        
        # Simple question analysis - can be enhanced with NLP
        question_lower = question.lower()
        
        # Detect question type
        if "solve" in question_lower or "find" in question_lower:
            analysis["question_type"] = "problem_solving"
        elif "explain" in question_lower or "why" in question_lower:
            analysis["question_type"] = "conceptual"
        elif "prove" in question_lower or "show" in question_lower:
            analysis["question_type"] = "proof"
        
        # Detect difficulty indicators
        if any(term in question_lower for term in ["advanced", "complex", "difficult"]):
            analysis["difficulty_indicators"].append("high")
        if any(term in question_lower for term in ["basic", "simple", "elementary"]):
            analysis["difficulty_indicators"].append("low")
        
        return analysis
    
    async def _adapt_response(
        self, 
        base_response: Dict[str, Any], 
        profile: StudentProfile, 
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt response based on student profile and question analysis"""
        
        adapted_response = base_response.copy()
        
        # Adapt explanation style
        adapted_response = await self._adapt_explanation_style(
            adapted_response, profile.learning_style
        )
        
        # Adapt difficulty level
        adapted_response = await self._adapt_difficulty_level(
            adapted_response, profile, question_analysis
        )
        
        # Add personalized examples
        adapted_response = await self._add_personalized_examples(
            adapted_response, profile, question_analysis
        )
        
        # Adapt visual elements
        adapted_response = await self._adapt_visual_elements(
            adapted_response, profile
        )
        
        # Add concept connections
        adapted_response = await self._add_concept_connections(
            adapted_response, profile, question_analysis
        )
        
        return adapted_response
    
    async def _adapt_explanation_style(
        self, 
        response: Dict[str, Any], 
        learning_style: LearningStyle
    ) -> Dict[str, Any]:
        """Adapt explanation style based on learning preference"""
        
        if learning_style == LearningStyle.STEP_BY_STEP:
            # Ensure detailed step-by-step breakdown
            response["explanation_style"] = "step_by_step"
            response["include_intermediate_steps"] = True
            
        elif learning_style == LearningStyle.CONCEPTUAL:
            # Focus on concepts and understanding
            response["explanation_style"] = "conceptual"
            response["emphasize_concepts"] = True
            response["include_concept_explanations"] = True
            
        elif learning_style == LearningStyle.VISUAL:
            # Emphasize visual and graphical elements
            response["explanation_style"] = "visual"
            response["include_diagrams"] = True
            response["visual_aids_suggested"] = True
            
        elif learning_style == LearningStyle.ANALYTICAL:
            # Focus on logical reasoning and analysis
            response["explanation_style"] = "analytical"
            response["include_reasoning_analysis"] = True
            response["emphasize_logic"] = True
            
        elif learning_style == LearningStyle.PRACTICAL:
            # Emphasize practical applications
            response["explanation_style"] = "practical"
            response["include_real_world_examples"] = True
            response["practical_applications"] = True
        
        return response
    
    async def _adapt_difficulty_level(
        self, 
        response: Dict[str, Any], 
        profile: StudentProfile, 
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt response difficulty based on student profile"""
        
        subject = question_analysis.get("subject", "unknown")
        
        # Determine appropriate difficulty level
        if subject in profile.strong_subjects:
            # Student is strong in this subject - can handle more advanced explanations
            response["difficulty_level"] = "advanced"
            response["include_advanced_concepts"] = True
            
        elif subject in profile.weak_subjects:
            # Student struggles with this subject - provide more basic explanations
            response["difficulty_level"] = "basic"
            response["include_prerequisite_review"] = True
            response["emphasize_fundamentals"] = True
            
        else:
            # Default to intermediate level
            response["difficulty_level"] = "intermediate"
        
        # Adjust explanation depth
        if profile.explanation_depth_preference > 0.7:
            response["explanation_depth"] = "detailed"
        elif profile.explanation_depth_preference < 0.3:
            response["explanation_depth"] = "concise"
        else:
            response["explanation_depth"] = "standard"
        
        return response
    
    async def _add_personalized_examples(
        self, 
        response: Dict[str, Any], 
        profile: StudentProfile, 
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add personalized examples based on student preferences"""
        
        # Determine number of examples based on preference
        example_count = min(profile.example_preference_count, 3)
        
        if example_count > 0:
            response["include_examples"] = True
            response["example_count"] = example_count
            
            # Tailor example types to learning style
            if profile.learning_style == LearningStyle.PRACTICAL:
                response["example_types"] = ["real_world", "application"]
            elif profile.learning_style == LearningStyle.VISUAL:
                response["example_types"] = ["graphical", "visual"]
            else:
                response["example_types"] = ["step_by_step", "conceptual"]
        
        return response
    
    async def _adapt_visual_elements(
        self, 
        response: Dict[str, Any], 
        profile: StudentProfile
    ) -> Dict[str, Any]:
        """Adapt visual elements based on preferences"""
        
        if profile.visual_aid_preference:
            response["include_visual_aids"] = True
            response["suggest_diagrams"] = True
            response["visual_learning_enhanced"] = True
        
        if profile.formula_preference:
            response["emphasize_formulas"] = True
            response["include_formula_derivations"] = True
        
        return response
    
    async def _add_concept_connections(
        self, 
        response: Dict[str, Any], 
        profile: StudentProfile, 
        question_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add connections to related concepts based on student's learning history"""
        
        topic = question_analysis.get("topic", "")
        subject = question_analysis.get("subject", "")
        
        # Find related concepts the student has studied
        related_concepts = []
        
        for concept, mastery_level in profile.concept_mastery.items():
            if (subject in concept or topic in concept) and mastery_level > 0.6:
                related_concepts.append(concept)
        
        if related_concepts:
            response["related_concepts"] = related_concepts[:3]  # Limit to top 3
            response["include_concept_connections"] = True
        
        # Suggest concepts to review based on weak areas
        weak_concepts = [
            concept for concept, mastery in profile.concept_mastery.items()
            if mastery < 0.4 and (subject in concept or topic in concept)
        ]
        
        if weak_concepts:
            response["concepts_to_review"] = weak_concepts[:2]
            response["include_review_suggestions"] = True
        
        return response
    
    async def _get_adaptations_applied(self, profile: StudentProfile) -> List[str]:
        """Get list of adaptations applied to the response"""
        adaptations = []
        
        adaptations.append(f"learning_style: {profile.learning_style.value}")
        adaptations.append(f"difficulty_preference: {profile.difficulty_preference.value}")
        
        if profile.visual_aid_preference:
            adaptations.append("visual_aids_enhanced")
        
        if profile.formula_preference:
            adaptations.append("formula_emphasis")
        
        if profile.strong_subjects:
            adaptations.append(f"advanced_for_subjects: {', '.join(profile.strong_subjects)}")
        
        if profile.weak_subjects:
            adaptations.append(f"basic_for_subjects: {', '.join(profile.weak_subjects)}")
        
        return adaptations
    
    async def _calculate_personalization_confidence(self, profile: StudentProfile) -> float:
        """Calculate confidence in personalization based on available data"""
        
        confidence_factors = []
        
        # Factor 1: Amount of historical data
        session_data_score = min(len(profile.session_patterns) / 10, 1.0)
        confidence_factors.append(session_data_score)
        
        # Factor 2: Topic performance data availability
        topic_data_score = min(len(profile.topic_performance) / 5, 1.0)
        confidence_factors.append(topic_data_score)
        
        # Factor 3: Learning pattern consistency
        pattern_consistency = profile.engagement_score
        confidence_factors.append(pattern_consistency)
        
        # Factor 4: Preference stability (simplified)
        preference_stability = 0.8  # Would be calculated based on preference changes over time
        confidence_factors.append(preference_stability)
        
        # Calculate weighted average
        if confidence_factors:
            overall_confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            overall_confidence = 0.5  # Default moderate confidence
        
        return round(overall_confidence, 2)
