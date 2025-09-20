"""
Doubt Processing Service
Core business logic for handling student doubts with AI processing
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import asyncio
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from app.models import Doubt, Student
from app.schemas import DoubtCreate, DoubtResponse, DoubtAskResponse
from app.core.llm.orchestrator import LLMOrchestrator
from app.core.pil.reasoning_engine import ReasoningEngine
from app.core.mcp.adaptive_engine import AdaptiveEngine
from app.core.rag.retriever import RAGRetriever
from app.exceptions import ServiceError, ValidationError, RateLimitError


logger = logging.getLogger(__name__)


class DoubtProcessingService:
    """Service for processing student doubts with AI"""
    
    def __init__(
        self,
        llm_orchestrator: LLMOrchestrator,
        reasoning_engine: ReasoningEngine,
        adaptive_engine: AdaptiveEngine,
        rag_retriever: RAGRetriever
    ):
        self.llm_orchestrator = llm_orchestrator
        self.reasoning_engine = reasoning_engine
        self.adaptive_engine = adaptive_engine
        self.rag_retriever = rag_retriever
    
    async def submit_doubt(
        self,
        doubt_data: DoubtCreate,
        student_id: uuid.UUID,
        session: AsyncSession
    ) -> DoubtAskResponse:
        """
        Submit a new doubt for processing
        
        Args:
            doubt_data: Doubt creation data
            student_id: ID of the student asking the doubt
            session: Database session
            
        Returns:
            Doubt submission response
        """
        try:
            # Validate student and check limits
            student = await self._get_student_with_validation(student_id, session)
            await self._check_daily_limits(student, session)
            
            # Create doubt record
            doubt = Doubt(
                student_id=student_id,
                subject_id=doubt_data.subject_id,
                topic_id=doubt_data.topic_id,
                question_text=doubt_data.question_text,
                question_type=doubt_data.question_type,
                question_images=doubt_data.question_images,
                difficulty_level=doubt_data.difficulty_level,
                question_source=doubt_data.question_source,
                processing_status="pending"
            )
            
            session.add(doubt)
            await session.flush()  # Get the ID
            
            # Update student's daily usage
            await self._update_daily_usage(student, session)
            
            # Start async processing
            asyncio.create_task(
                self._process_doubt_async(doubt.id, session)
            )
            
            await session.commit()
            
            return DoubtAskResponse(
                doubt_id=doubt.id,
                processing_status="pending",
                estimated_processing_time=30,  # seconds
                message="Your doubt has been submitted and is being processed by our AI system."
            )
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error submitting doubt: {e}", exc_info=True)
            raise ServiceError(f"Failed to submit doubt: {str(e)}")
    
    async def _get_student_with_validation(
        self,
        student_id: uuid.UUID,
        session: AsyncSession
    ) -> Student:
        """Get student and validate existence"""
        
        result = await session.execute(
            select(Student)
            .options(selectinload(Student.user))
            .where(Student.id == student_id)
        )
        
        student = result.scalar_one_or_none()
        
        if not student:
            raise ValidationError("Student not found")
        
        if not student.user.is_active:
            raise ValidationError("Student account is inactive")
        
        return student
    
    async def _check_daily_limits(
        self,
        student: Student,
        session: AsyncSession
    ) -> None:
        """Check if student has exceeded daily doubt limits"""
        
        if student.daily_doubts_used >= student.daily_doubt_limit:
            raise RateLimitError(
                f"Daily doubt limit of {student.daily_doubt_limit} exceeded. "
                f"Upgrade your subscription for more doubts."
            )
    
    async def _update_daily_usage(
        self,
        student: Student,
        session: AsyncSession
    ) -> None:
        """Update student's daily doubt usage"""
        
        await session.execute(
            update(Student)
            .where(Student.id == student.id)
            .values(
                daily_doubts_used=Student.daily_doubts_used + 1,
                total_doubts_asked=Student.total_doubts_asked + 1,
                last_activity_at=datetime.utcnow()
            )
        )
    
    async def _process_doubt_async(
        self,
        doubt_id: uuid.UUID,
        session: AsyncSession
    ) -> None:
        """Process doubt asynchronously with AI"""
        
        try:
            # Update status to processing
            await session.execute(
                update(Doubt)
                .where(Doubt.id == doubt_id)
                .values(processing_status="processing")
            )
            await session.commit()
            
            # Get doubt with related data
            result = await session.execute(
                select(Doubt)
                .options(
                    selectinload(Doubt.student).selectinload(Student.user),
                    selectinload(Doubt.subject),
                    selectinload(Doubt.topic)
                )
                .where(Doubt.id == doubt_id)
            )
            
            doubt = result.scalar_one()
            
            # Process with AI pipeline
            ai_response = await self._run_ai_pipeline(doubt, session)
            
            # Update doubt with results
            await session.execute(
                update(Doubt)
                .where(Doubt.id == doubt_id)
                .values(
                    processing_status="completed",
                    ai_response=ai_response["response"],
                    ai_explanation=ai_response["explanation"],
                    ai_confidence_score=ai_response["confidence_score"],
                    reasoning_steps=ai_response["reasoning_steps"],
                    llm_provider=ai_response["llm_provider"],
                    llm_model=ai_response["llm_model"],
                    tokens_used=ai_response["tokens_used"],
                    processing_cost=ai_response["cost"],
                    processing_time_ms=ai_response["processing_time_ms"],
                    adapted_explanation=ai_response.get("adapted_explanation"),
                    learning_level_adjusted=ai_response.get("learning_level_adjusted", False),
                    learning_style_adapted=ai_response.get("learning_style_adapted", False),
                    is_resolved=True,
                    resolved_at=datetime.utcnow(),
                    resolution_method="ai"
                )
            )
            
            # Update student statistics
            await session.execute(
                update(Student)
                .where(Student.id == doubt.student_id)
                .values(doubts_resolved=Student.doubts_resolved + 1)
            )
            
            await session.commit()
            
            logger.info(f"Successfully processed doubt {doubt_id}")
            
        except Exception as e:
            logger.error(f"Error processing doubt {doubt_id}: {e}", exc_info=True)
            
            # Update status to failed
            await session.execute(
                update(Doubt)
                .where(Doubt.id == doubt_id)
                .values(processing_status="failed")
            )
            await session.commit()
    
    async def _run_ai_pipeline(
        self,
        doubt: Doubt,
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Run the complete AI processing pipeline"""
        
        start_time = datetime.utcnow()
        
        # Step 1: Retrieve relevant context using RAG
        context = await self._retrieve_context(doubt)
        
        # Step 2: Generate initial response using LLM
        llm_response = await self._generate_llm_response(doubt, context)
        
        # Step 3: Validate and enhance with PIL reasoning
        validated_response = await self._validate_with_reasoning(
            doubt, llm_response, context
        )
        
        # Step 4: Personalize with MCP adaptive engine
        personalized_response = await self._personalize_response(
            doubt, validated_response, session
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            **personalized_response,
            "processing_time_ms": int(processing_time)
        }
    
    async def _retrieve_context(self, doubt: Doubt) -> Dict[str, Any]:
        """Retrieve relevant context using RAG"""
        
        try:
            # Build search query
            search_query = f"Subject: {doubt.subject.name} "
            if doubt.topic:
                search_query += f"Topic: {doubt.topic.name} "
            search_query += f"Question: {doubt.question_text}"
            
            # Retrieve similar content
            similar_content = await self.rag_retriever.retrieve_similar_content(
                query=search_query,
                subject=doubt.subject.code.lower(),
                top_k=5,
                min_score=0.7
            )
            
            # Retrieve examples
            examples = await self.rag_retriever.retrieve_examples(
                subject=doubt.subject.code.lower(),
                topic=doubt.topic.code.lower() if doubt.topic else None,
                difficulty=doubt.difficulty_level,
                count=3
            )
            
            return {
                "similar_content": similar_content,
                "examples": examples,
                "subject": doubt.subject.name,
                "topic": doubt.topic.name if doubt.topic else None
            }
            
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return {"similar_content": [], "examples": []}
    
    async def _generate_llm_response(
        self,
        doubt: Doubt,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response using LLM orchestrator"""
        
        # Build enhanced prompt
        prompt = self._build_enhanced_prompt(doubt, context)
        
        # Generate response
        response = await self.llm_orchestrator.generate_response(
            prompt=prompt,
            context={
                "subject": doubt.subject.code.lower(),
                "difficulty": doubt.difficulty_level,
                "question_type": doubt.question_type
            },
            max_tokens=2048,
            temperature=0.3  # Lower for more consistent educational content
        )
        
        return response
    
    def _build_enhanced_prompt(
        self,
        doubt: Doubt,
        context: Dict[str, Any]
    ) -> str:
        """Build enhanced prompt with context and examples"""
        
        prompt = f"""You are an expert {doubt.subject.name} tutor helping a student with their doubt.

Subject: {doubt.subject.name}
"""
        
        if doubt.topic:
            prompt += f"Topic: {doubt.topic.name}\n"
        
        if doubt.difficulty_level:
            prompt += f"Difficulty Level: {doubt.difficulty_level}\n"
        
        prompt += f"\nStudent's Question:\n{doubt.question_text}\n"
        
        # Add similar content context
        if context.get("similar_content"):
            prompt += "\nRelevant Background Information:\n"
            for content in context["similar_content"][:2]:  # Limit to top 2
                prompt += f"- {content.get('content', '')}\n"
        
        # Add examples
        if context.get("examples"):
            prompt += "\nSimilar Example Problems:\n"
            for example in context["examples"][:2]:  # Limit to top 2
                prompt += f"Example: {example.get('problem', '')}\n"
                prompt += f"Solution: {example.get('solution', '')}\n\n"
        
        prompt += """
Please provide:
1. A clear, step-by-step solution
2. Conceptual explanation of the underlying principles
3. Common mistakes to avoid
4. Tips for similar problems

Format your response with clear headings and step-by-step reasoning.
"""
        
        return prompt
    
    async def _validate_with_reasoning(
        self,
        doubt: Doubt,
        llm_response: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate response using PIL reasoning engine"""
        
        try:
            # Only validate mathematical subjects
            if doubt.subject.code.lower() in ["math", "physics", "chemistry"]:
                
                validation_result = await self.reasoning_engine.validate_solution(
                    problem=doubt.question_text,
                    solution=llm_response["response"],
                    subject=doubt.subject.code.lower()
                )
                
                # If validation fails, regenerate with corrections
                if not validation_result["is_valid"]:
                    logger.warning(f"Validation failed for doubt {doubt.id}: {validation_result['errors']}")
                    
                    # Try to correct the solution
                    corrected_response = await self._regenerate_with_corrections(
                        doubt, llm_response, validation_result
                    )
                    
                    return {
                        **corrected_response,
                        "reasoning_steps": validation_result["reasoning_steps"],
                        "validation_applied": True
                    }
                
                return {
                    **llm_response,
                    "reasoning_steps": validation_result["reasoning_steps"],
                    "validation_applied": True
                }
            
            else:
                # For non-mathematical subjects, return as-is
                return {
                    **llm_response,
                    "reasoning_steps": [],
                    "validation_applied": False
                }
                
        except Exception as e:
            logger.warning(f"Reasoning validation failed: {e}")
            return {
                **llm_response,
                "reasoning_steps": [],
                "validation_applied": False
            }
    
    async def _regenerate_with_corrections(
        self,
        doubt: Doubt,
        original_response: Dict[str, Any],
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Regenerate response with validation corrections"""
        
        correction_prompt = f"""The previous solution had validation errors. Please correct it.

Original Question: {doubt.question_text}

Previous Solution: {original_response['response']}

Validation Errors Found:
{chr(10).join(validation_result['errors'])}

Please provide a corrected solution that addresses these specific errors.
Focus on mathematical accuracy and proper step-by-step reasoning.
"""
        
        corrected_response = await self.llm_orchestrator.generate_response(
            prompt=correction_prompt,
            context={
                "subject": doubt.subject.code.lower(),
                "correction_mode": True
            },
            max_tokens=2048,
            temperature=0.1  # Very low for corrections
        )
        
        return corrected_response
    
    async def _personalize_response(
        self,
        doubt: Doubt,
        validated_response: Dict[str, Any],
        session: AsyncSession
    ) -> Dict[str, Any]:
        """Personalize response using MCP adaptive engine"""
        
        try:
            # Get student profile for personalization
            result = await session.execute(
                select(Student)
                .where(Student.id == doubt.student_id)
            )
            student = result.scalar_one()
            
            # Create student profile for MCP
            student_profile = {
                "grade": student.grade,
                "preparation_level": student.preparation_level,
                "learning_style": student.learning_style,
                "strong_subjects": student.strong_subjects or [],
                "weak_subjects": student.weak_subjects or [],
                "performance_data": student.performance_analytics
            }
            
            # Adapt response
            adapted_response = await self.adaptive_engine.adapt_response(
                original_response=validated_response["response"],
                student_profile=student_profile,
                subject=doubt.subject.code.lower(),
                difficulty=doubt.difficulty_level
            )
            
            return {
                **validated_response,
                "adapted_explanation": adapted_response["adapted_response"],
                "learning_level_adjusted": adapted_response["adjustments"]["difficulty_adjusted"],
                "learning_style_adapted": adapted_response["adjustments"]["style_adapted"],
                "personalization_applied": True
            }
            
        except Exception as e:
            logger.warning(f"Response personalization failed: {e}")
            return {
                **validated_response,
                "personalization_applied": False
            }
    
    async def get_doubt_status(
        self,
        doubt_id: uuid.UUID,
        session: AsyncSession
    ) -> DoubtResponse:
        """Get current status of a doubt"""
        
        result = await session.execute(
            select(Doubt)
            .where(Doubt.id == doubt_id)
        )
        
        doubt = result.scalar_one_or_none()
        
        if not doubt:
            raise ValidationError("Doubt not found")
        
        return DoubtResponse.from_orm(doubt)
    
    async def get_student_doubts(
        self,
        student_id: uuid.UUID,
        session: AsyncSession,
        limit: int = 20,
        offset: int = 0
    ) -> List[DoubtResponse]:
        """Get doubts for a specific student"""
        
        result = await session.execute(
            select(Doubt)
            .where(Doubt.student_id == student_id)
            .order_by(Doubt.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        doubts = result.scalars().all()
        
        return [DoubtResponse.from_orm(doubt) for doubt in doubts]
    
    async def mark_helpful(
        self,
        doubt_id: uuid.UUID,
        is_helpful: bool,
        session: AsyncSession
    ) -> None:
        """Mark a doubt response as helpful or not"""
        
        await session.execute(
            update(Doubt)
            .where(Doubt.id == doubt_id)
            .values(
                helpfulness_score=1.0 if is_helpful else 0.0,
                student_satisfaction=5 if is_helpful else 2
            )
        )
        
        await session.commit()


# Service factory function
async def get_doubt_processing_service() -> DoubtProcessingService:
    """Factory function to create doubt processing service with dependencies"""
    
    llm_orchestrator = LLMOrchestrator()
    reasoning_engine = ReasoningEngine()
    adaptive_engine = AdaptiveEngine()
    rag_retriever = RAGRetriever()
    
    return DoubtProcessingService(
        llm_orchestrator=llm_orchestrator,
        reasoning_engine=reasoning_engine,
        adaptive_engine=adaptive_engine,
        rag_retriever=rag_retriever
    )
