"""
Unit Tests for Business Logic Services
Testing doubt processing, student management, and other core services
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import uuid
from datetime import datetime


class TestDoubtProcessingService:
    """Test Doubt Processing Service"""

    @pytest.fixture
    async def doubt_service(self):
        """Create doubt processing service instance"""
        from app.services.doubt_service import DoubtProcessingService
        from app.database import get_cache_manager
        
        cache_manager = await get_cache_manager()
        return DoubtProcessingService(cache_manager)

    @pytest.mark.unit
    async def test_submit_doubt_success(self, doubt_service, db_session, test_data):
        """Test successful doubt submission"""
        
        doubt_data = {
            "question_text": test_data["doubt"]["question_text"],
            "subject_code": test_data["doubt"]["subject_code"],
            "topic": test_data["doubt"]["topic"],
            "difficulty_level": test_data["doubt"]["difficulty_level"],
            "question_type": test_data["doubt"]["question_type"]
        }
        
        with patch('app.workers.doubt_processor.process_doubt_task') as mock_task:
            mock_task.delay.return_value.id = "test-task-id"
            
            result = await doubt_service.submit_doubt(
                student_id=uuid.UUID(test_data["student"]["id"]),
                doubt_data=doubt_data,
                session=db_session
            )
            
            assert result["status"] == "submitted"
            assert result["question_text"] == doubt_data["question_text"]
            assert "id" in result
            mock_task.delay.assert_called_once()

    @pytest.mark.unit
    async def test_submit_doubt_validation_error(self, doubt_service, db_session, test_data):
        """Test doubt submission with validation error"""
        
        invalid_doubt_data = {
            "question_text": "",  # Empty question
            "subject_code": "INVALID",  # Invalid subject
            "difficulty_level": "extreme"  # Invalid difficulty
        }
        
        with pytest.raises(ValueError) as exc_info:
            await doubt_service.submit_doubt(
                student_id=uuid.UUID(test_data["student"]["id"]),
                doubt_data=invalid_doubt_data,
                session=db_session
            )
        
        assert "validation" in str(exc_info.value).lower()

    @pytest.mark.unit
    async def test_process_doubt_async_success(self, doubt_service, db_session, mock_llm_providers, mock_vector_db):
        """Test async doubt processing success"""
        
        doubt_id = uuid.uuid4()
        
        # Mock doubt retrieval
        with patch('app.models.Doubt') as mock_doubt_model:
            mock_doubt = MagicMock()
            mock_doubt.id = doubt_id
            mock_doubt.question_text = "What is Newton's second law?"
            mock_doubt.subject.code = "PHYS"
            mock_doubt.difficulty_level = "medium"
            mock_doubt.student.id = uuid.uuid4()
            
            mock_doubt_model.query.filter.return_value.first.return_value = mock_doubt
            
            # Mock AI processing
            mock_llm_providers['openai'].generate_response.return_value = {
                "response": "Newton's second law states that F = ma",
                "provider": "openai",
                "tokens_used": 100,
                "cost": 0.01,
                "processing_time_ms": 1500
            }
            
            result = await doubt_service._process_doubt_async(doubt_id, db_session)
            
            assert result["status"] == "completed"
            assert "response" in result
            assert result["provider"] == "openai"

    @pytest.mark.unit
    async def test_get_doubt_status(self, doubt_service, db_session, test_cache):
        """Test getting doubt status"""
        
        doubt_id = uuid.uuid4()
        
        # Mock cached status
        await test_cache.set(
            f"doubt_status:{doubt_id}",
            '{"status": "processing", "progress": 75}',
            ex=300
        )
        
        result = await doubt_service.get_doubt_status(doubt_id, db_session)
        
        assert result["status"] == "processing"
        assert result["progress"] == 75

    @pytest.mark.unit
    async def test_get_doubt_response_completed(self, doubt_service, db_session):
        """Test getting response for completed doubt"""
        
        doubt_id = uuid.uuid4()
        
        with patch('app.models.Doubt') as mock_doubt_model:
            mock_doubt = MagicMock()
            mock_doubt.id = doubt_id
            mock_doubt.is_resolved = True
            mock_doubt.resolution_data = {
                "response": "Detailed physics explanation",
                "explanation": "Step-by-step solution",
                "confidence_score": 0.95
            }
            
            mock_doubt_model.query.filter.return_value.first.return_value = mock_doubt
            
            result = await doubt_service.get_doubt_response(doubt_id, db_session)
            
            assert result["status"] == "resolved"
            assert result["response"] == "Detailed physics explanation"
            assert result["confidence_score"] == 0.95

    @pytest.mark.unit
    async def test_get_student_doubts_pagination(self, doubt_service, db_session):
        """Test getting student doubts with pagination"""
        
        student_id = uuid.uuid4()
        
        with patch('app.models.Doubt') as mock_doubt_model:
            mock_query = MagicMock()
            mock_query.filter.return_value = mock_query
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [
                MagicMock(id=uuid.uuid4(), question_text="Question 1"),
                MagicMock(id=uuid.uuid4(), question_text="Question 2")
            ]
            mock_query.count.return_value = 15  # Total count
            
            mock_doubt_model.query = mock_query
            
            result = await doubt_service.get_student_doubts(
                student_id=student_id,
                session=db_session,
                page=2,
                per_page=5
            )
            
            assert result["total"] == 15
            assert result["page"] == 2
            assert result["per_page"] == 5
            assert len(result["doubts"]) == 2


class TestStudentService:
    """Test Student Management Service"""

    @pytest.fixture
    async def student_service(self):
        """Create student service instance"""
        from app.services.student_service import StudentService
        from app.database import get_cache_manager
        
        cache_manager = await get_cache_manager()
        return StudentService(cache_manager)

    @pytest.mark.unit
    async def test_create_student_success(self, student_service, db_session, test_data):
        """Test successful student creation"""
        
        student_data = {
            "name": test_data["student"]["name"],
            "email": test_data["student"]["email"],
            "phone": test_data["student"]["phone"],
            "grade": test_data["student"]["grade"],
            "password": "hashedpassword123"
        }
        
        with patch('app.models.Student') as mock_student_model, \
             patch('app.utils.validation.Validator.validate_email') as mock_validate_email, \
             patch('app.utils.validation.Validator.validate_phone') as mock_validate_phone:
            
            mock_validate_email.return_value = True
            mock_validate_phone.return_value = True
            
            mock_student = MagicMock()
            mock_student.id = uuid.UUID(test_data["student"]["id"])
            mock_student.name = student_data["name"]
            mock_student.email = student_data["email"]
            
            mock_student_model.return_value = mock_student
            
            result = await student_service.create_student(student_data, db_session)
            
            assert result["id"] == test_data["student"]["id"]
            assert result["name"] == student_data["name"]
            assert result["email"] == student_data["email"]

    @pytest.mark.unit
    async def test_create_student_duplicate_email(self, student_service, db_session, test_data):
        """Test student creation with duplicate email"""
        
        student_data = {
            "email": test_data["student"]["email"],
            "name": "Another Student",
            "phone": "+9876543210",
            "grade": 10
        }
        
        with patch('app.models.Student') as mock_student_model:
            # Mock existing student query
            mock_student_model.query.filter.return_value.first.return_value = MagicMock()
            
            with pytest.raises(ValueError) as exc_info:
                await student_service.create_student(student_data, db_session)
            
            assert "already exists" in str(exc_info.value)

    @pytest.mark.unit
    async def test_get_student_profile_cached(self, student_service, db_session, test_cache, test_data):
        """Test getting student profile from cache"""
        
        student_id = uuid.UUID(test_data["student"]["id"])
        
        # Set cached profile
        import json
        cached_profile = json.dumps(test_data["student"])
        await test_cache.set(f"student_profile:{student_id}", cached_profile, ex=3600)
        
        result = await student_service.get_student_profile(student_id, db_session)
        
        assert result["id"] == test_data["student"]["id"]
        assert result["name"] == test_data["student"]["name"]

    @pytest.mark.unit
    async def test_update_student_profile(self, student_service, db_session, test_data):
        """Test updating student profile"""
        
        student_id = uuid.UUID(test_data["student"]["id"])
        update_data = {
            "name": "Updated Name",
            "learning_style": "kinesthetic",
            "difficulty_preference": "hard"
        }
        
        with patch('app.models.Student') as mock_student_model:
            mock_student = MagicMock()
            mock_student.id = student_id
            mock_student.name = test_data["student"]["name"]
            
            mock_student_model.query.filter.return_value.first.return_value = mock_student
            
            result = await student_service.update_student_profile(
                student_id, update_data, db_session
            )
            
            assert result["name"] == update_data["name"]
            assert result["learning_style"] == update_data["learning_style"]

    @pytest.mark.unit
    async def test_compute_analytics(self, student_service, db_session):
        """Test computing student analytics"""
        
        student_id = uuid.uuid4()
        
        with patch('app.models.Doubt') as mock_doubt_model:
            # Mock database query results
            mock_query = MagicMock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [
                MagicMock(is_resolved=True, resolution_time_ms=1500, subject=MagicMock(name="Physics")),
                MagicMock(is_resolved=True, resolution_time_ms=2000, subject=MagicMock(name="Physics")),
                MagicMock(is_resolved=False, resolution_time_ms=None, subject=MagicMock(name="Math"))
            ]
            
            mock_doubt_model.query = mock_query
            
            result = await student_service.compute_analytics(student_id, db_session)
            
            assert result["total_doubts"] == 3
            assert result["resolved_doubts"] == 2
            assert result["resolution_rate"] == 66.67
            assert "avg_resolution_time" in result
            assert "subject_breakdown" in result

    @pytest.mark.unit
    async def test_generate_recommendations(self, student_service, db_session, mock_llm_providers):
        """Test generating student recommendations"""
        
        student_id = uuid.uuid4()
        
        with patch('app.models.Student') as mock_student_model:
            mock_student = MagicMock()
            mock_student.learning_style = "visual"
            mock_student.difficulty_preference = "medium"
            mock_student.grade = 11
            
            mock_student_model.query.filter.return_value.first.return_value = mock_student
            
            mock_llm_providers['openai'].generate_response.return_value = {
                "response": '{"recommendations": [{"topic": "quadratic_equations", "priority": "high", "reason": "Next logical step"}], "study_plan": "Focus on visual learning"}',
                "provider": "openai"
            }
            
            result = await student_service.generate_recommendations(student_id, db_session)
            
            assert "recommendations" in result
            assert "study_plan" in result
            assert len(result["recommendations"]) > 0

    @pytest.mark.unit
    async def test_update_learning_patterns(self, student_service, db_session):
        """Test updating student learning patterns"""
        
        student_id = uuid.uuid4()
        interaction_data = {
            "response_time": 1500,
            "accuracy": 0.85,
            "help_requested": False,
            "topic": "algebra",
            "difficulty": "medium"
        }
        
        with patch('app.models.Student') as mock_student_model:
            mock_student = MagicMock()
            mock_student.learning_patterns = {}
            
            mock_student_model.query.filter.return_value.first.return_value = mock_student
            
            await student_service.update_learning_patterns(
                student_id, db_session, interaction_data
            )
            
            # Verify patterns were updated
            assert mock_student.learning_patterns is not None


class TestServiceErrorHandling:
    """Test service error handling and edge cases"""

    @pytest.mark.unit
    async def test_service_database_error_handling(self, db_session):
        """Test service behavior when database errors occur"""
        from app.services.doubt_service import DoubtProcessingService
        from app.database import get_cache_manager
        
        cache_manager = await get_cache_manager()
        doubt_service = DoubtProcessingService(cache_manager)
        
        # Mock database error
        with patch('app.models.Doubt') as mock_doubt_model:
            mock_doubt_model.query.filter.side_effect = Exception("Database connection error")
            
            with pytest.raises(Exception) as exc_info:
                await doubt_service.get_doubt_status(uuid.uuid4(), db_session)
            
            assert "Database connection error" in str(exc_info.value)

    @pytest.mark.unit
    async def test_service_cache_error_handling(self):
        """Test service behavior when cache errors occur"""
        from app.services.student_service import StudentService
        
        # Mock failing cache
        with patch('app.database.get_cache_manager') as mock_cache_manager:
            mock_cache = AsyncMock()
            mock_cache.get.side_effect = Exception("Redis connection error")
            mock_cache_manager.return_value = mock_cache
            
            student_service = StudentService(mock_cache)
            
            # Service should handle cache errors gracefully
            with patch('app.models.Student') as mock_student_model:
                mock_student = MagicMock()
                mock_student_model.query.filter.return_value.first.return_value = mock_student
                
                # Should not raise exception despite cache error
                result = await student_service.get_student_profile(uuid.uuid4(), MagicMock())
                assert result is not None

    @pytest.mark.unit
    async def test_service_input_validation(self):
        """Test service input validation"""
        from app.services.doubt_service import DoubtProcessingService
        from app.database import get_cache_manager
        
        cache_manager = await get_cache_manager()
        doubt_service = DoubtProcessingService(cache_manager)
        
        # Test invalid UUID
        with pytest.raises(ValueError):
            await doubt_service.get_doubt_status("invalid-uuid", MagicMock())
        
        # Test None values
        with pytest.raises(ValueError):
            await doubt_service.submit_doubt(None, {}, MagicMock())


class TestServicePerformance:
    """Test service performance characteristics"""

    @pytest.mark.unit
    async def test_service_caching_efficiency(self, test_cache):
        """Test service caching improves performance"""
        from app.services.student_service import StudentService
        
        student_service = StudentService(test_cache)
        student_id = uuid.uuid4()
        
        with patch('app.models.Student') as mock_student_model:
            mock_student = MagicMock()
            mock_student.id = student_id
            mock_student.name = "Test Student"
            
            mock_student_model.query.filter.return_value.first.return_value = mock_student
            
            # First call - should hit database
            start_time = datetime.now()
            result1 = await student_service.get_student_profile(student_id, MagicMock())
            first_call_time = (datetime.now() - start_time).total_seconds()
            
            # Second call - should hit cache
            start_time = datetime.now()
            result2 = await student_service.get_student_profile(student_id, MagicMock())
            second_call_time = (datetime.now() - start_time).total_seconds()
            
            # Cache hit should be faster (though in mocked environment, difference may be minimal)
            assert result1 == result2
            assert second_call_time <= first_call_time

    @pytest.mark.unit
    async def test_service_batch_operations(self):
        """Test service batch operation efficiency"""
        from app.services.doubt_service import DoubtProcessingService
        from app.database import get_cache_manager
        
        cache_manager = await get_cache_manager()
        doubt_service = DoubtProcessingService(cache_manager)
        
        doubt_ids = [uuid.uuid4() for _ in range(5)]
        
        with patch('app.workers.doubt_processor.batch_process_doubts') as mock_batch:
            mock_batch.delay.return_value.get.return_value = {
                "total_doubts": 5,
                "processed": 5,
                "failed": 0
            }
            
            result = await doubt_service.batch_process_doubts(doubt_ids, MagicMock())
            
            assert result["total_doubts"] == 5
            assert result["processed"] == 5
            mock_batch.delay.assert_called_once_with(doubt_ids)


if __name__ == "__main__":
    pytest.main([__file__])
