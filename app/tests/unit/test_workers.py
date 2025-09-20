"""
Unit Tests for Background Workers
Testing Celery workers with maximum precision
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import uuid
from datetime import datetime


class TestDoubtProcessorWorker:
    """Test Doubt Processing Worker Tasks"""

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_process_doubt_task_success(self, mock_llm_providers, mock_vector_db):
        """Test successful doubt processing task"""
        from app.workers.doubt_processor import process_doubt_task
        
        doubt_id = str(uuid.uuid4())
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session, \
             patch('app.services.doubt_service.get_doubt_processing_service') as mock_service:
            
            # Mock database session
            mock_session.return_value.__aenter__.return_value = MagicMock()
            
            # Mock doubt service
            mock_service_instance = AsyncMock()
            mock_service_instance._process_doubt_async.return_value = {
                "processing_time_ms": 2000,
                "provider": "openai",
                "tokens_used": 150,
                "cost": 0.015
            }
            mock_service.return_value = mock_service_instance
            
            # Execute task
            result = process_doubt_task(doubt_id)
            
            assert result["doubt_id"] == doubt_id
            assert result["status"] == "completed"
            assert result["processing_time"] == 2000
            assert result["provider"] == "openai"

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_process_doubt_task_retry_on_connection_error(self):
        """Test doubt processing task retry on connection error"""
        from app.workers.doubt_processor import process_doubt_task
        
        doubt_id = str(uuid.uuid4())
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            # Mock connection error
            mock_session.side_effect = ConnectionError("Database connection failed")
            
            # Create mock task instance for retry testing
            mock_task = MagicMock()
            mock_task.retry = MagicMock(side_effect=Exception("Retry called"))
            
            with patch.object(process_doubt_task, 'retry'):
                try:
                    # Bind the task to mock self
                    process_doubt_task.bind = True
                    process_doubt_task(mock_task, doubt_id)
                except Exception as e:
                    assert "Database connection failed" in str(e)

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_batch_process_doubts_success(self):
        """Test batch doubt processing success"""
        from app.workers.doubt_processor import batch_process_doubts
        
        doubt_ids = [str(uuid.uuid4()) for _ in range(3)]
        
        with patch('app.workers.doubt_processor.process_doubt_task') as mock_process:
            # Mock individual processing results
            mock_process.delay.return_value.get.side_effect = [
                {"doubt_id": doubt_ids[0], "status": "completed"},
                {"doubt_id": doubt_ids[1], "status": "completed"},
                {"doubt_id": doubt_ids[2], "status": "failed", "error": "Processing error"}
            ]
            
            # Create mock task instance
            mock_task = MagicMock()
            
            result = batch_process_doubts(mock_task, doubt_ids)
            
            assert result["total_doubts"] == 3
            assert result["processed"] == 2
            assert result["failed"] == 1
            assert len(result["results"]) == 3

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_validate_doubt_solution_correct(self, mock_llm_providers):
        """Test doubt solution validation for correct solution"""
        from app.workers.doubt_processor import validate_doubt_solution
        
        doubt_id = str(uuid.uuid4())
        solution = "F = ma (Newton's second law)"
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session, \
             patch('app.core.pil.reasoning_engine.ReasoningEngine') as mock_reasoning:
            
            mock_session.return_value.__aenter__.return_value = MagicMock()
            
            # Mock reasoning engine
            mock_reasoning_instance = AsyncMock()
            mock_reasoning_instance.validate_solution.return_value = {
                "is_valid": True,
                "confidence_score": 0.95,
                "errors": []
            }
            mock_reasoning.return_value = mock_reasoning_instance
            
            # Mock doubt retrieval
            mock_doubt = MagicMock()
            mock_doubt.question_text = "What is Newton's second law?"
            mock_doubt.subject.code = "PHYS"
            
            with patch('sqlalchemy.select'), patch('app.models.Doubt'):
                result = validate_doubt_solution(doubt_id, solution)
                
                assert result["is_valid"] is True
                assert result["confidence_score"] == 0.95

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_personalize_response_success(self):
        """Test response personalization success"""
        from app.workers.doubt_processor import personalize_response
        
        doubt_id = str(uuid.uuid4())
        response = "F = ma"
        student_profile = {
            "learning_style": "visual",
            "difficulty_preference": "medium"
        }
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session, \
             patch('app.core.mcp.adaptive_engine.AdaptiveEngine') as mock_adaptive:
            
            mock_session.return_value.__aenter__.return_value = MagicMock()
            
            # Mock adaptive engine
            mock_adaptive_instance = AsyncMock()
            mock_adaptive_instance.adapt_response.return_value = {
                "adapted_response": "Visual explanation: F = ma with diagrams",
                "adjustments": {
                    "difficulty_adjusted": False,
                    "style_adapted": True
                }
            }
            mock_adaptive.return_value = mock_adaptive_instance
            
            result = personalize_response(doubt_id, response, student_profile)
            
            assert "Visual explanation" in result["adapted_response"]
            assert result["adjustments"]["style_adapted"] is True

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_enrich_with_context_success(self, mock_vector_db):
        """Test context enrichment success"""
        from app.workers.doubt_processor import enrich_with_context
        
        question = "What is Newton's second law?"
        subject = "physics"
        topic = "mechanics"
        
        with patch('app.core.rag.retriever.RAGRetriever') as mock_rag:
            mock_rag_instance = AsyncMock()
            mock_rag_instance.retrieve_similar_content.return_value = [
                {"content": "Similar physics content", "score": 0.9}
            ]
            mock_rag_instance.retrieve_examples.return_value = [
                {"question": "Example question", "answer": "Example answer"}
            ]
            mock_rag.return_value = mock_rag_instance
            
            result = enrich_with_context(question, subject, topic)
            
            assert result["enriched"] is True
            assert len(result["similar_content"]) > 0
            assert len(result["examples"]) > 0

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_health_check_healthy(self):
        """Test doubt processor health check when healthy"""
        from app.workers.doubt_processor import health_check
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session.return_value.__aenter__.return_value.execute = AsyncMock()
            
            result = health_check()
            
            assert result["status"] == "healthy"
            assert result["database"] == "ok"
            assert "timestamp" in result

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_health_check_unhealthy(self):
        """Test doubt processor health check when unhealthy"""
        from app.workers.doubt_processor import health_check
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session.side_effect = Exception("Database unavailable")
            
            result = health_check()
            
            assert result["status"] == "unhealthy"
            assert "error" in result


class TestAnalyticsWorker:
    """Test Analytics Worker Tasks"""

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_compute_student_analytics_success(self):
        """Test successful student analytics computation"""
        from app.workers.analytics_worker import compute_student_analytics
        
        student_id = str(uuid.uuid4())
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session, \
             patch('app.services.student_service.get_student_service') as mock_service, \
             patch('app.database.get_cache_manager') as mock_cache:
            
            mock_session.return_value.__aenter__.return_value = MagicMock()
            mock_cache.return_value = MagicMock()
            
            # Mock student service
            mock_service_instance = AsyncMock()
            mock_service_instance.compute_analytics.return_value = {
                "total_doubts": 25,
                "resolved_doubts": 20,
                "resolution_rate": 80.0,
                "avg_resolution_time": 1800
            }
            mock_service.return_value = mock_service_instance
            
            # Create mock task instance
            mock_task = MagicMock()
            
            result = compute_student_analytics(mock_task, student_id)
            
            assert result["total_doubts"] == 25
            assert result["resolution_rate"] == 80.0

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_compute_system_metrics_success(self):
        """Test successful system metrics computation"""
        from app.workers.analytics_worker import compute_system_metrics
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock database query results
            mock_result = MagicMock()
            mock_result.first.return_value = MagicMock(
                total_doubts=100,
                resolved_doubts=80,
                avg_resolution_time=1500,
                active_students=25
            )
            mock_session_instance.execute.return_value = mock_result
            
            result = compute_system_metrics()
            
            assert "daily" in result
            assert "weekly" in result
            assert "monthly" in result
            assert result["daily"]["total_doubts"] == 100

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_generate_performance_report_success(self):
        """Test successful performance report generation"""
        from app.workers.analytics_worker import generate_performance_report
        
        period_days = 7
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock query results
            mock_hourly_result = [
                MagicMock(hour=9, total_doubts=10, resolved_doubts=8, avg_resolution_time=1500),
                MagicMock(hour=10, total_doubts=15, resolved_doubts=12, avg_resolution_time=1600)
            ]
            
            mock_daily_result = [
                MagicMock(date=datetime.now().date(), total_doubts=50, resolved_doubts=40),
                MagicMock(date=datetime.now().date(), total_doubts=60, resolved_doubts=48)
            ]
            
            mock_session_instance.execute.side_effect = [
                mock_hourly_result,
                mock_daily_result
            ]
            
            result = generate_performance_report(period_days)
            
            assert result["period"]["days"] == period_days
            assert "hourly_performance" in result
            assert "daily_trends" in result
            assert "summary" in result

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_update_student_recommendations_success(self):
        """Test successful student recommendations update"""
        from app.workers.analytics_worker import update_student_recommendations
        
        student_id = str(uuid.uuid4())
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session, \
             patch('app.services.student_service.get_student_service') as mock_service, \
             patch('app.database.get_cache_manager') as mock_cache:
            
            mock_session.return_value.__aenter__.return_value = MagicMock()
            mock_cache.return_value = MagicMock()
            
            # Mock student service
            mock_service_instance = AsyncMock()
            mock_service_instance.generate_recommendations.return_value = {
                "recommendations": [
                    {"topic": "quadratic_equations", "priority": "high"},
                    {"topic": "trigonometry", "priority": "medium"}
                ],
                "study_plan": "Focus on algebra foundations"
            }
            mock_service.return_value = mock_service_instance
            
            result = update_student_recommendations(student_id)
            
            assert "recommendations" in result
            assert len(result["recommendations"]) == 2
            assert "study_plan" in result

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_analytics_health_check_healthy(self):
        """Test analytics worker health check when healthy"""
        from app.workers.analytics_worker import analytics_health_check
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock successful database query
            mock_result = MagicMock()
            mock_result.scalar.return_value = 100  # Total doubts count
            mock_session_instance.execute.return_value = mock_result
            
            result = analytics_health_check()
            
            assert result["status"] == "healthy"
            assert result["database"] == "ok"
            assert result["total_doubts"] == 100


class TestNotificationWorker:
    """Test Notification Worker Tasks"""

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_send_doubt_resolved_notification_success(self):
        """Test successful doubt resolved notification"""
        from app.workers.notification_worker import send_doubt_resolved_notification
        
        student_id = str(uuid.uuid4())
        doubt_id = str(uuid.uuid4())
        notification_data = {"response_quality": "excellent"}
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock student and doubt retrieval
            mock_student = MagicMock()
            mock_student.name = "Test Student"
            
            mock_doubt = MagicMock()
            mock_doubt.subject.name = "Physics"
            
            mock_result = MagicMock()
            mock_result.scalar_one.side_effect = [mock_student, mock_doubt]
            mock_session_instance.execute.return_value = mock_result
            
            result = send_doubt_resolved_notification(student_id, doubt_id, notification_data)
            
            assert result["status"] == "sent"
            assert "notification_id" in result
            assert "push" in result["channels"]

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_send_daily_summary_success(self):
        """Test successful daily summary notification"""
        from app.workers.notification_worker import send_daily_summary
        
        student_id = str(uuid.uuid4())
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock student retrieval
            mock_student = MagicMock()
            mock_student.name = "Test Student"
            mock_student.id = uuid.UUID(student_id)
            
            # Mock activity stats
            mock_stats = MagicMock()
            mock_stats.total_doubts = 5
            mock_stats.resolved_doubts = 4
            
            mock_result = MagicMock()
            mock_result.scalar_one.return_value = mock_student
            mock_result.first.return_value = mock_stats
            mock_session_instance.execute.return_value = mock_result
            
            result = send_daily_summary(student_id)
            
            assert result["status"] == "sent"
            assert result["summary"]["data"]["total_doubts"] == 5
            assert result["summary"]["data"]["resolved_doubts"] == 4

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_send_achievement_notification_success(self):
        """Test successful achievement notification"""
        from app.workers.notification_worker import send_achievement_notification
        
        student_id = str(uuid.uuid4())
        achievement = {
            "name": "Physics Master",
            "description": "completed 50 physics problems",
            "level": "gold"
        }
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock student retrieval
            mock_student = MagicMock()
            mock_student.name = "Test Student"
            
            mock_result = MagicMock()
            mock_result.scalar_one.return_value = mock_student
            mock_session_instance.execute.return_value = mock_result
            
            result = send_achievement_notification(student_id, achievement)
            
            assert result["status"] == "sent"
            assert "Achievement Unlocked" in result["notification"]["title"]
            assert achievement["description"] in result["notification"]["message"]

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_send_reminder_notification_types(self):
        """Test different types of reminder notifications"""
        from app.workers.notification_worker import send_reminder_notification
        
        student_id = str(uuid.uuid4())
        
        reminder_types = [
            ("study_time", {}),
            ("pending_doubts", {"count": 3}),
            ("practice_session", {"subject": "Mathematics"}),
            ("streak_continuation", {"streak_days": 7})
        ]
        
        for reminder_type, reminder_data in reminder_types:
            with patch('app.database.db_manager.get_postgres_session') as mock_session:
                mock_session_instance = MagicMock()
                mock_session.return_value.__aenter__.return_value = mock_session_instance
                
                # Mock student retrieval
                mock_student = MagicMock()
                mock_student.name = "Test Student"
                
                mock_result = MagicMock()
                mock_result.scalar_one.return_value = mock_student
                mock_session_instance.execute.return_value = mock_result
                
                result = send_reminder_notification(student_id, reminder_type, reminder_data)
                
                assert result["status"] == "sent"
                assert result["notification"]["subtype"] == reminder_type

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_notifications_health_check_healthy(self):
        """Test notifications worker health check when healthy"""
        from app.workers.notification_worker import notifications_health_check
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.execute = AsyncMock()
            
            result = notifications_health_check()
            
            assert result["status"] == "healthy"
            assert result["database"] == "ok"
            assert result["notification_services"] == "ok"


class TestDataSyncWorker:
    """Test Data Sync Worker Tasks"""

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_sync_vector_embeddings_success(self, mock_vector_db):
        """Test successful vector embeddings sync"""
        from app.workers.data_sync_worker import sync_vector_embeddings
        
        batch_size = 50
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session, \
             patch('app.core.rag.retriever.RAGRetriever') as mock_rag:
            
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock doubts needing sync
            mock_doubts = [
                MagicMock(id=uuid.uuid4(), question_text="Question 1", vector_synced=False),
                MagicMock(id=uuid.uuid4(), question_text="Question 2", vector_synced=False)
            ]
            
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = mock_doubts
            mock_session_instance.execute.return_value = mock_result
            
            # Mock RAG retriever
            mock_rag_instance = AsyncMock()
            mock_rag_instance.store_doubt_embedding.return_value = True
            mock_rag.return_value = mock_rag_instance
            
            # Create mock task instance
            mock_task = MagicMock()
            
            result = sync_vector_embeddings(mock_task, batch_size)
            
            assert result["status"] == "completed"
            assert result["synced"] == 2
            assert result["failed"] == 0

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_backup_doubt_data_success(self):
        """Test successful doubt data backup"""
        from app.workers.data_sync_worker import backup_doubt_data
        
        period_days = 7
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock backup data
            mock_backup_data = [
                (MagicMock(id=uuid.uuid4(), question_text="Question 1"), "Student 1", "Physics"),
                (MagicMock(id=uuid.uuid4(), question_text="Question 2"), "Student 2", "Math")
            ]
            
            mock_session_instance.execute.return_value = mock_backup_data
            
            result = backup_doubt_data(period_days)
            
            assert result["status"] == "completed"
            assert result["records_count"] == 2
            assert "backup_file" in result

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_cleanup_old_data_success(self):
        """Test successful old data cleanup"""
        from app.workers.data_sync_worker import cleanup_old_data
        
        retention_days = 90
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock delete operation
            mock_delete_result = MagicMock()
            mock_delete_result.rowcount = 15  # Number of deleted rows
            mock_session_instance.execute.return_value = mock_delete_result
            
            result = cleanup_old_data(retention_days)
            
            assert result["status"] == "completed"
            assert result["unresolved_deleted"] == 15
            assert result["retention_days"] == retention_days

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_sync_student_analytics_success(self):
        """Test successful student analytics sync"""
        from app.workers.data_sync_worker import sync_student_analytics
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock active students
            mock_students = [
                MagicMock(id=uuid.uuid4(), name="Student 1"),
                MagicMock(id=uuid.uuid4(), name="Student 2")
            ]
            
            # Mock analytics stats
            mock_stats = MagicMock()
            mock_stats.total_doubts = 10
            mock_stats.resolved_doubts = 8
            mock_stats.avg_resolution_time = 1800
            
            mock_students_result = MagicMock()
            mock_students_result.scalars.return_value.all.return_value = mock_students
            
            mock_stats_result = MagicMock()
            mock_stats_result.first.return_value = mock_stats
            
            mock_session_instance.execute.side_effect = [mock_students_result, mock_stats_result, mock_stats_result]
            
            result = sync_student_analytics()
            
            assert result["status"] == "completed"
            assert result["students_synced"] == 2
            assert result["total_students"] == 2

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_refresh_cache_data_success(self):
        """Test successful cache data refresh"""
        from app.workers.data_sync_worker import refresh_cache_data
        
        result = refresh_cache_data()
        
        assert result["status"] == "completed"
        assert result["refreshed_items"] == 4  # Based on mock data
        assert result["cleared_keys"] == 0

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_data_sync_health_check_healthy(self):
        """Test data sync worker health check when healthy"""
        from app.workers.data_sync_worker import data_sync_health_check
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.execute = AsyncMock()
            
            result = data_sync_health_check()
            
            assert result["status"] == "healthy"
            assert result["database"] == "ok"
            assert result["vector_db"] == "ok"
            assert result["cache"] == "ok"


class TestWorkerErrorHandling:
    """Test worker error handling and resilience"""

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_worker_task_timeout_handling(self):
        """Test worker behavior on task timeout"""
        from app.workers.doubt_processor import process_doubt_task
        
        doubt_id = str(uuid.uuid4())
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            # Mock timeout error
            mock_session.side_effect = TimeoutError("Task execution timeout")
            
            result = process_doubt_task(doubt_id)
            
            assert result["status"] == "failed"
            assert "timeout" in result["error"].lower()

    @pytest.mark.unit
    @pytest.mark.worker
    async def test_worker_invalid_input_handling(self):
        """Test worker behavior with invalid inputs"""
        from app.workers.analytics_worker import compute_student_analytics
        
        # Test with invalid UUID
        invalid_student_id = "invalid-uuid"
        
        # Create mock task instance
        mock_task = MagicMock()
        
        result = compute_student_analytics(mock_task, invalid_student_id)
        
        assert result["error"] is not None
        assert "timestamp" in result

    @pytest.mark.unit
    @pytest.mark.worker  
    async def test_worker_resource_cleanup(self):
        """Test worker resource cleanup on task completion"""
        from app.workers.data_sync_worker import sync_vector_embeddings
        
        with patch('app.database.db_manager.get_postgres_session') as mock_session:
            mock_session_instance = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            # Mock empty result (no work to do)
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session_instance.execute.return_value = mock_result
            
            # Create mock task instance
            mock_task = MagicMock()
            
            result = sync_vector_embeddings(mock_task, 100)
            
            assert result["status"] == "no_work"
            assert result["message"] == "No doubts to sync"
            
            # Verify session cleanup
            mock_session.return_value.__aenter__.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
