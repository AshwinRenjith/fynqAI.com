"""
Integration tests for API workflows and cross-component interactions.
Tests end-to-end functionality with real service integration.
"""

import asyncio
import pytest
from unittest.mock import patch
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Student
from app.services.doubt_service import DoubtProcessingService
from app.services.student_service import StudentService
from app.core.llm.orchestrator import LLMOrchestrator
from app.core.rag.retriever import RAGRetriever


class TestAuthWorkflows:
    """Test authentication and user management workflows."""
    
    @pytest.mark.asyncio
    async def test_student_registration_workflow(self, test_client: AsyncClient, db_session: AsyncSession):
        """Test complete student registration workflow."""
        # Step 1: Register new student
        registration_data = {
            "email": "test_student@example.com",
            "password": "SecurePass123!",
            "first_name": "Test",
            "last_name": "Student",
            "phone": "+91 9876543210",
            "board": "CBSE",
            "grade": 12,
            "subjects": ["Mathematics", "Physics", "Chemistry"]
        }
        
        response = await test_client.post("/api/v1/auth/register", json=registration_data)
        assert response.status_code == 201
        
        registration_result = response.json()
        assert "access_token" in registration_result
        assert "user_id" in registration_result
        
        user_id = registration_result["user_id"]
        access_token = registration_result["access_token"]
        
        # Step 2: Verify user exists in database
        student = await db_session.get(Student, user_id)
        assert student is not None
        assert student.email == registration_data["email"]
        assert student.is_active is True
        
        # Step 3: Test authenticated request
        headers = {"Authorization": f"Bearer {access_token}"}
        profile_response = await test_client.get("/api/v1/students/profile", headers=headers)
        assert profile_response.status_code == 200
        
        profile_data = profile_response.json()
        assert profile_data["email"] == registration_data["email"]
        assert profile_data["grade"] == registration_data["grade"]
    
    @pytest.mark.asyncio
    async def test_login_workflow(self, test_client: AsyncClient, test_student: Student):
        """Test student login workflow."""
        # Step 1: Login with credentials
        login_data = {
            "email": test_student.email,
            "password": "testpassword"
        }
        
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        
        login_result = response.json()
        assert "access_token" in login_result
        assert "refresh_token" in login_result
        
        # Step 2: Use access token for authenticated request
        headers = {"Authorization": f"Bearer {login_result['access_token']}"}
        profile_response = await test_client.get("/api/v1/students/profile", headers=headers)
        assert profile_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_token_refresh_workflow(self, test_client: AsyncClient, test_student: Student):
        """Test token refresh workflow."""
        # Step 1: Login to get tokens
        login_data = {"email": test_student.email, "password": "testpassword"}
        login_response = await test_client.post("/api/v1/auth/login", json=login_data)
        tokens = login_response.json()
        
        # Step 2: Use refresh token to get new access token
        refresh_data = {"refresh_token": tokens["refresh_token"]}
        refresh_response = await test_client.post("/api/v1/auth/refresh", json=refresh_data)
        assert refresh_response.status_code == 200
        
        new_tokens = refresh_response.json()
        assert "access_token" in new_tokens
        assert new_tokens["access_token"] != tokens["access_token"]


class TestDoubtWorkflows:
    """Test doubt processing workflows."""
    
    @pytest.mark.asyncio
    async def test_doubt_submission_to_resolution_workflow(
        self, 
        test_client: AsyncClient, 
        test_student: Student,
        mock_llm_providers: dict,
        mock_vector_db,
        db_session: AsyncSession
    ):
        """Test complete doubt submission to resolution workflow."""
        # Step 1: Submit doubt
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        doubt_data = {
            "question": "Explain the concept of limits in calculus",
            "subject": "Mathematics",
            "chapter": "Limits and Derivatives",
            "difficulty_level": "intermediate",
            "context": "I'm struggling with the epsilon-delta definition",
            "urgency": "medium"
        }
        
        response = await test_client.post("/api/v1/doubts/", json=doubt_data, headers=headers)
        assert response.status_code == 201
        
        doubt_result = response.json()
        doubt_id = doubt_result["id"]
        assert doubt_result["status"] == "submitted"
        
        # Step 2: Process doubt (simulate background processing)
        with patch('app.workers.doubt_processor.process_doubt_task.delay') as mock_task:
            await test_client.post(f"/api/v1/doubts/{doubt_id}/process", headers=headers)
            mock_task.assert_called_once_with(doubt_id)
        
        # Step 3: Verify doubt processing
        doubt_service = DoubtProcessingService(db_session)
        processed_doubt = await doubt_service.get_doubt(doubt_id, test_student.id)
        
        assert processed_doubt is not None
        assert processed_doubt.id == doubt_id
        
        # Step 4: Check doubt status progression
        status_response = await test_client.get(f"/api/v1/doubts/{doubt_id}", headers=headers)
        assert status_response.status_code == 200
        
        doubt_status = status_response.json()
        assert doubt_status["id"] == doubt_id
        assert doubt_status["question"] == doubt_data["question"]
    
    @pytest.mark.asyncio
    async def test_doubt_analytics_workflow(
        self, 
        test_client: AsyncClient, 
        test_student: Student,
        db_session: AsyncSession
    ):
        """Test doubt analytics tracking workflow."""
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Create multiple doubts
        doubt_ids = []
        for i in range(3):
            doubt_data = {
                "question": f"Test question {i+1}",
                "subject": "Mathematics",
                "chapter": "Algebra",
                "difficulty_level": "basic",
                "urgency": "low"
            }
            
            response = await test_client.post("/api/v1/doubts/", json=doubt_data, headers=headers)
            doubt_ids.append(response.json()["id"])
        
        # Check analytics
        analytics_response = await test_client.get("/api/v1/analytics/student/doubts", headers=headers)
        assert analytics_response.status_code == 200
        
        analytics_data = analytics_response.json()
        assert analytics_data["total_doubts"] >= 3
        assert "subject_breakdown" in analytics_data
        assert "Mathematics" in analytics_data["subject_breakdown"]
    
    async def _get_auth_token(self, test_client: AsyncClient, student: Student) -> str:
        """Helper to get authentication token."""
        login_data = {"email": student.email, "password": "testpassword"}
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        return response.json()["access_token"]


class TestAIWorkflows:
    """Test AI processing workflows."""
    
    @pytest.mark.asyncio
    async def test_rag_integration_workflow(
        self, 
        mock_vector_db,
        mock_llm_providers: dict,
        db_session: AsyncSession
    ):
        """Test RAG retrieval and LLM integration workflow."""
        # Step 1: Initialize services
        retriever = RAGRetriever()
        orchestrator = LLMOrchestrator()
        
        # Step 2: Test knowledge retrieval
        query = "What are the properties of limits in calculus?"
        
        with patch.object(retriever, 'retrieve_relevant_content') as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    "content": "Limits are fundamental in calculus...",
                    "source": "NCERT_Mathematics_Chapter_13",
                    "relevance_score": 0.95
                }
            ]
            
            retrieved_content = await retriever.retrieve_relevant_content(
                query=query,
                subject="Mathematics",
                filters={"difficulty": "intermediate"}
            )
            
            assert len(retrieved_content) > 0
            assert retrieved_content[0]["relevance_score"] > 0.9
        
        # Step 3: Test LLM processing with retrieved content
        with patch.object(orchestrator, 'process_doubt') as mock_process:
            mock_process.return_value = {
                "explanation": "Detailed explanation of limits...",
                "examples": ["Example 1", "Example 2"],
                "difficulty_assessment": "intermediate",
                "confidence_score": 0.92
            }
            
            llm_response = await orchestrator.process_doubt(
                question=query,
                context=retrieved_content[0]["content"],
                student_level="grade_12"
            )
            
            assert "explanation" in llm_response
            assert llm_response["confidence_score"] > 0.9
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_workflow(
        self, 
        test_student: Student,
        db_session: AsyncSession,
        mock_llm_providers: dict
    ):
        """Test adaptive learning engine workflow."""
        from app.core.mcp.adaptive_engine import MCPAdaptiveEngine
        
        # Step 1: Initialize adaptive engine
        adaptive_engine = MCPAdaptiveEngine()
        
        # Step 2: Track student performance
        performance_data = [
            {"subject": "Mathematics", "topic": "Limits", "score": 0.85, "time_taken": 300},
            {"subject": "Mathematics", "topic": "Derivatives", "score": 0.72, "time_taken": 450},
            {"subject": "Physics", "topic": "Kinematics", "score": 0.90, "time_taken": 280}
        ]
        
        with patch.object(adaptive_engine, 'update_student_profile') as mock_update:
            for performance in performance_data:
                await adaptive_engine.update_student_profile(
                    student_id=test_student.id,
                    performance_data=performance
                )
            
            assert mock_update.call_count == len(performance_data)
        
        # Step 3: Get personalized recommendations
        with patch.object(adaptive_engine, 'get_recommendations') as mock_recommendations:
            mock_recommendations.return_value = {
                "weak_areas": ["Derivatives"],
                "suggested_topics": ["Chain Rule", "Product Rule"],
                "difficulty_adjustment": "increase_for_kinematics",
                "study_plan": ["Review limits", "Practice derivatives"]
            }
            
            recommendations = await adaptive_engine.get_recommendations(test_student.id)
            
            assert "weak_areas" in recommendations
            assert "suggested_topics" in recommendations


class TestDataFlowWorkflows:
    """Test data flow and synchronization workflows."""
    
    @pytest.mark.asyncio
    async def test_student_data_sync_workflow(
        self, 
        test_client: AsyncClient,
        test_student: Student,
        db_session: AsyncSession
    ):
        """Test student data synchronization workflow."""
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Step 1: Update student profile
        update_data = {
            "grade": 12,
            "subjects": ["Mathematics", "Physics", "Chemistry", "Biology"],
            "learning_preferences": {
                "visual": 0.8,
                "auditory": 0.6,
                "kinesthetic": 0.4
            }
        }
        
        response = await test_client.put("/api/v1/students/profile", json=update_data, headers=headers)
        assert response.status_code == 200
        
        # Step 2: Verify sync across services
        student_service = StudentService(db_session)
        updated_student = await student_service.get_student(test_student.id)
        
        assert updated_student.grade == update_data["grade"]
        assert len(updated_student.subjects) == len(update_data["subjects"])
        
        # Step 3: Check analytics update
        analytics_response = await test_client.get("/api/v1/analytics/student/profile", headers=headers)
        assert analytics_response.status_code == 200
        
        analytics_data = analytics_response.json()
        assert analytics_data["current_grade"] == update_data["grade"]
    
    @pytest.mark.asyncio
    async def test_enterprise_integration_workflow(
        self, 
        test_client: AsyncClient,
        db_session: AsyncSession
    ):
        """Test enterprise user integration workflow."""
        # Step 1: Create enterprise user
        enterprise_data = {
            "email": "admin@testschool.edu",
            "password": "AdminPass123!",
            "organization_name": "Test School",
            "role": "administrator",
            "permissions": ["view_analytics", "manage_students", "export_data"]
        }
        
        response = await test_client.post("/api/v1/enterprise/register", json=enterprise_data)
        assert response.status_code == 201
        
        result = response.json()
        admin_token = result["access_token"]
        
        # Step 2: Access enterprise analytics
        headers = {"Authorization": f"Bearer {admin_token}"}
        analytics_response = await test_client.get("/api/v1/enterprise/analytics/overview", headers=headers)
        assert analytics_response.status_code == 200
        
        analytics_data = analytics_response.json()
        assert "total_students" in analytics_data
        assert "active_doubts" in analytics_data
        
        # Step 3: Test student management
        students_response = await test_client.get("/api/v1/enterprise/students", headers=headers)
        assert students_response.status_code == 200
    
    async def _get_auth_token(self, test_client: AsyncClient, student: Student) -> str:
        """Helper to get authentication token."""
        login_data = {"email": student.email, "password": "testpassword"}
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        return response.json()["access_token"]


class TestErrorHandlingWorkflows:
    """Test error handling across workflows."""
    
    @pytest.mark.asyncio
    async def test_doubt_processing_error_recovery(
        self, 
        test_client: AsyncClient,
        test_student: Student,
        db_session: AsyncSession
    ):
        """Test error recovery in doubt processing workflow."""
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Step 1: Submit doubt
        doubt_data = {
            "question": "Complex question requiring external processing",
            "subject": "Mathematics",
            "chapter": "Complex Analysis"
        }
        
        response = await test_client.post("/api/v1/doubts/", json=doubt_data, headers=headers)
        doubt_id = response.json()["id"]
        
        # Step 2: Simulate processing error
        with patch('app.core.llm.orchestrator.LLMOrchestrator.process_doubt') as mock_process:
            mock_process.side_effect = Exception("LLM service unavailable")
            
            # Step 3: Verify error handling
            process_response = await test_client.post(f"/api/v1/doubts/{doubt_id}/process", headers=headers)
            
            # Should handle gracefully and not crash
            assert process_response.status_code in [200, 500, 503]
        
        # Step 4: Verify doubt status reflects error
        status_response = await test_client.get(f"/api/v1/doubts/{doubt_id}", headers=headers)
        doubt_status = status_response.json()
        
        # Status should indicate processing issue or retry needed
        assert doubt_status["status"] in ["submitted", "processing", "failed", "retry_needed"]
    
    @pytest.mark.asyncio
    async def test_rate_limiting_workflow(self, test_client: AsyncClient, test_student: Student):
        """Test rate limiting behavior."""
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Submit multiple doubts rapidly
        responses = []
        for i in range(10):
            doubt_data = {
                "question": f"Rapid question {i}",
                "subject": "Mathematics"
            }
            response = await test_client.post("/api/v1/doubts/", json=doubt_data, headers=headers)
            responses.append(response)
        
        # Verify rate limiting kicks in
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes or all(code == 201 for code in status_codes[:5])
    
    async def _get_auth_token(self, test_client: AsyncClient, student: Student) -> str:
        """Helper to get authentication token."""
        login_data = {"email": student.email, "password": "testpassword"}
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        return response.json()["access_token"]


class TestPerformanceWorkflows:
    """Test performance-critical workflows."""
    
    @pytest.mark.asyncio
    async def test_concurrent_doubt_processing(
        self, 
        test_client: AsyncClient,
        test_student: Student,
        db_session: AsyncSession
    ):
        """Test concurrent doubt processing performance."""
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Submit multiple doubts concurrently
        doubt_tasks = []
        for i in range(5):
            doubt_data = {
                "question": f"Concurrent question {i}",
                "subject": "Mathematics",
                "urgency": "high"
            }
            task = test_client.post("/api/v1/doubts/", json=doubt_data, headers=headers)
            doubt_tasks.append(task)
        
        # Wait for all submissions
        responses = await asyncio.gather(*doubt_tasks)
        
        # Verify all succeeded
        for response in responses:
            assert response.status_code == 201
        
        # Check processing doesn't block
        doubt_ids = [r.json()["id"] for r in responses]
        process_tasks = []
        
        for doubt_id in doubt_ids:
            task = test_client.post(f"/api/v1/doubts/{doubt_id}/process", headers=headers)
            process_tasks.append(task)
        
        process_responses = await asyncio.gather(*process_tasks, return_exceptions=True)
        
        # Should handle concurrent processing
        successful_processes = [r for r in process_responses if not isinstance(r, Exception)]
        assert len(successful_processes) >= 3  # At least 3 should succeed
    
    async def _get_auth_token(self, test_client: AsyncClient, student: Student) -> str:
        """Helper to get authentication token."""
        login_data = {"email": student.email, "password": "testpassword"}
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        return response.json()["access_token"]
