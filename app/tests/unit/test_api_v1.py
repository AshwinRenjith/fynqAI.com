"""
Unit Tests for API v1 Endpoints
Maximum precision testing for all API functionality
"""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
import uuid


class TestHealthEndpoints:
    """Test health check endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    async def test_health_check(self, test_client: AsyncClient):
        """Test basic health check endpoint"""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    @pytest.mark.unit
    @pytest.mark.api
    async def test_health_detailed(self, test_client: AsyncClient):
        """Test detailed health check with all services"""
        with patch('app.database.db_manager.check_postgres_health') as mock_db, \
             patch('app.database.cache_manager.check_redis_health') as mock_redis:
            
            mock_db.return_value = {"status": "healthy", "latency_ms": 5}
            mock_redis.return_value = {"status": "healthy", "latency_ms": 2}
            
            response = await test_client.get("/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["services"]["database"]["status"] == "healthy"
            assert data["services"]["cache"]["status"] == "healthy"


class TestAuthEndpoints:
    """Test authentication endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    async def test_register_student_success(self, test_client: AsyncClient, test_data):
        """Test successful student registration"""
        registration_data = {
            "name": test_data["student"]["name"],
            "email": test_data["student"]["email"],
            "phone": test_data["student"]["phone"],
            "password": "SecurePassword123!",
            "grade": test_data["student"]["grade"]
        }
        
        with patch('app.services.student_service.get_student_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.create_student.return_value = {
                "id": test_data["student"]["id"],
                **registration_data
            }
            mock_service.return_value = mock_service_instance
            
            response = await test_client.post("/api/v1/auth/register", json=registration_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == test_data["student"]["id"]
            assert data["email"] == registration_data["email"]
            assert "password" not in data  # Ensure password is not returned

    @pytest.mark.unit
    @pytest.mark.api
    async def test_register_student_validation_error(self, test_client: AsyncClient):
        """Test registration with invalid data"""
        invalid_data = {
            "name": "",  # Empty name
            "email": "invalid-email",  # Invalid email format
            "phone": "123",  # Invalid phone
            "password": "weak",  # Weak password
            "grade": 15  # Invalid grade
        }
        
        response = await test_client.post("/api/v1/auth/register", json=invalid_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert len(data["detail"]) > 0  # Should have validation errors

    @pytest.mark.unit
    @pytest.mark.api
    async def test_login_success(self, test_client: AsyncClient, test_data):
        """Test successful login"""
        login_data = {
            "email": test_data["student"]["email"],
            "password": "SecurePassword123!"
        }
        
        with patch('app.utils.security.PasswordManager.verify_password') as mock_verify, \
             patch('app.utils.security.JWTManager.create_access_token') as mock_jwt, \
             patch('app.services.student_service.get_student_service') as mock_service:
            
            mock_verify.return_value = True
            mock_jwt.return_value = test_data["auth"]["valid_token"]
            
            mock_service_instance = AsyncMock()
            mock_service_instance.get_student_by_email.return_value = test_data["student"]
            mock_service.return_value = mock_service_instance
            
            response = await test_client.post("/api/v1/auth/login", json=login_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["access_token"] == test_data["auth"]["valid_token"]
            assert data["token_type"] == "bearer"

    @pytest.mark.unit
    @pytest.mark.api
    async def test_login_invalid_credentials(self, test_client: AsyncClient):
        """Test login with invalid credentials"""
        login_data = {
            "email": "wrong@example.com",
            "password": "wrongpassword"
        }
        
        with patch('app.services.student_service.get_student_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_student_by_email.return_value = None
            mock_service.return_value = mock_service_instance
            
            response = await test_client.post("/api/v1/auth/login", json=login_data)
            
            assert response.status_code == 401
            data = response.json()
            assert "Invalid credentials" in data["detail"]


class TestDoubtEndpoints:
    """Test doubt management endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    async def test_submit_doubt_success(self, authenticated_client: AsyncClient, test_data):
        """Test successful doubt submission"""
        doubt_data = {
            "question_text": test_data["doubt"]["question_text"],
            "subject_code": test_data["doubt"]["subject_code"],
            "topic": test_data["doubt"]["topic"],
            "difficulty_level": test_data["doubt"]["difficulty_level"],
            "question_type": test_data["doubt"]["question_type"]
        }
        
        with patch('app.services.doubt_service.get_doubt_processing_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.submit_doubt.return_value = {
                "id": test_data["doubt"]["id"],
                "status": "submitted",
                **doubt_data
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.post("/api/v1/doubts/submit", json=doubt_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == test_data["doubt"]["id"]
            assert data["status"] == "submitted"
            assert data["question_text"] == doubt_data["question_text"]

    @pytest.mark.unit
    @pytest.mark.api
    async def test_submit_doubt_unauthorized(self, test_client: AsyncClient, test_data):
        """Test doubt submission without authentication"""
        doubt_data = {
            "question_text": test_data["doubt"]["question_text"],
            "subject_code": test_data["doubt"]["subject_code"]
        }
        
        response = await test_client.post("/api/v1/doubts/submit", json=doubt_data)
        
        assert response.status_code == 401

    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_doubt_status(self, authenticated_client: AsyncClient, test_data):
        """Test getting doubt status"""
        doubt_id = test_data["doubt"]["id"]
        
        with patch('app.services.doubt_service.get_doubt_processing_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_doubt_status.return_value = {
                "id": doubt_id,
                "status": "processing",
                "progress": 75,
                "estimated_completion": "2024-01-01T12:00:00Z"
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.get(f"/api/v1/doubts/{doubt_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == doubt_id
            assert data["status"] == "processing"
            assert data["progress"] == 75

    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_doubt_response(self, authenticated_client: AsyncClient, test_data):
        """Test getting doubt response"""
        doubt_id = test_data["doubt"]["id"]
        
        with patch('app.services.doubt_service.get_doubt_processing_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_doubt_response.return_value = {
                "id": doubt_id,
                "status": "resolved",
                "response": "Newton's second law states that F = ma",
                "explanation": "Detailed explanation here",
                "examples": ["Example 1", "Example 2"],
                "confidence_score": 0.95,
                "processing_time_ms": 2500
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.get(f"/api/v1/doubts/{doubt_id}/response")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == doubt_id
            assert data["status"] == "resolved"
            assert data["confidence_score"] == 0.95

    @pytest.mark.unit
    @pytest.mark.api
    async def test_list_student_doubts(self, authenticated_client: AsyncClient, test_data):
        """Test listing student's doubts"""
        with patch('app.services.doubt_service.get_doubt_processing_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_student_doubts.return_value = {
                "doubts": [
                    {
                        "id": test_data["doubt"]["id"],
                        "question_text": test_data["doubt"]["question_text"],
                        "status": "resolved",
                        "created_at": "2024-01-01T10:00:00Z"
                    }
                ],
                "total": 1,
                "page": 1,
                "per_page": 10
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.get("/api/v1/doubts/my-doubts")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["doubts"]) == 1
            assert data["doubts"][0]["id"] == test_data["doubt"]["id"]


class TestStudentEndpoints:
    """Test student management endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_student_profile(self, authenticated_client: AsyncClient, test_data):
        """Test getting student profile"""
        with patch('app.services.student_service.get_student_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_student_profile.return_value = test_data["student"]
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.get("/api/v1/students/profile")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == test_data["student"]["id"]
            assert data["name"] == test_data["student"]["name"]
            assert data["email"] == test_data["student"]["email"]

    @pytest.mark.unit
    @pytest.mark.api
    async def test_update_student_profile(self, authenticated_client: AsyncClient, test_data):
        """Test updating student profile"""
        update_data = {
            "name": "Updated Name",
            "learning_style": "kinesthetic",
            "difficulty_preference": "hard"
        }
        
        with patch('app.services.student_service.get_student_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.update_student_profile.return_value = {
                **test_data["student"],
                **update_data
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.put("/api/v1/students/profile", json=update_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == update_data["name"]
            assert data["learning_style"] == update_data["learning_style"]

    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_student_analytics(self, authenticated_client: AsyncClient, test_data):
        """Test getting student analytics"""
        with patch('app.services.student_service.get_student_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_student_analytics.return_value = {
                "total_doubts": 25,
                "resolved_doubts": 20,
                "resolution_rate": 80.0,
                "avg_resolution_time": 1800,
                "subject_breakdown": {
                    "physics": {"total": 10, "resolved": 8},
                    "mathematics": {"total": 15, "resolved": 12}
                },
                "learning_patterns": {
                    "most_active_time": "evening",
                    "preferred_difficulty": "medium",
                    "strong_subjects": ["mathematics"],
                    "improvement_areas": ["physics"]
                }
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.get("/api/v1/students/analytics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_doubts"] == 25
            assert data["resolution_rate"] == 80.0
            assert "subject_breakdown" in data
            assert "learning_patterns" in data


class TestFeedbackEndpoints:
    """Test feedback management endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    async def test_submit_feedback_success(self, authenticated_client: AsyncClient, test_data):
        """Test successful feedback submission"""
        feedback_data = {
            "doubt_id": test_data["doubt"]["id"],
            "rating": 5,
            "feedback_text": "Excellent explanation, very clear and helpful!",
            "feedback_type": "positive"
        }
        
        with patch('app.services.feedback_service.get_feedback_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.submit_feedback.return_value = {
                "id": str(uuid.uuid4()),
                "status": "submitted",
                **feedback_data
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.post("/api/v1/feedback/submit", json=feedback_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data["rating"] == feedback_data["rating"]
            assert data["feedback_text"] == feedback_data["feedback_text"]
            assert data["status"] == "submitted"

    @pytest.mark.unit
    @pytest.mark.api
    async def test_submit_feedback_invalid_rating(self, authenticated_client: AsyncClient, test_data):
        """Test feedback submission with invalid rating"""
        feedback_data = {
            "doubt_id": test_data["doubt"]["id"],
            "rating": 6,  # Invalid rating (should be 1-5)
            "feedback_text": "Test feedback"
        }
        
        response = await authenticated_client.post("/api/v1/feedback/submit", json=feedback_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestAnalyticsEndpoints:
    """Test analytics endpoints"""

    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_system_metrics(self, authenticated_client: AsyncClient):
        """Test getting system metrics"""
        with patch('app.api.v1.analytics.get_analytics_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.get_system_metrics.return_value = {
                "daily": {
                    "total_doubts": 150,
                    "resolved_doubts": 120,
                    "resolution_rate": 80.0,
                    "active_students": 45
                },
                "weekly": {
                    "total_doubts": 800,
                    "resolved_doubts": 650,
                    "resolution_rate": 81.25,
                    "active_students": 200
                },
                "monthly": {
                    "total_doubts": 3200,
                    "resolved_doubts": 2600,
                    "resolution_rate": 81.5,
                    "active_students": 500
                }
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.get("/api/v1/analytics/system-metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert "daily" in data
            assert "weekly" in data
            assert "monthly" in data
            assert data["daily"]["total_doubts"] == 150

    @pytest.mark.unit
    @pytest.mark.api
    async def test_get_performance_report(self, authenticated_client: AsyncClient):
        """Test getting performance report"""
        with patch('app.api.v1.analytics.get_analytics_service') as mock_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.generate_performance_report.return_value = {
                "period": {
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-07T23:59:59Z",
                    "days": 7
                },
                "summary": {
                    "total_doubts": 500,
                    "overall_resolution_rate": 82.0,
                    "avg_resolution_time": 1850,
                    "peak_hour": 19
                },
                "trends": [
                    {"date": "2024-01-01", "total_doubts": 70, "resolution_rate": 80.0},
                    {"date": "2024-01-02", "total_doubts": 75, "resolution_rate": 82.0}
                ]
            }
            mock_service.return_value = mock_service_instance
            
            response = await authenticated_client.get("/api/v1/analytics/performance-report?period_days=7")
            
            assert response.status_code == 200
            data = response.json()
            assert data["period"]["days"] == 7
            assert data["summary"]["total_doubts"] == 500
            assert len(data["trends"]) == 2


# Error handling tests
class TestErrorHandling:
    """Test API error handling"""

    @pytest.mark.unit
    @pytest.mark.api
    async def test_404_not_found(self, test_client: AsyncClient):
        """Test 404 error handling"""
        response = await test_client.get("/api/v1/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    @pytest.mark.unit
    @pytest.mark.api
    async def test_method_not_allowed(self, test_client: AsyncClient):
        """Test 405 method not allowed"""
        response = await test_client.patch("/health")  # GET only endpoint
        
        assert response.status_code == 405

    @pytest.mark.unit
    @pytest.mark.api
    async def test_validation_error_format(self, test_client: AsyncClient):
        """Test validation error response format"""
        invalid_data = {"invalid": "data"}
        
        response = await test_client.post("/api/v1/auth/register", json=invalid_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)


if __name__ == "__main__":
    pytest.main([__file__])
