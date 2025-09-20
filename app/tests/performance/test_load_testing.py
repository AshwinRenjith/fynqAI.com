"""
Performance tests for load testing, stress testing, and scalability validation.
Tests system behavior under various load conditions.
"""

import asyncio
import pytest
import time
import statistics
from unittest.mock import patch
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Student
from app.services.doubt_service import DoubtProcessingService
from app.core.llm.orchestrator import LLMOrchestrator
from app.core.rag.retriever import RAGRetriever


class TestAPIPerformance:
    """Test API endpoint performance under load."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_doubt_submission_performance(self, test_client: AsyncClient, test_student: Student):
        """Test doubt submission endpoint performance under load."""
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Prepare test data
        doubt_data = {
            "question": "Performance test question for load testing",
            "subject": "Mathematics",
            "chapter": "Performance Testing",
            "difficulty_level": "intermediate",
            "urgency": "medium"
        }
        
        # Performance metrics
        response_times = []
        success_count = 0
        
        # Load test parameters
        num_requests = 50
        
        async def submit_doubt():
            start_time = time.time()
            try:
                response = await test_client.post("/api/v1/doubts/", json=doubt_data, headers=headers)
                end_time = time.time()
                response_times.append(end_time - start_time)
                
                if response.status_code == 201:
                    return "success"
                else:
                    return "error"
            except Exception:
                end_time = time.time()
                response_times.append(end_time - start_time)
                return "error"
        
        # Execute concurrent requests
        tasks = [submit_doubt() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Count results
        success_count = results.count("success")
        
        # Calculate performance metrics
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        
        # Assertions for performance requirements
        assert avg_response_time < 2.0, f"Average response time {avg_response_time:.2f}s exceeds 2s limit"
        assert p95_response_time < 5.0, f"95th percentile {p95_response_time:.2f}s exceeds 5s limit"
        assert p99_response_time < 10.0, f"99th percentile {p99_response_time:.2f}s exceeds 10s limit"
        
        # Success rate should be high
        success_rate = success_count / num_requests
        assert success_rate >= 0.95, f"Success rate {success_rate:.2f} below 95% threshold"
        
        print("Performance Results:")
        print(f"  Requests: {num_requests}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Avg Response Time: {avg_response_time:.3f}s")
        print(f"  95th Percentile: {p95_response_time:.3f}s")
        print(f"  99th Percentile: {p99_response_time:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_authentication_performance(self, test_client: AsyncClient):
        """Test authentication endpoint performance."""
        # Create test credentials
        login_data = {
            "email": "performance@test.com",
            "password": "TestPassword123!"
        }
        
        # Register user first
        registration_data = {
            **login_data,
            "first_name": "Performance",
            "last_name": "Test",
            "phone": "+91 9876543210",
            "board": "CBSE",
            "grade": 12,
            "subjects": ["Mathematics"]
        }
        
        await test_client.post("/api/v1/auth/register", json=registration_data)
        
        # Performance test login
        response_times = []
        num_requests = 30
        
        async def login_request():
            start_time = time.time()
            response = await test_client.post("/api/v1/auth/login", json=login_data)
            end_time = time.time()
            response_times.append(end_time - start_time)
            return response.status_code == 200
        
        # Execute concurrent logins
        tasks = [login_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Performance assertions
        avg_response_time = statistics.mean(response_times)
        success_rate = sum(results) / num_requests
        
        assert avg_response_time < 1.0, f"Login average response time {avg_response_time:.3f}s exceeds 1s"
        assert success_rate >= 0.98, f"Login success rate {success_rate:.2%} below 98%"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_search_performance(self, test_client: AsyncClient, test_student: Student):
        """Test search endpoint performance."""
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Test various search queries
        search_queries = [
            "limits in calculus",
            "derivatives mathematics",
            "physics kinematics",
            "chemistry organic compounds",
            "algebra linear equations"
        ]
        
        response_times = []
        
        async def search_request(query):
            start_time = time.time()
            response = await test_client.get(f"/api/v1/doubts/search?q={query}", headers=headers)
            end_time = time.time()
            response_times.append(end_time - start_time)
            return response.status_code == 200
        
        # Execute search requests
        tasks = [search_request(query) for query in search_queries for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Performance assertions
        avg_response_time = statistics.mean(response_times)
        success_rate = sum(results) / len(results)
        
        assert avg_response_time < 1.5, f"Search average response time {avg_response_time:.3f}s exceeds 1.5s"
        assert success_rate >= 0.95, f"Search success rate {success_rate:.2%} below 95%"
    
    async def _get_auth_token(self, test_client: AsyncClient, student: Student) -> str:
        """Helper to get authentication token."""
        login_data = {"email": student.email, "password": "testpassword"}
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        return response.json()["access_token"]


class TestDatabasePerformance:
    """Test database operations performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_bulk_doubt_creation(self, db_session: AsyncSession, test_student: Student):
        """Test bulk doubt creation performance."""
        doubt_service = DoubtProcessingService(db_session)
        
        # Prepare bulk data
        num_doubts = 100
        doubt_data_list = []
        
        for i in range(num_doubts):
            doubt_data = {
                "question": f"Bulk test question {i}",
                "subject": "Mathematics",
                "chapter": "Performance Testing",
                "difficulty_level": "intermediate",
                "context": f"Context for question {i}",
                "urgency": "medium"
            }
            doubt_data_list.append(doubt_data)
        
        # Measure bulk creation time
        start_time = time.time()
        
        created_doubts = []
        for doubt_data in doubt_data_list:
            doubt = await doubt_service.create_doubt(
                student_id=test_student.id,
                doubt_data=doubt_data
            )
            created_doubts.append(doubt)
        
        end_time = time.time()
        
        # Performance assertions
        total_time = end_time - start_time
        avg_time_per_doubt = total_time / num_doubts
        
        assert total_time < 30.0, f"Bulk creation took {total_time:.2f}s, exceeds 30s limit"
        assert avg_time_per_doubt < 0.3, f"Average per doubt {avg_time_per_doubt:.3f}s exceeds 0.3s"
        assert len(created_doubts) == num_doubts, "Not all doubts were created"
        
        print("Bulk Creation Performance:")
        print(f"  Total Doubts: {num_doubts}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Time per Doubt: {avg_time_per_doubt:.3f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_query_performance(self, db_session: AsyncSession, test_student: Student):
        """Test database query performance."""
        doubt_service = DoubtProcessingService(db_session)
        
        # Create test data first
        for i in range(50):
            await doubt_service.create_doubt(
                student_id=test_student.id,
                doubt_data={
                    "question": f"Query test question {i}",
                    "subject": "Mathematics" if i % 2 == 0 else "Physics",
                    "chapter": f"Chapter {i % 10}",
                    "difficulty_level": "intermediate"
                }
            )
        
        # Test various query patterns
        query_times = []
        
        # Test 1: Get all doubts for student
        start_time = time.time()
        all_doubts = await doubt_service.get_student_doubts(test_student.id)
        query_times.append(time.time() - start_time)
        
        # Test 2: Filter by subject
        start_time = time.time()
        math_doubts = await doubt_service.get_student_doubts(
            test_student.id, 
            filters={"subject": "Mathematics"}
        )
        query_times.append(time.time() - start_time)
        
        # Test 3: Pagination
        start_time = time.time()
        paginated_doubts = await doubt_service.get_student_doubts(
            test_student.id, 
            limit=10, 
            offset=0
        )
        query_times.append(time.time() - start_time)
        
        # Performance assertions
        avg_query_time = statistics.mean(query_times)
        max_query_time = max(query_times)
        
        assert avg_query_time < 0.5, f"Average query time {avg_query_time:.3f}s exceeds 0.5s"
        assert max_query_time < 1.0, f"Max query time {max_query_time:.3f}s exceeds 1.0s"
        
        # Verify query results
        assert len(all_doubts) >= 50
        assert len(math_doubts) >= 20
        assert len(paginated_doubts) == 10


class TestAIPerformance:
    """Test AI processing performance."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_llm_processing_performance(self, mock_llm_providers: dict):
        """Test LLM processing performance."""
        orchestrator = LLMOrchestrator()
        
        # Mock LLM responses for consistent testing
        with patch.object(orchestrator, 'process_doubt') as mock_process:
            mock_process.return_value = {
                "explanation": "Detailed mathematical explanation...",
                "examples": ["Example 1", "Example 2"],
                "confidence_score": 0.92,
                "processing_time": 0.5
            }
            
            # Test concurrent processing
            num_requests = 20
            processing_times = []
            
            async def process_doubt():
                start_time = time.time()
                result = await orchestrator.process_doubt(
                    question="Test performance question",
                    context="Mathematical context",
                    student_level="grade_12"
                )
                end_time = time.time()
                processing_times.append(end_time - start_time)
                return result
            
            # Execute concurrent processing
            tasks = [process_doubt() for _ in range(num_requests)]
            results = await asyncio.gather(*tasks)
            
            # Performance assertions
            avg_processing_time = statistics.mean(processing_times)
            p95_processing_time = statistics.quantiles(processing_times, n=20)[18]
            
            assert avg_processing_time < 1.0, f"Average processing time {avg_processing_time:.3f}s exceeds 1s"
            assert p95_processing_time < 2.0, f"95th percentile {p95_processing_time:.3f}s exceeds 2s"
            assert len(results) == num_requests, "Not all requests completed"
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_rag_retrieval_performance(self, mock_vector_db):
        """Test RAG retrieval performance."""
        retriever = RAGRetriever()
        
        with patch.object(retriever, 'retrieve_relevant_content') as mock_retrieve:
            # Mock retrieval with varying response times
            async def mock_retrieval(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate processing time
                return [
                    {
                        "content": "Relevant content for query...",
                        "source": "NCERT_Mathematics",
                        "relevance_score": 0.95
                    }
                ]
            
            mock_retrieve.side_effect = mock_retrieval
            
            # Test concurrent retrievals
            num_queries = 15
            retrieval_times = []
            
            async def retrieve_content(query_id):
                start_time = time.time()
                result = await retriever.retrieve_relevant_content(
                    query=f"Performance test query {query_id}",
                    subject="Mathematics",
                    filters={"difficulty": "intermediate"}
                )
                end_time = time.time()
                retrieval_times.append(end_time - start_time)
                return result
            
            # Execute concurrent retrievals
            tasks = [retrieve_content(i) for i in range(num_queries)]
            results = await asyncio.gather(*tasks)
            
            # Performance assertions
            avg_retrieval_time = statistics.mean(retrieval_times)
            max_retrieval_time = max(retrieval_times)
            
            assert avg_retrieval_time < 0.5, f"Average retrieval time {avg_retrieval_time:.3f}s exceeds 0.5s"
            assert max_retrieval_time < 1.0, f"Max retrieval time {max_retrieval_time:.3f}s exceeds 1.0s"
            assert len(results) == num_queries, "Not all retrievals completed"


class TestMemoryPerformance:
    """Test memory usage and optimization."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_under_load(self, test_client: AsyncClient, test_student: Student):
        """Test memory usage under high load."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Generate load
        num_requests = 100
        tasks = []
        
        for i in range(num_requests):
            doubt_data = {
                "question": f"Memory test question {i} with some additional context to test memory usage",
                "subject": "Mathematics",
                "chapter": "Memory Testing",
                "difficulty_level": "intermediate"
            }
            task = test_client.post("/api/v1/doubts/", json=doubt_data, headers=headers)
            tasks.append(task)
        
        # Execute all requests
        await asyncio.gather(*tasks)
        
        # Check memory usage after load
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase dramatically
        memory_increase_per_request = memory_increase / num_requests
        
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB, exceeds 100MB limit"
        assert memory_increase_per_request < 1.0, f"Memory per request {memory_increase_per_request:.3f}MB exceeds 1MB"
        
        print("Memory Performance:")
        print(f"  Initial Memory: {initial_memory:.2f}MB")
        print(f"  Final Memory: {final_memory:.2f}MB")
        print(f"  Memory Increase: {memory_increase:.2f}MB")
        print(f"  Per Request: {memory_increase_per_request:.3f}MB")
    
    async def _get_auth_token(self, test_client: AsyncClient, student: Student) -> str:
        """Helper to get authentication token."""
        login_data = {"email": student.email, "password": "testpassword"}
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        return response.json()["access_token"]


class TestScalabilityPerformance:
    """Test system scalability characteristics."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_user_simulation(self, test_client: AsyncClient):
        """Test system behavior with concurrent users."""
        # Create multiple test users
        num_users = 10
        users_data = []
        
        for i in range(num_users):
            user_data = {
                "email": f"scalability_user_{i}@test.com",
                "password": "TestPassword123!",
                "first_name": f"User{i}",
                "last_name": "Test",
                "phone": f"+91 987654321{i}",
                "board": "CBSE",
                "grade": 12,
                "subjects": ["Mathematics", "Physics"]
            }
            users_data.append(user_data)
        
        # Register all users
        registration_tasks = [
            test_client.post("/api/v1/auth/register", json=user_data)
            for user_data in users_data
        ]
        registration_responses = await asyncio.gather(*registration_tasks)
        
        # Extract tokens
        tokens = [response.json()["access_token"] for response in registration_responses]
        
        # Simulate concurrent user activity
        activity_times = []
        
        async def user_activity(token, user_id):
            headers = {"Authorization": f"Bearer {token}"}
            start_time = time.time()
            
            # Each user performs multiple actions
            actions = [
                test_client.get("/api/v1/students/profile", headers=headers),
                test_client.post("/api/v1/doubts/", json={
                    "question": f"Scalability test question from user {user_id}",
                    "subject": "Mathematics"
                }, headers=headers),
                test_client.get("/api/v1/analytics/student/doubts", headers=headers)
            ]
            
            await asyncio.gather(*actions)
            end_time = time.time()
            activity_times.append(end_time - start_time)
        
        # Execute concurrent user activities
        user_tasks = [user_activity(token, i) for i, token in enumerate(tokens)]
        await asyncio.gather(*user_tasks)
        
        # Performance assertions
        avg_activity_time = statistics.mean(activity_times)
        max_activity_time = max(activity_times)
        
        assert avg_activity_time < 5.0, f"Average user activity time {avg_activity_time:.2f}s exceeds 5s"
        assert max_activity_time < 10.0, f"Max user activity time {max_activity_time:.2f}s exceeds 10s"
        
        print("Scalability Results:")
        print(f"  Concurrent Users: {num_users}")
        print(f"  Avg Activity Time: {avg_activity_time:.2f}s")
        print(f"  Max Activity Time: {max_activity_time:.2f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput_limits(self, test_client: AsyncClient, test_student: Student):
        """Test system throughput limits."""
        headers = {"Authorization": f"Bearer {await self._get_auth_token(test_client, test_student)}"}
        
        # Test increasing load patterns
        load_patterns = [10, 25, 50, 75, 100]
        throughput_results = []
        
        for load in load_patterns:
            start_time = time.time()
            
            # Generate load
            tasks = []
            for i in range(load):
                doubt_data = {
                    "question": f"Throughput test {i} for load {load}",
                    "subject": "Mathematics"
                }
                task = test_client.post("/api/v1/doubts/", json=doubt_data, headers=headers)
                tasks.append(task)
            
            # Execute and measure
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            successful_requests = len([r for r in responses if not isinstance(r, Exception)])
            throughput = successful_requests / total_time  # requests per second
            
            throughput_results.append({
                "load": load,
                "throughput": throughput,
                "success_rate": successful_requests / load,
                "total_time": total_time
            })
        
        # Analyze throughput degradation
        for i, result in enumerate(throughput_results):
            print(f"Load {result['load']}: {result['throughput']:.2f} req/s, "
                  f"Success: {result['success_rate']:.2%}, Time: {result['total_time']:.2f}s")
            
            # Basic throughput requirements
            if result['load'] <= 50:
                assert result['throughput'] >= 10, f"Throughput {result['throughput']:.2f} below 10 req/s for load {result['load']}"
                assert result['success_rate'] >= 0.95, f"Success rate {result['success_rate']:.2%} below 95% for load {result['load']}"
    
    async def _get_auth_token(self, test_client: AsyncClient, student: Student) -> str:
        """Helper to get authentication token."""
        login_data = {"email": student.email, "password": "testpassword"}
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        return response.json()["access_token"]
