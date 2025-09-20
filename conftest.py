"""
PyTest Configuration and Global Fixtures
Maximum precision testing infrastructure for fynqAI backend
"""

import pytest
import asyncio
import warnings
from unittest.mock import AsyncMock, patch
import os

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"  
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API endpoint test"
    )
    config.addinivalue_line(
        "markers", "worker: mark test as background worker test"
    )
    config.addinivalue_line(
        "markers", "ai: mark test as AI/LLM functionality test"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Test environment settings"""
    # Override settings for testing
    os.environ.update({
        "ENVIRONMENT": "test",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/fynqai_test",
        "REDIS_URL": "redis://localhost:6379/1",
        "PINECONE_API_KEY": "test-pinecone-key",
        "OPENAI_API_KEY": "test-openai-key",
        "GOOGLE_API_KEY": "test-google-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "MISTRAL_API_KEY": "test-mistral-key",
        "JWT_SECRET_KEY": "test-jwt-secret-key-for-testing-only",
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_KEY": "test-supabase-key",
        "CELERY_BROKER_URL": "redis://localhost:6379/2",
        "CELERY_RESULT_BACKEND": "redis://localhost:6379/2",
        "LOG_LEVEL": "ERROR"  # Reduce logging noise during tests
    })
    
    from app.config import Settings
    return Settings()


@pytest.fixture(scope="session") 
async def test_app(test_settings):
    """Create test FastAPI application"""
    from app.main import create_app
    
    app = create_app()
    
    # Override dependencies for testing
    from app.dependencies import get_settings
    app.dependency_overrides[get_settings] = lambda: test_settings
    
    yield app
    
    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
async def test_client(test_app):
    """Create test HTTP client"""
    from httpx import AsyncClient
    
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
async def db_session():
    """Create test database session with rollback"""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    # Create test engine
    engine = create_async_engine(
        "postgresql+asyncpg://test:test@localhost:5432/fynqai_test",
        echo=False,
        pool_pre_ping=True
    )
    
    # Create test session factory
    TestSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        # Begin transaction
        transaction = await session.begin()
        
        try:
            yield session
        finally:
            # Rollback transaction to keep test database clean
            await transaction.rollback()
            await session.close()
    
    await engine.dispose()


@pytest.fixture(scope="function")
async def test_cache():
    """Create test Redis cache connection"""
    import redis.asyncio as redis
    
    cache = redis.Redis.from_url(
        "redis://localhost:6379/1",
        decode_responses=True
    )
    
    # Clear test cache before each test
    await cache.flushdb()
    
    yield cache
    
    # Cleanup after test
    await cache.flushdb()
    await cache.close()


@pytest.fixture(scope="function")
def mock_llm_providers():
    """Mock all LLM providers for testing"""
    mocks = {}
    
    # Mock OpenAI
    with patch('app.core.llm.providers.openai.OpenAIProvider') as mock_openai:
        mock_openai_instance = AsyncMock()
        mock_openai_instance.generate_response.return_value = {
            "response": "Test response from OpenAI",
            "provider": "openai",
            "model": "gpt-4",
            "tokens_used": 100,
            "cost": 0.01,
            "processing_time_ms": 1500
        }
        mock_openai.return_value = mock_openai_instance
        mocks['openai'] = mock_openai_instance
        
        # Mock Gemini
        with patch('app.core.llm.providers.gemini.GeminiProvider') as mock_gemini:
            mock_gemini_instance = AsyncMock()
            mock_gemini_instance.generate_response.return_value = {
                "response": "Test response from Gemini",
                "provider": "gemini",
                "model": "gemini-pro",
                "tokens_used": 80,
                "cost": 0.008,
                "processing_time_ms": 1200
            }
            mock_gemini.return_value = mock_gemini_instance
            mocks['gemini'] = mock_gemini_instance
            
            # Mock Anthropic
            with patch('app.core.llm.providers.anthropic.AnthropicProvider') as mock_anthropic:
                mock_anthropic_instance = AsyncMock()
                mock_anthropic_instance.generate_response.return_value = {
                    "response": "Test response from Anthropic",
                    "provider": "anthropic", 
                    "model": "claude-3-sonnet",
                    "tokens_used": 90,
                    "cost": 0.009,
                    "processing_time_ms": 1300
                }
                mock_anthropic.return_value = mock_anthropic_instance
                mocks['anthropic'] = mock_anthropic_instance
                
                # Mock Mistral
                with patch('app.core.llm.providers.mistral.MistralProvider') as mock_mistral:
                    mock_mistral_instance = AsyncMock()
                    mock_mistral_instance.generate_response.return_value = {
                        "response": "Test response from Mistral",
                        "provider": "mistral",
                        "model": "mistral-large",
                        "tokens_used": 85,
                        "cost": 0.0085,
                        "processing_time_ms": 1100
                    }
                    mock_mistral.return_value = mock_mistral_instance
                    mocks['mistral'] = mock_mistral_instance
                    
                    yield mocks


@pytest.fixture(scope="function")
def mock_vector_db():
    """Mock Pinecone vector database"""
    with patch('app.core.rag.retriever.RAGRetriever') as mock_rag:
        mock_rag_instance = AsyncMock()
        
        # Mock retrieve methods
        mock_rag_instance.retrieve_similar_content.return_value = [
            {
                "id": "test_content_1",
                "content": "Test similar content 1",
                "score": 0.95,
                "metadata": {"subject": "physics", "topic": "mechanics"}
            },
            {
                "id": "test_content_2", 
                "content": "Test similar content 2",
                "score": 0.87,
                "metadata": {"subject": "physics", "topic": "thermodynamics"}
            }
        ]
        
        mock_rag_instance.retrieve_examples.return_value = [
            {
                "id": "example_1",
                "question": "Test example question 1",
                "answer": "Test example answer 1",
                "subject": "physics"
            }
        ]
        
        mock_rag_instance.store_doubt_embedding.return_value = True
        
        mock_rag.return_value = mock_rag_instance
        yield mock_rag_instance


@pytest.fixture(scope="function")
def mock_celery_workers():
    """Mock Celery workers for testing"""
    mocks = {}
    
    # Mock doubt processor tasks
    with patch('app.workers.doubt_processor.process_doubt_task') as mock_process:
        mock_process.delay.return_value.get.return_value = {
            "doubt_id": "test-doubt-id",
            "status": "completed",
            "processing_time": 2000,
            "provider": "openai",
            "tokens_used": 150,
            "cost": 0.015
        }
        mocks['process_doubt'] = mock_process
        
        # Mock analytics tasks
        with patch('app.workers.analytics_worker.compute_student_analytics') as mock_analytics:
            mock_analytics.delay.return_value.get.return_value = {
                "student_id": "test-student-id",
                "total_doubts": 10,
                "resolved_doubts": 8,
                "resolution_rate": 80.0,
                "avg_resolution_time": 1800
            }
            mocks['analytics'] = mock_analytics
            
            # Mock notification tasks
            with patch('app.workers.notification_worker.send_doubt_resolved_notification') as mock_notification:
                mock_notification.delay.return_value.get.return_value = {
                    "status": "sent",
                    "notification_id": "test-notification-id",
                    "channels": ["push", "email"]
                }
                mocks['notification'] = mock_notification
                
                yield mocks


@pytest.fixture(scope="function")
def test_data():
    """Test data fixtures"""
    import uuid
    
    return {
        "student": {
            "id": str(uuid.uuid4()),
            "name": "Test Student",
            "email": "test.student@example.com",
            "phone": "+1234567890",
            "grade": 12,
            "subjects": ["physics", "mathematics", "chemistry"],
            "learning_style": "visual",
            "difficulty_preference": "medium"
        },
        "doubt": {
            "id": str(uuid.uuid4()),
            "question_text": "What is Newton's second law of motion?",
            "subject_code": "PHYS",
            "topic": "mechanics",
            "difficulty_level": "medium",
            "question_type": "conceptual",
            "attachments": []
        },
        "subject": {
            "id": str(uuid.uuid4()),
            "code": "PHYS",
            "name": "Physics",
            "description": "Physical sciences and natural phenomena"
        },
        "auth": {
            "valid_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token",
            "expired_token": "expired.jwt.token",
            "invalid_token": "invalid.token.format"
        }
    }


@pytest.fixture(scope="function")
async def authenticated_client(test_client, test_data):
    """Create authenticated test client"""
    # Mock authentication
    from app.utils.security import JWTManager
    
    with patch.object(JWTManager, 'verify_token') as mock_verify:
        mock_verify.return_value = {
            "sub": test_data["student"]["id"],
            "email": test_data["student"]["email"],
            "exp": 9999999999  # Far future expiry
        }
        
        # Set authorization header
        test_client.headers.update({
            "Authorization": f"Bearer {test_data['auth']['valid_token']}"
        })
        
        yield test_client


@pytest.fixture(scope="function", autouse=True)
def cleanup_files():
    """Cleanup temporary files after each test"""
    yield
    
    # Cleanup any temporary files created during testing
    import glob
    import os
    
    temp_patterns = [
        "/tmp/fynqai_test_*",
        "/tmp/test_upload_*",
        "/tmp/test_backup_*"
    ]
    
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
            except (OSError, FileNotFoundError):
                pass


# Pytest plugins and hooks
def pytest_collection_modifyitems(config, items):
    """Modify test items to add markers based on file location"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add markers based on test name patterns
        if "test_api" in item.name:
            item.add_marker(pytest.mark.api)
        elif "test_worker" in item.name:
            item.add_marker(pytest.mark.worker) 
        elif "test_ai" in item.name or "test_llm" in item.name:
            item.add_marker(pytest.mark.ai)


def pytest_runtest_setup(item):
    """Setup before each test"""
    # Skip external tests if no internet connection
    if item.get_closest_marker("external"):
        pytest.skip("External service tests disabled")


def pytest_sessionstart(session):
    """Setup before test session starts"""
    print("\n🚀 Starting fynqAI Test Suite - Maximum Precision Mode")
    

def pytest_sessionfinish(session, exitstatus):
    """Cleanup after test session ends"""
    print("\n✅ fynqAI Test Suite Complete")


# Custom pytest markers for better test organization
pytestmark = [
    pytest.mark.asyncio,  # Enable async support for all tests
]
