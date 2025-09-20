"""
Unit Tests for AI Core Modules
Testing PIL, MCP, RAG, and LLM Orchestration with maximum precision
"""

import pytest
from unittest.mock import patch, MagicMock


class TestPILReasoningEngine:
    """Test Process Intelligence Layer (PIL) Reasoning Engine"""

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_analyze_problem_structure_success(self, mock_llm_providers):
        """Test successful problem structure analysis"""
        from app.core.pil.reasoning_engine import ReasoningEngine
        
        reasoning_engine = ReasoningEngine()
        
        problem = "A ball is thrown upward with initial velocity 20 m/s. Find the maximum height."
        subject = "physics"
        
        # Mock LLM response for problem analysis
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"problem_type": "kinematics", "difficulty": "medium", "concepts": ["projectile_motion", "velocity", "acceleration"], "prerequisites": ["basic_algebra", "quadratic_equations"]}',
            "provider": "openai",
            "tokens_used": 120,
            "cost": 0.012
        }
        
        result = await reasoning_engine.analyze_problem_structure(problem, subject)
        
        assert result["problem_type"] == "kinematics"
        assert result["difficulty"] == "medium"
        assert "projectile_motion" in result["concepts"]
        assert "basic_algebra" in result["prerequisites"]

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_validate_solution_correct(self, mock_llm_providers):
        """Test solution validation for correct solution"""
        from app.core.pil.reasoning_engine import ReasoningEngine
        
        reasoning_engine = ReasoningEngine()
        
        problem = "Find the derivative of x^2"
        solution = "The derivative of x^2 is 2x using the power rule"
        subject = "mathematics"
        
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"is_valid": true, "confidence_score": 0.95, "errors": [], "reasoning": "Correct application of power rule"}',
            "provider": "openai"
        }
        
        result = await reasoning_engine.validate_solution(problem, solution, subject)
        
        assert result["is_valid"] is True
        assert result["confidence_score"] == 0.95
        assert len(result["errors"]) == 0

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_validate_solution_incorrect(self, mock_llm_providers):
        """Test solution validation for incorrect solution"""
        from app.core.pil.reasoning_engine import ReasoningEngine
        
        reasoning_engine = ReasoningEngine()
        
        problem = "Find the derivative of x^2"
        solution = "The derivative of x^2 is x^2"  # Incorrect
        subject = "mathematics"
        
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"is_valid": false, "confidence_score": 0.1, "errors": ["Incorrect application of derivative rule"], "reasoning": "Should be 2x, not x^2"}',
            "provider": "openai"
        }
        
        result = await reasoning_engine.validate_solution(problem, solution, subject)
        
        assert result["is_valid"] is False
        assert result["confidence_score"] == 0.1
        assert len(result["errors"]) > 0

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_generate_step_by_step_solution(self, mock_llm_providers):
        """Test step-by-step solution generation"""
        from app.core.pil.reasoning_engine import ReasoningEngine
        
        reasoning_engine = ReasoningEngine()
        
        problem = "Solve 2x + 5 = 13"
        subject = "mathematics"
        
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"steps": [{"step": 1, "action": "Subtract 5 from both sides", "equation": "2x = 8"}, {"step": 2, "action": "Divide both sides by 2", "equation": "x = 4"}], "final_answer": "x = 4"}',
            "provider": "openai"
        }
        
        result = await reasoning_engine.generate_step_by_step_solution(problem, subject)
        
        assert len(result["steps"]) == 2
        assert result["steps"][0]["action"] == "Subtract 5 from both sides"
        assert result["final_answer"] == "x = 4"

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_reasoning_engine_error_handling(self, mock_llm_providers):
        """Test reasoning engine error handling"""
        from app.core.pil.reasoning_engine import ReasoningEngine
        
        reasoning_engine = ReasoningEngine()
        
        # Mock LLM failure
        mock_llm_providers['openai'].generate_response.side_effect = Exception("LLM service unavailable")
        
        with pytest.raises(Exception) as exc_info:
            await reasoning_engine.analyze_problem_structure("test problem", "physics")
        
        assert "LLM service unavailable" in str(exc_info.value)


class TestMCPAdaptiveEngine:
    """Test Multi-Context Personalization (MCP) Adaptive Engine"""

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_adapt_response_visual_learner(self, mock_llm_providers):
        """Test response adaptation for visual learner"""
        from app.core.mcp.adaptive_engine import AdaptiveEngine
        
        adaptive_engine = AdaptiveEngine()
        
        original_response = "Force equals mass times acceleration (F = ma)"
        student_profile = {
            "learning_style": "visual",
            "difficulty_preference": "medium",
            "grade": 11,
            "strong_subjects": ["mathematics"],
            "weak_subjects": ["physics"]
        }
        
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"adapted_response": "Imagine a diagram: Force equals mass times acceleration (F = ma). Picture a heavy box (mass) being pushed with force - the harder you push, the faster it accelerates.", "adjustments": {"difficulty_adjusted": false, "style_adapted": true, "visual_elements_added": true}}',
            "provider": "openai"
        }
        
        result = await adaptive_engine.adapt_response(
            original_response, student_profile, "physics", "medium"
        )
        
        assert "diagram" in result["adapted_response"] or "Picture" in result["adapted_response"]
        assert result["adjustments"]["style_adapted"] is True

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_adapt_response_difficulty_adjustment(self, mock_llm_providers):
        """Test response adaptation with difficulty adjustment"""
        from app.core.mcp.adaptive_engine import AdaptiveEngine
        
        adaptive_engine = AdaptiveEngine()
        
        original_response = "The derivative represents the instantaneous rate of change"
        student_profile = {
            "learning_style": "kinesthetic",
            "difficulty_preference": "easy",
            "grade": 9,
            "performance_level": "beginner"
        }
        
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"adapted_response": "A derivative is like the speedometer in a car - it tells you how fast something is changing at that exact moment.", "adjustments": {"difficulty_adjusted": true, "style_adapted": true, "examples_added": true}}',
            "provider": "openai"
        }
        
        result = await adaptive_engine.adapt_response(
            original_response, student_profile, "mathematics", "hard"
        )
        
        assert "speedometer" in result["adapted_response"]
        assert result["adjustments"]["difficulty_adjusted"] is True

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_personalize_learning_path(self, mock_llm_providers):
        """Test learning path personalization"""
        from app.core.mcp.adaptive_engine import AdaptiveEngine
        
        adaptive_engine = AdaptiveEngine()
        
        student_profile = {
            "current_topics": ["algebra", "geometry"],
            "mastered_topics": ["arithmetic", "fractions"],
            "struggling_topics": ["trigonometry"],
            "learning_style": "auditory",
            "pace": "fast"
        }
        
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"recommended_path": [{"topic": "quadratic_equations", "priority": "high", "reasoning": "Natural progression from algebra"}, {"topic": "coordinate_geometry", "priority": "medium", "reasoning": "Combines algebra and geometry knowledge"}], "study_approach": "auditory_focused", "estimated_timeline": "2_weeks"}',
            "provider": "openai"
        }
        
        result = await adaptive_engine.personalize_learning_path(student_profile, "mathematics")
        
        assert len(result["recommended_path"]) >= 1
        assert result["study_approach"] == "auditory_focused"
        assert "estimated_timeline" in result

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_adapt_difficulty_level(self):
        """Test difficulty level adaptation logic"""
        from app.core.mcp.adaptive_engine import AdaptiveEngine
        
        adaptive_engine = AdaptiveEngine()
        
        # Test difficulty adjustment for struggling student
        result = adaptive_engine._adjust_difficulty_level(
            current_difficulty="hard",
            student_performance=0.4,  # Low performance
            learning_pace="slow"
        )
        
        assert result["adjusted_difficulty"] == "easy"
        assert result["adjustment_reason"] == "performance_based_reduction"
        
        # Test difficulty adjustment for excelling student
        result = adaptive_engine._adjust_difficulty_level(
            current_difficulty="easy",
            student_performance=0.95,  # High performance
            learning_pace="fast"
        )
        
        assert result["adjusted_difficulty"] == "medium"
        assert result["adjustment_reason"] == "performance_based_increase"


class TestRAGRetriever:
    """Test Retrieval-Augmented Generation (RAG) System"""

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_retrieve_similar_content_success(self, mock_vector_db):
        """Test successful similar content retrieval"""
        from app.core.rag.retriever import RAGRetriever
        
        rag_retriever = RAGRetriever()
        
        query = "Newton's laws of motion"
        subject = "physics"
        
        result = await rag_retriever.retrieve_similar_content(
            query=query,
            subject=subject,
            top_k=5,
            min_score=0.7
        )
        
        assert len(result) == 2  # Based on mock data
        assert result[0]["score"] == 0.95
        assert "content" in result[0]
        assert "metadata" in result[0]

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_retrieve_examples_by_subject(self, mock_vector_db):
        """Test example retrieval by subject"""
        from app.core.rag.retriever import RAGRetriever
        
        rag_retriever = RAGRetriever()
        
        result = await rag_retriever.retrieve_examples(
            subject="physics",
            topic="mechanics",
            count=3
        )
        
        assert len(result) == 1  # Based on mock data
        assert result[0]["subject"] == "physics"
        assert "question" in result[0]
        assert "answer" in result[0]

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_store_doubt_embedding(self, mock_vector_db):
        """Test storing doubt embedding"""
        from app.core.rag.retriever import RAGRetriever
        
        rag_retriever = RAGRetriever()
        
        result = await rag_retriever.store_doubt_embedding(
            doubt_id="test-doubt-id",
            question="What is acceleration?",
            answer="Acceleration is the rate of change of velocity",
            subject="physics",
            metadata={"difficulty": "easy", "topic": "mechanics"}
        )
        
        assert result is True  # Based on mock

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_generate_embedding(self):
        """Test embedding generation"""
        from app.core.rag.retriever import RAGRetriever
        
        rag_retriever = RAGRetriever()
        
        with patch('app.core.rag.retriever.OpenAIEmbeddings') as mock_embeddings:
            mock_embeddings_instance = MagicMock()
            mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
            mock_embeddings.return_value = mock_embeddings_instance
            
            result = await rag_retriever._generate_embedding("test text")
            
            assert len(result) == 4
            assert result == [0.1, 0.2, 0.3, 0.4]


class TestLLMOrchestrator:
    """Test LLM Orchestration System"""

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_select_optimal_provider_cost_optimization(self):
        """Test provider selection with cost optimization"""
        from app.core.llm.orchestrator import LLMOrchestrator
        
        orchestrator = LLMOrchestrator()
        
        # Mock provider costs and capabilities
        with patch.object(orchestrator, '_get_provider_costs') as mock_costs:
            mock_costs.return_value = {
                "openai": {"input": 0.01, "output": 0.03},
                "gemini": {"input": 0.005, "output": 0.015},
                "anthropic": {"input": 0.008, "output": 0.024},
                "mistral": {"input": 0.006, "output": 0.018}
            }
            
            result = orchestrator._select_optimal_provider(
                prompt="Simple question",
                optimization_strategy="cost",
                estimated_tokens=100
            )
            
            assert result["provider"] == "gemini"  # Cheapest option
            assert result["reasoning"] == "cost_optimized"

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_select_optimal_provider_performance_optimization(self):
        """Test provider selection with performance optimization"""
        from app.core.llm.orchestrator import LLMOrchestrator
        
        orchestrator = LLMOrchestrator()
        
        # Mock performance metrics
        with patch.object(orchestrator, '_get_provider_performance') as mock_performance:
            mock_performance.return_value = {
                "openai": {"avg_latency": 1500, "success_rate": 0.98},
                "gemini": {"avg_latency": 1200, "success_rate": 0.96},
                "anthropic": {"avg_latency": 1800, "success_rate": 0.99},
                "mistral": {"avg_latency": 1100, "success_rate": 0.94}
            }
            
            result = orchestrator._select_optimal_provider(
                prompt="Complex reasoning task",
                optimization_strategy="performance",
                task_complexity="high"
            )
            
            assert result["provider"] == "anthropic"  # Best success rate for complex tasks
            assert result["reasoning"] == "performance_optimized"

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_generate_response_with_fallback(self, mock_llm_providers):
        """Test response generation with fallback mechanism"""
        from app.core.llm.orchestrator import LLMOrchestrator
        
        orchestrator = LLMOrchestrator()
        
        # Mock primary provider failure
        mock_llm_providers['openai'].generate_response.side_effect = Exception("Service unavailable")
        
        # Mock fallback provider success
        mock_llm_providers['gemini'].generate_response.return_value = {
            "response": "Fallback response from Gemini",
            "provider": "gemini",
            "tokens_used": 80,
            "cost": 0.008
        }
        
        with patch.object(orchestrator, '_select_optimal_provider') as mock_select:
            mock_select.return_value = {"provider": "openai", "reasoning": "test"}
            
            result = await orchestrator.generate_response(
                prompt="Test prompt",
                context="Test context",
                optimization_strategy="balanced"
            )
            
            assert result["provider"] == "gemini"  # Fallback provider
            assert "Fallback response" in result["response"]

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_track_usage_metrics(self):
        """Test usage metrics tracking"""
        from app.core.llm.orchestrator import LLMOrchestrator
        
        orchestrator = LLMOrchestrator()
        
        # Mock metrics storage
        with patch.object(orchestrator, '_store_metrics') as mock_store:
            await orchestrator._track_usage_metrics(
                provider="openai",
                tokens_used=150,
                cost=0.015,
                latency_ms=1500,
                success=True
            )
            
            mock_store.assert_called_once()
            call_args = mock_store.call_args[0][0]
            assert call_args["provider"] == "openai"
            assert call_args["tokens_used"] == 150
            assert call_args["cost"] == 0.015
            assert call_args["success"] is True

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from app.core.llm.orchestrator import LLMOrchestrator
        
        orchestrator = LLMOrchestrator()
        
        # Mock rate limiter
        with patch.object(orchestrator, '_check_rate_limit') as mock_rate_limit:
            mock_rate_limit.return_value = {
                "allowed": False,
                "retry_after": 60,
                "reason": "quota_exceeded"
            }
            
            result = await orchestrator._handle_rate_limiting("openai")
            
            assert result["allowed"] is False
            assert result["retry_after"] == 60

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_context_optimization(self):
        """Test context optimization for token limits"""
        from app.core.llm.orchestrator import LLMOrchestrator
        
        orchestrator = LLMOrchestrator()
        
        long_context = "This is a very long context. " * 1000  # Simulate long context
        
        optimized = orchestrator._optimize_context(
            context=long_context,
            max_tokens=4000,
            preserve_priority=["important", "key"]
        )
        
        assert len(optimized) < len(long_context)
        assert "important" in optimized or "key" in optimized  # Priority preservation


class TestAIIntegration:
    """Test AI module integration"""

    @pytest.mark.unit
    @pytest.mark.ai
    async def test_pil_mcp_rag_integration(self, mock_llm_providers, mock_vector_db):
        """Test integration between PIL, MCP, and RAG"""
        from app.core.pil.reasoning_engine import ReasoningEngine
        from app.core.mcp.adaptive_engine import AdaptiveEngine
        from app.core.rag.retriever import RAGRetriever
        
        reasoning_engine = ReasoningEngine()
        adaptive_engine = AdaptiveEngine()
        rag_retriever = RAGRetriever()
        
        # Simulate integrated workflow
        problem = "Solve quadratic equation x^2 + 5x + 6 = 0"
        student_profile = {"learning_style": "visual", "difficulty_preference": "medium"}
        
        # Step 1: RAG retrieval
        similar_content = await rag_retriever.retrieve_similar_content(
            query=problem,
            subject="mathematics"
        )
        
        # Step 2: PIL analysis
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"problem_type": "quadratic_equation", "difficulty": "medium"}',
            "provider": "openai"
        }
        
        analysis = await reasoning_engine.analyze_problem_structure(problem, "mathematics")
        
        # Step 3: MCP adaptation
        mock_llm_providers['openai'].generate_response.return_value = {
            "response": '{"adapted_response": "Visual solution with graphs", "adjustments": {"style_adapted": true}}',
            "provider": "openai"
        }
        
        adapted_response = await adaptive_engine.adapt_response(
            "Standard solution", student_profile, "mathematics", "medium"
        )
        
        # Verify integration
        assert len(similar_content) > 0
        assert analysis["problem_type"] == "quadratic_equation"
        assert adapted_response["adjustments"]["style_adapted"] is True


if __name__ == "__main__":
    pytest.main([__file__])
