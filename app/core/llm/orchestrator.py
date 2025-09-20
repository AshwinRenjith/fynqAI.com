"""
LLM Orchestrator
Intelligent routing and management of multiple LLM providers
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
import time

from app.config import get_settings
from app.core.llm.providers.gemini import GeminiProvider
from app.core.llm.providers.openai import OpenAIProvider
from app.core.llm.providers.anthropic import AnthropicProvider
from app.core.llm.providers.mistral import MistralProvider
from app.exceptions import LLMProviderError


logger = logging.getLogger(__name__)
settings = get_settings()


class ModelTier(Enum):
    """Model performance tiers for routing decisions"""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class LLMOrchestrator:
    """Intelligent LLM provider orchestration with cost optimization"""
    
    def __init__(self):
        self.providers = self._initialize_providers()
        self.fallback_chain = ["gemini", "openai", "anthropic", "mistral"]
        self.cost_tracker = {}
        self.performance_metrics = {}
    
    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize all available LLM providers"""
        providers = {}
        
        if settings.GEMINI_API_KEY:
            providers["gemini"] = GeminiProvider()
        
        if settings.OPENAI_API_KEY:
            providers["openai"] = OpenAIProvider()
        
        if settings.ANTHROPIC_API_KEY:
            providers["anthropic"] = AnthropicProvider()
        
        if settings.MISTRAL_API_KEY:
            providers["mistral"] = MistralProvider()
        
        logger.info(f"Initialized {len(providers)} LLM providers: {list(providers.keys())}")
        return providers
    
    async def generate_response(
        self,
        prompt: str,
        context: Dict[str, Any] = None,
        preferred_provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate response using optimal provider selection
        
        Args:
            prompt: Input prompt for the LLM
            context: Additional context for provider selection
            preferred_provider: Force use of specific provider
            max_tokens: Maximum tokens for response
            temperature: Generation temperature
        
        Returns:
            Generated response with metadata
        """
        try:
            start_time = time.time()
            
            # Select optimal provider
            if preferred_provider and preferred_provider in self.providers:
                selected_provider = preferred_provider
            else:
                selected_provider = await self._select_optimal_provider(prompt, context or {})
            
            # Generate response with fallback handling
            response = await self._generate_with_fallback(
                prompt=prompt,
                selected_provider=selected_provider,
                max_tokens=max_tokens,
                temperature=temperature,
                context=context or {}
            )
            
            # Track performance and costs
            generation_time = time.time() - start_time
            await self._track_usage(
                provider=response.get("provider"),
                tokens=response.get("tokens_used", 0),
                cost=response.get("cost", 0.0),
                generation_time=generation_time
            )
            
            # Add orchestration metadata
            response["orchestration"] = {
                "selected_provider": selected_provider,
                "generation_time": generation_time,
                "fallback_used": response.get("fallback_used", False),
                "cost_optimization": await self._get_cost_optimization_info(selected_provider)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"LLM orchestration failed: {e}", exc_info=True)
            raise LLMProviderError("orchestrator", f"Generation failed: {str(e)}")
    
    async def _select_optimal_provider(
        self, 
        prompt: str, 
        context: Dict[str, Any]
    ) -> str:
        """Select optimal provider based on prompt characteristics and context"""
        
        # Analyze prompt requirements
        prompt_analysis = await self._analyze_prompt(prompt, context)
        
        # Check daily cost limits
        within_budget_providers = await self._get_within_budget_providers()
        
        # Score available providers
        provider_scores = {}
        
        for provider_name in self.providers.keys():
            if provider_name in within_budget_providers:
                score = await self._score_provider(provider_name, prompt_analysis, context)
                provider_scores[provider_name] = score
        
        # Select highest scoring provider
        if provider_scores:
            selected_provider = max(provider_scores, key=provider_scores.get)
            
            logger.info(
                f"Selected provider: {selected_provider}",
                extra={
                    "provider_scores": provider_scores,
                    "prompt_analysis": prompt_analysis
                }
            )
            
            return selected_provider
        else:
            # Fallback to first available provider
            return self.fallback_chain[0] if self.fallback_chain else "gemini"
    
    async def _analyze_prompt(
        self, 
        prompt: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze prompt characteristics for provider selection"""
        
        analysis = {
            "length": len(prompt),
            "complexity": "medium",
            "domain": "general",
            "requires_math": False,
            "requires_reasoning": False,
            "requires_creativity": False,
            "suggested_tier": ModelTier.STANDARD
        }
        
        prompt_lower = prompt.lower()
        
        # Detect mathematical content
        math_indicators = ["solve", "calculate", "derivative", "integral", "equation", "formula"]
        if any(indicator in prompt_lower for indicator in math_indicators):
            analysis["requires_math"] = True
            analysis["domain"] = "mathematics"
        
        # Detect reasoning requirements
        reasoning_indicators = ["explain", "analyze", "prove", "reason", "logic"]
        if any(indicator in prompt_lower for indicator in reasoning_indicators):
            analysis["requires_reasoning"] = True
        
        # Detect creative requirements
        creative_indicators = ["creative", "generate", "write", "compose"]
        if any(indicator in prompt_lower for indicator in creative_indicators):
            analysis["requires_creativity"] = True
        
        # Determine complexity
        if len(prompt) > 2000 or analysis["requires_reasoning"]:
            analysis["complexity"] = "high"
            analysis["suggested_tier"] = ModelTier.PREMIUM
        elif len(prompt) < 500 and not analysis["requires_math"]:
            analysis["complexity"] = "low"
            analysis["suggested_tier"] = ModelTier.BASIC
        
        # Add context-based analysis
        subject = context.get("subject", "")
        if subject in ["mathematics", "physics", "chemistry"]:
            analysis["domain"] = "science"
            analysis["requires_math"] = True
        
        return analysis
    
    async def _get_within_budget_providers(self) -> List[str]:
        """Get providers that are within daily cost budget"""
        within_budget = []
        
        for provider_name in self.providers.keys():
            daily_cost = self.cost_tracker.get(provider_name, {}).get("daily_cost", 0.0)
            
            # Simple budget check - can be made more sophisticated
            max_daily_cost = settings.MAX_DAILY_COST_USD / len(self.providers)
            
            if daily_cost < max_daily_cost:
                within_budget.append(provider_name)
        
        return within_budget if within_budget else list(self.providers.keys())
    
    async def _score_provider(
        self, 
        provider_name: str, 
        prompt_analysis: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> float:
        """Score provider based on capabilities and cost efficiency"""
        
        base_score = 0.5
        
        # Provider-specific scoring
        provider_capabilities = {
            "gemini": {
                "math_capability": 0.9,
                "reasoning_capability": 0.85,
                "cost_efficiency": 0.95,
                "speed": 0.9
            },
            "openai": {
                "math_capability": 0.95,
                "reasoning_capability": 0.9,
                "cost_efficiency": 0.7,
                "speed": 0.85
            },
            "anthropic": {
                "math_capability": 0.85,
                "reasoning_capability": 0.95,
                "cost_efficiency": 0.6,
                "speed": 0.8
            },
            "mistral": {
                "math_capability": 0.8,
                "reasoning_capability": 0.8,
                "cost_efficiency": 0.85,
                "speed": 0.9
            }
        }
        
        capabilities = provider_capabilities.get(provider_name, {})
        
        # Calculate score based on requirements
        if prompt_analysis["requires_math"]:
            base_score += capabilities.get("math_capability", 0.5) * 0.3
        
        if prompt_analysis["requires_reasoning"]:
            base_score += capabilities.get("reasoning_capability", 0.5) * 0.3
        
        # Factor in cost efficiency
        base_score += capabilities.get("cost_efficiency", 0.5) * 0.2
        
        # Factor in speed for real-time applications
        base_score += capabilities.get("speed", 0.5) * 0.2
        
        # Adjust based on recent performance
        recent_performance = self.performance_metrics.get(provider_name, {}).get("success_rate", 1.0)
        base_score *= recent_performance
        
        return min(base_score, 1.0)
    
    async def _generate_with_fallback(
        self,
        prompt: str,
        selected_provider: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response with automatic fallback on failure"""
        
        # Try primary provider
        try:
            provider = self.providers[selected_provider]
            response = await provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                context=context
            )
            response["provider"] = selected_provider
            response["fallback_used"] = False
            return response
            
        except Exception as e:
            logger.warning(f"Primary provider {selected_provider} failed: {e}")
            
            # Try fallback providers
            for fallback_provider in self.fallback_chain:
                if fallback_provider == selected_provider:
                    continue  # Skip the failed provider
                
                if fallback_provider not in self.providers:
                    continue  # Skip unavailable providers
                
                try:
                    logger.info(f"Trying fallback provider: {fallback_provider}")
                    provider = self.providers[fallback_provider]
                    response = await provider.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        context=context
                    )
                    response["provider"] = fallback_provider
                    response["fallback_used"] = True
                    response["original_provider_failed"] = selected_provider
                    return response
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback provider {fallback_provider} failed: {fallback_error}")
                    continue
            
            # All providers failed
            raise LLMProviderError(
                "all_providers", 
                "All available LLM providers failed"
            )
    
    async def _track_usage(
        self,
        provider: str,
        tokens: int,
        cost: float,
        generation_time: float
    ):
        """Track provider usage metrics"""
        
        if provider not in self.cost_tracker:
            self.cost_tracker[provider] = {
                "daily_cost": 0.0,
                "daily_tokens": 0,
                "daily_requests": 0
            }
        
        if provider not in self.performance_metrics:
            self.performance_metrics[provider] = {
                "success_rate": 1.0,
                "avg_generation_time": 0.0,
                "total_requests": 0
            }
        
        # Update cost tracking
        self.cost_tracker[provider]["daily_cost"] += cost
        self.cost_tracker[provider]["daily_tokens"] += tokens
        self.cost_tracker[provider]["daily_requests"] += 1
        
        # Update performance metrics
        metrics = self.performance_metrics[provider]
        metrics["total_requests"] += 1
        
        # Update average generation time
        current_avg = metrics["avg_generation_time"]
        total_requests = metrics["total_requests"]
        metrics["avg_generation_time"] = (
            (current_avg * (total_requests - 1) + generation_time) / total_requests
        )
    
    async def _get_cost_optimization_info(self, provider: str) -> Dict[str, Any]:
        """Get cost optimization information for the selected provider"""
        
        provider_costs = {
            "gemini": {"cost_per_1k_tokens": 0.001, "tier": "cost_effective"},
            "openai": {"cost_per_1k_tokens": 0.03, "tier": "premium"},
            "anthropic": {"cost_per_1k_tokens": 0.08, "tier": "premium"},
            "mistral": {"cost_per_1k_tokens": 0.002, "tier": "balanced"}
        }
        
        optimization_info = provider_costs.get(provider, {})
        optimization_info["daily_usage"] = self.cost_tracker.get(provider, {})
        
        return optimization_info
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers"""
        
        health_status = {
            "overall_status": "healthy",
            "providers": {},
            "timestamp": time.time()
        }
        
        for provider_name, provider in self.providers.items():
            try:
                # Simple health check - ping the provider
                await provider.health_check()
                health_status["providers"][provider_name] = {
                    "status": "healthy",
                    "last_check": time.time()
                }
            except Exception as e:
                health_status["providers"][provider_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": time.time()
                }
                health_status["overall_status"] = "degraded"
        
        return health_status
    
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        
        total_cost = sum(
            tracker.get("daily_cost", 0) 
            for tracker in self.cost_tracker.values()
        )
        
        total_requests = sum(
            tracker.get("daily_requests", 0) 
            for tracker in self.cost_tracker.values()
        )
        
        statistics = {
            "total_daily_cost": total_cost,
            "total_daily_requests": total_requests,
            "cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "provider_breakdown": self.cost_tracker,
            "performance_metrics": self.performance_metrics,
            "budget_utilization": total_cost / settings.MAX_DAILY_COST_USD if settings.MAX_DAILY_COST_USD > 0 else 0
        }
        
        return statistics
