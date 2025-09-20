"""
Anthropic Provider Implementation
Claude models integration with advanced reasoning
"""

import logging
from typing import Dict, Any, Optional, List
import time

from app.config import get_settings
from app.exceptions import LLMProviderError


logger = logging.getLogger(__name__)
settings = get_settings()


class AnthropicProvider:
    """Anthropic Claude provider for advanced reasoning tasks"""
    
    def __init__(self):
        self.api_key = settings.ANTHROPIC_API_KEY
        self.model_name = settings.ANTHROPIC_MODEL_NAME
        self.base_url = "https://api.anthropic.com/v1"
        
        # Cost mapping for Claude models
        self.cost_mapping = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-2.1": {"input": 0.008, "output": 0.024},
            "claude-2.0": {"input": 0.008, "output": 0.024}
        }
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate response using Anthropic API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens for response
            temperature: Generation temperature (0.0-1.0)
            context: Additional context for generation
        
        Returns:
            Generated response with metadata
        """
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = await self._prepare_messages(prompt, context or {})
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens or 2048,
                "temperature": temperature if temperature is not None else 0.7,
                "top_p": 1.0,
                "stream": False
            }
            
            # Add system message if provided
            system_prompt = (context or {}).get("system_prompt", "")
            if system_prompt:
                payload["system"] = system_prompt
            
            # Make API request
            import requests
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            response = requests.post(
                url=f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"Anthropic API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise LLMProviderError("anthropic", error_msg)
            
            result = response.json()
            
            # Extract response
            generated_text = self._extract_response(result)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            return {
                "response": generated_text,
                "provider": "anthropic",
                "model": self.model_name,
                "tokens_used": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "generation_time": generation_time,
                "raw_response": result
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic API request failed: {e}", exc_info=True)
            raise LLMProviderError("anthropic", f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}", exc_info=True)
            raise LLMProviderError("anthropic", f"Generation failed: {str(e)}")
    
    async def _prepare_messages(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> list[Dict[str, str]]:
        """Prepare messages for Anthropic message format"""
        
        messages = []
        
        # Add conversation history if provided
        conversation_history = context.get("conversation_history", [])
        for message in conversation_history:
            role = message.get("role", "user")
            # Convert system messages to user messages with prefix
            if role == "system":
                content = f"System: {message.get('content', '')}"
                role = "user"
            else:
                content = message.get("content", "")
            
            messages.append({
                "role": role,
                "content": content
            })
        
        # Add current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def _extract_response(self, result: Dict[str, Any]) -> str:
        """Extract generated text from API response"""
        
        try:
            content = result.get("content", [])
            if not content:
                raise ValueError("No content in response")
            
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            
            generated_text = "".join(text_parts)
            
            if not generated_text:
                # Check stop reason
                stop_reason = result.get("stop_reason", "")
                if stop_reason == "max_tokens":
                    logger.warning("Response truncated due to max tokens limit")
                raise ValueError("Empty response generated")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to extract Anthropic response: {e}")
            logger.error(f"Raw response: {result}")
            raise ValueError(f"Invalid response format: {str(e)}")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        
        costs = self.cost_mapping.get(self.model_name, {"input": 0.008, "output": 0.024})
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    async def generate_with_reasoning(
        self,
        prompt: str,
        reasoning_type: str = "analytical",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate response with specific reasoning approach"""
        
        reasoning_prompts = {
            "analytical": "Think through this step by step, analyzing each component carefully.",
            "creative": "Approach this creatively, considering multiple perspectives and innovative solutions.",
            "logical": "Use logical reasoning and formal analysis to work through this problem.",
            "scientific": "Apply the scientific method and evidence-based reasoning to this question.",
            "mathematical": "Use mathematical rigor and precise logical steps to solve this problem."
        }
        
        enhanced_context = context.copy() if context else {}
        
        # Add reasoning instruction to system prompt
        system_prompt = enhanced_context.get("system_prompt", "")
        reasoning_instruction = reasoning_prompts.get(reasoning_type, reasoning_prompts["analytical"])
        
        if system_prompt:
            enhanced_context["system_prompt"] = f"{system_prompt}\n\n{reasoning_instruction}"
        else:
            enhanced_context["system_prompt"] = reasoning_instruction
        
        response = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            context=enhanced_context
        )
        
        response["reasoning_type"] = reasoning_type
        return response
    
    async def health_check(self) -> bool:
        """Perform health check on Anthropic API"""
        
        try:
            test_response = await self.generate(
                prompt="Hello, this is a health check. Please respond with 'OK'.",
                max_tokens=10,
                temperature=0.0
            )
            
            return "ok" in test_response.get("response", "").lower()
            
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        
        model_specs = {
            "claude-3-opus-20240229": {
                "max_tokens": 4096,
                "context_window": 200000,
                "tier": "premium",
                "strengths": ["Complex reasoning", "Creative tasks", "Analysis"]
            },
            "claude-3-sonnet-20240229": {
                "max_tokens": 4096,
                "context_window": 200000,
                "tier": "standard",
                "strengths": ["Balanced performance", "Reasoning", "Writing"]
            },
            "claude-3-haiku-20240307": {
                "max_tokens": 4096,
                "context_window": 200000,
                "tier": "basic",
                "strengths": ["Speed", "Cost efficiency", "Simple tasks"]
            },
            "claude-2.1": {
                "max_tokens": 4096,
                "context_window": 200000,
                "tier": "standard",
                "strengths": ["Long context", "Analysis", "Writing"]
            },
            "claude-2.0": {
                "max_tokens": 4096,
                "context_window": 100000,
                "tier": "standard",
                "strengths": ["Reasoning", "Analysis", "Conversation"]
            }
        }
        
        specs = model_specs.get(self.model_name, model_specs["claude-3-sonnet-20240229"])
        costs = self.cost_mapping.get(self.model_name, {"input": 0.003, "output": 0.015})
        
        return {
            "provider": "anthropic",
            "model_name": self.model_name,
            "max_tokens": specs["max_tokens"],
            "context_window": specs["context_window"],
            "supports_functions": False,  # Claude doesn't support function calling yet
            "tier": specs["tier"],
            "cost_per_1k_input_tokens": costs["input"],
            "cost_per_1k_output_tokens": costs["output"],
            "strengths": specs["strengths"] + [
                "Advanced reasoning",
                "Long context understanding",
                "Safety and alignment",
                "Nuanced responses"
            ],
            "best_for": [
                "Complex reasoning tasks",
                "Long document analysis",
                "Creative writing",
                "Ethical considerations",
                "Detailed explanations"
            ]
        }
    
    def get_reasoning_types(self) -> List[str]:
        """Get available reasoning types for enhanced generation"""
        
        return [
            "analytical",
            "creative", 
            "logical",
            "scientific",
            "mathematical"
        ]
