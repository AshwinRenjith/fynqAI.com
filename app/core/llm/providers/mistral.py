"""
Mistral Provider Implementation
Mistral AI models integration with competitive performance
"""

import logging
from typing import Dict, Any, Optional
import time

from app.config import get_settings
from app.exceptions import LLMProviderError


logger = logging.getLogger(__name__)
settings = get_settings()


class MistralProvider:
    """Mistral AI provider for efficient and cost-effective generation"""
    
    def __init__(self):
        self.api_key = settings.MISTRAL_API_KEY
        self.model_name = settings.MISTRAL_MODEL_NAME
        self.base_url = "https://api.mistral.ai/v1"
        
        # Cost mapping for Mistral models
        self.cost_mapping = {
            "mistral-large-latest": {"input": 0.008, "output": 0.024},
            "mistral-medium-latest": {"input": 0.0027, "output": 0.0081},
            "mistral-small-latest": {"input": 0.002, "output": 0.006},
            "open-mistral-7b": {"input": 0.00025, "output": 0.00025},
            "open-mixtral-8x7b": {"input": 0.0007, "output": 0.0007},
            "open-mixtral-8x22b": {"input": 0.002, "output": 0.006}
        }
        
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate response using Mistral API
        
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
                "stream": False,
                "safe_prompt": True  # Enable safety filtering
            }
            
            # Make API request
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"Mistral API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise LLMProviderError("mistral", error_msg)
            
            result = response.json()
            
            # Extract response
            generated_text = self._extract_response(result)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            return {
                "response": generated_text,
                "provider": "mistral",
                "model": self.model_name,
                "tokens_used": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "generation_time": generation_time,
                "raw_response": result
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Mistral API request failed: {e}", exc_info=True)
            raise LLMProviderError("mistral", f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Mistral generation failed: {e}", exc_info=True)
            raise LLMProviderError("mistral", f"Generation failed: {str(e)}")
    
    async def _prepare_messages(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> list[Dict[str, str]]:
        """Prepare messages for Mistral chat format"""
        
        messages = []
        
        # Add system message if provided
        system_prompt = context.get("system_prompt", "")
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history if provided
        conversation_history = context.get("conversation_history", [])
        for message in conversation_history:
            messages.append({
                "role": message.get("role", "user"),
                "content": message.get("content", "")
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
            choices = result.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            if not content:
                # Check finish reason
                finish_reason = choice.get("finish_reason", "")
                if finish_reason == "length":
                    logger.warning("Response truncated due to length limit")
                elif finish_reason == "content_filter":
                    raise ValueError("Response blocked by content filter")
                raise ValueError("Empty response generated")
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to extract Mistral response: {e}")
            logger.error(f"Raw response: {result}")
            raise ValueError(f"Invalid response format: {str(e)}")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        
        costs = self.cost_mapping.get(self.model_name, {"input": 0.002, "output": 0.006})
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    async def generate_with_specialization(
        self,
        prompt: str,
        specialization: str = "general",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate response with domain specialization"""
        
        specialization_prompts = {
            "mathematics": "You are a mathematics expert. Solve problems step-by-step with clear explanations.",
            "science": "You are a science expert. Provide accurate, evidence-based explanations.",
            "programming": "You are a programming expert. Write clean, efficient code with explanations.",
            "writing": "You are a writing expert. Focus on clarity, style, and effective communication.",
            "analysis": "You are an analytical expert. Provide systematic, logical analysis.",
            "general": "You are a helpful assistant. Provide clear and accurate responses."
        }
        
        enhanced_context = context.copy() if context else {}
        
        # Add specialization to system prompt
        system_prompt = enhanced_context.get("system_prompt", "")
        specialization_instruction = specialization_prompts.get(specialization, specialization_prompts["general"])
        
        if system_prompt:
            enhanced_context["system_prompt"] = f"{system_prompt}\n\n{specialization_instruction}"
        else:
            enhanced_context["system_prompt"] = specialization_instruction
        
        response = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            context=enhanced_context
        )
        
        response["specialization"] = specialization
        return response
    
    async def health_check(self) -> bool:
        """Perform health check on Mistral API"""
        
        try:
            test_response = await self.generate(
                prompt="Hello, this is a health check. Please respond with 'OK'.",
                max_tokens=10,
                temperature=0.0
            )
            
            return "ok" in test_response.get("response", "").lower()
            
        except Exception as e:
            logger.error(f"Mistral health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        
        model_specs = {
            "mistral-large-latest": {
                "max_tokens": 4096,
                "context_window": 32768,
                "tier": "premium",
                "strengths": ["Complex reasoning", "Multi-language", "Code generation"]
            },
            "mistral-medium-latest": {
                "max_tokens": 4096,
                "context_window": 32768,
                "tier": "standard",
                "strengths": ["Balanced performance", "Efficiency", "Reasoning"]
            },
            "mistral-small-latest": {
                "max_tokens": 4096,
                "context_window": 32768,
                "tier": "basic",
                "strengths": ["Speed", "Cost efficiency", "Simple tasks"]
            },
            "open-mistral-7b": {
                "max_tokens": 4096,
                "context_window": 32768,
                "tier": "basic",
                "strengths": ["Open source", "Cost effective", "Fast inference"]
            },
            "open-mixtral-8x7b": {
                "max_tokens": 4096,
                "context_window": 32768,
                "tier": "standard",
                "strengths": ["MoE architecture", "Efficiency", "Performance"]
            },
            "open-mixtral-8x22b": {
                "max_tokens": 4096,
                "context_window": 65536,
                "tier": "premium",
                "strengths": ["Large context", "Advanced reasoning", "Multi-task"]
            }
        }
        
        specs = model_specs.get(self.model_name, model_specs["mistral-small-latest"])
        costs = self.cost_mapping.get(self.model_name, {"input": 0.002, "output": 0.006})
        
        return {
            "provider": "mistral",
            "model_name": self.model_name,
            "max_tokens": specs["max_tokens"],
            "context_window": specs["context_window"],
            "supports_functions": False,  # Mistral doesn't support function calling yet
            "tier": specs["tier"],
            "cost_per_1k_input_tokens": costs["input"],
            "cost_per_1k_output_tokens": costs["output"],
            "strengths": specs["strengths"] + [
                "Multilingual support",
                "Efficient inference",
                "European AI alternative",
                "Competitive pricing"
            ],
            "best_for": [
                "Cost-sensitive applications",
                "Multilingual tasks",
                "European data compliance",
                "Balanced performance needs",
                "Code generation"
            ]
        }
    
    def get_specializations(self) -> list[str]:
        """Get available specializations for enhanced generation"""
        
        return [
            "mathematics",
            "science",
            "programming",
            "writing",
            "analysis",
            "general"
        ]
