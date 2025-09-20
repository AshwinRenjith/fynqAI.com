"""
OpenAI Provider Implementation
OpenAI GPT models integration with function calling
"""

import logging
from typing import Dict, Any, Optional, List
import time

from app.config import get_settings
from app.exceptions import LLMProviderError


logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIProvider:
    """OpenAI GPT provider with function calling support"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL_NAME
        self.base_url = "https://api.openai.com/v1"
        
        # Cost mapping for different models
        self.cost_mapping = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
        }
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Dict[str, Any] = None,
        functions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens for response
            temperature: Generation temperature (0.0-2.0)
            context: Additional context for generation
            functions: Function definitions for function calling
        
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
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }
            
            # Add functions if provided
            if functions:
                payload["functions"] = functions
                payload["function_call"] = "auto"
            
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
                error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise LLMProviderError("openai", error_msg)
            
            result = response.json()
            
            # Extract response
            generated_text, function_call = self._extract_response(result)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            response_data = {
                "response": generated_text,
                "provider": "openai",
                "model": self.model_name,
                "tokens_used": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "generation_time": generation_time,
                "raw_response": result
            }
            
            # Add function call if present
            if function_call:
                response_data["function_call"] = function_call
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}", exc_info=True)
            raise LLMProviderError("openai", f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}", exc_info=True)
            raise LLMProviderError("openai", f"Generation failed: {str(e)}")
    
    async def _prepare_messages(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI chat format"""
        
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
    
    def _extract_response(self, result: Dict[str, Any]) -> tuple[str, Optional[Dict[str, Any]]]:
        """Extract generated text and function call from API response"""
        
        try:
            choices = result.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            
            choice = choices[0]
            message = choice.get("message", {})
            
            # Extract text content
            content = message.get("content", "")
            
            # Extract function call if present
            function_call = message.get("function_call")
            
            # Handle cases where response is empty but function call exists
            if not content and function_call:
                content = f"Function call: {function_call.get('name', 'unknown')}"
            
            return content, function_call
            
        except Exception as e:
            logger.error(f"Failed to extract OpenAI response: {e}")
            logger.error(f"Raw response: {result}")
            raise ValueError(f"Invalid response format: {str(e)}")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        
        costs = self.cost_mapping.get(self.model_name, {"input": 0.03, "output": 0.06})
        
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    async def generate_with_functions(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate response with function calling capability"""
        
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
            functions=functions
        )
    
    async def health_check(self) -> bool:
        """Perform health check on OpenAI API"""
        
        try:
            test_response = await self.generate(
                prompt="Hello, this is a health check. Please respond with 'OK'.",
                max_tokens=10,
                temperature=0.0
            )
            
            return "ok" in test_response.get("response", "").lower()
            
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        
        model_specs = {
            "gpt-4": {
                "max_tokens": 8192,
                "context_window": 8192,
                "supports_functions": True,
                "tier": "premium"
            },
            "gpt-4-turbo": {
                "max_tokens": 4096,
                "context_window": 128000,
                "supports_functions": True,
                "tier": "premium"
            },
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "context_window": 4096,
                "supports_functions": True,
                "tier": "standard"
            },
            "gpt-3.5-turbo-16k": {
                "max_tokens": 4096,
                "context_window": 16384,
                "supports_functions": True,
                "tier": "standard"
            }
        }
        
        specs = model_specs.get(self.model_name, model_specs["gpt-3.5-turbo"])
        costs = self.cost_mapping.get(self.model_name, {"input": 0.03, "output": 0.06})
        
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "max_tokens": specs["max_tokens"],
            "context_window": specs["context_window"],
            "supports_functions": specs["supports_functions"],
            "tier": specs["tier"],
            "cost_per_1k_input_tokens": costs["input"],
            "cost_per_1k_output_tokens": costs["output"],
            "strengths": [
                "High quality responses",
                "Function calling support",
                "Extensive training data",
                "Good reasoning capabilities",
                "Code generation"
            ],
            "best_for": [
                "Complex reasoning tasks",
                "Function calling applications",
                "Code generation",
                "Creative writing",
                "Advanced problem solving"
            ]
        }
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get standard function definitions for educational tasks"""
        
        return [
            {
                "name": "solve_math_problem",
                "description": "Solve a mathematical problem step by step",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problem": {
                            "type": "string",
                            "description": "The mathematical problem to solve"
                        },
                        "subject": {
                            "type": "string",
                            "enum": ["algebra", "calculus", "geometry", "trigonometry", "physics", "chemistry"],
                            "description": "The subject area of the problem"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["basic", "intermediate", "advanced"],
                            "description": "The difficulty level of the problem"
                        }
                    },
                    "required": ["problem", "subject"]
                }
            },
            {
                "name": "explain_concept",
                "description": "Explain a concept in simple terms with examples",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "The concept to explain"
                        },
                        "subject": {
                            "type": "string",
                            "description": "The subject area"
                        },
                        "grade_level": {
                            "type": "string",
                            "enum": ["elementary", "middle", "high", "college"],
                            "description": "The appropriate grade level"
                        }
                    },
                    "required": ["concept", "subject"]
                }
            }
        ]
