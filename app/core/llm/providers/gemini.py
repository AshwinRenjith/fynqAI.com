"""
Gemini Provider Implementation
Google's Gemini Pro/Ultra models integration
"""

import logging
from typing import Dict, Any, Optional
import time

from app.config import get_settings
from app.exceptions import LLMProviderError


logger = logging.getLogger(__name__)
settings = get_settings()


class GeminiProvider:
    """Google Gemini LLM provider with cost optimization"""
    
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL_NAME
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.cost_per_1k_tokens = 0.001  # Competitive pricing
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate response using Gemini API
        
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
            
            # Prepare request payload
            payload = await self._prepare_payload(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                context=context or {}
            )
            
            # Make API request (using requests for now, can be upgraded to aiohttp)
            import requests
            
            headers = {
                "Content-Type": "application/json"
            }
            
            url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
            
            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = f"Gemini API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise LLMProviderError("gemini", error_msg)
            
            result = response.json()
            
            # Extract response
            generated_text = self._extract_response(result)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(generated_text.split()) * 1.3
            total_tokens = input_tokens + output_tokens
            
            cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            
            return {
                "response": generated_text,
                "provider": "gemini",
                "model": self.model_name,
                "tokens_used": int(total_tokens),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "cost": cost,
                "generation_time": generation_time,
                "raw_response": result
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}", exc_info=True)
            raise LLMProviderError("gemini", f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}", exc_info=True)
            raise LLMProviderError("gemini", f"Generation failed: {str(e)}")
    
    async def _prepare_payload(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: Optional[float],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare API request payload"""
        
        # Build the contents array
        contents = [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Add context if provided
        system_prompt = context.get("system_prompt", "")
        if system_prompt:
            contents.insert(0, {
                "parts": [
                    {
                        "text": f"System: {system_prompt}"
                    }
                ]
            })
        
        # Prepare generation config
        generation_config = {}
        
        if max_tokens:
            generation_config["maxOutputTokens"] = min(max_tokens, 8192)  # Gemini limit
        
        if temperature is not None:
            generation_config["temperature"] = max(0.0, min(1.0, temperature))
        
        # Set safety settings for educational content
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        payload = {
            "contents": contents,
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }
        
        return payload
    
    def _extract_response(self, result: Dict[str, Any]) -> str:
        """Extract generated text from API response"""
        
        try:
            candidates = result.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")
            
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise ValueError("No parts in response content")
            
            generated_text = parts[0].get("text", "")
            
            if not generated_text:
                # Check for safety filters
                finish_reason = candidate.get("finishReason", "")
                if finish_reason == "SAFETY":
                    raise ValueError("Response blocked by safety filters")
                else:
                    raise ValueError("Empty response generated")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to extract Gemini response: {e}")
            logger.error(f"Raw response: {result}")
            raise ValueError(f"Invalid response format: {str(e)}")
    
    async def health_check(self) -> bool:
        """Perform health check on Gemini API"""
        
        try:
            test_response = await self.generate(
                prompt="Hello, this is a health check. Please respond with 'OK'.",
                max_tokens=10,
                temperature=0.0
            )
            
            return "ok" in test_response.get("response", "").lower()
            
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        
        return {
            "provider": "gemini",
            "model_name": self.model_name,
            "max_tokens": 8192,
            "supports_functions": True,
            "supports_vision": self.model_name in ["gemini-pro-vision", "gemini-ultra-vision"],
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "strengths": [
                "Cost-effective",
                "Fast inference",
                "Good reasoning capabilities",
                "Mathematics proficiency",
                "Multilingual support"
            ],
            "best_for": [
                "Educational content",
                "Mathematical problems",
                "Cost-sensitive applications",
                "Real-time interactions"
            ]
        }
