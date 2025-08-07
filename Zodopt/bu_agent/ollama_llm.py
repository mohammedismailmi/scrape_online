# ollama_llm.py
import httpx
import logging
from pydantic import BaseModel
from typing import Optional

# Setup logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Token usage schema
class ChatInvokeUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# LLM Response schema
class LLMResponse(BaseModel):
    content: str
    usage: ChatInvokeUsage

# Ollama-compatible LLM wrapper
class OllamaLLM:
    provider = "ollama"
    model = "deepseek-r1:7b"
    model_name = "deepseek-r1:7b"  # Required by browser_use

    async def ainvoke(self, prompt, images=None, **kwargs) -> LLMResponse:
        if not isinstance(prompt, str):
            prompt = getattr(prompt, 'content', str(prompt))

        logger.debug(f"Prompt sent to LLM: {prompt}")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")
                logger.debug(f"LLM response: {result}")
                return {
                    "content": result,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }

        except httpx.RequestError as e:
            logger.error(f"Request to Ollama failed: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama: {e}")
            raise
