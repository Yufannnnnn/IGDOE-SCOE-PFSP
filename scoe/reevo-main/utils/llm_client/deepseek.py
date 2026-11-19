import logging
from typing import Optional
from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = 'openai'

logger = logging.getLogger(__name__)

class DeepSeekClient(BaseClient):
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature)
        
        if isinstance(OpenAI, str):
            logger.fatal(f"Package `{OpenAI}` is required")
            exit(-1)
        
        # DeepSeek API endpoint
        self.base_url = base_url 
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            stream=False,
        )
        return response.choices

