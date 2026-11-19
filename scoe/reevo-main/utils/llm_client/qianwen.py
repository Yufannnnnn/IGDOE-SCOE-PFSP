import logging
from typing import Optional
from .base import BaseClient

try:
    import dashscope
except ImportError:
    dashscope = 'dashscope'

logger = logging.getLogger(__name__)

class QianwenClient(BaseClient):
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature)
        
        if isinstance(dashscope, str):
            logger.fatal(f"Package `{dashscope}` is required")
            exit(-1)
        
        self.api_key = api_key
        dashscope.api_key = api_key
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        responses = []
        for _ in range(n):
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                result_format='message',  # 返回格式为message
                stream=False,
            )
            
            if response.status_code == 200:
                # 创建一个与OpenAI格式兼容的Choice对象
                class Choice:
                    def __init__(self, message_content):
                        self.message = type('Message', (), {'content': message_content})
                
                choice = Choice(response.output.choices[0]['message']['content'])
                responses.append(choice)
            else:
                logger.error(f"通义千问API调用失败: {response.code}, {response.message}")
                raise Exception(f"通义千问API调用失败: {response.code}, {response.message}")
                
        return responses 