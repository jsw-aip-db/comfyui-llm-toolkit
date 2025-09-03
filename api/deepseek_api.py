# api/deepseek_api.py

import logging
from typing import List, Dict, Any, Optional
from .openai_api import send_openai_request # Re-use the openai compatible request sender

logger = logging.getLogger(__name__)

async def send_deepseek_request(
    api_url: Optional[str],
    base64_images: Optional[List[str]],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str],
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    tools: Optional[Any] = None,
    tool_choice: Optional[Any] = None,
    **kwargs, # absorb extra arguments
) -> Dict[str, Any]:
    """
    Sends a request to the DeepSeek API using the OpenAI-compatible endpoint.
    """
    deepseek_api_url = api_url or "https://api.deepseek.com/v1/chat/completions"
    
    logger.info(f"Sending request to DeepSeek API: model={model}")

    # DeepSeek API is OpenAI-compatible, so we can reuse the same request function.
    return await send_openai_request(
        api_url=deepseek_api_url,
        base64_images=base64_images,
        model=model,
        system_message=system_message,
        user_message=user_message,
        messages=messages,
        api_key=api_key,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        tools=tools,
        tool_choice=tool_choice,
    )
