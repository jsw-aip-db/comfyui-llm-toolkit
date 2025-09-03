# api/anthropic_api.py
import aiohttp
import json
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

async def send_anthropic_request(
    api_url: Optional[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: Optional[str],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    top_k: int = 40,
    stop_sequences: Optional[List[str]] = None,
    **kwargs, # absorb extra arguments
) -> Dict[str, Any]:
    if not api_key:
        return {"choices": [{"message": {"content": "Error: API key is required for Anthropic requests"}}]}

    endpoint = api_url or "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    # Construct messages list for Anthropic
    anthropic_messages = []
    if messages:
        for msg in messages:
            # Skip system messages if we are passing it as a top-level parameter
            if msg.get("role") != "system":
                anthropic_messages.append(msg)

    # Add current user message
    if user_message:
        anthropic_messages.append({"role": "user", "content": user_message})

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    if system_message:
        body["system"] = system_message
    if stop_sequences:
        body["stop_sequences"] = stop_sequences

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=body) as response:
                if response.status != 200:
                    err_txt = await response.text()
                    logger.error(f"Anthropic API error: {response.status} - {err_txt}")
                    return {"choices": [{"message": {"content": f"Anthropic API error: {response.status}. {err_txt}"}}]}
                
                data = await response.json()
                
                # Convert response to OpenAI-compatible format
                text_content = ""
                if "content" in data and isinstance(data["content"], list):
                    for block in data["content"]:
                        if block.get("type") == "text":
                            text_content += block.get("text", "")
                
                return {"choices": [{"message": {"content": text_content}}]}

    except Exception as e:
        logger.error(f"Exception during Anthropic API call: {e}", exc_info=True)
        return {"choices": [{"message": {"content": f"Exception during Anthropic API call: {str(e)}"}}]}
