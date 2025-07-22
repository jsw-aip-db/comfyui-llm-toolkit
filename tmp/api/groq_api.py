import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

# Re-use the OpenAI utilities for message preparation so we stay DRY
try:
    from api.openai_api import prepare_openai_messages
except ImportError:  # Fallback stub so import errors do not crash plugin load
    def prepare_openai_messages(*args, **kwargs):
        raise RuntimeError("prepare_openai_messages missing – openai_api not available")


async def send_groq_request(
    *,
    base64_images: Optional[List[str]] = None,
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    reasoning_format: Optional[str] = None,
    tools: Optional[Any] = None,
    tool_choice: Optional[Any] = None,
    **kwargs,
) -> Union[str, Dict[str, Any]]:
    """Send a chat completion request to Groq's OpenAI-compatible endpoint.

    Parameters largely mirror the OpenAI helper.  `reasoning_format` is a Groq-
    specific extra that controls how <think> content is returned for supported
    reasoning models ("raw", "hidden", "parsed").
    """
    api_url = "https://api.groq.com/openai/v1/chat/completions"

    # Validate essentials
    if not api_key:
        logger.error("Groq API key missing – cannot send request")
        return {"choices": [{"message": {"content": "Error: Missing GROQ_API_KEY"}}]}
    if not model:
        logger.error("Groq request missing model param")
        return {"choices": [{"message": {"content": "Error: Missing model"}}]}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # --- Vision model checks ---
    # Only certain Groq models support vision. If images are provided for a
    # non-vision model, we must strip them to avoid an API error.
    is_vision_model = "scout" in model or "maverick" in model
    images_to_send = base64_images

    if images_to_send and not is_vision_model:
        logger.warning("Model '%s' may not support images. Sending request without them.", model)
        images_to_send = None
    elif images_to_send and len(images_to_send) > 5:
        logger.warning("Groq supports a max of 5 images, but %s were provided. Taking the first 5.", len(images_to_send))
        images_to_send = images_to_send[:5]

    groq_messages = prepare_openai_messages(
        images_to_send,
        system_message,
        user_message,
        messages,
    )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": groq_messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens, # Correct key for Groq API
        "top_p": top_p,
    }
    # Optional params
    if seed is not None:
        payload["seed"] = seed
    if reasoning_format is not None:
        payload["reasoning_format"] = reasoning_format
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    # Debug – mask key
    logger.debug(
        "Sending Groq request: %s", {k: (v if k != "messages" else f"[{len(v)} messages]") for k, v in payload.items()}
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logger.error("Groq API error %s: %s", resp.status, err_text[:200])
                    return {"choices": [{"message": {"content": f"Groq API error {resp.status}: {err_text}"}}]}
                data = await resp.json()
                return data
    except Exception as exc:
        logger.error("Exception when calling Groq API: %s", exc, exc_info=True)
        return {"choices": [{"message": {"content": str(exc)}}]} 