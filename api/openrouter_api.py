# api/openrouter_api.py
import aiohttp
import logging
import json
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

async def send_openrouter_image_generation_request(
    api_key: str,
    model: str,
    prompt: str,
    n: int = 1,
    timeout: int = 120,
    input_image_base64: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Sends a request to OpenRouter's chat completions endpoint for image generation,
    optionally including input images for context.
    """
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build the message content
    content_parts = []
    if prompt:
        content_parts.append({"type": "text", "text": prompt})
    
    if input_image_base64:
        for b64_image in input_image_base64:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_image}"
                }
            })

    messages = [{"role": "user", "content": content_parts}]
    
    payload = {
        "model": model,
        "messages": messages,
        "modalities": ["image", "text"],
    }

    # Add 'n' if it's greater than 1, as some models may support it.
    if n > 1:
        payload['n'] = n
        
    # Add only SEED from kwargs as it's a documented advanced parameter.
    # 'size' is not a standard for chat completions and likely cause 400 errors.
    if 'seed' in kwargs and kwargs.get('seed') != -1 and kwargs.get('seed') is not None:
        payload['seed'] = kwargs['seed']

    # Log a sanitized version of the payload for debugging
    log_payload = payload.copy()
    if log_payload.get("messages") and log_payload["messages"][0].get("content"):
        import copy
        log_payload = copy.deepcopy(payload)
        for part in log_payload["messages"][0]["content"]:
            if part.get("type") == "image_url":
                part["image_url"]["url"] = "[base64 data omitted]"
    logger.info(f"Sending image generation request to OpenRouter with payload: {json.dumps(log_payload, indent=2)}")

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                response_json = await response.json()
                
                if response_json.get("choices"):
                    image_data = []
                    for choice in response_json["choices"]:
                        message = choice.get("message", {})
                        if "images" in message and message["images"]:
                            for img in message["images"]:
                                base64_string = img.get("image_url", {}).get("url", "")
                                if "base64," in base64_string:
                                    base64_string = base64_string.split("base64,")[1]
                                image_data.append({"b64_json": base64_string})
                    
                    if image_data:
                        return {"data": image_data}

                logger.error(f"OpenRouter API response did not contain expected image data. Response: {response_json}")
                return {"error": "No image data in response", "data": []}

    except aiohttp.ClientResponseError as e:
        logger.error(f"HTTP error from OpenRouter API: {e.status} {e.message}")
        error_body = "Could not read error body."
        try:
            error_json = await e.json()
            error_body = str(error_json)
            logger.error(f"Error Body (JSON): {error_body}")
        except Exception:
            try:
                error_body = await e.text()
                logger.error(f"Error Body (text): {error_body}")
            except Exception:
                pass
        return {"error": f"HTTP error: {e.status} {e.message}. Body: {error_body}", "data": []}
    except Exception as e:
        logger.error(f"Error during OpenRouter API call: {e}", exc_info=True)
        return {"error": str(e), "data": []}

async def send_openrouter_request(
    api_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    **kwargs,
) -> Dict[str, Any]:
    """Generic OpenRouter chat completions handler."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, **kwargs}
    payload = {k: v for k, v in payload.items() if v is not None}
    
    # Log sanitized payload
    log_payload = payload.copy()
    logger.info(f"Sending request to OpenRouter with payload: {json.dumps(log_payload, indent=2)}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientResponseError as e:
        logger.error(f"HTTP error from OpenRouter API: {e.status} {e.message}")
        error_body = await response.text()
        logger.error(f"Error body: {error_body}")
        return {"error": f"HTTP error: {e.status} {e.message}. Body: {error_body}"}
    except Exception as e:
        logger.error(f"Error during OpenRouter API call: {e}", exc_info=True)
        return {"error": str(e)}
