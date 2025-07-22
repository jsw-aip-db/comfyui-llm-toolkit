import aiohttp
import json
import logging
import base64
from typing import Any, Dict, List, Optional, Union

# Re-use the same message preparation helper from *openai_api* so that the
# logic for turning ComfyUI inputs into the OpenAI/Gemini JSON schema stays
# centralised in one place.  If the import fails we fall back to a local
# minimal implementation.
try:
    from .openai_api import prepare_openai_messages  # type: ignore
except (ImportError, ModuleNotFoundError):

    def prepare_openai_messages(
        base64_images: List[str],
        system_message: str,
        user_message: str,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Minimal local fallback used only if *openai_api* is unavailable."""

        openai_messages: List[Dict[str, Any]] = []
        if system_message:
            openai_messages.append({"role": "system", "content": system_message})
        for m in messages:
            openai_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        if base64_images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
            for img_b64 in base64_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                )
            openai_messages.append({"role": "user", "content": content})
        else:
            openai_messages.append({"role": "user", "content": user_message})
        return openai_messages


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Chat / Completions (text)
# -----------------------------------------------------------------------------


async def send_gemini_request(
    *,
    api_url: Optional[str],
    base64_images: List[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    repeat_penalty: float = 1.0,
    tools: Optional[Any] = None,
    tool_choice: Optional[Any] = None,
) -> Dict[str, Any]:
    """Send a chat/completions request to the Gemini API (OpenAI-compatible).

    The function mirrors *openai_api.send_openai_request* so other modules can
    switch between providers by simply changing *llm_provider*.
    """

    if not api_key:
        err = "API key is required for Gemini requests"
        logger.error(err)
        return {"choices": [{"message": {"content": f"Error: {err}"}}]}

    if not model:
        err = "Model parameter is required for Gemini requests"
        logger.error(err)
        return {"choices": [{"message": {"content": f"Error: {err}"}}]}

    # Default URL (OpenAI compatibility layer)
    if not api_url:
        api_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    openai_messages = prepare_openai_messages(
        base64_images, system_message, user_message, messages or []
    )

    data: Dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    if seed is not None:
        data["seed"] = seed
    if tools:
        data["tools"] = tools
    if tool_choice:
        data["tool_choice"] = tool_choice

    logger.info(f"Gemini request – model: {model}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=data) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("Gemini API error %s: %s", resp.status, text)
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": f"Gemini API error {resp.status}: {text}"}}
                        ]
                    }
                result = await resp.json()

                if tools:
                    return result  # Let caller handle tool-based response directly

                # Normalise to same minimal shape used elsewhere
                choices = result.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    txt = msg.get("content", "")
                    return {"choices": [{"message": {"content": txt}}]}
                logger.warning("Gemini response without choices: %s", result)
                return {"choices": [{"message": {"content": ""}}]}
    except aiohttp.ClientError as exc:
        logger.error("Gemini HTTP error: %s", exc)
        return {"choices": [{"message": {"content": str(exc)}}]}
    except Exception as exc:
        logger.error("Gemini request exception: %s", exc, exc_info=True)
        return {"choices": [{"message": {"content": str(exc)}}]}


# -----------------------------------------------------------------------------
# Image generation (Imagen 3, etc.)
# -----------------------------------------------------------------------------


async def send_gemini_image_generation_request(
    *,
    api_key: str,
    model: str,
    prompt: str,
    n: int = 1,
    size: str = "1024x1024",
    response_format: str = "b64_json",
    user: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an image with the Gemini *imagen-3* family via OpenAI style API.

    The endpoint mimics the OpenAI *images/generations* path which Google
    exposes under its `/openai/` compatibility layer.
    """

    if not api_key:
        raise ValueError("API key is required for Gemini image generation.")

    api_url = "https://generativelanguage.googleapis.com/v1beta/openai/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": response_format,
    }
    if user:
        payload["user"] = user

    logger.info(f"Gemini image generation – model: {model}, prompt preview: {prompt[:60]}")

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as resp:
            txt = await resp.text()
            if resp.status != 200:
                logger.error("Gemini image API error %s: %s", resp.status, txt)
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=txt,
                    headers=resp.headers,
                )
            logger.debug("Gemini image API raw response: %s", txt[:500])
            return json.loads(txt)


# -----------------------------------------------------------------------------
# Embeddings helper (simple wrapper around /embeddings path)
# -----------------------------------------------------------------------------


async def create_gemini_compatible_embedding(
    api_base: str,
    model: str,
    input: Union[str, List[str]],
    api_key: Optional[str] = None,
) -> List[float]:
    """Create text embeddings using the Gemini OpenAI-compat endpoint."""

    api_base = (api_base or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    if not api_base.endswith("/embeddings"):
        url = f"{api_base}/embeddings"
    else:
        url = api_base

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "input": input, "encoding_format": "float"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"Embedding error {resp.status}: {data}")
            if data.get("data") and data["data"][0].get("embedding"):
                return data["data"][0]["embedding"]
            raise ValueError("Unexpected embedding response format") 