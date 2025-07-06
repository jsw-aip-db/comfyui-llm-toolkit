import asyncio
import time
import logging
from typing import Dict, Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = "https://api.sunoapi.org"


async def _request(method: str, url: str, api_key: str, *, json_payload: Optional[Dict[str, Any]] = None,
                   params: Optional[Dict[str, str]] = None, timeout: int = 60) -> Dict[str, Any]:
    """Internal helper for performing authenticated HTTP requests."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.request(method, url, headers=headers, json=json_payload, params=params) as response:
            response.raise_for_status()
            return await response.json()


async def send_suno_music_generation_request(
    *,
    api_key: str,
    prompt: str,
    style: Optional[str] = None,
    title: Optional[str] = None,
    custom_mode: bool = False,
    instrumental: bool = False,
    model: str = "V3_5",
    negative_tags: Optional[str] = None,
    callback_url: str = "https://example.com/callback",
    poll: bool = True,
    poll_interval: float = 5.0,
    max_poll_seconds: int = 300,
) -> Dict[str, Any]:
    """Create music with Suno API and optionally poll until complete.

    Returns the final record-info payload (or the initial response if polling is disabled).
    """
    payload = {
        "prompt": prompt,
        "style": style,
        "title": title,
        "customMode": custom_mode,
        "instrumental": instrumental,
        "model": model,
        "negativeTags": negative_tags,
        "callBackUrl": callback_url,
    }
    # Remove None values that the API may reject
    payload = {k: v for k, v in payload.items() if v is not None}

    logger.info(f"Suno music request payload: {payload}")
    initial_url = f"{BASE_URL}/api/v1/generate"
    init_resp = await _request("POST", initial_url, api_key, json_payload=payload)
    logger.info(f"Suno initial response: {init_resp}")

    # Handle error codes or missing data early
    if init_resp.get("code") != 200 or init_resp.get("data") is None:
        return {"error": "Suno generation request failed", "initial_response": init_resp}

    if not poll:
        return init_resp

    data_block = init_resp.get("data") or {}
    task_id = data_block.get("task_id") or data_block.get("taskId")
    if not task_id:
        logger.warning("No task_id returned by Suno, cannot poll.")
        return init_resp

    info_url = f"{BASE_URL}/api/v1/generate/record-info"
    start_time = time.time()

    while True:
        await asyncio.sleep(poll_interval)
        elapsed = time.time() - start_time
        if elapsed > max_poll_seconds:
            logger.warning("Polling for Suno music generation timed out.")
            break
        try:
            info_resp = await _request("GET", info_url, api_key, params={"taskId": task_id})
        except Exception as e:
            logger.error(f"Error polling Suno record-info: {e}")
            continue

        status_code = info_resp.get("code", 0)
        if status_code == 200:
            status_data = info_resp.get("data", {})
            callback_type = status_data.get("callbackType") or status_data.get("status")
            if callback_type in {"SUCCESS", "complete", "FIRST_SUCCESS", "FIRST", "COMPLETE"}:
                logger.info("Suno music generation completed.")
                return info_resp
        # Continue polling otherwise

    return {"error": "Timeout waiting for Suno music generation", "initial_response": init_resp}


async def send_suno_lyrics_generation_request(
    *,
    api_key: str,
    prompt: str,
    callback_url: str = "https://example.com/callback",
    poll: bool = True,
    poll_interval: float = 5.0,
    max_poll_seconds: int = 180,
) -> Dict[str, Any]:
    """Generate lyrics with Suno API and optionally poll until complete."""
    payload = {
        "prompt": prompt,
        "callBackUrl": callback_url,
    }
    lyrics_url = f"{BASE_URL}/api/v1/lyrics"
    init_resp = await _request("POST", lyrics_url, api_key, json_payload=payload)

    if not poll:
        return init_resp

    task_id = init_resp.get("data", {}).get("task_id") or init_resp.get("data", {}).get("taskId")
    if not task_id:
        logger.warning("No task_id returned by Suno lyrics endpoint, cannot poll.")
        return init_resp

    info_url = f"{BASE_URL}/api/v1/lyrics/record-info"
    start_time = time.time()

    while True:
        await asyncio.sleep(poll_interval)
        if time.time() - start_time > max_poll_seconds:
            logger.warning("Polling for Suno lyrics generation timed out.")
            break
        try:
            info_resp = await _request("GET", info_url, api_key, params={"taskId": task_id})
        except Exception as e:
            logger.error(f"Error polling Suno lyrics record-info: {e}")
            continue

        if info_resp.get("code") == 200 and info_resp.get("data", {}).get("status") == "SUCCESS":
            return info_resp

    return {"error": "Timeout waiting for Suno lyrics generation", "initial_response": init_resp}


async def send_suno_upload_cover_request(
    *,
    api_key: str,
    upload_url: str,
    prompt: str,
    style: Optional[str],
    title: Optional[str],
    custom_mode: bool,
    instrumental: bool,
    model: str,
    negative_tags: Optional[str],
    callback_url: str,
    poll: bool = True,
) -> Dict[str, Any]:
    """Call Suno upload-cover endpoint."""
    payload = {
        "uploadUrl": upload_url,
        "prompt": prompt,
        "style": style,
        "title": title,
        "customMode": custom_mode,
        "instrumental": instrumental,
        "model": model,
        "negativeTags": negative_tags,
        "callBackUrl": callback_url,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    resp = await _request("POST", f"{BASE_URL}/api/v1/generate/upload-cover", api_key, json_payload=payload)
    if not poll:
        return resp
    # Re-use record-info polling
    task_id = resp.get("data", {}).get("task_id")
    if not task_id:
        return resp
    return await send_suno_music_generation_request(
        api_key=api_key,
        prompt=prompt,
        custom_mode=custom_mode,
        instrumental=instrumental,
        model=model,
        poll=True,
        callback_url=callback_url,
    )


async def send_suno_upload_extend_request(
    *,
    api_key: str,
    upload_url: str,
    default_param_flag: bool,
    prompt: Optional[str],
    style: Optional[str],
    title: Optional[str],
    continue_at: Optional[int],
    instrumental: bool,
    model: str,
    negative_tags: Optional[str],
    callback_url: str,
    poll: bool = True,
) -> Dict[str, Any]:
    payload = {
        "uploadUrl": upload_url,
        "defaultParamFlag": default_param_flag,
        "instrumental": instrumental,
        "prompt": prompt,
        "style": style,
        "title": title,
        "continueAt": continue_at,
        "model": model,
        "negativeTags": negative_tags,
        "callBackUrl": callback_url,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    resp = await _request("POST", f"{BASE_URL}/api/v1/generate/upload-extend", api_key, json_payload=payload)
    if not poll:
        return resp
    task_id = resp.get("data", {}).get("task_id")
    if not task_id:
        return resp
    # Poll via record-info
    return await send_suno_music_generation_request(
        api_key=api_key,
        prompt=prompt or "",
        model=model,
        poll=True,
        callback_url=callback_url,
    )


# -----------------------------------------------------------------------------
# Account management helper
# -----------------------------------------------------------------------------


async def get_suno_remaining_credits(*, api_key: str, timeout: int = 30) -> Dict[str, Any]:
    """Return the remaining credit balance for the given Suno account.

    Response format (success): {"code": 200, "msg": "success", "data": <int>} where data is credits.
    On error returns the raw JSON with non-200 code or {"error": ...} if unexpected exception.
    """
    try:
        url = f"{BASE_URL}/api/v1/generate/credit"
        resp = await _request("GET", url, api_key, timeout=timeout)
        return resp
    except Exception as exc:
        logger.error("Error fetching Suno remaining credits: %s", exc, exc_info=True)
        return {"error": str(exc)} 