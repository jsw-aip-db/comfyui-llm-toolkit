import aiohttp
import asyncio
import base64
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


async def send_bfl_image_generation_request(
    api_key: str,
    prompt: str,
    aspect_ratio: str = "1:1",
    # Editing support
    input_image_base64: Optional[str] = None,
    # Internal tuning
    seed: Optional[int] = None,
    prompt_upsampling: bool = False,
    safety_tolerance: int = 2,
    output_format: Optional[str] = None,
    polling_interval: float = 0.5,
    max_wait_seconds: int = 120,
) -> Dict[str, Any]:
    """Submit a text-to-image or image-edit request to the BFL FLUX Kontext MAX endpoint.

    The helper wraps the asynchronous flow required by the BFL API:
    1.  Send a POST request to ``/flux-kontext-max`` with the requested parameters.
    2.  Poll the provided ``polling_url`` until the generation is *Ready* (or fails).
    3.  Download the resulting image, convert it to **Base64** and re-shape the
        response so that downstream utilities (e.g. *process_images_for_comfy*)
        can handle it transparently.  The returned structure mimics the OpenAI
        ``{'data': [{'b64_json': ...}]}`` schema.

    Parameters
    ----------
    api_key : str
        The BFL account API key.  Required.
    prompt : str
        A description of the desired image or the edit operation.
    aspect_ratio : str, optional
        Ratio in the format ``"W:H"`` (e.g. ``"1:1"``, ``"16:9"``).  Defaults
        to ``"1:1"`` which results in a 1024×1024 output.
    input_image_base64 : str or None, optional
        When provided the request is interpreted as an *edit* operation.  The
        value **must** be a Base64-encoded string (without the data-URL prefix).
    seed, prompt_upsampling, safety_tolerance, output_format : various, optional
        Direct mappings of the official BFL parameters.  ``None`` omits the
        field allowing the server-side default (currently *png*).
    polling_interval : float, optional
        Delay between polling attempts in seconds.  Defaults to **0.5 s**.
    max_wait_seconds : int, optional
        Upper bound for the total time spent polling.  An exception is raised
        when the limit is exceeded (default: **120 s**).

    Returns
    -------
    dict
        A dictionary shaped like the OpenAI response (``{"data": [{"b64_json": …}]}``).
    """

    if not api_key:
        raise ValueError("BFL image generation requires a valid API key")

    # ------------------------------------------------------------------
    #  1. Submit generation request
    # ------------------------------------------------------------------
    endpoint = "https://api.bfl.ai/v1/flux-kontext-max"
    headers = {
        "accept": "application/json",
        "x-key": api_key,
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "prompt": prompt,
    }
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    if seed is not None:
        payload["seed"] = seed
    if prompt_upsampling:
        payload["prompt_upsampling"] = prompt_upsampling
    if safety_tolerance is not None:
        payload["safety_tolerance"] = safety_tolerance
    if output_format is not None:
        payload["output_format"] = output_format
    # Image editing
    if input_image_base64:
        payload["input_image"] = input_image_base64

    logger.info(
        "BFL API generation request → %s | aspect_ratio=%s | edit=%s",
        endpoint,
        aspect_ratio,
        bool(input_image_base64),
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(
                    f"BFL API error: status {resp.status} – {text[:200]}…"
                )
            first_response = await resp.json()

    request_id = first_response.get("id")
    polling_url = first_response.get("polling_url")
    if not polling_url or not request_id:
        raise RuntimeError(
            f"Unexpected BFL response: missing id/polling_url → {first_response}"
        )

    # ------------------------------------------------------------------
    #  2. Poll for completion
    # ------------------------------------------------------------------
    poll_headers = {"accept": "application/json", "x-key": api_key}
    elapsed = 0.0
    status = None
    result_json: Optional[Dict[str, Any]] = None

    async with aiohttp.ClientSession() as session:
        while elapsed < max_wait_seconds:
            await asyncio.sleep(polling_interval)
            elapsed += polling_interval

            async with session.get(
                polling_url, headers=poll_headers, params={"id": request_id}
            ) as poll_resp:
                if poll_resp.status != 200:
                    text = await poll_resp.text()
                    raise RuntimeError(
                        f"BFL polling error: status {poll_resp.status} – {text[:200]}…"
                    )
                result_json = await poll_resp.json()
                status = result_json.get("status")

            if status == "Ready":
                break
            if status in {"Error", "Failed"}:
                raise RuntimeError(
                    f"BFL generation failed: {result_json}"  # type: ignore[arg-type]
                )
        else:
            raise TimeoutError(
                f"BFL generation polling timed-out after {max_wait_seconds} seconds"
            )

    assert result_json is not None  # mypy appeasement

    sample_url: Optional[str] = (
        result_json.get("result", {}).get("sample") if isinstance(result_json, dict) else None
    )
    if not sample_url:
        raise RuntimeError(
            f"BFL response missing sample URL → {result_json}"  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    #  3. Download image & convert to Base64
    # ------------------------------------------------------------------
    async with aiohttp.ClientSession() as session:
        async with session.get(sample_url) as img_resp:
            if img_resp.status != 200:
                raise RuntimeError(
                    f"Could not download generated image: HTTP {img_resp.status}"
                )
            img_bytes = await img_resp.read()

    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    # Shape to OpenAI-like structure so downstream code can reuse the same logic
    return {"data": [{"b64_json": img_b64}]} 