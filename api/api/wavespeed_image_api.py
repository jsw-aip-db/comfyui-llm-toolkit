# wavespeed_image_api.py
import asyncio
import logging
import httpx
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

async def send_wavespeed_image_edit_request(
    api_key: str,
    model: str,
    prompt: str,
    image_base64: str,
    guidance_scale: Optional[float] = None,
    seed: int = -1,
) -> Dict[str, Any]:
    """
    Sends a request to the WaveSpeedAI Image Edit/Enhancement APIs and polls for the result.
    Handles models like SeedEdit V3 and Portrait.
    """
    if not api_key:
        raise ValueError("WaveSpeed API key is required.")
    if not image_base64:
        raise ValueError("Input image is required for this model.")

    # Always request base64 output for direct use in ComfyUI
    payload = {
        "prompt": prompt,
        "image": f"data:image/png;base64,{image_base64}",
        "enable_base64_output": True,
    }
    if seed != -1:
        payload["seed"] = seed
    
    # Conditionally add guidance_scale for models that support it
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale

    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    request_id = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Submitting image edit task to WaveSpeedAI model: {model}")
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            if result.get("data") and result["data"].get("id"):
                request_id = result["data"]["id"]
                logger.info(f"Task submitted successfully. Request ID: {request_id}")
            else:
                logger.error(f"WaveSpeedAI API Error: Unexpected submission response format. {result}")
                return {"error": "Unexpected submission response format.", "details": result}

        except httpx.HTTPStatusError as e:
            logger.error(f"WaveSpeedAI API Error on submission: {e.response.status_code} - {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            logger.error(f"Error submitting to WaveSpeedAI: {e}", exc_info=True)
            return {"error": str(e)}

        if not request_id:
            return {"error": "Failed to get a request ID from WaveSpeedAI."}

        # --- Polling for result ---
        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}
        
        for attempt in range(60): # Poll for up to 60 seconds (60 * 1s)
            try:
                await asyncio.sleep(1) # 1-second polling interval
                logger.debug(f"Polling for result... (Attempt {attempt + 1})")
                
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    logger.info("WaveSpeedAI task completed.")
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    
                    # Expecting a list with one base64 string
                    b64_data = outputs[0]
                    # The API returns the full data URI, strip the prefix
                    if "base64," in b64_data:
                        b64_json = b64_data.split("base64,")[1]
                    else:
                        b64_json = b64_data
                    
                    return {"data": [{"b64_json": b64_json}]}

                elif status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    logger.error(f"WaveSpeedAI task failed: {error_msg}")
                    return {"error": f"Task failed: {error_msg}"}
                else:
                    logger.info(f"Task status: {status}. Continuing to poll.")
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"WaveSpeedAI API Error during polling: {e.response.status_code} - {e.response.text}")
                return {"error": f"HTTP {e.response.status_code} while polling", "details": e.response.text}
            except Exception as e:
                logger.error(f"Error polling WaveSpeedAI: {e}", exc_info=True)
                return {"error": f"Polling failed: {str(e)}"}

    return {"error": "Polling timed out after 60 seconds."}


async def send_wavespeed_flux_request(
    api_key: str,
    model: str,
    prompt: str,
    image_base64: Optional[str] = None, # For single image models
    images_base64: Optional[list[str]] = None, # For multi-image models
    size: Optional[str] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 2.5,
    num_images: int = 1,
    seed: int = -1,
    enable_safety_checker: bool = True,
) -> Dict[str, Any]:
    """
    Sends a request to the WaveSpeedAI FLUX Kontext Dev API and polls for the result.
    """
    if not api_key:
        raise ValueError("WaveSpeed API key is required.")

    payload = {
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "enable_safety_checker": enable_safety_checker,
        "enable_base64_output": True,
    }
    if seed != -1:
        payload["seed"] = seed
    if size:
        payload["size"] = size
    if images_base64:
        payload["images"] = [f"data:image/png;base64,{b64}" for b64 in images_base64]
    elif image_base64:
        payload["image"] = f"data:image/png;base64,{image_base64}"
    
    submit_url = f"https://api.wavespeed.ai/api/v3/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    request_id = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            logger.info(f"Submitting task to WaveSpeedAI Flux model: {model}")
            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            if result.get("data") and result["data"].get("id"):
                request_id = result["data"]["id"]
                logger.info(f"Task submitted successfully. Request ID: {request_id}")
            else:
                logger.error(f"WaveSpeedAI API Error: Unexpected submission response. {result}")
                return {"error": "Unexpected submission response format.", "details": result}

        except httpx.HTTPStatusError as e:
            logger.error(f"WaveSpeedAI API Error on submission: {e.response.status_code} - {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}", "details": e.response.text}
        except Exception as e:
            logger.error(f"Error submitting to WaveSpeedAI: {e}", exc_info=True)
            return {"error": str(e)}

        if not request_id:
            return {"error": "Failed to get a request ID from WaveSpeedAI."}

        # --- Polling for result ---
        result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        polling_headers = {"Authorization": f"Bearer {api_key}"}
        
        for attempt in range(120): # Poll for up to 120 seconds
            try:
                await asyncio.sleep(1)
                logger.debug(f"Polling for result... (Attempt {attempt + 1})")
                
                poll_response = await client.get(result_url, headers=polling_headers)
                poll_response.raise_for_status()
                
                poll_result = poll_response.json().get("data", {})
                status = poll_result.get("status")

                if status == "completed":
                    logger.info("WaveSpeedAI Flux task completed.")
                    outputs = poll_result.get("outputs", [])
                    if not outputs:
                        return {"error": "Task completed but no outputs found."}
                    
                    # Process multiple potential outputs
                    b64_jsons = []
                    for out in outputs:
                        if "base64," in out:
                            b64_jsons.append(out.split("base64,")[1])
                        else:
                            b64_jsons.append(out)
                    
                    return {"data": [{"b64_json": b64} for b64 in b64_jsons]}

                elif status == "failed":
                    error_msg = poll_result.get("error", "Unknown error.")
                    logger.error(f"WaveSpeedAI Flux task failed: {error_msg}")
                    return {"error": f"Task failed: {error_msg}"}
                else:
                    logger.info(f"Task status: {status}. Continuing to poll.")
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"WaveSpeedAI API Error during polling: {e.response.status_code} - {e.response.text}")
                return {"error": f"HTTP {e.response.status_code} while polling", "details": e.response.text}
            except Exception as e:
                logger.error(f"Error polling WaveSpeedAI: {e}", exc_info=True)
                return {"error": f"Polling failed: {str(e)}"}

    return {"error": "Polling timed out after 120 seconds."} 