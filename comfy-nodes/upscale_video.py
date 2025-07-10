# comfy-nodes/upscale_video.py
"""Upscale Video using WaveSpeedAI.

This node takes a local video file, uploads it, and then calls the WaveSpeedAI
video upscaler endpoint. It polls for completion and downloads the result.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from context_payload import extract_context

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from llmtoolkit_utils import get_api_key
except ImportError:
    get_api_key = None  # type: ignore

try:
    import folder_paths
except ImportError:
    folder_paths = None  # type: ignore

logger = logging.getLogger(__name__)

class UpscaleVideo:
    """Node to handle video upscaling via WaveSpeedAI."""
    
    MODEL_ID = "wavespeed-ai/video-upscaler"
    CATEGORY = "llm_toolkit/utils/video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"placeholder": "/path/to/your/video.mp4"}),
                "target_resolution": (["1080p", "720p", "2k", "4k"], {"default": "1080p"}),
                "copy_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("upscaled_video_path",)
    FUNCTION = "upscale"

    def upscale(
        self,
        video_path: str,
        target_resolution: str,
        copy_audio: bool,
        context: Optional[Any] = None,
    ) -> Tuple[str]:
        logger.info("UpscaleVideo node executing…")

        # --- Context and API Key Handling ---
        if context is None:
            ctx: Dict[str, Any] = {}
        elif isinstance(context, dict):
            ctx = context.copy()
        else:
            ctx = extract_context(context)

        provider_cfg = ctx.get("provider_config", {})
        api_key = provider_cfg.get("api_key", "").strip()
        if (not api_key or api_key == "1234") and get_api_key:
            try:
                api_key = get_api_key("WAVESPEED_API_KEY", "wavespeed")
            except ValueError:
                api_key = ""

        if not api_key:
            err = "UpscaleVideo: missing WaveSpeed API key"
            logger.error(err)
            raise ValueError(err)

        # --- Video Path and Upload ---
        video_path = video_path.strip().replace("\\", "/")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"UpscaleVideo: Input video not found at {video_path}")

        video_url = ""
        try:
            with open(video_path, "rb") as f:
                upload_url = "https://api.wavespeed.ai/api/v2/media/upload/binary"
                file_name = os.path.basename(video_path)
                files = {"file": (file_name, f, "video/mp4")}
                headers = {"Authorization": f"Bearer {api_key}"}
                
                logger.info("UpscaleVideo: Uploading video to WaveSpeedAI...")
                up_resp = requests.post(upload_url, headers=headers, files=files, timeout=180)
                up_resp.raise_for_status()
                
                up_data = up_resp.json()
                if up_data.get("code") != 200:
                    raise RuntimeError(f"Upload failed: {up_data.get('message', 'Unknown error')}")
                
                video_url = up_data["data"]["download_url"]
                logger.info(f"UpscaleVideo: Video uploaded successfully. URL: {video_url}")

        except Exception as up_exc:
            logger.error("UpscaleVideo: Video upload failed - %s", up_exc, exc_info=True)
            raise

        # --- API Call and Polling ---
        post_url = f"https://api.wavespeed.ai/api/v3/{self.MODEL_ID}"
        payload = {
            "video": video_url,
            "target_resolution": target_resolution,
            "copy_audio": copy_audio,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            resp = requests.post(post_url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                err = f"WaveSpeed POST failed {resp.status_code}: {resp.text[:120]}"
                logger.error(err)
                raise RuntimeError(err)

            result = resp.json().get("data", {})
            request_id = result.get("id")
            if not request_id:
                err = "WaveSpeed API response missing request id"
                logger.error(err)
                raise RuntimeError(err)

            # --- Poll for results ---
            poll_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
            waited = 0
            max_wait = 1800  # 30 minutes
            poll_interval = 10
            
            while waited < max_wait:
                time.sleep(poll_interval)
                pr = requests.get(poll_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
                if pr.status_code != 200:
                    logger.warning(f"Polling failed with status {pr.status_code}, retrying...")
                    continue

                pr_data = pr.json().get("data", {})
                status = pr_data.get("status")
                logger.info(f"UpscaleVideo: Polling status for {request_id} - {status}")

                if status == "completed":
                    outputs = pr_data.get("outputs", [])
                    if not outputs:
                        raise RuntimeError("Upscaling completed but no outputs returned.")

                    # --- Download the upscaled video ---
                    if folder_paths and hasattr(folder_paths, "get_output_directory"):
                        out_dir = Path(folder_paths.get_output_directory()) / "upscaled_videos"
                    else:
                        out_dir = Path(os.getcwd()) / "output" / "upscaled_videos"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    video_dl_url = outputs[0]
                    video_resp = requests.get(video_dl_url, stream=True, timeout=300)
                    video_resp.raise_for_status()
                    
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    fname = f"{base_name}_upscaled_{target_resolution}_{request_id}.mp4"
                    tgt_path = out_dir / fname
                    
                    with open(tgt_path, "wb") as f_out:
                        for chunk in video_resp.iter_content(chunk_size=8192):
                            f_out.write(chunk)
                    
                    logger.info("UpscaleVideo: Saved upscaled video to %s", tgt_path)
                    return (str(tgt_path),)

                elif status == "failed":
                    err_msg = pr_data.get('error', 'Unknown error')
                    raise RuntimeError(f"Upscaling job failed: {err_msg}")
                
                waited += poll_interval

            raise TimeoutError("Upscaling job timed out after 30 minutes.")

        except Exception as exc:
            logger.error("UpscaleVideo: API error - %s", exc, exc_info=True)
            raise

NODE_CLASS_MAPPINGS = {"UpscaleVideo": UpscaleVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"UpscaleVideo": "Upscale Video (LLMToolkit)"} 