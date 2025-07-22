import os
import sys
import logging
import tempfile
from typing import Any, Dict, Optional, Tuple

import torch
import requests
import torchaudio
import json
import urllib.parse
from io import BytesIO
import numpy as np
from PIL import Image
import folder_paths
import time

# Ensure parent directory (project root) is on sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api.suno_api import (
    send_suno_music_generation_request,
    send_suno_lyrics_generation_request,
    get_suno_remaining_credits,
)
from llmtoolkit_utils import get_api_key
from context_payload import extract_context
from send_request import run_async

logger = logging.getLogger(__name__)


class GenerateMusic:  # noqa: N801 – follow existing naming pattern
    """Generate music with Suno API and return an AUDIO tensor compatible with nodes_audio."""

    DEFAULT_PROVIDER = "suno"
    DEFAULT_MODEL = "V3_5"
    DEFAULT_PROMPT = "A calm and relaxing piano track with soft melodies"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": cls.DEFAULT_PROMPT},
                ),
            },
            "optional": {
                "save_name": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Folder name inside ComfyUI/outputs where audio, image and metadata will be saved.",
                    },
                ),
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "credits")
    FUNCTION = "generate"
    CATEGORY = "llm_toolkit/generators"
    OUTPUT_NODE = True

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def _download_audio_to_tensor(self, audio_url: str, dest_dir: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """Download a remote audio file and convert to ComfyUI audio tensor.

        If *dest_dir* is provided the file is saved there (using *filename* or the name parsed from the URL).
        Otherwise a temporary file is used and deleted afterwards.
        """
        try:
            # Determine destination path (if requested)
            save_path = None
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
                if not filename:
                    filename = os.path.basename(urllib.parse.urlparse(audio_url).path) or "audio.mp3"
                save_path = os.path.join(dest_dir, filename)

            # Stream-download the audio
            logger.info("Downloading generated audio from %s", audio_url)
            with requests.get(audio_url, stream=True, timeout=60) as resp:
                resp.raise_for_status()

                # Choose path – temp or final
                if save_path:
                    tmp_path = save_path
                    f_handle = open(tmp_path, "wb")
                else:
                    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tmp_path = tmp_file.name
                    f_handle = tmp_file

                with f_handle as out_f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            out_f.write(chunk)

            # Load waveform via torchaudio
            waveform, sample_rate = torchaudio.load(tmp_path)

            # Clean up temp file (when not persisting)
            if not save_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            logger.error("Failed to download or decode audio: %s", e, exc_info=True)
            return {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}

    def _download_image_to_tensor(self, image_url: str, dest_dir: Optional[str] = None, filename: Optional[str] = None):
        """Download the cover image and return as ComfyUI IMAGE tensor (NHWC)."""
        try:
            logger.info("Downloading generated image from %s", image_url)
            resp = requests.get(image_url, timeout=60)
            resp.raise_for_status()
            img_bytes = resp.content

            # Persist if requested
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
                if not filename:
                    filename = os.path.basename(urllib.parse.urlparse(image_url).path) or "cover.jpg"
                with open(os.path.join(dest_dir, filename), "wb") as f:
                    f.write(img_bytes)

            # Convert to tensor
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0  # scale to 0-1
            img_tensor = torch.from_numpy(img_np)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)  # add batch dim
            return img_tensor
        except Exception as e:
            logger.error("Failed to download or process image: %s", e, exc_info=True)
            return torch.zeros((1, 64, 64, 3))  # placeholder small black image

    # ---------------------------------------------------------------------
    # Main generate() implementation
    # ---------------------------------------------------------------------
    def generate(self, prompt: str, save_name: str = "", context: Optional[Dict[str, Any]] = None):
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            context = extract_context(context) or {}

        provider_cfg = context.get("provider_config", {})
        llm_provider = provider_cfg.get("provider_name", self.DEFAULT_PROVIDER)

        if llm_provider != "suno":
            err = f"GenerateMusic node supports only provider 'suno'. Received '{llm_provider}'."
            logger.error(err)
            context["error"] = err
            placeholder = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}
            credits_resp = run_async(get_suno_remaining_credits(api_key=api_key))
            credits_str = str(credits_resp.get("data")) if isinstance(credits_resp, dict) else ""
            context["suno_credits"] = credits_str
            return (context, credits_str)

        # ------------------------------------------------------------------
        # API Key resolution
        # ------------------------------------------------------------------
        api_key = provider_cfg.get("api_key", "")
        if not api_key or api_key in {"", "1234"}:
            try:
                api_key = get_api_key("SUNO_API_KEY", "suno")
            except ValueError:
                api_key = ""

        if not api_key:
            err = "Missing Suno API key. Please set SUNO_API_KEY environment variable or pass via ProviderSelector."
            logger.error(err)
            context["error"] = err
            placeholder = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}
            credits_resp = run_async(get_suno_remaining_credits(api_key=api_key))
            credits_str = str(credits_resp.get("data")) if isinstance(credits_resp, dict) else ""
            context["suno_credits"] = credits_str
            return (context, credits_str)

        # ------------------------------------------------------------------
        # Read generation_config overrides from context (optional)
        # ------------------------------------------------------------------
        gen_cfg = context.get("generation_config", {})
        style = gen_cfg.get("style")
        title = gen_cfg.get("title")
        custom_mode = gen_cfg.get("custom_mode", False)
        instrumental = gen_cfg.get("instrumental", False)
        model_name = provider_cfg.get("llm_model") or gen_cfg.get("model") or self.DEFAULT_MODEL
        negative_tags = gen_cfg.get("negative_tags")
        callback_url = gen_cfg.get("callback_url") or "https://example.com/callback"

        # ------------------------------------------------------------------
        # Call Suno async helper via run_async
        # ------------------------------------------------------------------
        api_response = run_async(
            send_suno_music_generation_request(
                api_key=api_key,
                prompt=prompt,
                style=style,
                title=title,
                custom_mode=custom_mode,
                instrumental=instrumental,
                model=model_name,
                negative_tags=negative_tags,
                callback_url=callback_url,
                poll=True,
            )
        )

        audio_tensor_dict: Optional[Dict[str, Any]] = None  # no longer used for output
        image_tensor = None  # not used for output
        lyrics_text = ""
        title = ""
        if isinstance(api_response, dict):
            data_block = api_response.get("data", {})
            # Attempt to gather tracks from multiple possible keys
            tracks = []
            if isinstance(data_block, dict):
                if isinstance(data_block.get("data"), list):
                    tracks = data_block.get("data", [])
                elif isinstance(data_block.get("sunoData"), list):
                    tracks = data_block.get("sunoData", [])
                elif isinstance(data_block.get("response"), dict):
                    resp_block = data_block["response"]
                    if isinstance(resp_block.get("sunoData"), list):
                        tracks = resp_block.get("sunoData", [])
            # Fallback: search one level deeper for any list of dicts containing audioUrl
            if not tracks:
                for v in data_block.values():
                    if isinstance(v, list):
                        if all(isinstance(x, dict) for x in v):
                            if any("audioUrl" in x or "audio_url" in x for x in v):
                                tracks = v
                                break

            for tr in tracks:
                audio_url = (
                    tr.get("audio_url")
                    or tr.get("audioUrl")
                    or tr.get("source_audio_url")
                    or tr.get("sourceAudioUrl")
                )
                image_url = (
                    tr.get("image_url")
                    or tr.get("imageUrl")
                    or tr.get("sourceImageUrl")
                    or tr.get("source_image_url")
                )
                title = tr.get("title", title)
                lyrics_text = tr.get("prompt", lyrics_text)

                if audio_url and "suno_audio_url" not in context:
                    context["suno_audio_url"] = audio_url
                if image_url and "suno_image_url" not in context:
                    context["suno_image_url"] = image_url

            # store title & lyrics in context
            if title:
                context["suno_title"] = title
            if lyrics_text:
                context["suno_lyrics"] = lyrics_text

        # If Suno didn't give us URLs, note error
        if "suno_audio_url" not in context:
            context["warning"] = "Suno response did not include audio URL yet."

        # Fetch remaining credits regardless of success or failure
        credits_resp = run_async(get_suno_remaining_credits(api_key=api_key))
        credits_str = str(credits_resp.get("data")) if isinstance(credits_resp, dict) else ""
        context["suno_credits"] = credits_str
        return (context, credits_str)


class GenerateLyrics:  # noqa: N801
    """Generate song lyrics with Suno API."""

    DEFAULT_PROMPT = "A song about peaceful night in the city"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": cls.DEFAULT_PROMPT},
                ),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "lyrics")
    FUNCTION = "generate_lyrics"
    CATEGORY = "llm_toolkit/generators"
    OUTPUT_NODE = True

    def generate_lyrics(self, prompt: str, context: Optional[Dict[str, Any]] = None):
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            context = extract_context(context) or {}

        provider_cfg = context.get("provider_config", {})
        llm_provider = provider_cfg.get("provider_name", "suno")
        if llm_provider != "suno":
            err = f"GenerateLyrics node supports only provider 'suno'. Got '{llm_provider}'."
            context["error"] = err
            return (context, err)

        api_key = provider_cfg.get("api_key", "")
        if not api_key or api_key in {"", "1234"}:
            try:
                api_key = get_api_key("SUNO_API_KEY", "suno")
            except ValueError:
                api_key = ""
        if not api_key:
            err = "Missing Suno API key."
            context["error"] = err
            return (context, err)

        api_response = run_async(
            send_suno_lyrics_generation_request(
                api_key=api_key,
                prompt=prompt,
                callback_url="https://example.com/callback",
                poll=True,
            )
        )

        lyrics_text = ""
        if isinstance(api_response, dict):
            data_block = api_response.get("data", {})
            lyrics_list = data_block.get("lyricsData", [])
            if lyrics_list:
                lyrics_text = lyrics_list[0].get("text", "")

        context["suno_lyrics_response"] = api_response
        return (context, lyrics_text)


# -------------------------------------------------------------------------
# Node registration for ComfyUI
# -------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "GenerateMusic": GenerateMusic,
    "GenerateLyrics": GenerateLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateMusic": "Generate Music (LLMToolkit)",
    "GenerateLyrics": "Generate Lyrics (LLMToolkit)",
} 