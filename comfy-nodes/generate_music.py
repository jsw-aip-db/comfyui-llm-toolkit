import os
import sys
import logging
import tempfile
from typing import Any, Dict, Optional, Tuple

import torch
import requests
import torchaudio

# Ensure parent directory (project root) is on sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from suno_api import (
    send_suno_music_generation_request,
    send_suno_lyrics_generation_request,
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
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "AUDIO")
    RETURN_NAMES = ("context", "audio")
    FUNCTION = "generate"
    CATEGORY = "llm_toolkit/generators"
    OUTPUT_NODE = True

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    def _download_audio_to_tensor(self, audio_url: str) -> Dict[str, Any]:
        """Download a remote audio file (MP3/OGG/WAV) and convert to ComfyUI audio tensor."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                logger.info("Downloading generated audio from %s", audio_url)
                with requests.get(audio_url, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                tmp.flush()
                temp_path = tmp.name

            waveform, sample_rate = torchaudio.load(temp_path)
            os.unlink(temp_path)

            # Ensure batch dimension [B, C, T]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            logger.error("Failed to download or decode audio: %s", e, exc_info=True)
            # Return 1-second silent placeholder tensor on failure
            return {
                "waveform": torch.zeros((1, 1, 44100)),
                "sample_rate": 44100,
            }

    # ---------------------------------------------------------------------
    # Main generate() implementation
    # ---------------------------------------------------------------------
    def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Any]:
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
            return (context, placeholder)

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
            return (context, placeholder)

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
        callback_url = gen_cfg.get("callback_url", "https://example.com/callback")

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

        audio_tensor_dict: Optional[Dict[str, Any]] = None
        if isinstance(api_response, dict):
            data_block = api_response.get("data", {})
            tracks = data_block.get("data", []) if isinstance(data_block, dict) else []
            if tracks:
                first_track = tracks[0]
                audio_url = first_track.get("audio_url") or first_track.get("source_audio_url")
                if audio_url:
                    audio_tensor_dict = self._download_audio_to_tensor(audio_url)
                    context["suno_audio_url"] = audio_url

        if audio_tensor_dict is None:
            logger.warning("Audio not ready or failed to download. Returning silent placeholder.")
            audio_tensor_dict = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}
            context["error"] = "Audio generation failed or timed out."

        context["suno_raw_response"] = api_response
        return (context, audio_tensor_dict)


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