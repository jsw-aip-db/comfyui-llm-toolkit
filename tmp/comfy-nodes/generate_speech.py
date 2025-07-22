# comfy-nodes/generate_speech.py
"""Generate speech (text-to-speech) using Gemini 2.5 TTS models.

The node consumes *provider_config* and *generation_config* from the
context (populated e.g. via GeminiProviderNode & ConfigGenerateSpeech) and
saves a WAV file with the audio.
"""

import os
import sys
import logging
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from context_payload import extract_context

# Add parent dir to path for utils
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


class GenerateSpeech:
    """Node that converts text to speech with Gemini TTS models."""

    DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Hello world!"}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "audio_path")
    FUNCTION = "generate"
    CATEGORY = "llm_toolkit/generators"

    def _save_wav(self, pcm_bytes: bytes, file_path: Path, channels: int = 1, rate: int = 24000, sample_width: int = 2):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm_bytes)

    # ------------------------------------------------------------------
    # Main node function
    # ------------------------------------------------------------------
    def generate(self, prompt: str, context: Optional[Any] = None):
        logger.info("GenerateSpeech node executing…")

        # -------------------- Unwrap context ---------------------------
        if context is None:
            context_dict: Dict[str, Any] = {}
        elif isinstance(context, dict):
            context_dict = context.copy()
        else:
            context_dict = extract_context(context)
            if not isinstance(context_dict, dict):
                context_dict = {"passthrough_data": context}

        provider_cfg = context_dict.get("provider_config", {})
        gen_cfg = context_dict.get("generation_config", {})

        llm_provider = provider_cfg.get("provider_name", "gemini").lower()
        if llm_provider not in {"gemini", "google"}:
            err = f"GenerateSpeech currently supports Gemini only, received provider '{llm_provider}'."
            logger.error(err)
            context_dict["error"] = err
            return (context_dict, "")

        llm_model = provider_cfg.get("llm_model", self.DEFAULT_MODEL)
        if not llm_model:
            llm_model = self.DEFAULT_MODEL

        # -------------------- API key resolution ----------------------
        api_key = provider_cfg.get("api_key", "").strip()
        if (not api_key or api_key == "1234") and get_api_key is not None:
            try:
                api_key = get_api_key("GEMINI_API_KEY", "gemini")
                logger.info("GenerateSpeech: fetched API key from environment/.env")
            except ValueError:
                pass
        if not api_key:
            err = "GenerateSpeech: missing Gemini API key."
            logger.error(err)
            context_dict["error"] = err
            return (context_dict, "")

        # -------------------- Import Google GenAI ---------------------
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            err = "google-generativeai package not installed. Run 'pip install google-generativeai'."
            logger.error(err)
            context_dict["error"] = err
            return (context_dict, "")

        # -------------------- Build configs ---------------------------
        voice_name = gen_cfg.get("voice_name", "Kore")
        sample_rate = int(gen_cfg.get("sample_rate", 24000))
        channels = int(gen_cfg.get("channels", 1))

        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )

        generation_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        )

        # -------------------- Call API -------------------------------
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=llm_model,
                contents=prompt,
                config=generation_config,
            )
        except Exception as api_exc:
            err = f"Gemini TTS API error: {api_exc}"
            logger.error(err, exc_info=True)
            context_dict["error"] = err
            return (context_dict, "")

        # Extract PCM bytes
        try:
            audio_bytes: bytes = response.candidates[0].content.parts[0].inline_data.data  # type: ignore
        except Exception as e:
            err = f"Unexpected TTS response format: {e}"
            logger.error(err, exc_info=True)
            context_dict["error"] = err
            return (context_dict, "")

        # Save WAV
        # Use ComfyUI's standard output directory
        if folder_paths and hasattr(folder_paths, 'get_output_directory'):
            base_output_dir = folder_paths.get_output_directory()
            out_dir = Path(base_output_dir) / "tts_audio"
        else:
            # Fallback to current directory if folder_paths not available
            out_dir = Path(os.getcwd()) / "output" / "tts_audio"
        
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving audio to: {out_dir}")
        
        filename = f"gemini_tts_{voice_name.lower()}_{os.getpid()}.wav"
        file_path = out_dir / filename
        try:
            self._save_wav(audio_bytes, file_path, channels=channels, rate=sample_rate)
        except Exception as save_exc:
            err = f"Failed saving WAV: {save_exc}"
            logger.error(err, exc_info=True)
            context_dict["error"] = err
            return (context_dict, "")

        logger.info("GenerateSpeech: saved audio to %s", file_path)
        context_dict["generated_audio_path"] = str(file_path)

        return (context_dict, str(file_path))


NODE_CLASS_MAPPINGS = {"GenerateSpeech": GenerateSpeech}
NODE_DISPLAY_NAME_MAPPINGS = {"GenerateSpeech": "Generate Speech (LLMToolkit)"} 