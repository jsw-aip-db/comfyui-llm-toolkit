# comfy-nodes/config_generate_speech.py
"""ComfyUI helper node – prepares generation_config for Gemini TTS.

The node mirrors *config_generate_image.py* but specialises on the
parameters needed by the Gemini 2.5 TTS models.
"""

import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

from context_payload import extract_context

# Ensure parent dir in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

# Pre-built voice names published by Google (see docs)
VOICE_OPTIONS = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]


class ConfigGenerateSpeech:
    """Adds text-to-speech parameters to the context for downstream nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "voice_name": (VOICE_OPTIONS, {"default": "Kore"}),
                # Sample rate & channels kept here for future flexibility
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000}),
                "channels": ("INT", {"default": 1, "min": 1, "max": 2}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/speech"

    def configure(
        self,
        context: Optional[Dict[str, Any]] = None,
        voice_name: str = "Kore",
        sample_rate: int = 24000,
        channels: int = 1,
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateSpeech executing…")

        # Unwrap / copy context
        if context is None:
            output_context: Dict[str, Any] = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            output_context = extract_context(context)
            if not isinstance(output_context, dict):
                output_context = {"passthrough_data": context}

        gen_cfg = output_context.get("generation_config", {})
        if not isinstance(gen_cfg, dict):
            gen_cfg = {}

        gen_cfg.update(
            {
                "voice_name": voice_name,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )

        output_context["generation_config"] = gen_cfg
        logger.info("ConfigGenerateSpeech: Updated generation_config with voice %s", voice_name)
        return (output_context,)


NODE_CLASS_MAPPINGS = {"ConfigGenerateSpeech": ConfigGenerateSpeech}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateSpeech": "Configure Speech Generation (LLMToolkit)",
} 