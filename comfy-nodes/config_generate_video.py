# comfy-nodes/config_generate_video.py
"""Configure Video Generation (Veo) – adds GenerateVideosConfig values to context."""

import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

from context_payload import extract_context

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

ASPECT_OPTIONS = ["16:9", "9:16"]
PERSON_OPTIONS = ["dont_allow", "allow_adult"]  # Removed "allow_all" - not supported by Veo API

class ConfigGenerateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "aspect_ratio": (ASPECT_OPTIONS, {"default": "16:9"}),
                "person_generation": (PERSON_OPTIONS, {"default": "dont_allow"}),
                "number_of_videos": ("INT", {"default": 1, "min": 1, "max": 2}),
                "duration_seconds": ("INT", {"default": 6, "min": 5, "max": 8}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "🔗llm_toolkit/config/video"

    def configure(
        self,
        context: Optional[Dict[str, Any]] = None,
        aspect_ratio: str = "16:9",
        person_generation: str = "dont_allow",
        number_of_videos: int = 1,
        duration_seconds: int = 6,
        negative_prompt: str = "",
        enhance_prompt: bool = True,
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideo executing…")

        if context is None:
            out_ctx: Dict[str, Any] = {}
        elif isinstance(context, dict):
            out_ctx = context.copy()
        else:
            out_ctx = extract_context(context)
            if not isinstance(out_ctx, dict):
                out_ctx = {"passthrough_data": context}

        gen_cfg = out_ctx.get("generation_config", {})
        if not isinstance(gen_cfg, dict):
            gen_cfg = {}

        gen_cfg.update(
            {
                "aspect_ratio": aspect_ratio,
                "person_generation": person_generation,
                "number_of_videos": number_of_videos,
                "duration_seconds": duration_seconds,
                "negative_prompt": negative_prompt.strip(),
                "enhance_prompt": enhance_prompt,
            }
        )

        out_ctx["generation_config"] = gen_cfg
        logger.info("ConfigGenerateVideo: saved config %s", gen_cfg)
        return (out_ctx,)

NODE_CLASS_MAPPINGS = {"ConfigGenerateVideo": ConfigGenerateVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"ConfigGenerateVideo": "Configure Video Generation (🔗LLMToolkit)"} 