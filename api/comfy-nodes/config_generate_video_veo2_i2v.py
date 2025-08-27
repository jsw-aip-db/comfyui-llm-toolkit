# comfy-nodes/config_generate_video_veo2_i2v.py
"""Configure Video Generation for WaveSpeedAI Veo2 I2V.

Handles image-to-video parameters specific to the
`wavespeed-ai/veo2-i2v` endpoint. Prompt text is supplied via
PromptManager or GenerateVideo; this node only adds model-specific settings to
generation_config.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

from context_payload import extract_context  # type: ignore

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

class ConfigGenerateVideoVeo2I2V:
    MODEL_ID = "wavespeed-ai/veo2-i2v"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "image_url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Input image URL (.jpg/.png)",
                    },
                ),
                "aspect_ratio": (
                    ["16:9", "9:16", "1:1", "4:3", "3:4"],
                    {"default": "16:9"}
                ),
                "duration": (
                    ["5s", "6s", "7s", "8s"],
                    {"default": "5s"}
                ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/video/veo"

    def configure(
        self,
        context: Optional[Any] = None,
        image_url: str = "",
        aspect_ratio: str = "16:9",
        duration: str = "5s",
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideoVeo2I2V executing…")

        # Prepare context dict
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

        # Only set/update keys relevant to this model, keep others intact
        gen_cfg.update(
            {
                "model_id": self.MODEL_ID,
                **({"image": image_url.strip()} if image_url.strip() else {}),
                "aspect_ratio": aspect_ratio,
                "duration": duration,
            }
        )

        out_ctx["generation_config"] = gen_cfg
        logger.info("ConfigGenerateVideoVeo2I2V: saved config %s", gen_cfg)
        return (out_ctx,)


NODE_CLASS_MAPPINGS = {
    "ConfigGenerateVideoVeo2I2V": ConfigGenerateVideoVeo2I2V
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateVideoVeo2I2V": "Configure Veo2 I2V (LLMToolkit)"
} 