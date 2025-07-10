# comfy-nodes/config_generate_video_seedance_pro_i2v_720p.py
"""Configure Video Generation for Bytedance Seedance Pro I2V (WaveSpeedAI).

Handles image-to-video parameters specific to the
`bytedance-seedance-v1-pro-i2v-720p` endpoint. Prompt text is supplied via
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

class ConfigGenerateVideoSeedanceProI2V:
    MODEL_ID = "bytedance-seedance-v1-pro-i2v-720p"

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
                "duration": ("INT", {"default": 5, "min": 5, "max": 10}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/video/seedance"

    def configure(
        self,
        context: Optional[Any] = None,
        image_url: str = "",
        duration: int = 5,
        seed: int = -1,
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideoSeedanceProI2V executing…")

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
                "duration": int(duration),
                "seed": int(seed),
            }
        )

        out_ctx["generation_config"] = gen_cfg
        logger.info("ConfigGenerateVideoSeedanceProI2V: saved config %s", gen_cfg)
        return (out_ctx,)


NODE_CLASS_MAPPINGS = {
    "ConfigGenerateVideoSeedanceProI2V": ConfigGenerateVideoSeedanceProI2V
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateVideoSeedanceProI2V": "Configure Seedance Pro I2V (LLMToolkit)"
} 