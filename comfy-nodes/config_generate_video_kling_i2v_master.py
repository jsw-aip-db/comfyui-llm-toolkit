# comfy-nodes/config_generate_video_kling_i2v_master.py
"""Configure Video Generation for Kling V2.1 I2V Master.

Handles image-to-video parameters specific to the
`kwaivgi/kling-v2.1-i2v-master` endpoint.
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

class ConfigGenerateVideoKlingI2VMaster:
    MODEL_ID = "kwaivgi/kling-v2.1-i2v-master"

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
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "guidance_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "duration": (["5", "10"], {"default": "5"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "🔗llm_toolkit/config/video/kling"

    def configure(
        self,
        context: Optional[Any] = None,
        image_url: str = "",
        negative_prompt: str = "",
        guidance_scale: float = 0.5,
        duration: str = "5",
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideoKlingI2VMaster executing…")

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
                "model_id": self.MODEL_ID,
                **({"image": image_url.strip()} if image_url.strip() else {}),
                **({"negative_prompt": negative_prompt.strip()} if negative_prompt.strip() else {}),
                "guidance_scale": float(guidance_scale),
                "duration": duration,
            }
        )

        out_ctx["generation_config"] = gen_cfg
        logger.info("ConfigGenerateVideoKlingI2VMaster: saved config %s", gen_cfg)
        return (out_ctx,)


NODE_CLASS_MAPPINGS = {
    "ConfigGenerateVideoKlingI2VMaster": ConfigGenerateVideoKlingI2VMaster
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateVideoKlingI2VMaster": "Configure Kling 2.1 I2V Master (🔗LLMToolkit)"
} 