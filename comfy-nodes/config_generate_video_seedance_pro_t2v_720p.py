# comfy-nodes/config_generate_video_seedance_pro_t2v_720p.py
"""Configure Video Generation for Bytedance Seedance Pro T2V (WaveSpeedAI).

This node follows the same pattern as *config_generate_video.py* but exposes the
parameters that are specific to the *bytedance-seedance-v1-pro-t2v-720p* model
on WaveSpeedAI.
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

ASPECT_OPTIONS = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]

class ConfigGenerateVideoSeedanceProT2V:
    """Builds *generation_config* for the Seedance Pro T2V WaveSpeed model."""

    MODEL_ID = "bytedance-seedance-v1-pro-t2v-720p"

    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "aspect_ratio": (ASPECT_OPTIONS, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 5, "max": 10}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "🔗llm_toolkit/config/video/seedance"

    # ------------------------------------------------------------------
    def configure(
        self,
        context: Optional[Any] = None,
        aspect_ratio: str = "16:9",
        duration: int = 5,
        seed: int = -1,
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideoSeedanceProT2V executing…")

        # Convert incoming context to dict or create new one
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
                "aspect_ratio": aspect_ratio,
                "duration": int(duration),
                "seed": int(seed),
            }
        )

        out_ctx["generation_config"] = gen_cfg
        logger.info("ConfigGenerateVideoSeedanceProT2V: saved config %s", gen_cfg)
        return (out_ctx,)


# ------------------------------------------------------------------
# Node registration
# ------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateVideoSeedanceProT2V": ConfigGenerateVideoSeedanceProT2V
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateVideoSeedanceProT2V": "Configure Seedance Pro T2V (🔗LLMToolkit)"
} 