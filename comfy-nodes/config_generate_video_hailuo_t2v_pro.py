# comfy-nodes/config_generate_video_hailuo_t2v_pro.py
"""Configure Video Generation for Minimax Hailuo 02 T2V Pro.

Handles text-to-video parameters specific to the
`minimax/hailuo-02/t2v-pro` endpoint.
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

class ConfigGenerateVideoHailuoT2VPro:
    MODEL_ID = "minimax/hailuo-02/t2v-pro"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/video/hailuo"

    def configure(
        self,
        context: Optional[Any] = None,
        enable_prompt_expansion: bool = True,
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideoHailuoT2VPro executing…")

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
                "enable_prompt_expansion": enable_prompt_expansion,
            }
        )

        out_ctx["generation_config"] = gen_cfg
        logger.info("ConfigGenerateVideoHailuoT2VPro: saved config %s", gen_cfg)
        return (out_ctx,)


NODE_CLASS_MAPPINGS = {
    "ConfigGenerateVideoHailuoT2VPro": ConfigGenerateVideoHailuoT2VPro
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateVideoHailuoT2VPro": "Configure Hailuo 02 T2V Pro (LLMToolkit)"
} 