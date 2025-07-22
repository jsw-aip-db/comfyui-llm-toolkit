# comfy-nodes/config_generate_video_veo3_fast.py
"""Configure Video Generation for Google VEO3 Fast.

Handles text-to-video parameters specific to the
`google/veo3-fast` endpoint. Prompt text is supplied via
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

class ConfigGenerateVideoVeo3Fast:
    MODEL_ID = "google/veo3-fast"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "aspect_ratio": (
                    ["16:9", "9:16", "1:1", "4:3", "3:4"],
                    {"default": "16:9"}
                ),
                "duration": ("INT", {"default": 8, "min": 8, "max": 8}), # Fixed at 8s
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/video/veo"

    def configure(
        self,
        context: Optional[Any] = None,
        aspect_ratio: str = "16:9",
        duration: int = 8,
        negative_prompt: str = "",
        enable_prompt_expansion: bool = True,
        generate_audio: bool = False,
        seed: int = -1,
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideoVeo3Fast executing…")

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
                "enable_prompt_expansion": enable_prompt_expansion,
                "generate_audio": generate_audio,
                "seed": int(seed),
                **({"negative_prompt": negative_prompt.strip()} if negative_prompt.strip() else {}),
            }
        )

        out_ctx["generation_config"] = gen_cfg
        logger.info("ConfigGenerateVideoVeo3Fast: saved config %s", gen_cfg)
        return (out_ctx,)


NODE_CLASS_MAPPINGS = {
    "ConfigGenerateVideoVeo3Fast": ConfigGenerateVideoVeo3Fast
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateVideoVeo3Fast": "Configure VEO3 Fast (LLMToolkit)"
} 