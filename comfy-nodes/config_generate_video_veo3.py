# comfy-nodes/config_generate_video_veo3.py
"""Configure Video Generation for Google Veo 3 (WaveSpeedAI).

Veo 3 is Google DeepMind's text-to-video model with native audio.  This
configuration node handles model-specific parameters (currently only *seed*).
Prompt text itself is supplied via the *Prompt Manager* node or the
GenerateVideo node argument – we intentionally keep it out of
*generation_config* so later nodes can override it cleanly.
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

class ConfigGenerateVideoVeo3:
    MODEL_ID = "google-veo3"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                # prompt handled via PromptManager; not included here
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "🔗llm_toolkit/config/video/veo"

    # ------------------------------------------------------------------
    def configure(
        self,
        context: Optional[Any] = None,
        seed: int = -1,
    ) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateVideoVeo3 executing…")

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
                # 'prompt' handled elsewhere; only include seed here
                "seed": int(seed),
            }
        )

        out_ctx["generation_config"] = gen_cfg
        logger.info("ConfigGenerateVideoVeo3: saved config %s", gen_cfg)
        return (out_ctx,)


# ------------------------------------------------------------------
# Node registration
# ------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateVideoVeo3": ConfigGenerateVideoVeo3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateVideoVeo3": "Configure Veo 3 (🔗LLMToolkit)"
} 