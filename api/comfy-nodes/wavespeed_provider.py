# comfy-nodes/wavespeed_provider.py
"""WaveSpeed Provider Node for ComfyUI-LLM-Toolkit

This node exposes WaveSpeedAI video & image generation models via a simple
combo-box dropdown and outputs a *provider_config* dictionary inside the
pipeline context so that downstream generator nodes can send requests without
having to know any WaveSpeed specifics.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root utilities are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    import requests
    from llmtoolkit_utils import get_api_key  # Re-use existing key helper
except ImportError as e:
    raise ImportError(
        "WaveSpeedProviderNode: failed to import dependencies – " + str(e)
    )

logger = logging.getLogger(__name__)


class WaveSpeedProviderNode:
    """Dedicated ComfyUI node that prepares *provider_config* for WaveSpeedAI."""

    PROVIDER_NAME = "wavespeed"

    # A concise fallback list – will be extended dynamically when the live
    # endpoint is reachable.  These cover popular video models supported by
    # WaveSpeedAI at the time of writing.
    _DEFAULT_MODELS: List[str] = [
        # WaveSpeed AI Models
        "wavespeed-ai/flux-kontext-dev-ultra-fast",
        "wavespeed-ai/flux-kontext-dev/multi-ultra-fast",
        # Bytedance Image Editing/Enhancement
        "bytedance/portrait",
        "bytedance/seededit-v3",
        # Bytedance Seedance
        "bytedance-seedance-v1-pro-t2v-480p",
        "bytedance-seedance-v1-pro-t2v-720p",
        "bytedance-seedance-v1-pro-t2v-1080p",
        "bytedance-seedance-v1-pro-i2v-480p",
        "bytedance-seedance-v1-pro-i2v-720p",
        "bytedance-seedance-v1-pro-i2v-1080p",
        # Minimax Hailuo
        "minimax/hailuo-02/t2v-standard",
        "minimax/hailuo-02/t2v-pro",
        "minimax/hailuo-02/i2v-pro",
        "minimax/hailuo-02/i2v-standard",
        # Kuaishou Kling
        "kwaivgi/kling-v2.1-i2v-standard",
        "kwaivgi/kling-v2.1-i2v-pro",
        "kwaivgi/kling-v2.1-i2v-master",
        # Wan 2.1
        #"wan-2.1-t2v-720p",
        #"wan-2.1-i2v-720p",
        # Google Veo
        "google/veo3-fast",
        "wavespeed-ai/veo2-t2v",
        "wavespeed-ai/veo2-i2v",
        # Hunyuan
        #"hunyuan-t2v",
        # Seedance Lite
        #"bytedance-seedance-v1-lite-i2v-720p",
    ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def _fetch_models(cls, api_key: Optional[str]) -> List[str]:
        """Return list of WaveSpeedAI model identifiers.

        Tries the official `/models` endpoint. Falls back to a static list when
        the request fails (e.g. missing/invalid key or network issues).
        """
        # 2024-07-29: Skipping live fetch since endpoint is 404.
        logger.debug("WaveSpeedProviderNode: skipping model fetch, using defaults.")
        return cls._DEFAULT_MODELS

    # ------------------------------------------------------------------
    # ComfyUI node interface
    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        # Attempt env key fetch so UI dropdown is populated immediately.
        api_key_env = os.getenv("WAVESPEED_API_KEY", "").strip()
        model_list = cls._fetch_models(api_key_env)
        default_model = model_list[0] if model_list else "bytedance-seedance-v1-pro-t2v-720p"
        return {
            "required": {
                "llm_model": (model_list, {"default": default_model}),
            },
            "optional": {
                "external_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "WaveSpeed API key (optional – overrides env/.env)",
                    },
                ),
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "select_wavespeed"
    CATEGORY = "llm_toolkit/providers"

    # ------------------------------------------------------------------
    # IS_CHANGED – refresh when key or model selection changes
    # ------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, llm_model: str, external_api_key: str = "", context: Any = None):
        import hashlib

        key_hash = hashlib.md5(external_api_key.encode()).hexdigest() if external_api_key else "nokey"
        state = f"{llm_model}-{key_hash}"
        return hashlib.md5(state.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def select_wavespeed(
        self,
        llm_model: str,
        external_api_key: str = "",
        context: Optional[Any] = None,
    ) -> Tuple[Any]:
        """Validate key, build *provider_config*, and merge it into context."""

        # 1) Resolve API key
        external_api_key = external_api_key.strip()
        api_key: str = ""
        if external_api_key:
            api_key = external_api_key  # Assume valid – WaveSpeed has no cheap validation endpoint
            logger.info("WaveSpeedProviderNode: using external API key provided by user")
        else:
            try:
                api_key = get_api_key("WAVESPEED_API_KEY", self.PROVIDER_NAME)
            except ValueError:
                api_key = ""  # Leave blank – downstream will error out gracefully

        # Allow placeholder key for offline testing so graph stays functional.
        if not api_key:
            api_key = "1234"

        provider_config: Dict[str, Any] = {
            "provider_name": self.PROVIDER_NAME,
            "llm_model": llm_model,
            "api_key": api_key,
            "base_ip": None,
            "port": None,
        }

        # Merge with incoming context or create a new one
        if context is not None:
            if isinstance(context, dict):
                merged = context.copy()
                merged["provider_config"] = provider_config
                result: Any = merged
            else:
                result = {"provider_config": provider_config, "passthrough_data": context}
        else:
            result = {"provider_config": provider_config}

        return (result,)


# ------------------------------------------------------------------
# Node registration – discovered by ComfyUI automatically
# ------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"WaveSpeedProviderNode": WaveSpeedProviderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WaveSpeedProviderNode": "WaveSpeed Provider (LLMToolkit)"} 