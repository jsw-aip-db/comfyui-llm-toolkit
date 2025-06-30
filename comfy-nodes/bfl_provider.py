# bfl_provider.py
"""BFL (Flux Kontext MAX) Provider Node for ComfyUI-LLM-Toolkit

This lightweight node outputs a *provider_config* dictionary that instructs
other nodes (e.g. *Generate Image*) to use the BFL "Flux Kontext MAX" model.
It mimics the behaviour of *OpenAIProviderNode* but is much simpler because
BFL currently exposes a single public model through the `/flux-kontext-max`
endpoint.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Tuple

# Ensure the toolkit root is in *sys.path* so we can import shared utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from llmtoolkit_utils import get_api_key
except ImportError as e:
    raise ImportError(
        "BFLProviderNode: Unable to import llmtoolkit_utils."
    ) from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BFLProviderNode:
    """Minimal provider-configuration node for BFL Flux Kontext."""

    PROVIDER_NAME = "bfl"
    DEFAULT_MODEL = "flux-kontext-max"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_MODEL,
                        "tooltip": "Model name (currently only flux-kontext-max)",
                    },
                ),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure_bfl"
    CATEGORY = "llm_toolkit"

    @classmethod
    def IS_CHANGED(cls, llm_model: str, context: Any = None):
        # Recompute when model name changes (future-proof)
        return llm_model

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------
    def configure_bfl(self, llm_model: str, context: Any = None) -> Tuple[Any]:
        logger.info("BFLProviderNode: selected model '%s'", llm_model)

        # Resolve API key (optional – downstream node will flag missing key)
        api_key = ""
        try:
            api_key = get_api_key("BFL_API_KEY", self.PROVIDER_NAME)
            logger.info("BFLProviderNode: API key fetched from environment / .env file")
        except ValueError as e:
            logger.warning("BFLProviderNode: BFL_API_KEY not found – %s", e)
            # Leave empty; GenerateImage will signal auth error.

        provider_config = {
            "provider_name": self.PROVIDER_NAME,
            "llm_model": llm_model.strip() or self.DEFAULT_MODEL,
            "api_key": api_key,
            # BFL does not need IP/port
            "base_ip": None,
            "port": None,
        }

        if context is not None:
            if isinstance(context, dict):
                context["provider_config"] = provider_config
                output = context
            else:
                output = {"provider_config": provider_config, "passthrough_data": context}
        else:
            output = {"provider_config": provider_config}

        return (output,)


NODE_CLASS_MAPPINGS = {
    "BFLProviderNode": BFLProviderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BFLProviderNode": "BFL Provider (LLMToolkit)",
} 