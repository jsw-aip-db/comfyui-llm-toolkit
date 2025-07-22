# openai_provider.py
"""OpenAI Provider Node for ComfyUI‑LLM‑Toolkit

This node is a streamlined version of the generic LLMToolkitProviderSelector that is
hard‑wired to the OpenAI provider.  It only exposes the *model* selection to the UI
while everything else (provider name and API key) is resolved automatically from the
environment / .env file with the existing helper utilities.
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Tuple

# Ensure parent directory (repository root) is on sys.path so we can import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import utility helpers from the main toolkit
try:
    from llmtoolkit_utils import get_api_key, get_models
except ImportError as e:
    raise ImportError("OpenAIProviderNode: Failed to import llmtoolkit_utils. "
                      "Ensure it exists and is error‑free.") from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------------
# ComfyUI Server Endpoint for Model List (optional but convenient)
# -------------------------------------------------------------------------
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/ComfyLLMToolkit/get_openai_models")
    async def get_openai_models_endpoint(request):
        """Return the list of available OpenAI chat/completion models for the account."""
        try:
            api_key = ""
            try:
                api_key = get_api_key("OPENAI_API_KEY", OpenAIProviderNode.PROVIDER_NAME)
            except ValueError:
                logger.warning("get_openai_models_endpoint: OPENAI_API_KEY not set; returning empty list")

            models = get_models(OpenAIProviderNode.PROVIDER_NAME, None, None, api_key)

            # Ensure GPT-Image-1 is always present in the dropdown, even if the
            # account list endpoint does not yet expose it (common while the
            # model is still in limited beta).
            if not models:
                models = ["gpt-image-1"]
            elif "gpt-image-1" not in models:
                models.insert(0, "gpt-image-1")  # prepend for visibility

            return web.json_response(models)
        except Exception as e:
            logger.error("get_openai_models_endpoint: error %s", e, exc_info=True)
            return web.json_response(["Error fetching models"], status=500)

    logger.info("OpenAIProviderNode: /ComfyLLMToolkit/get_openai_models endpoint registered")

except (ImportError, AttributeError) as e:
    # Running outside ComfyUI server context; skip endpoint.
    logger.debug("OpenAIProviderNode: PromptServer not available (%s); endpoint not registered", e)


class OpenAIProviderNode:
    """A minimal provider‑configuration node dedicated to OpenAI."""

    PROVIDER_NAME = "openai"

    # Dummy placeholder value – JS will replace this with a dynamic combo‑box.
    PLACEHOLDER_MODEL = "gpt-4o-mini"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Backend sees a STRING; frontend JS converts to COMBO.
                "llm_model": (
                    "STRING",
                    {
                        "default": cls.PLACEHOLDER_MODEL,
                        "tooltip": "OpenAI model name. Populated dynamically from your account.",
                    },
                ),
            },
            "optional": {
                # Pass through pipeline data
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure_openai"
    CATEGORY = "llm_toolkit/providers"

    # This allows ComfyUI to invalidate the node if inputs that matter change.
    @classmethod
    def IS_CHANGED(cls, llm_model: str, context: Any = None):
        # Model name is the only changing factor that should trigger a recompute.
        return llm_model

    # ---------------------------------------------------------------------
    # Main execution
    # ---------------------------------------------------------------------
    def configure_openai(self, llm_model: str, context: Any = None) -> Tuple[Any]:
        """Build a provider‑config dictionary for OpenAI and merge with `context`."""
        logger.info("OpenAIProviderNode: configuring model '%s'", llm_model)

        # ------------------------------------------------------------------
        # Resolve API key (env / .env). If not present we return an error but
        # still allow the graph to proceed so the user gets feedback.
        # ------------------------------------------------------------------
        api_key = ""
        try:
            api_key = get_api_key("OPENAI_API_KEY", self.PROVIDER_NAME)
            logger.info("OpenAIProviderNode: API key fetched from environment / .env")
        except ValueError as e:
            logger.warning("OpenAIProviderNode: could not retrieve OPENAI_API_KEY – %s", e)
            # Leave api_key=""; downstream node will handle auth error.

        # ------------------------------------------------------------------
        # Prepare provider configuration dictionary
        # ------------------------------------------------------------------
        # Forward the selected model name verbatim so downstream nodes receive it.
        # If the user did not pick a model (empty string), we leave it empty and
        # Generate_Text will choose a sensible default.
        provider_config = {
            "provider_name": self.PROVIDER_NAME,
            "llm_model": llm_model.strip(),  # pass through as‑is
            "api_key": api_key,
            "base_ip": None,
            "port": None,
        }

        # ------------------------------------------------------------------
        # Merge with incoming `context` object to preserve pipeline data
        # ------------------------------------------------------------------
        if context is not None:
            if isinstance(context, dict):
                context["provider_config"] = provider_config
                output = context
            else:
                output = {"provider_config": provider_config, "passthrough_data": context}
        else:
            # When no incoming context, start a new one and attach provider_config under the
            # well-known key expected by downstream nodes (e.g. GenerateImage, GenerateText).
            # Previously we returned the provider_config dictionary directly which meant
            # downstream nodes could not find it via context.get("provider_config", {}).
            output = {"provider_config": provider_config}

        return (output,)


# -------------------------------------------------------------------------
# ComfyUI Node registration helpers
# -------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "OpenAIProviderNode": OpenAIProviderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIProviderNode": "OpenAI Provider (LLMToolkit)",
} 