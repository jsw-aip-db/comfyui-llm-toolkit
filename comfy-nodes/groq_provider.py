# groq_provider.py
"""Groq Provider Node for ComfyUI-LLM-Toolkit

This node mirrors *openai_provider.py* but is hard-wired to the Groq API.
It resolves the API key automatically (``GROQ_API_KEY``), exposes only the
model selector to the UI, and merges the chosen configuration into the
pipeline context so downstream generator nodes (Generate Text / Generate Image)
can issue requests via the Groq backend.
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

# -----------------------------------------------------------------------------
# Toolkit helpers
# -----------------------------------------------------------------------------
try:
    from llmtoolkit_utils import get_api_key, get_models
except ImportError as e:
    raise ImportError("GroqProviderNode: Failed to import llmtoolkit_utils."
                      " Ensure it exists and is error-free.") from e

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Optional ComfyUI server endpoint to expose dynamic model list
# -----------------------------------------------------------------------------
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/ComfyLLMToolkit/get_groq_models")
    async def get_groq_models_endpoint(request):
        """Return the list of Groq chat/completion models available for the account."""
        try:
            api_key = ""
            try:
                api_key = get_api_key("GROQ_API_KEY", GroqProviderNode.PROVIDER_NAME)
            except ValueError:
                logger.warning("get_groq_models_endpoint: GROQ_API_KEY not set; returning empty list")

            models = get_models(GroqProviderNode.PROVIDER_NAME, None, None, api_key)
            # Provide a sensible default if the list is empty
            if not models:
                models = [GroqProviderNode.PLACEHOLDER_MODEL]
            elif GroqProviderNode.PLACEHOLDER_MODEL not in models:
                models.insert(0, GroqProviderNode.PLACEHOLDER_MODEL)

            return web.json_response(models)
        except Exception as exc:
            logger.error("get_groq_models_endpoint: error %s", exc, exc_info=True)
            return web.json_response(["Error fetching models"], status=500)

    logger.info("GroqProviderNode: /ComfyLLMToolkit/get_groq_models endpoint registered")

except (ImportError, AttributeError) as e:
    # Likely running outside the ComfyUI server context
    logger.debug("GroqProviderNode: PromptServer not available (%s); endpoint not registered", e)


# -----------------------------------------------------------------------------
# Groq Provider Node definition
# -----------------------------------------------------------------------------
class GroqProviderNode:
    """A minimal provider-configuration node dedicated to Groq."""

    PROVIDER_NAME = "groq"
    # Reasonable default placeholder (updated when Groq model names evolve)
    PLACEHOLDER_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Backend sees a STRING; frontend JS converts it to a dynamic combo box
                "llm_model": (
                    "STRING",
                    {
                        "default": cls.PLACEHOLDER_MODEL,
                        "tooltip": "Groq model name. Populated dynamically from your account.",
                    },
                ),
            },
            "optional": {
                "context": ("*", {}),  # Pass-through pipeline data
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure_groq"
    CATEGORY = "llm_toolkit/providers"

    @classmethod
    def IS_CHANGED(cls, llm_model: str, context: Any = None):
        """Let ComfyUI know when the node output must be recomputed."""
        return llm_model  # Only the selected model affects the output

    # ---------------------------------------------------------------------
    # Main execution
    # ---------------------------------------------------------------------
    def configure_groq(self, llm_model: str, context: Any = None) -> Tuple[Any]:
        logger.info("GroqProviderNode: configuring model '%s'", llm_model)

        # --------------------------------------------------------------
        # Resolve API key (env / .env). Continue with empty string if missing
        # so downstream nodes can raise a clear authentication error.
        # --------------------------------------------------------------
        api_key = ""
        try:
            api_key = get_api_key("GROQ_API_KEY", self.PROVIDER_NAME)
            # Masked logging to avoid leaking full key
            # Show only first 5 characters when debug logging is enabled
            masked_key = (api_key[:5] + "…") if api_key else ""  # never more than 5 chars
            logger.debug("GroqProviderNode: retrieved GROQ_API_KEY (%s).", masked_key)
            logger.info("GroqProviderNode: API key fetched from environment / .env")
        except ValueError as err:
            logger.warning("GroqProviderNode: could not retrieve GROQ_API_KEY – %s", err)

        # --------------------------------------------------------------
        # Build provider configuration dictionary
        # --------------------------------------------------------------
        provider_config = {
            "provider_name": self.PROVIDER_NAME,
            "llm_model": llm_model.strip(),  # forward exact value (may be empty)
            "api_key": api_key,
            "base_ip": None,
            "port": None,
        }

        # --------------------------------------------------------------
        # Merge with existing context if present
        # --------------------------------------------------------------
        if context is not None:
            if isinstance(context, dict):
                context["provider_config"] = provider_config
                output = context
            else:
                output = {"provider_config": provider_config, "passthrough_data": context}
        else:
            output = {"provider_config": provider_config}

        return (output,)


# -----------------------------------------------------------------------------
# Node registration
# -----------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "GroqProviderNode": GroqProviderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqProviderNode": "Groq Provider (LLMToolkit)",
} 