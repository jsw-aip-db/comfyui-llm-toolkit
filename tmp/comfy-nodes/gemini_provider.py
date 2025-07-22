# comfy-nodes/gemini_provider.py
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
    from llmtoolkit_utils import (
        get_api_key,
        get_models,
        validate_gemini_key,
    )
except ImportError as e:
    raise ImportError(
        "GeminiProviderNode: failed to import helpers from llmtoolkit_utils – "
        + str(e)
    )

logger = logging.getLogger(__name__)


class GeminiProviderNode:
    """Dedicated ComfyUI node that prepares *provider_config* for Gemini.

    The node fetches available Gemini models (if an API key is available),
    exposes them as a dropdown, and returns a context dictionary containing a
    *provider_config* block so that downstream generator nodes can send the
    request without having to know any Gemini specifics.
    """

    # Fallback list used when live fetching fails or no API key is present.
    _DEFAULT_MODELS: List[str] = [
        # Text models
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-pro",
        "gemini-2.0-flash",
        "gemini-pro",
        "gemini-pro-vision",
        # Native Gemini image generation
        "gemini-2.0-flash-preview-image-generation",
        # Imagen models (stable versions as per docs)
        "imagen-3.0-generate-002",
        # Imagen models (preview versions)
        "imagen-4.0-generate-preview-06-06",
        "imagen-4.0-ultra-generate-preview-06-06",
        # Legacy naming (for compatibility)
        "imagen-3.0-generate-001",
        "imagen-4.0-generate-001",
        "imagen-4.0-ultra-generate-001",
        "imagen-3.0-generate-preview-06-06",
        "imagen-3-light-alpha",
    ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def _fetch_models(cls, api_key: Optional[str]) -> List[str]:
        """Return list of model names, falling back to defaults on error."""
        try:
            models = get_models("gemini", base_ip="localhost", port="0", api_key=api_key)
            if models and isinstance(models, list):
                return models
        except Exception as e:
            logger.warning(f"GeminiProviderNode: model fetch failed – {e}")
        return cls._DEFAULT_MODELS

    # ------------------------------------------------------------------
    # ComfyUI node interface
    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        # Attempt to fetch models using env key at import time so the dropdown
        # has a useful list even before the user enters a key in the UI.
        api_key_env = os.getenv("GEMINI_API_KEY", "").strip()
        model_list = cls._fetch_models(api_key_env) or cls._DEFAULT_MODELS
        default_model = model_list[0] if model_list else "gemini-pro"
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
                        "tooltip": "Gemini API key (optional – overrides env/.env)",
                    },
                ),
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "select_gemini"
    CATEGORY = "llm_toolkit/providers"

    # ------------------------------------------------------------------
    # IS_CHANGED → forces UI to refresh dropdown when API key changes
    # ------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, llm_model, external_api_key="", context=None):
        import hashlib

        key_hash = hashlib.md5(external_api_key.encode()).hexdigest() if external_api_key else "nokey"
        state = f"{llm_model}-{key_hash}"
        return hashlib.md5(state.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def select_gemini(
        self,
        llm_model: str,
        external_api_key: str = "",
        context: Optional[Any] = None,
    ) -> Tuple[Any]:
        """Validate key, build *provider_config*, and merge it into context."""

        external_api_key = external_api_key.strip()
        final_api_key = ""

        # 1) Validate external key if supplied
        if external_api_key:
            if validate_gemini_key(external_api_key):
                final_api_key = external_api_key
                logger.info("GeminiProviderNode: using valid external API key")
            else:
                logger.warning(
                    "GeminiProviderNode: provided external API key failed validation"
                )

        # 2) Fallback to env or .env file
        if not final_api_key:
            try:
                final_api_key = get_api_key("GEMINI_API_KEY", "gemini")
            except ValueError:
                final_api_key = ""

        # 3) Allow dummy key for local/offline testing so other providers stay unaffected
        if not final_api_key:
            final_api_key = "1234"

        provider_config: Dict[str, Any] = {
            "provider_name": "gemini",
            "llm_model": llm_model,
            "api_key": final_api_key,
            "base_ip": None,  # Cloud provider – no IP/port
            "port": None,
        }

        # Merge with incoming context if present
        if context is not None:
            if isinstance(context, dict):
                merged = context.copy()
                merged["provider_config"] = provider_config
                result: Any = merged
            else:
                result = {"provider_config": provider_config, "passthrough_data": context}
        else:
            # Always return a proper context dictionary
            result = {"provider_config": provider_config}

        return (result,)


# ------------------------------------------------------------------
# Node registration – discovered automatically by ComfyUI
# ------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"GeminiProviderNode": GeminiProviderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiProviderNode": "Gemini Provider Selector (LLMToolkit)"} 