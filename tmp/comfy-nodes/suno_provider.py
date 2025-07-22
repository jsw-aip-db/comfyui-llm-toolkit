import logging
from typing import Any, Dict, Tuple

from llmtoolkit_utils import get_models, get_api_key

logger = logging.getLogger(__name__)


class SunoProviderSelector:
    """Dedicated provider selector for Suno music API."""

    SUNO_MODELS = ["V3_5", "V4", "V4_5"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (cls.SUNO_MODELS, {"default": "V3_5"}),
            },
            "optional": {
                "external_api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optionally supply SUNO API key directly. Otherwise SUNO_API_KEY env var will be used.",
                    },
                ),
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "select_provider"
    CATEGORY = "llm_toolkit/providers"

    def select_provider(
        self,
        llm_model: str,
        external_api_key: str = "",
        context: Any = None,
    ) -> Tuple[Any]:
        api_key = external_api_key.strip() or ""
        if not api_key:
            try:
                api_key = get_api_key("SUNO_API_KEY", "suno")
            except ValueError:
                api_key = ""

        if not api_key:
            logger.warning("SunoProviderSelector: API key not provided or found. Calls will fail until set.")

        provider_config: Dict[str, Any] = {
            "provider_name": "suno",
            "llm_model": llm_model,
            "api_key": api_key or "",
        }

        if context is not None:
            if isinstance(context, dict):
                context["provider_config"] = provider_config
                return (context,)
            else:
                wrapped = {"provider_config": provider_config, "passthrough_data": context}
                return (wrapped,)
        return (provider_config,)


NODE_CLASS_MAPPINGS = {"SunoProviderSelector": SunoProviderSelector}
NODE_DISPLAY_NAME_MAPPINGS = {"SunoProviderSelector": "Suno Provider Selector (LLMToolkit)"} 