# llm_provider_selector.py
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Model Mappings ---
# Group models by provider for easier management and UI population
MODEL_MAPPING = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "ollama": [
        "llama3",
        "qwen:14b",
        "qwen-vl:7b",
        "llava",
        "bakllava",
        "gemma:7b",
        "mistral",
        "mixtral",
    ],
    "qwen": [
        "Qwen/Qwen-VL-3B-Instruct",
        "Qwen/Qwen-VL-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "Qwen/Qwen3-4B-AWQ",
        "Qwen/Qwen3-8B-AWQ",   
        "SoybeanMilk/Kimi-VL-A3B-Thinking-2506-BNB-4bit",
    ],
    # For local transformers, the list is discovered dynamically
    "transformers": ["Discovering local models..."],
}

class LLMProviderSelector:
    """
    A node that allows the user to select an LLM provider and a model from a
    pre-defined list. It then outputs a provider_config dictionary that can be
    used by other nodes in the toolkit.
    """
    # Provider names should be lowercase for consistency
    PROVIDERS = sorted(list(MODEL_MAPPING.keys()))

    @classmethod
    def INPUT_TYPES(cls):
        # Flatten the model list for the initial view, but we'll update it dynamically
        all_models = [model for sublist in MODEL_MAPPING.values() for model in sublist]
        
        return {
            "required": {
                "llm_provider": (cls.PROVIDERS, {"default": cls.PROVIDERS[0]}),
                "llm_model": (all_models, {}),
                "base_ip": ("STRING", {"default": "localhost"}),
                "port": ("STRING", {"default": "11434"}),
                "external_api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure_provider"
    CATEGORY = "llm_toolkit/providers"

    def configure_provider(
        self,
        llm_provider: str,
        llm_model: str,
        base_ip: str,
        port: str,
        external_api_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any]]:
        """
        Creates a provider_config dictionary and merges it into the context.
        """
        logger.info(f"Configuring LLM Provider: {llm_provider}, Model: {llm_model}")

        provider_config = {
            "provider_name": llm_provider.lower().strip(),
            "llm_model": llm_model.strip(),
            "base_ip": base_ip.strip() if base_ip else None,
            "port": port.strip() if port else None,
            "api_key": external_api_key.strip() if external_api_key else None,
        }

        # Merge with incoming context to retain pipeline data
        if context is not None and isinstance(context, dict):
            context_out = context.copy()
            context_out["provider_config"] = provider_config
        else:
            context_out = {"provider_config": provider_config, "passthrough_data": context}
        
        return (context_out,)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Simple change detection: if any input changes, re-run.
        return float("nan")

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "LLMProviderSelector": LLMProviderSelector
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMProviderSelector": "LLM Provider Selector (LLMToolkit)"
} 