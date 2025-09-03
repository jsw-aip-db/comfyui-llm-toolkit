# api_key_input.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from context_payload import extract_context

logger = logging.getLogger(__name__)

class APIKeyInput:
    """
    A node to input API keys for various providers through context.
    This allows dynamic API key management without hardcoding keys in provider nodes.
    The API key is passed through context and can override provider-specific keys.
    """
    
    SUPPORTED_PROVIDERS = [
        "openai", "gemini", "google", "anthropic", "groq", "huggingface", 
        "bfl", "wavespeed", "ollama", "lmstudio", "textgen", "kobold", 
        "llamacpp", "vllm", "transformers", "custom"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (cls.SUPPORTED_PROVIDERS, {
                    "default": "gemini", 
                    "tooltip": "Select the provider this API key is for"
                }),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": "", 
                    "tooltip": "Enter the API key for the selected provider"
                }),
            },
            "optional": {
                "context": ("*", {"tooltip": "Optional context to merge with"}),
                "override_existing": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "Whether to override existing API keys in context/provider config"
                }),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "set_api_key"
    CATEGORY = "🔗llm_toolkit/config"

    def set_api_key(
        self, 
        provider: str, 
        api_key: str, 
        context: Optional[Dict[str, Any]] = None,
        override_existing: bool = False
    ) -> Tuple[Dict[str, Any]]:
        """
        Sets an API key for a specific provider in the context.
        
        Args:
            provider: The provider name (e.g., 'gemini', 'openai')
            api_key: The API key string
            context: Existing context to merge with
            override_existing: Whether to override existing API keys
        
        Returns:
            Updated context with API key information
        """
        logger.info(f"APIKeyInput: Setting API key for provider '{provider}'")
        
        # Validate inputs
        if not api_key or not api_key.strip():
            logger.warning(f"APIKeyInput: Empty API key provided for provider '{provider}'")
        
        api_key = api_key.strip()
        
        # Initialize or copy context
        if context is None:
            output_context = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            # Handle ContextPayload or other formats
            unwrapped = extract_context(context)
            if isinstance(unwrapped, dict):
                output_context = unwrapped.copy()
                output_context.setdefault("passthrough_data", context)
            else:
                output_context = {"passthrough_data": context}

        # Initialize api_keys section in context
        if "api_keys" not in output_context:
            output_context["api_keys"] = {}
        
        # Check if we should set the key
        existing_key = output_context["api_keys"].get(provider)
        
        if existing_key and not override_existing:
            # Secure logging: only show first 5 characters
            masked_existing = existing_key[:5] + "..." if len(existing_key) > 5 else "..."
            masked_new = api_key[:5] + "..." if len(api_key) > 5 else "..."
            logger.info(f"APIKeyInput: Provider '{provider}' already has API key ({masked_existing}), keeping existing (override=False)")
        else:
            # Set the new key
            output_context["api_keys"][provider] = api_key
            
            # Secure logging: only show first 5 characters
            masked_key = api_key[:5] + "..." if len(api_key) > 5 else "..."
            action = "overriding" if existing_key else "setting"
            logger.info(f"APIKeyInput: {action} API key for provider '{provider}' ({masked_key})")
        
        # For backward compatibility, also update provider_config if it exists and matches
        provider_config = output_context.get("provider_config")
        if provider_config and isinstance(provider_config, dict):
            config_provider = provider_config.get("provider_name", "").lower()
            
            # Check if this API key is for the current provider config
            if config_provider == provider.lower() or (provider == "google" and config_provider == "gemini"):
                if override_existing or not provider_config.get("api_key") or provider_config.get("api_key") == "1234":
                    provider_config["api_key"] = api_key
                    logger.info(f"APIKeyInput: Updated provider_config API key for '{config_provider}'")
        
        logger.info(f"APIKeyInput: Context now contains API keys for: {list(output_context['api_keys'].keys())}")
        
        return (output_context,)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "APIKeyInput": APIKeyInput
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "APIKeyInput": "API Key Input (🔗LLMToolkit)"
}