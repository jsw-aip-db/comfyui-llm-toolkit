# llmtoolkit_providers.py
import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add ComfyUI directory to path if necessary (adjust path as needed)
comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for required dependencies
missing_deps = []
try:
    import requests
except ImportError:
    missing_deps.append("requests")
try:
    import yaml
except ImportError:
    missing_deps.append("pyyaml")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    missing_deps.append("python-dotenv")
    
if missing_deps:
    logger.warning(f"Missing dependencies: {', '.join(missing_deps)}. Some functionality may not work.")
    logger.warning("Please install missing dependencies: pip install " + " ".join(missing_deps))

# --- Imports from your existing project ---
try:
    # Add parent directory to sys.path THIS TIME to ensure utils is found
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logger.info(f"Added parent directory to sys.path for utils import: {parent_dir}")
    
    # Now import directly from the renamed file
    from llmtoolkit_utils import get_models
    logger.info("Successfully imported functions from main llmtoolkit_utils.py")

except ImportError as e:
    logger.error(f"CRITICAL ERROR: Could not import required functions from llmtoolkit_utils: {e}")
    logger.error("Please ensure llmtoolkit_utils.py exists in the parent directory and has no import errors itself.")
    # Removed dummy functions - raise error instead
    raise ImportError(f"Failed to import functions from llmtoolkit_utils: {e}")
# --- End Imports ---

try:
    import folder_paths # ComfyUI specific import
except ImportError:
    print("Error: Could not import folder_paths. Make sure ComfyUI environment is set up.")
    folder_paths = None

# --- ComfyUI Server Integration ---
try:
    from server import PromptServer
    from aiohttp import web

    # Define the endpoint when the module is imported
    @PromptServer.instance.routes.post("/ComfyLLMToolkit/get_provider_models")
    async def get_llmtoolkit_provider_models_endpoint(request):
        """API endpoint for the frontend to fetch available models for a selected provider."""
        try:
            data = await request.json()
            llm_provider = data.get("llm_provider")
            base_ip = data.get("base_ip", "localhost")
            port = data.get("port", "11434")
            
            # Enhanced logging for debugging
            logger.info(f"API /ComfyLLMToolkit/get_provider_models called with provider: {llm_provider}, ip: {base_ip}, port: {port}")
            print(f"Fetching models for {llm_provider} at {base_ip}:{port}")

            # For local providers, `get_models` doesn't need a real key. A placeholder is fine.
            api_key = "1234"

            # --- Call get_models from utils ---
            models = get_models(llm_provider, base_ip, port, api_key)
            model_count = len(models) if models else 0
            logger.info(f"Fetched {model_count} models for {llm_provider}.")
            print(f"Retrieved {model_count} models for {llm_provider}: {models[:5]}{'...' if model_count > 5 else ''}")
            
            # Ensure we're returning a valid JSON array
            if not models or not isinstance(models, list):
                models = ["No models found"]
                logger.warning(f"No valid models returned for {llm_provider}, using fallback.")
            
            return web.json_response(models)

        except Exception as e:
            logger.error(f"Error in /ComfyLLMToolkit/get_provider_models endpoint: {str(e)}", exc_info=True)
            print(f"Error in get_llmtoolkit_provider_models_endpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            return web.json_response(["Error fetching models"], status=500)

    # Print startup confirmation
    logger.info("ComfyUI-LLM-Toolkit API routes registered!")

except (ImportError, AttributeError) as e:
    logger.warning(f"ComfyUI PromptServer or aiohttp not available. Dynamic model fetching from UI will not work. Error: {e}")
# --- End ComfyUI Server Integration ---


class LLMToolkitProviderSelector:
    """
    Selects the LLM Provider and Model for local-only services and outputs a
    configuration dictionary for downstream generator nodes.
    """
    SUPPORTED_PROVIDERS = [
        "llamacpp", "ollama", "kobold", "lmstudio", "textgen", "vllm"
    ]
    REQUIRES_IP_PORT = ["ollama", "llamacpp", "kobold", "lmstudio", "textgen", "vllm"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_provider": (cls.SUPPORTED_PROVIDERS, {"default": "ollama"}),
                "llm_model": ("STRING", {"default": "Provider not selected or models not fetched", "tooltip": "Select the model. Updates when provider/connection changes."}),
            },
            "optional": {
                "base_ip": ("STRING", {"default": "localhost", "tooltip": "IP address for local providers"}),
                "port": ("STRING", {"default": "11434", "tooltip": "Port for local providers"}),
                "context": ("*", {}),  # Accept context input type for maximum flexibility
            }
        }

    # Define the output type as a generic wildcard that will contain provider config
    RETURN_TYPES = ("*",)
    # Define the output name
    RETURN_NAMES = ("context",)

    FUNCTION = "select_provider"
    CATEGORY = "llm_toolkit/providers"

    @classmethod
    def IS_CHANGED(cls, llm_provider, llm_model, base_ip="localhost", port="11434", context=None):
        """Check if inputs that affect model list or validity have changed."""
        import hashlib
        
        # Track connection details for local providers
        connection_details = ""
        if llm_provider in cls.REQUIRES_IP_PORT:
            connection_details = f"-{base_ip}-{port}"

        # Create a unique state hash based only on provider and connection
        state = f"{llm_provider}{connection_details}"
        logger.debug(f"IS_CHANGED computing state hash from: {state}")
        
        # Return a hash that will change whenever these inputs change
        state_hash = hashlib.md5(state.encode()).hexdigest()
        logger.debug(f"IS_CHANGED hash: {state_hash}")
        return state_hash

    def select_provider(self, llm_provider: str, llm_model: str, base_ip: str, port: str, context=None) -> Tuple[Any]:
        """
        Outputs a provider config dictionary for the selected local provider.
        """
        logger.info(f"ProviderNode executing for: {llm_provider} / {llm_model}")
        
        # API Key is not used for local providers, so we use a placeholder.
        final_api_key = "1234"
        logger.info(f"Using placeholder API key for local provider '{llm_provider}'.")

        # Model Selection Handling
        if not llm_model or llm_model == "Provider not selected or models not fetched":
             logger.warning(f"No valid model selected for {llm_provider}. Using empty string.")
             llm_model_out = ""
        else:
            llm_model_out = llm_model
            logger.info(f"Passing selected model '{llm_model_out}' for provider '{llm_provider}'.")

        # Determine relevant IP/Port
        final_base_ip = base_ip if llm_provider in self.REQUIRES_IP_PORT else None
        final_port = port if llm_provider in self.REQUIRES_IP_PORT else None

        # Create the provider config dictionary
        provider_config = {
            "provider_name": llm_provider,
            "llm_model": llm_model_out,
            "api_key": final_api_key,
            "base_ip": final_base_ip,
            "port": final_port
        }

        # Prepare output
        if context is not None:
            if isinstance(context, dict):
                context["provider_config"] = provider_config
                result = context
                logger.info(f"Merged provider_config into existing dict")
            else:
                result = {
                    "provider_config": provider_config,
                    "passthrough_data": context
                }
                logger.info(f"Wrapped non-dict 'context' input with provider_config")
        else:
            result = provider_config
            logger.info(f"Using provider_config directly as output")
        
        logger.info(f"ProviderNode returning config type: {type(result)}")
        
        # Return the combined result
        return (result,)


# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "LLMToolkitProviderSelector": LLMToolkitProviderSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMToolkitProviderSelector": "LLM Provider Selector (LLMToolkit)"
}
# --- End Node Mappings ---