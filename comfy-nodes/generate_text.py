# generate_text.py
import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import ComfyUI-specific modules
try:
    import folder_paths
except ImportError:
    logger.warning("Could not import folder_paths. Make sure ComfyUI environment is set up.")
    folder_paths = None

# Check for required dependencies
missing_deps = []
try:
    import aiohttp
except ImportError:
    missing_deps.append("aiohttp")

if missing_deps:
    logger.warning(f"Missing dependencies: {', '.join(missing_deps)}. Some functionality may not work.")
    logger.warning("Please install missing dependencies: pip install " + " ".join(missing_deps))

# Import the send_request function for making API calls to different providers
from send_request import send_request, run_async

class LLMToolkitTextGenerator:
    """
    Simple text generator with minimal UI controls.
    Uses the local Ollama daemon by default and dynamically lists the models
    that are currently installed on the machine.  If the Ollama daemon cannot
    be reached, it falls back to a single default model (``llama3``).
    """

    DEFAULT_PROVIDER = "ollama"

    # ------------------------------------------------------------------
    # Dynamically query the local Ollama daemon for installed models.
    # ------------------------------------------------------------------
    @staticmethod
    def _query_local_ollama_models(base_ip: str = "localhost", port: Union[str, int] = 11434) -> List[str]:
        """Return a list of model names installed in the local Ollama daemon.

        Parameters
        ----------
        base_ip : str
            IP address where the Ollama daemon is listening (default ``localhost``).
        port : str | int
            Port of the Ollama daemon (default ``11434``).

        Returns
        -------
        List[str]
            The list of model names or an empty list if an error occurs.
        """
        try:
            import requests  # local import to avoid dependency when unused
            url = f"http://{base_ip}:{port}/api/tags"
            logger.debug(f"Querying Ollama models via {url}")
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()

            data = resp.json()
            # Two possible shapes have been observed in the wild:
            # 1. {"models": [{"name": "llama3"}, {...}, ...]}
            # 2. [ {"name": "llama3"}, {...} ]
            if isinstance(data, dict) and "models" in data:
                data = data["models"]

            if isinstance(data, list):
                models = [m.get("name") or m for m in data]
                # Filter out context ``None`` or empty names and keep tags
                cleaned: List[str] = []
                for entry in models:
                    if not entry:
                        continue
                    # Keep the full entry, including tag (e.g., "llama3:latest")
                    cleaned.append(entry)
                # Sort models alphabetically for consistency
                cleaned.sort()
                return cleaned
        except Exception as exc:
            logger.warning(f"Unable to query local Ollama models: {exc}")
        # On error return empty list so the caller can fall back
        return []

    # Fetch models once at class‑definition time
    _INSTALLED_OLLAMA_MODELS: List[str] = _query_local_ollama_models.__func__()  # type: ignore[misc]

    # Fallback model list if the daemon could not be reached
    _FALLBACK_MODELS: List[str] = ["gemma3:1b"]

    MODEL_LIST: List[str] = _INSTALLED_OLLAMA_MODELS or _FALLBACK_MODELS

    # The default model should always be the first element of MODEL_LIST
    DEFAULT_MODEL: str = MODEL_LIST[0]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODEL_LIST, {"default": cls.DEFAULT_MODEL}),
                "prompt": ("STRING", {"multiline": True, "default": "Write a short story about a robot learning to paint."})
            },
            "optional": {
                "context": ("*", {})  # Accept context input for maximum flexibility
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "generate"
    CATEGORY = "llm_toolkit"

    def generate(self, model, prompt, context=None):
        """
        Generate text using OpenAI by default or another provider if specified in provider_config.
        Only uses the prompt and model from UI unless overridden by provider_config.
        
        Provider config can be contained within the 'context' input parameter.
        """
        try:
            # Initialize with defaults
            params = {
                "llm_provider": self.DEFAULT_PROVIDER,
                "llm_model": model,
                "system_message": "You are a helpful, creative, and concise assistant.",
                "user_message": prompt,
                "base_ip": "localhost",  # default Ollama daemon
                "port": "11434",
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": None,
                "keep_alive": True,
                "messages": []
            }
            
            # Extract provider config from 'context' input if available
            provider_config = None
            if context is not None:
                # Case 1: 'context' is itself a provider config dictionary
                if isinstance(context, dict) and "provider_name" in context:
                    provider_config = context
                    logger.info("Found provider config directly in 'context' input")
                
                # Case 2: 'context' is a dictionary containing a provider_config key
                elif isinstance(context, dict) and "provider_config" in context:
                    provider_config = context["provider_config"]
                    logger.info("Found provider_config inside 'context' dictionary")
            
            # Override with provider_config if available
            if provider_config and isinstance(provider_config, dict):
                # Extract and map provider config parameters
                if "provider_name" in provider_config:
                    params["llm_provider"] = provider_config["provider_name"]
                if "model_name" in provider_config and provider_config["model_name"]:
                    params["llm_model"] = provider_config["model_name"]
                if "api_key" in provider_config:
                    params["llm_api_key"] = provider_config["api_key"]
                if "base_ip" in provider_config:
                    params["base_ip"] = provider_config["base_ip"]
                if "port" in provider_config:
                    params["port"] = provider_config["port"]
                # Extract context additional parameters that might be in the config
                for key in provider_config:
                    if key not in ["provider_name", "model_name", "api_key", "base_ip", "port"]:
                        params[key] = provider_config[key]
                        
                # If user_prompt is in provider_config, override the prompt from UI
                if "user_prompt" in provider_config:
                    params["user_message"] = provider_config["user_prompt"]
            elif provider_config:
                logger.warning(f"provider_config is not a dictionary, it's a {type(provider_config)}. Using defaults.")
            
            # Log request details (hide API key)
            log_params = {**params}
            if "llm_api_key" in log_params:
                log_params["llm_api_key"] = "****" if log_params["llm_api_key"] else "None"
            logger.info(f"Making LLM request with params: {log_params}")
            
            # Always dispatch via send_request for modularity
            try:
                logger.info(f"Dispatching via send_request for provider {params['llm_provider']}")
                response_data = run_async(
                    send_request(
                        llm_provider=params["llm_provider"],
                        base_ip=params.get("base_ip", "localhost"),
                        port=params.get("port", "11434"),
                        images=None,
                        llm_model=params["llm_model"],
                        system_message=params["system_message"],
                        user_message=params["user_message"],
                        messages=params["messages"],
                        seed=params.get("seed"),
                        temperature=params["temperature"],
                        max_tokens=params["max_tokens"],
                        random=params.get("random", False),
                        top_k=params["top_k"],
                        top_p=params["top_p"],
                        repeat_penalty=params["repeat_penalty"],
                        stop=params.get("stop"),
                        keep_alive=params.get("keep_alive", True),
                        llm_api_key=params.get("llm_api_key"),
                    )
                )
            except Exception as e:
                logger.error(f"Error in send_request call: {e}", exc_info=True)
                response_data = {"choices": [{"message": {"content": f"Error calling send_request: {str(e)}"}}]}
            
            # Extract the response content
            if response_data is None:
                content = "Error: Received None response from send_request"
                logger.error(content)
            elif isinstance(response_data, dict):
                if "choices" in response_data and response_data["choices"]:
                    message = response_data["choices"][0].get("message", {})
                    content = message.get("content", "")
                    if content is None:
                        content = "Error: Null content in response message"
                        logger.error(f"Response has null content: {str(response_data)}")
                elif "response" in response_data:
                    content = response_data["response"]
                else:
                    content = f"Error: Unexpected response format: {str(response_data)}"
                    logger.error(content)
            elif isinstance(response_data, str):
                content = response_data
            else:
                content = f"Error: Unexpected response type: {type(response_data)}"
                logger.error(content)
            
            # Update the 'context' output with the response
            if context is not None and isinstance(context, dict):
                # Add our response to the dict without overwriting the existing data
                context_out = context.copy()
                context_out["llm_response"] = content
                context_out["llm_raw_response"] = response_data
            else:
                # If 'context' was not a dict or was None, create a new dict with our response
                context_out = {
                    "llm_response": content,
                    "llm_raw_response": response_data,
                    "passthrough_data": context
                }
            
            # Return only the updated 'context' structure
            return (context_out,)
            
        except Exception as e:
            error_message = f"Error generating text: {str(e)}"
            logger.error(error_message, exc_info=True)
            
            # Return error information within the 'context' structure if possible
            error_output = {
                "error": error_message,
                "original_input": context
            }
            return (error_output,)


# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LLMToolkitTextGenerator": LLMToolkitTextGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMToolkitTextGenerator": "Generate Text (LLMToolkit)"
}