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
try:
    # Try module-level import in multiple ways
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Trying to import send_request from parent_dir: {parent_dir}")
    
    # Ensure parent directory is in the path
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        # Try direct import first
        from send_request import send_request, run_async
        logger.info("Successfully imported send_request module directly")
    except ImportError as e1:
        logger.warning(f"Direct import failed: {e1}, trying alternative approaches")
        
        try:
            # Try with the parent module name
            module_name = os.path.basename(parent_dir)
            import_stmt = f"from {module_name}.send_request import send_request, run_async"
            logger.info(f"Trying: {import_stmt}")
            
            # Dynamic import
            module = __import__(f"{module_name}.send_request", fromlist=["send_request", "run_async"])
            send_request = getattr(module, "send_request")
            run_async = getattr(module, "run_async")
            logger.info("Successfully imported via module name")
            
        except (ImportError, AttributeError) as e2:
            logger.error(f"All import methods failed: {e2}")
            
            # Last resort: define stubs
            logger.critical("Using stub functions for send_request and run_async")
            def send_request(*args, **kwargs):
                logger.error("Stub send_request called - real function not found")
                return {"choices": [{"message": {"content": "Error: send_request module not found"}}]}
            
            def run_async(coroutine):
                logger.error("Stub run_async called - real function not found")
                return None

except Exception as e:
    logger.error(f"Unexpected error during import: {e}", exc_info=True)
    def send_request(*args, **kwargs):
        return {"choices": [{"message": {"content": f"Error: Exception during import: {str(e)}"}}]}
    def run_async(coroutine):
        return None

class LLMToolkitTextGenerator:
    """
    Simple text generator with minimal UI controls.
    Uses OpenAI's gpt-4o-mini by default, but can be overridden via provider_config.
    """
    # Default model and provider
    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # List of models for the dropdown
    MODEL_LIST = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "gpt-4o-2024-05-13"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (cls.MODEL_LIST, {"default": "gpt-4o-mini"}),
                "prompt": ("STRING", {"multiline": True, "default": "Write a short story about a robot learning to paint."})
            },
            "optional": {
                "any": ("*", {})  # Accept any input for maximum flexibility
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("any",)
    FUNCTION = "generate"
    CATEGORY = "llm_toolkit"

    def generate(self, model, prompt, any=None):
        """
        Generate text using OpenAI by default or another provider if specified in provider_config.
        Only uses the prompt and model from UI unless overridden by provider_config.
        
        Provider config can be contained within the 'any' input parameter.
        """
        try:
            # Initialize with defaults
            params = {
                "llm_provider": self.DEFAULT_PROVIDER,
                "llm_model": model,
                "system_message": "You are a helpful, creative, and concise assistant.",
                "user_message": prompt,
                "base_ip": "localhost",
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
            
            # Extract provider config from 'any' input if available
            provider_config = None
            if any is not None:
                # Case 1: 'any' is itself a provider config dictionary
                if isinstance(any, dict) and "provider_name" in any:
                    provider_config = any
                    logger.info("Found provider config directly in 'any' input")
                
                # Case 2: 'any' is a dictionary containing a provider_config key
                elif isinstance(any, dict) and "provider_config" in any:
                    provider_config = any["provider_config"]
                    logger.info("Found provider_config inside 'any' dictionary")
            
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
                # Extract any additional parameters that might be in the config
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
            
            # Check if we're using OpenAI - implement direct API call if needed
            if params["llm_provider"] == "openai" and params.get("llm_api_key"):
                logger.info("Detected OpenAI provider with API key, using direct implementation")
                try:
                    # Direct implementation of OpenAI API call to bypass import issues
                    import aiohttp
                    import asyncio
                    
                    async def direct_openai_request():
                        api_url = "https://api.openai.com/v1/chat/completions"
                        headers = {
                            "Authorization": f"Bearer {params.get('llm_api_key')}",
                            "Content-Type": "application/json"
                        }
                        
                        # Prepare a simplified message format
                        messages = []
                        if params["system_message"]:
                            messages.append({"role": "system", "content": params["system_message"]})
                        messages.append({"role": "user", "content": params["user_message"]})
                        
                        data = {
                            "model": params["llm_model"],
                            "messages": messages,
                            "temperature": params["temperature"],
                            "max_tokens": params["max_tokens"],
                            "presence_penalty": params["repeat_penalty"],
                            "top_p": params["top_p"],
                        }
                        
                        logger.info(f"Making direct OpenAI API call with model: {params['llm_model']}")
                        
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.post(api_url, headers=headers, json=data) as response:
                                    if response.status != 200:
                                        error_text = await response.text()
                                        logger.error(f"OpenAI API error: {response.status}, {error_text}")
                                        return {"choices": [{"message": {"content": f"Error: OpenAI API returned status {response.status}: {error_text}"}}]}
                                    
                                    response_data = await response.json()
                                    logger.info(f"OpenAI API response received: {type(response_data)}")
                                    return response_data
                        except Exception as e:
                            logger.error(f"Error in direct OpenAI API call: {e}", exc_info=True)
                            return {"choices": [{"message": {"content": f"Error in OpenAI API call: {str(e)}"}}]}
                    
                    # Create a new event loop if needed
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run the direct OpenAI call
                    response_data = loop.run_until_complete(direct_openai_request())
                    
                except Exception as e:
                    logger.error(f"Failed in direct OpenAI implementation: {e}", exc_info=True)
                    response_data = {"choices": [{"message": {"content": f"Error in direct implementation: {str(e)}"}}]}
            else:
                # Fallback to using the regular send_request method
                try:
                    logger.info(f"Making API call via send_request for {params['llm_provider']}")
                    response_data = run_async(
                        send_request(
                            llm_provider=params["llm_provider"],
                            base_ip=params["base_ip"],
                            port=params["port"],
                            images=None,  # No image processing as per requirements
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
                            llm_api_key=params.get("llm_api_key")
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
            
            # Update the 'any' output with the response
            if any is not None and isinstance(any, dict):
                # Add our response to the dict without overwriting the existing data
                any_out = any.copy()
                any_out["llm_response"] = content
                any_out["llm_raw_response"] = response_data
            else:
                # If 'any' was not a dict or was None, create a new dict with our response
                any_out = {
                    "llm_response": content,
                    "llm_raw_response": response_data,
                    "passthrough_data": any
                }
            
            # Return only the updated 'any' structure
            return (any_out,)
            
        except Exception as e:
            error_message = f"Error generating text: {str(e)}"
            logger.error(error_message, exc_info=True)
            
            # Return error information within the 'any' structure if possible
            error_output = {
                "error": error_message,
                "original_input": any
            }
            return (error_output,)


# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LLMToolkitTextGenerator": LLMToolkitTextGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMToolkitTextGenerator": "Text Generator (LLMToolkit)"
}