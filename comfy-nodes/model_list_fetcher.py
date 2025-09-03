"""
Model List Fetcher Node - For updating provider model lists
This node fetches the latest models from each provider's API and generates
Python code with hardcoded lists that can be copied into the provider nodes.
"""

import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llmtoolkit_utils import get_api_key, get_models

class ModelListFetcherNode:
    """
    A utility node for fetching and updating model lists from API providers.
    This node is used to generate hardcoded model lists for the actual provider nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        providers = [
            "openai",
            "gemini", 
            "groq",
            "anthropic",
            "cohere",
            "perplexity",
            "together",
            "mistral",
            "deepseek",
            "xai",
            "ollama",
            "huggingface",
            "openrouter"
        ]
        
        return {
            "required": {
                "provider": (providers, {"default": "openai"}),
                "fetch_models": ("BOOLEAN", {"default": True, "tooltip": "Fetch latest models from API"}),
            },
            "optional": {
                "external_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API key (optional - will use env if not provided)"
                }),
                "base_ip": ("STRING", {"default": "localhost", "tooltip": "For local providers like Ollama"}),
                "port": ("STRING", {"default": "11434", "tooltip": "For local providers like Ollama"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_list", "python_code")
    FUNCTION = "fetch_models"
    CATEGORY = "🔗llm_toolkit/utils"
    
    def fetch_models(self, provider: str, fetch_models: bool, 
                    external_api_key: str = "", base_ip: str = "localhost", 
                    port: str = "11434") -> Tuple[str, str]:
        """
        Fetch models from the provider's API and generate Python code.
        """
        
        if not fetch_models:
            return ("Fetching disabled", "# Fetching disabled")
        
        try:
            # Get API key
            api_key = external_api_key.strip() if external_api_key else ""
            
            if not api_key:
                # Try to get from environment
                env_key_map = {
                    "openai": "OPENAI_API_KEY",
                    "gemini": "GEMINI_API_KEY",
                    "groq": "GROQ_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "cohere": "COHERE_API_KEY",
                    "perplexity": "PERPLEXITY_API_KEY",
                    "together": "TOGETHER_API_KEY",
                    "mistral": "MISTRAL_API_KEY",
                    "deepseek": "DEEPSEEK_API_KEY",
                    "xai": "XAI_API_KEY",
                    "huggingface": "HUGGINGFACE_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY",
                }
                
                env_key_name = env_key_map.get(provider)
                if env_key_name:
                    try:
                        api_key = get_api_key(env_key_name, provider)
                    except:
                        api_key = ""
            
            # Fetch models
            print(f"Fetching models for {provider}...")
            if provider == "openrouter":
                try:
                    response = requests.get("https://openrouter.ai/api/v1/models")
                    response.raise_for_status()
                    data = response.json()
                    models = sorted([item['id'] for item in data['data']])
                except Exception as e:
                    print(f"Could not fetch models from OpenRouter: {e}")
                    models = [f"Error fetching models: {e}"]
            else:
                models = get_models(provider, base_ip, port, api_key)
            
            if not models:
                models = [f"No models found for {provider}"]
            
            # Sort models alphabetically
            models = sorted(models)
            
            # Generate display text
            display_text = f"=== {provider.upper()} Models (Total: {len(models)}) ===\n\n"
            for i, model in enumerate(models, 1):
                display_text += f"{i:3}. {model}\n"
            
            # Generate Python code
            python_code = self._generate_python_code(provider, models)
            
            # Save to file for easy copying
            self._save_to_file(provider, models, python_code)
            
            return (display_text, python_code)
            
        except Exception as e:
            error_msg = f"Error fetching models for {provider}: {str(e)}"
            return (error_msg, f"# {error_msg}")
    
    def _generate_python_code(self, provider: str, models: List[str]) -> str:
        """Generate Python code for the model list."""
        
        timestamp = datetime.now().isoformat()
        
        code = f'''# Auto-generated model list for {provider.upper()}
# Generated: {timestamp}
# Total models: {len(models)}

{provider.upper()}_MODELS = ['''
        
        for model in models:
            code += f'\n    "{model}",'
        
        code += f'''
]

# Usage in node:
# @classmethod
# def INPUT_TYPES(cls):
#     return {{
#         "required": {{
#             "llm_model": ({provider.upper()}_MODELS, {{"default": "{models[0] if models else 'none'}"}}),
#         }}
#     }}
'''
        
        return code
    
    def _save_to_file(self, provider: str, models: List[str], python_code: str):
        """Save the model list and code to files for easy access."""
        
        try:
            # Create output directory
            output_dir = os.path.join(parent_dir, "model_lists_output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model list as JSON
            json_file = os.path.join(output_dir, f"{provider}_models.json")
            with open(json_file, 'w') as f:
                json.dump({
                    "provider": provider,
                    "generated": datetime.now().isoformat(),
                    "models": models
                }, f, indent=2)
            
            # Save Python code
            py_file = os.path.join(output_dir, f"{provider}_models.py")
            with open(py_file, 'w') as f:
                f.write(python_code)
            
            # Also create a consolidated file with all providers
            self._update_consolidated_file(provider, models)
            
            print(f"Saved model list to: {json_file}")
            print(f"Saved Python code to: {py_file}")
            
        except Exception as e:
            print(f"Error saving files: {e}")
    
    def _update_consolidated_file(self, provider: str, models: List[str]):
        """Update the consolidated provider_models.py file."""
        
        try:
            consolidated_file = os.path.join(parent_dir, "provider_models_hardcoded.py")
            
            # Read existing file if it exists
            existing_data = {}
            if os.path.exists(consolidated_file):
                try:
                    # Parse existing file
                    with open(consolidated_file, 'r') as f:
                        content = f.read()
                        if "PROVIDER_MODELS = {" in content:
                            # Extract the dictionary
                            import ast
                            start = content.find("PROVIDER_MODELS = {")
                            end = content.find("}", start) + 1
                            dict_str = content[start:end].replace("PROVIDER_MODELS = ", "")
                            existing_data = ast.literal_eval(dict_str)
                except:
                    pass
            
            # Update with new data
            existing_data[provider] = models
            
            # Generate new consolidated file
            code = f'''"""
Hardcoded model lists for LLM Toolkit providers
Last updated: {datetime.now().isoformat()}

This file contains model lists fetched from provider APIs.
Use these lists in provider nodes to avoid API calls at startup.
"""

PROVIDER_MODELS = {{'''
            
            for prov in sorted(existing_data.keys()):
                code += f'\n    "{prov}": ['
                for model in existing_data[prov]:
                    code += f'\n        "{model}",'
                code += '\n    ],'
            
            code += '''
}

def get_models_for_provider(provider: str) -> list:
    """Get the hardcoded model list for a specific provider."""
    return PROVIDER_MODELS.get(provider, ["No models available"])

def get_all_providers() -> list:
    """Get list of all available providers."""
    return sorted(list(PROVIDER_MODELS.keys()))
'''
            
            with open(consolidated_file, 'w') as f:
                f.write(code)
            
            print(f"Updated consolidated file: {consolidated_file}")
            
        except Exception as e:
            print(f"Error updating consolidated file: {e}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "ModelListFetcherNode": ModelListFetcherNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelListFetcherNode": "Model List Fetcher (Dev Tool)"
}