#!/usr/bin/env python3
"""
Simple test script to verify that the JS and Python integration for model fetching works.
Run this script from the command line with:
    python test_js_integration.py
"""
import os
import sys
import json
import logging
import requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import utility functions if possible
try:
    from utils import get_api_key, get_models
    import_success = True
except ImportError as e:
    logger.error(f"Could not import required functions: {e}")
    import_success = False


def test_model_fetch(provider, base_ip="localhost", port="11434", api_key=None):
    """Test fetching models for a given provider."""
    if not import_success:
        return f"ERROR: Required modules not imported. Cannot run test."
    
    try:
        if not api_key:
            try:
                api_key_name = f"{provider.upper()}_API_KEY"
                api_key = get_api_key(api_key_name, provider)
                logger.info(f"Using API key from environment for {provider}")
            except Exception as e:
                logger.warning(f"Could not get API key for {provider}: {e}")
                api_key = None
        
        logger.info(f"Fetching models for {provider} (IP: {base_ip}, Port: {port}, API Key: {'Present' if api_key else 'None'})")
        
        models = get_models(provider, base_ip, port, api_key)
        
        if models:
            logger.info(f"SUCCESS: Found {len(models)} models for {provider}: {models[:5]}" + 
                       (f" and {len(models)-5} more..." if len(models) > 5 else ""))
            return models
        else:
            logger.warning(f"WARNING: No models found for {provider}")
            return []
    
    except Exception as e:
        logger.error(f"ERROR testing {provider}: {str(e)}")
        return f"ERROR: {str(e)}"


def test_endpoint(provider, base_ip="localhost", port="11434", api_key=None):
    """Test the ComfyUI API endpoint for fetching models."""
    logger.info("Testing ComfyUI API endpoint - this requires ComfyUI to be running")
    
    try:
        url = "http://localhost:8188/ComfyLLMToolkit/get_provider_models"
        data = {
            "llm_provider": provider,
            "base_ip": base_ip,
            "port": port,
            "external_api_key": api_key or ""
        }
        
        response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            models = response.json()
            logger.info(f"API Endpoint SUCCESS: Found {len(models)} models for {provider}")
            return models
        else:
            logger.error(f"API Endpoint ERROR: Status code {response.status_code}")
            return f"ERROR: Status code {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        logger.error("API Endpoint ERROR: Could not connect to ComfyUI server. Is it running?")
        return "ERROR: ConnectionError - Is ComfyUI running?"
    except Exception as e:
        logger.error(f"API Endpoint ERROR: {str(e)}")
        return f"ERROR: {str(e)}"


if __name__ == "__main__":
    # Test a few providers
    providers_to_test = ["openai", "ollama"]
    
    print("\n=== Testing Direct Model Fetching ===")
    for provider in providers_to_test:
        result = test_model_fetch(provider)
        print(f"{provider}: {'SUCCESS' if isinstance(result, list) else 'FAILED'}")
    
    print("\n=== Testing API Endpoint (requires ComfyUI running) ===")
    for provider in providers_to_test:
        result = test_endpoint(provider)
        print(f"{provider}: {'SUCCESS' if isinstance(result, list) else 'FAILED'}")
    
    print("\nTests completed. Check logs for details.") 