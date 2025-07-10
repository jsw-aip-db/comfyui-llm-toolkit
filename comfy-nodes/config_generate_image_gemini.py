# config_generate_image_gemini.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

from context_payload import extract_context

# Ensure parent directory is in path if running standalone for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

class ConfigGenerateImageGemini:
    """
    Configures parameters specifically for Gemini/Imagen image generation.
    """
    # Gemini/Imagen specific options
    ASPECT_RATIO_OPTIONS = ["1:1", "3:4", "4:3", "9:16", "16:9"]
    PERSON_GENERATION_OPTIONS = ["dont_allow", "allow_adult", "allow_all"]
    SAFETY_FILTER_OPTIONS = ["block_few", "block_some", "block_most"]
    LANGUAGE_OPTIONS = ["auto", "en", "es-MX", "ja-JP", "zh-CN", "hi-IN"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                # Common parameters
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Number of images (1-4). Imagen 4 Ultra only supports 1."}),
                "aspect_ratio": (cls.ASPECT_RATIO_OPTIONS, {"default": "1:1", "tooltip": "Aspect ratio for generated images."}),
                
                # Imagen specific
                "person_generation": (cls.PERSON_GENERATION_OPTIONS, {"default": "allow_adult", "tooltip": "Policy for generating people in images."}),
                "safety_filter_level": (cls.SAFETY_FILTER_OPTIONS, {"default": "block_some", "tooltip": "Content safety filter level."}),
                "language": (cls.LANGUAGE_OPTIONS, {"default": "auto", "tooltip": "Language hint for generation (best results with listed languages)."}),
                
                # Gemini native specific
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Generation temperature for Gemini native models."}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768, "step": 1, "tooltip": "Max tokens for Gemini native generation response."}),
                
                # Seed
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Seed for reproducible generation (-1 for random)."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/image/gemini"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        Adds Gemini/Imagen-specific image generation parameters to the context.
        """
        logger.info("ConfigGenerateImageGemini executing...")

        # Initialize or copy the context
        if context is None:
            output_context = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            # Try to unwrap ContextPayload
            unwrapped = extract_context(context)
            if isinstance(unwrapped, dict):
                output_context = unwrapped.copy()
                output_context.setdefault("passthrough_data", context)
            else:
                output_context = {"passthrough_data": context}

        # Initialize generation_config dictionary
        generation_config = output_context.get("generation_config", {})
        if not isinstance(generation_config, dict):
            generation_config = {}

        # Add Gemini/Imagen parameters
        generation_config['n'] = kwargs.get('n', 1)
        generation_config['aspect_ratio'] = kwargs.get('aspect_ratio', '1:1')
        generation_config['person_generation'] = kwargs.get('person_generation', 'allow_adult')
        generation_config['safety_filter_level'] = kwargs.get('safety_filter_level', 'block_some')
        
        language = kwargs.get('language', 'auto')
        if language != 'auto':
            generation_config['language'] = language
        
        # Gemini native specific (store with _gemini suffix for clarity)
        generation_config['temperature_gemini'] = kwargs.get('temperature', 0.7)
        generation_config['max_tokens_gemini'] = kwargs.get('max_tokens', 8192)
        
        # Seed handling
        seed_val = kwargs.get('seed', -1)
        if seed_val != -1:
            generation_config['seed'] = seed_val

        # Add the config to the main context
        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImageGemini: Updated context with generation_config")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateImageGemini": ConfigGenerateImageGemini
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateImageGemini": "Configure Image Generation - Gemini (LLMToolkit)"
} 