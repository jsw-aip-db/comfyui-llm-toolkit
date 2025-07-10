# config_generate_image_split.py
"""Provider-specific image generation configuration nodes.

These nodes provide cleaner interfaces by only showing parameters relevant
to each specific provider, avoiding confusion from universal parameter sets.
"""

import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

from context_payload import extract_context

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)


class ConfigGenerateImageOpenAI:
    """Configuration node specifically for OpenAI/DALL-E image generation."""
    
    SIZE_OPTIONS_DALLE2 = ["256x256", "512x512", "1024x1024"]
    SIZE_OPTIONS_DALLE3 = ["1024x1024", "1792x1024", "1024x1792"]
    ALL_SIZE_OPTIONS = sorted(list(set(SIZE_OPTIONS_DALLE2 + SIZE_OPTIONS_DALLE3)))
    
    QUALITY_OPTIONS_DALLE3 = ["standard", "hd"]
    STYLE_OPTIONS_DALLE3 = ["vivid", "natural"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                # Common parameters
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "Number of images (1-10). DALL-E 3 only supports 1."}),
                "size": (cls.ALL_SIZE_OPTIONS, {"default": "1024x1024", "tooltip": "Image dimensions. Supported sizes vary by model."}),
                "response_format": (["url", "b64_json"], {"default": "b64_json", "tooltip": "Return format (b64_json recommended for ComfyUI)."}),
                "user": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional user ID for tracking."}),
                
                # DALL-E 3 specific
                "quality": (cls.QUALITY_OPTIONS_DALLE3, {"default": "standard", "tooltip": "DALL-E 3 quality setting."}),
                "style": (cls.STYLE_OPTIONS_DALLE3, {"default": "vivid", "tooltip": "DALL-E 3 style setting."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/image/openai"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateImageOpenAI executing...")
        
        # Initialize or copy context
        if context is None:
            output_context = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            unwrapped = extract_context(context)
            if isinstance(unwrapped, dict):
                output_context = unwrapped.copy()
                output_context.setdefault("passthrough_data", context)
            else:
                output_context = {"passthrough_data": context}
        
        # Initialize generation_config
        generation_config = output_context.get("generation_config", {})
        if not isinstance(generation_config, dict):
            generation_config = {}
        
        # Add OpenAI-specific parameters
        generation_config['n'] = kwargs.get('n', 1)
        generation_config['size'] = kwargs.get('size', '1024x1024')
        generation_config['response_format'] = kwargs.get('response_format', 'b64_json')
        
        user = kwargs.get('user', "").strip()
        if user:
            generation_config['user'] = user
        
        # DALL-E 3 specific
        generation_config['quality_dalle3'] = kwargs.get('quality', 'standard')
        generation_config['style_dalle3'] = kwargs.get('style', 'vivid')
        
        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImageOpenAI: Updated context with config: {generation_config}")
        
        return (output_context,)


class ConfigGenerateImageGemini:
    """Configuration node specifically for Gemini/Imagen image generation."""
    
    ASPECT_RATIO_OPTIONS = ["1:1", "3:4", "4:3", "9:16", "16:9"]
    PERSON_GENERATION_OPTIONS = ["dont_allow", "allow_adult", "allow_all"]
    SAFETY_FILTER_OPTIONS = ["block_low_and_above", "block_medium_and_above", "block_high_and_above"]
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
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Seed for generation (-1 for random)."}),
                
                # Imagen specific
                "person_generation": (cls.PERSON_GENERATION_OPTIONS, {"default": "allow_adult", "tooltip": "Person generation policy (allow_all not available in EU/UK/CH/MENA)."}),
                "safety_filter_level": (cls.SAFETY_FILTER_OPTIONS, {"default": "block_low_and_above", "tooltip": "Safety filter level."}),
                # Note: API expects block_low_and_above etc.
                "language": (cls.LANGUAGE_OPTIONS, {"default": "auto", "tooltip": "Language hint for generation."}),
                
                # Gemini native specific
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1, "tooltip": "Generation temperature (Gemini native only)."}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768, "step": 1, "tooltip": "Max tokens (Gemini native only)."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/image/gemini"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateImageGemini executing...")
        
        # Initialize or copy context
        if context is None:
            output_context = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            unwrapped = extract_context(context)
            if isinstance(unwrapped, dict):
                output_context = unwrapped.copy()
                output_context.setdefault("passthrough_data", context)
            else:
                output_context = {"passthrough_data": context}
        
        # Initialize generation_config
        generation_config = output_context.get("generation_config", {})
        if not isinstance(generation_config, dict):
            generation_config = {}
        
        # Add Gemini/Imagen-specific parameters
        generation_config['n'] = kwargs.get('n', 1)
        generation_config['aspect_ratio'] = kwargs.get('aspect_ratio', '1:1')
        generation_config['person_generation'] = kwargs.get('person_generation', 'allow_adult')
        generation_config['safety_filter_level'] = kwargs.get('safety_filter_level', 'block_some')
        
        language = kwargs.get('language', 'auto')
        if language != 'auto':
            generation_config['language'] = language
        
        # Gemini native specific
        generation_config['temperature_gemini'] = kwargs.get('temperature', 0.7)
        generation_config['max_tokens_gemini'] = kwargs.get('max_tokens', 8192)
        
        # Seed handling
        seed_val = kwargs.get('seed', -1)
        if seed_val != -1:
            generation_config['seed'] = seed_val
        
        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImageGemini: Updated context with config: {generation_config}")
        
        return (output_context,)


class ConfigGenerateImageBFL:
    """Configuration node specifically for BFL Flux Kontext Max image generation."""
    
    ASPECT_RATIO_OPTIONS = ["1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]
    OUTPUT_FORMAT_OPTIONS = ["png", "jpeg"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                # BFL specific parameters
                "aspect_ratio": (cls.ASPECT_RATIO_OPTIONS, {"default": "1:1", "tooltip": "Aspect ratio for generated images."}),
                "prompt_upsampling": ("BOOLEAN", {"default": False, "tooltip": "Enhance prompt for more creative generation."}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1, "tooltip": "Safety tolerance (0=strict, 6=relaxed). Max 2 for editing."}),
                "output_format": (cls.OUTPUT_FORMAT_OPTIONS, {"default": "png", "tooltip": "Output image format."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Seed for generation (-1 for random)."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/image/bfl"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateImageBFL executing...")
        
        # Initialize or copy context
        if context is None:
            output_context = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            unwrapped = extract_context(context)
            if isinstance(unwrapped, dict):
                output_context = unwrapped.copy()
                output_context.setdefault("passthrough_data", context)
            else:
                output_context = {"passthrough_data": context}
        
        # Initialize generation_config
        generation_config = output_context.get("generation_config", {})
        if not isinstance(generation_config, dict):
            generation_config = {}
        
        # Add BFL-specific parameters
        generation_config['aspect_ratio'] = kwargs.get('aspect_ratio', '1:1')
        generation_config['prompt_upsampling'] = kwargs.get('prompt_upsampling', False)
        generation_config['safety_tolerance'] = kwargs.get('safety_tolerance', 2)
        generation_config['output_format_bfl'] = kwargs.get('output_format', 'png')
        
        # BFL always generates 1 image at a time
        generation_config['n'] = 1
        generation_config['response_format'] = 'b64_json'  # BFL always returns b64
        
        # Seed handling
        seed_val = kwargs.get('seed', -1)
        if seed_val != -1:
            generation_config['seed'] = seed_val
        
        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImageBFL: Updated context with config: {generation_config}")
        
        return (output_context,)


# Node Mappings
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateImageOpenAI": ConfigGenerateImageOpenAI,
    "ConfigGenerateImageGemini": ConfigGenerateImageGemini,
    "ConfigGenerateImageBFL": ConfigGenerateImageBFL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateImageOpenAI": "Configure Image Generation - OpenAI/DALL-E (LLMToolkit)",
    "ConfigGenerateImageGemini": "Configure Image Generation - Gemini/Imagen (LLMToolkit)",
    "ConfigGenerateImageBFL": "Configure Image Generation - BFL Flux (LLMToolkit)",
} 