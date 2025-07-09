# config_generate_image_bfl.py
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

class ConfigGenerateImageBFL:
    """
    Configures parameters specifically for BFL Flux Kontext Max image generation.
    """
    # BFL specific options
    ASPECT_RATIO_OPTIONS = ["1:1", "3:4", "4:3", "9:16", "16:9", "21:9", "9:21"]
    OUTPUT_FORMAT_OPTIONS = ["png", "jpeg"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                # BFL parameters
                "aspect_ratio": (cls.ASPECT_RATIO_OPTIONS, {"default": "1:1", "tooltip": "Aspect ratio between 21:9 and 9:21."}),
                "prompt_upsampling": ("BOOLEAN", {"default": False, "tooltip": "Automatically enhance prompt for more creative generation."}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1, "tooltip": "Safety tolerance (0=strict, 6=relaxed). Limit of 2 for image editing."}),
                "output_format": (cls.OUTPUT_FORMAT_OPTIONS, {"default": "png", "tooltip": "Output image format."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Seed for reproducible generation (-1 for random)."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/bfl"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        Adds BFL-specific image generation parameters to the context.
        """
        logger.info("ConfigGenerateImageBFL executing...")

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

        # Add BFL parameters
        generation_config['aspect_ratio'] = kwargs.get('aspect_ratio', '1:1')
        generation_config['prompt_upsampling'] = kwargs.get('prompt_upsampling', False)
        generation_config['safety_tolerance'] = kwargs.get('safety_tolerance', 2)
        generation_config['output_format_bfl'] = kwargs.get('output_format', 'png')
        
        # Seed handling
        seed_val = kwargs.get('seed', -1)
        if seed_val != -1:
            generation_config['seed'] = seed_val

        # BFL always generates 1 image at a time
        generation_config['n'] = 1

        # Add the config to the main context
        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImageBFL: Updated context with generation_config")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateImageBFL": ConfigGenerateImageBFL
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateImageBFL": "Configure Image Generation - BFL (LLMToolkit)"
} 