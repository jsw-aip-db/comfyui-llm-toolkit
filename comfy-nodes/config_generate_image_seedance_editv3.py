# config_generate_image_seedance_editv3.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

# Ensure parent directory is in path if running standalone for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from context_payload import extract_context

logger = logging.getLogger(__name__)

class ConfigGenerateImageSeedanceEditV3:
    """
    Configures parameters specifically for WaveSpeedAI SeedEdit V3 image editing.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "guidance_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Controls how strongly the model follows the editing instruction. Higher values mean stronger edits."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "Seed for reproducible generation (-1 for random)."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/image/wavespeed"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        Adds SeedEdit V3-specific image generation parameters to the context.
        """
        logger.info("ConfigGenerateImageSeedanceEditV3 executing...")

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

        generation_config = output_context.get("generation_config", {})
        if not isinstance(generation_config, dict):
            generation_config = {}

        # Add SeedEdit V3 parameters
        generation_config['guidance_scale'] = kwargs.get('guidance_scale', 0.5)
        
        seed_val = kwargs.get('seed', -1)
        if seed_val != -1:
            generation_config['seed'] = seed_val

        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImageSeedanceEditV3: Updated context with generation_config")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateImageSeedanceEditV3": ConfigGenerateImageSeedanceEditV3
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateImageSeedanceEditV3": "Configure Image Generation - SeedEdit v3 (LLMToolkit)"
} 