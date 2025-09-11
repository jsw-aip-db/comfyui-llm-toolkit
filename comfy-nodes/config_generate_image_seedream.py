# config_generate_image_seedream.py
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

class ConfigGenerateImageSeedream:
    """
    Unified configuration for WaveSpeedAI's Seedream V4 and SeedEdit V3 models.
    """
    
    MODEL_FAMILIES = ["seedream_v4", "seededit_v3"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_family": (cls.MODEL_FAMILIES, {"default": "seedream_v4"}),
            },
            "optional": {
                "context": ("*", {}),
                # Seedream V4 parameters
                "size": ("STRING", {"default": "2048*2048", "tooltip": "[V4] The size of the generated media in pixels (width*height)."}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1, "tooltip": "[V4] Number of images for sequential models."}),
                
                # SeedEdit V3 parameters
                "guidance_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "[V3] Controls how strongly the model follows the editing instruction."}),
                
                # Common parameters
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "[V3 & V4] Seed for reproducible generation (-1 for random)."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "🔗llm_toolkit/config/image/wavespeed"

    def configure(self, model_family: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        Adds model-specific image generation parameters to the context based on the selected family.
        """
        logger.info(f"ConfigGenerateImageSeedream executing for model family: {model_family}")

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

        # Apply parameters based on the selected model family
        if model_family == "seedream_v4":
            generation_config['size'] = kwargs.get('size', "2048*2048")
            generation_config['max_images'] = kwargs.get('max_images', 1)
            # 'n' is a common parameter for number of images, so let's set it for v4
            generation_config['n'] = kwargs.get('max_images', 1)

        elif model_family == "seededit_v3":
            generation_config['guidance_scale'] = kwargs.get('guidance_scale', 0.5)

        # Apply common parameters
        seed_val = kwargs.get('seed', -1)
        if seed_val != -1:
            generation_config['seed'] = seed_val

        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImageSeedream: Updated context with generation_config for {model_family}: {generation_config}")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateImageSeedream": ConfigGenerateImageSeedream
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateImageSeedream": "Configure Image Gen - Seedream (🔗LLMToolkit)"
}
