# comfy-nodes/config_generate_image_flux_dev.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from context_payload import extract_context

logger = logging.getLogger(__name__)

class ConfigGenerateImageFluxDev:
    """
    Configures parameters for WaveSpeedAI FLUX.1 Kontext Multi Ultra Fast [dev].
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Number of images to generate (1-4)."}),
                "size": ("STRING", {"default": "1024x1024", "multiline": False, "tooltip": "The size of the generated image (e.g., '1024x1024')."}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 50, "step": 1, "tooltip": "Number of inference steps to perform."}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "How closely the model follows the prompt."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "Seed for reproducible generation (-1 for random)."}),
                "enable_safety_checker": ("BOOLEAN", {"default": True, "tooltip": "Enable the safety checker."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/image/wavespeed"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateImageFluxDev executing...")

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

        # Add Flux Dev parameters
        generation_config['n'] = kwargs.get('n', 1)
        generation_config['size'] = kwargs.get('size', '1024x1024')
        generation_config['num_inference_steps'] = kwargs.get('num_inference_steps', 28)
        generation_config['guidance_scale'] = kwargs.get('guidance_scale', 2.5)
        generation_config['enable_safety_checker'] = kwargs.get('enable_safety_checker', True)
        
        seed_val = kwargs.get('seed', -1)
        if seed_val != -1:
            generation_config['seed'] = seed_val

        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImageFluxDev: Updated context with generation_config")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateImageFluxDev": ConfigGenerateImageFluxDev
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateImageFluxDev": "Configure Image Generation - Flux Dev (LLMToolkit)"
} 