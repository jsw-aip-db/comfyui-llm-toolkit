# config_generate_image.py
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

class ConfigGenerateImage:
    """
    Configures parameters for image generation APIs (like DALL-E, GPT-Image)
    and adds them to a generation_config dictionary within the context.
    """
    # Define options based on Images API documentation
    # Use 'auto' as default where applicable, let backend handle specifics
    SIZE_OPTIONS_GPT = ["auto", "1024x1024", "1024x1536", "1536x1024"]
    SIZE_OPTIONS_DALLE2 = ["256x256", "512x512", "1024x1024"]
    SIZE_OPTIONS_DALLE3 = ["1024x1024", "1792x1024", "1024x1792"]
    # Combine for flexibility, or make dynamic later if needed
    ALL_SIZE_OPTIONS = sorted(list(set(SIZE_OPTIONS_GPT + SIZE_OPTIONS_DALLE2 + SIZE_OPTIONS_DALLE3)))

    QUALITY_OPTIONS_GPT = ["auto", "low", "medium", "high"]
    QUALITY_OPTIONS_DALLE3 = ["standard", "hd"]
    QUALITY_OPTIONS_DALLE2 = ["standard"] # Only one option
    ALL_QUALITY_OPTIONS = sorted(list(set(QUALITY_OPTIONS_GPT + QUALITY_OPTIONS_DALLE3 + QUALITY_OPTIONS_DALLE2)))

    STYLE_OPTIONS_DALLE3 = ["vivid", "natural"]

    BACKGROUND_OPTIONS_GPT = ["auto", "opaque", "transparent"]
    OUTPUT_FORMAT_OPTIONS_GPT = ["png", "jpeg", "webp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {}, # No required inputs, reads from context or uses defaults
            "optional": {
                "context": ("*", {}),
                # Common parameters
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "Number of images (1-10). DALL-E 3 only supports 1."}),
                "size": (cls.ALL_SIZE_OPTIONS, {"default": "1024x1024", "tooltip": "Image dimensions. Supported sizes vary by model."}),
                "response_format": (["url", "b64_json"], {"default": "b64_json", "tooltip": "Return format (b64_json recommended for ComfyUI). gpt-image-1 always uses b64_json."}),
                "user": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional user ID for moderation tracking."}),

                # DALL-E 3 specific
                "quality_dalle3": (cls.QUALITY_OPTIONS_DALLE3, {"default": "standard", "tooltip": "DALL-E 3 quality (standard/hd)."}),
                "style_dalle3": (cls.STYLE_OPTIONS_DALLE3, {"default": "vivid", "tooltip": "DALL-E 3 style (vivid/natural)."}),

                # GPT-Image-1 specific
                "quality_gpt": (cls.QUALITY_OPTIONS_GPT, {"default": "auto", "tooltip": "GPT-Image-1 quality (low/medium/high/auto)."}),
                "background_gpt": (cls.BACKGROUND_OPTIONS_GPT, {"default": "auto", "tooltip": "GPT-Image-1 background (opaque/transparent/auto)."}),
                "output_format_gpt": (cls.OUTPUT_FORMAT_OPTIONS_GPT, {"default": "png", "tooltip": "GPT-Image-1 output format (png/jpeg/webp)."}),
                "moderation_gpt": (["auto", "low"], {"default": "auto", "tooltip": "GPT-Image-1 content moderation level."}),
                "output_compression_gpt": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1, "tooltip": "GPT-Image-1 compression (0-100) for webp/jpeg."}),

                # Seed (Common but may not be supported by all APIs)
                # "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "tooltip": "Seed for generation (-1 for random). API support varies."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/utils"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        Adds image generation parameters to the context.
        """
        logger.info("ConfigGenerateImage executing...")

        # Initialize or copy the context
        if context is None:
            output_context = {}
            logger.debug("ConfigGenerateImage: Initializing new context.")
        elif isinstance(context, dict):
            output_context = context.copy()
            logger.debug("ConfigGenerateImage: Copied input context.")
        else:
            # Try to unwrap ContextPayload
            unwrapped = extract_context(context)
            if isinstance(unwrapped, dict):
                output_context = unwrapped.copy()
                output_context.setdefault("passthrough_data", context)
                logger.debug("ConfigGenerateImage: Unwrapped context from payload object.")
            else:
                output_context = {"passthrough_data": context}
                logger.warning("ConfigGenerateImage: Received non-dict context input. Wrapping it.")

        # Initialize generation_config dictionary
        generation_config = output_context.get("generation_config", {})
        if not isinstance(generation_config, dict):
            logger.warning("ConfigGenerateImage: Existing 'generation_config' in context is not a dict. Overwriting.")
            generation_config = {}

        # Add parameters from kwargs to the config, applying defaults from INPUT_TYPES if needed
        # Note: We don't need separate defaults here as ComfyUI handles widget defaults
        generation_config['n'] = kwargs.get('n', 1)
        generation_config['size'] = kwargs.get('size', '1024x1024')
        generation_config['response_format'] = kwargs.get('response_format', 'b64_json')
        user = kwargs.get('user', "").strip()
        if user: generation_config['user'] = user

        # DALL-E 3 specific (using distinct names to avoid clashes)
        generation_config['quality_dalle3'] = kwargs.get('quality_dalle3', 'standard')
        generation_config['style_dalle3'] = kwargs.get('style_dalle3', 'vivid')

        # GPT-Image-1 specific (using distinct names)
        generation_config['quality_gpt'] = kwargs.get('quality_gpt', 'auto')
        generation_config['background_gpt'] = kwargs.get('background_gpt', 'auto')
        generation_config['output_format_gpt'] = kwargs.get('output_format_gpt', 'png')
        generation_config['moderation_gpt'] = kwargs.get('moderation_gpt', 'auto')
        generation_config['output_compression_gpt'] = kwargs.get('output_compression_gpt', 100)

        # Seed handling (optional, API support varies)
        # seed_val = kwargs.get('seed', -1)
        # if seed_val != -1:
        #     generation_config['seed'] = seed_val

        # Add the config to the main context
        output_context["generation_config"] = generation_config
        logger.info(f"ConfigGenerateImage: Updated context with generation_config: {generation_config}")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateImage": ConfigGenerateImage
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateImage": "Configure Image Generation (LLMToolkit)"
} 