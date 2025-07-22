# config_generate_image_openai.py
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
    """
    Configures parameters specifically for OpenAI's DALL-E and GPT-Image models.
    """
    # DALL-E 2/3 Options
    SIZE_OPTIONS_DALLE3 = ["1024x1024", "1792x1024", "1024x1792"]
    SIZE_OPTIONS_DALLE2 = ["256x256", "512x512", "1024x1024"]
    ALL_DALLE_SIZES = sorted(list(set(SIZE_OPTIONS_DALLE3 + SIZE_OPTIONS_DALLE2)))
    
    QUALITY_OPTIONS_DALLE3 = ["standard", "hd"]
    STYLE_OPTIONS_DALLE3 = ["vivid", "natural"]

    # GPT-Image-1 Options
    QUALITY_OPTIONS_GPT = ["auto", "low", "medium", "high"]
    BACKGROUND_OPTIONS_GPT = ["auto", "opaque", "transparent"]
    OUTPUT_FORMAT_OPTIONS_GPT = ["png", "jpeg", "webp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                # Common parameters
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "Number of images (DALL-E 3 only supports 1)."}),
                "size": (cls.ALL_DALLE_SIZES, {"default": "1024x1024", "tooltip": "Image dimensions (supported sizes vary by model)."}),
                "response_format": (["url", "b64_json"], {"default": "b64_json", "tooltip": "Return format (b64_json recommended for ComfyUI). gpt-image-1 always uses b64_json."}),
                "user": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional user ID for moderation tracking."}),

                # DALL-E 3 specific
                "quality_dalle3": (cls.QUALITY_OPTIONS_DALLE3, {"default": "standard", "tooltip": "DALL-E 3 quality (standard/hd)."}),
                "style_dalle3": (cls.STYLE_OPTIONS_DALLE3, {"default": "vivid", "tooltip": "DALL-E 3 style (vivid/natural)."}),

                # GPT-Image-1 specific
                "quality_gpt": (cls.QUALITY_OPTIONS_GPT, {"default": "auto", "tooltip": "GPT-Image-1 quality."}),
                "background_gpt": (cls.BACKGROUND_OPTIONS_GPT, {"default": "auto", "tooltip": "GPT-Image-1 background."}),
                "output_format_gpt": (cls.OUTPUT_FORMAT_OPTIONS_GPT, {"default": "png", "tooltip": "GPT-Image-1 output format."}),
                "moderation_gpt": (["auto", "low"], {"default": "auto", "tooltip": "GPT-Image-1 content moderation level."}),
                "output_compression_gpt": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1, "tooltip": "GPT-Image-1 compression (0-100) for webp/jpeg."}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/config/image/openai"

    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        logger.info("ConfigGenerateImageOpenAI executing...")

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

        # Common
        generation_config['n'] = kwargs.get('n', 1)
        generation_config['size'] = kwargs.get('size', '1024x1024')
        generation_config['response_format'] = kwargs.get('response_format', 'b64_json')
        if kwargs.get('user', "").strip():
            generation_config['user'] = kwargs.get('user').strip()

        # DALL-E 3
        generation_config['quality_dalle3'] = kwargs.get('quality_dalle3', 'standard')
        generation_config['style_dalle3'] = kwargs.get('style_dalle3', 'vivid')

        # GPT-Image-1
        generation_config['quality_gpt'] = kwargs.get('quality_gpt', 'auto')
        generation_config['background_gpt'] = kwargs.get('background_gpt', 'auto')
        generation_config['output_format_gpt'] = kwargs.get('output_format_gpt', 'png')
        generation_config['moderation_gpt'] = kwargs.get('moderation_gpt', 'auto')
        generation_config['output_compression_gpt'] = kwargs.get('output_compression_gpt', 100)

        output_context["generation_config"] = generation_config
        logger.info("ConfigGenerateImageOpenAI: Updated context with generation_config.")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "ConfigGenerateImageOpenAI": ConfigGenerateImageOpenAI
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigGenerateImageOpenAI": "Configure Image Generation - OpenAI (LLMToolkit)"
} 