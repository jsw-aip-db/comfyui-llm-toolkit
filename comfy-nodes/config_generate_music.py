# config_generate_music.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

# Helper to unwrap ContextPayload when a raw payload object is passed instead of dict
from context_payload import extract_context

# Ensure parent directory is on path (supports running file standalone for testing)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import model list from Suno provider – keep import local to avoid circular deps if provider imports utils
try:
    from suno_provider import SunoProviderSelector  # noqa: E402 – placed after sys.path tweak

    SUNO_MODEL_OPTIONS = SunoProviderSelector.SUNO_MODELS  # ["V3_5", "V4", "V4_5"]
except Exception:
    # Fallback – if import fails (e.g., when file executed in isolation) just hard-code the list
    SUNO_MODEL_OPTIONS = ["V3_5", "V4", "V4_5"]

logger = logging.getLogger(__name__)


class ConfigGenerateMusic:
    """Attach Suno music-generation parameters to the shared *context* dict.

    This node mirrors *ConfigGenerateImage* but targets the Suno API so that downstream
    nodes (e.g. a future *GenerateMusic* node) can read a *generation_config* section
    from the pipeline context and translate it into the correct Suno API request.

    The widget options are intentionally broad to cover all Suno endpoints:
    • /generate
    • /generate/extend
    • /generate/upload-cover
    • /generate/upload-extend
    • /lyrics
    """

    # ------------------------------------------------------------------
    # ComfyUI schema helpers
    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 – ComfyUI naming convention
        # Only *context* is strictly required – every other field is optional so that
        # the same node can configure multiple endpoint variants without clutter.
        return {
            "required": {},
            "optional": {
                "context": ("*", {}),
                # Core prompt and music attributes
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Prompt / lyrics input. Optional depending on endpoint & mode.",
                    },
                ),
                "style": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Music style / genre (e.g. Jazz, Classical). Optional.",
                    },
                ),
                "title": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Track title (max 80 chars). Optional.",
                    },
                ),
                # Suno mode switches
                "custom_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable Custom Mode (requires additional fields depending on instrumental).",
                    },
                ),
                "instrumental": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Generate instrumental track (no sung lyrics).",
                    },
                ),
                # Model & negative tags
                "model": (SUNO_MODEL_OPTIONS, {"default": "V3_5", "tooltip": "Suno model version."}),
                "negative_tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Comma-separated tags/styles to avoid in generation.",
                    },
                ),
                # Callback & polling
                "callback_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "URL that Suno will POST results to (optional).",
                    },
                ),
                "poll": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, downstream node should poll Suno until complete.",
                    },
                ),
                "poll_interval": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 60,
                        "step": 1,
                        "tooltip": "Polling interval in seconds.",
                    },
                ),
                "max_poll_seconds": (
                    "INT",
                    {
                        "default": 300,
                        "min": 30,
                        "max": 1800,
                        "step": 30,
                        "tooltip": "Maximum time to poll before giving up.",
                    },
                ),
                # Upload / extend specific
                "upload_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "URL of the uploaded audio file (for /upload-cover & /upload-extend).",
                    },
                ),
                "default_param_flag": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "When true, /upload-extend uses custom parameters; false = use original.\nMatches Suno defaultParamFlag field.",
                    },
                ),
                "audio_id": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Audio ID of the source track (for /extend).",
                    },
                ),
                "continue_at": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Start time in seconds for extension endpoints.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "🔗llm_toolkit/config/music"

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def configure(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        """Merge provided kwargs into *generation_config* inside *context* dict."""
        logger.info("ConfigGenerateMusic executing…")

        # 1. Normalise incoming context (support raw ContextPayload or None)
        if context is None:
            output_context: Dict[str, Any] = {}
            logger.debug("ConfigGenerateMusic: Initialising new context dict.")
        elif isinstance(context, dict):
            output_context = context.copy()
            logger.debug("ConfigGenerateMusic: Copied existing context dict.")
        else:
            unwrapped = extract_context(context)
            if isinstance(unwrapped, dict):
                output_context = unwrapped.copy()
                output_context.setdefault("passthrough_data", context)
                logger.debug("ConfigGenerateMusic: Extracted context from payload object.")
            else:
                output_context = {"passthrough_data": context}
                logger.warning(
                    "ConfigGenerateMusic: Received non-dict context input. Wrapping in dict under 'passthrough_data'."
                )

        # 2. Get or create generation_config sub-dict
        generation_config = output_context.get("generation_config", {})
        if not isinstance(generation_config, dict):
            logger.warning(
                "ConfigGenerateMusic: Existing 'generation_config' in context is not a dict. Overwriting it."
            )
            generation_config = {}

        # 3. Transfer parameters from kwargs (ComfyUI already applies widget defaults)
        #    We simply copy everything verbatim; downstream nodes know which fields to use.
        copy_keys = [
            "prompt",
            "style",
            "title",
            "custom_mode",
            "instrumental",
            "model",
            "negative_tags",
            "callback_url",
            "poll",
            "poll_interval",
            "max_poll_seconds",
            "upload_url",
            "default_param_flag",
            "audio_id",
            "continue_at",
        ]
        for key in copy_keys:
            if key in kwargs:
                generation_config[key] = kwargs[key]

        # 4. Persist back into context
        output_context["generation_config"] = generation_config
        logger.info("ConfigGenerateMusic: Updated generation_config with keys: %s", list(generation_config.keys()))

        return (output_context,)


# ------------------------------------------------------------------
# Node registration maps expected by ComfyUI
# ------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {"ConfigGenerateMusic": ConfigGenerateMusic}
NODE_DISPLAY_NAME_MAPPINGS = {"ConfigGenerateMusic": "Configure Music Generation (🔗LLMToolkit)"} 