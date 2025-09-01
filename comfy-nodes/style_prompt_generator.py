import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import logging

# Ensure parent directory in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

# ------------------------- Style Loading -------------------------

# Cache loaded styles
_styles_cache: Optional[Dict[str, Dict]] = None
_style_names_cache: Optional[List[str]] = None
_styles_path: Optional[Path] = None
_load_error: Optional[str] = None


def get_styles_path() -> Path:
    """Get the path to styles.json."""
    global _styles_path
    if _styles_path is None:
        project_root = Path(current_dir).parent
        _styles_path = project_root / "presets" / "styles.json"
    return _styles_path


def get_style_names() -> List[str]:
    """Get just the style names without loading full style data - faster for UI."""
    global _style_names_cache, _load_error
    
    if _style_names_cache is not None:
        return _style_names_cache
    
    if _load_error is not None:
        return ["Error: " + _load_error]
    
    try:
        styles_path = get_styles_path()
        if not styles_path.exists():
            _load_error = "styles.json not found"
            return ["Error: " + _load_error]
        
        # Quick parse just for style names
        with open(styles_path, "r", encoding="utf-8") as f:
            styles_data = json.load(f)
            # Sort styles alphabetically for consistent dropdown order
            sorted_styles = sorted(styles_data, key=lambda x: x.get("style", ""))
            _style_names_cache = [entry["style"] for entry in sorted_styles if isinstance(entry, dict) and entry.get("style")]
            logger.info(f"Loaded {len(_style_names_cache)} style names")
            return _style_names_cache
    except Exception as e:
        _load_error = str(e)
        logger.error(f"Error loading style names: {e}")
        return ["Error: " + _load_error]


def load_styles() -> Dict[str, Dict]:
    """Lazily load full style data only when needed."""
    global _styles_cache, _load_error
    
    if _styles_cache is not None:
        return _styles_cache
    
    try:
        styles_path = get_styles_path()
        if not styles_path.exists():
            raise FileNotFoundError(f"styles.json not found at {styles_path}")
        
        with open(styles_path, "r", encoding="utf-8") as f:
            # Sort styles alphabetically for consistent dropdown order
            styles_data = sorted(json.load(f), key=lambda x: x.get("style", ""))
            _styles_cache = {entry["style"]: entry for entry in styles_data}
            logger.debug(f"Loaded {len(_styles_cache)} styles (full data)")
            return _styles_cache
    except Exception as e:
        logger.error(f"Error loading styles: {e}")
        _load_error = str(e)
        return {}

# ------------------------- Prompt Formatting Helpers -------------------------

def _lit(d: dict) -> str:
    """Formats the lighting dictionary into a string."""
    base = f'{d.get("type", "ambient")} at {d.get("intensity", "medium")} intensity, {d.get("direction", "uniform")}'
    opts = []
    if d.get("accent_colors"):
        opts.append("accent colors " + ", ".join(d["accent_colors"]))
    for k in ("reflections", "refractions", "dispersion_effects", "bloom"):
        if d.get(k):
            opts.append(k.replace("_", " "))
    return base + (" with " + ", ".join(opts) if opts else "")

def _color(d: dict) -> str:
    """Formats the color scheme dictionary into a string."""
    return (
        f'primary {d.get("primary", "black")}, secondary {d.get("secondary", "white")}, '
        f'highlights {d.get("highlights", "gray")}, rim-light {d.get("rim_light", "none")}'
    )

# ------------------------- System Prompt Builder -------------------------

def build_system_prompt(style: dict) -> str:
    """Builds the final system prompt string from a style dictionary."""
    bg = style.get("background", {})
    pp_items = style.get("post_processing", {})
    pp = ", ".join(k.replace("_", " ") for k, v in pp_items.items() if v) if pp_items else "none"

    return (
        "You are a creative prompt engineer. Your mission is to analyze the provided image "
        "and generate exactly 1 distinct image transformation *instruction*.\n\n"
        "The brief:\n\n"
        f'Transform the image into the **"{style.get("style", "default")}"** style, featuring '
        f'{style.get("material", "default material")}, {style.get("surface_texture", "smooth")}. '
        f'Lighting: {_lit(style.get("lighting", {}))}. '
        f'Color scheme: {_color(style.get("color_scheme", {}))}. '
        f'Background: {bg.get("color", "neutral")} with {bg.get("texture", "plain")}. '
        f'Post-processing: {pp}.\n\n'
        "Your response must consist of exactly 1 numbered line (1-1).\n\n"
        "Each line *is* a complete, concise instruction ready for the image editing AI. "
        "Do not add any conversational text, explanations, or deviations; only the 1 instruction."
    )

# ------------------------- Node Definition -------------------------

class StylePromptGenerator:
    """
    A node that loads styles from presets/styles.json, allows selecting one,
    and generates a system prompt for an LLM based on the chosen style.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Use the optimized style name loading - doesn't load full style data
        style_names = get_style_names()
        
        return {
            "required": {
                "style": (style_names, {"default": style_names[0] if style_names else ""}),
                "output_as_text": ("BOOLEAN", {"default": False, "tooltip": "If enabled, outputs prompt as text only without adding to context"}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "system_prompt")
    FUNCTION = "generate_prompt"
    CATEGORY = "llm_toolkit/prompt"

    def generate_prompt(self, style: str, output_as_text: bool = False, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """
        Generates a system prompt based on the selected style and updates the context.
        """
        # Initialize or copy the context
        if context is None:
            output_context = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            output_context = {"passthrough_data": context}

        # Only load full style data when actually executing
        styles = load_styles()
        if not styles:
            error_message = f"Error: Failed to load styles."
            logger.error(error_message)
            if not output_as_text:
                # Ensure provider_config exists before trying to update it
                if "provider_config" not in output_context:
                    output_context["provider_config"] = {}
                output_context["provider_config"]["system_message"] = error_message
            return (output_context, error_message)
        
        selected_style_data = styles.get(style)
        if not selected_style_data:
            error_message = f"Error: Style '{style}' not found in loaded styles."
            logger.error(error_message)
            if not output_as_text:
                # Ensure provider_config exists before trying to update it
                if "provider_config" not in output_context:
                    output_context["provider_config"] = {}
                output_context["provider_config"]["system_message"] = error_message
            return (output_context, error_message)

        # Build the system prompt from the style data
        system_prompt = build_system_prompt(selected_style_data)

        if output_as_text:
            # When switch is ON: output prompt as text directly in prompt_config
            prompt_config = output_context.get("prompt_config", {})
            if not isinstance(prompt_config, dict):
                prompt_config = {}
            prompt_config["text"] = system_prompt
            output_context["prompt_config"] = prompt_config
            logger.info(f"StylePromptGenerator: Generated prompt for style '{style}' as text output.")
        else:
            # When switch is OFF (default): use the normal system_message approach
            # Update the context with the new system prompt
            # This will be picked up by downstream nodes like the TextGenerator
            provider_config = output_context.get("provider_config", {})
            if not isinstance(provider_config, dict):
                logger.warning(f"Warning: Overwriting non-dict 'provider_config' in context.")
                provider_config = {}
            provider_config["system_message"] = system_prompt
            output_context["provider_config"] = provider_config
            logger.info(f"StylePromptGenerator: Generated prompt for style '{style}' as system message.")

        return (output_context, system_prompt)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "StylePromptGenerator": StylePromptGenerator
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StylePromptGenerator": "Style Prompt Generator (LLMToolkit)"
}