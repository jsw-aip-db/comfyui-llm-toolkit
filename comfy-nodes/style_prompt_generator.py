import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ------------------------- Style Loading -------------------------

# Cache loaded styles
_styles_cache: Optional[Dict[str, Dict]] = None

def load_styles() -> Dict[str, Dict]:
    """Return a dict keyed by style name."""
    global _styles_cache
    if _styles_cache is not None:
        return _styles_cache

    # The node file is in comfy-nodes, so we go up one level to the project root
    project_root = Path(current_dir).parent
    styles_path = project_root / "presets" / "styles.json"

    if not styles_path.exists():
        raise FileNotFoundError(f"styles.json not found at {styles_path}")

    with open(styles_path, "r", encoding="utf-8") as f:
        # Sort styles alphabetically for consistent dropdown order
        styles_data = sorted(json.load(f), key=lambda x: x.get("style", ""))
        _styles_cache = {entry["style"]: entry for entry in styles_data}
        return _styles_cache

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
        f'Transform the image into the **“{style.get("style", "default")}”** style, featuring '
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
    _style_names: list[str] = []

    @classmethod
    def INPUT_TYPES(cls):
        try:
            styles = load_styles()
            cls._style_names = list(styles.keys())
        except FileNotFoundError as e:
            print(f"Error loading styles for node: {e}")
            cls._style_names = ["Error: styles.json not found"]

        return {
            "required": {
                "style": (cls._style_names, {"default": cls._style_names[0] if cls._style_names else ""}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "system_prompt")
    FUNCTION = "generate_prompt"
    CATEGORY = "llm_toolkit/prompt"

    def generate_prompt(self, style: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
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

        # Load all available styles
        styles = load_styles()
        selected_style_data = styles.get(style)

        if not selected_style_data:
            error_message = f"Error: Style '{style}' not found in loaded styles."
            print(error_message)
            # Ensure provider_config exists before trying to update it
            if "provider_config" not in output_context:
                output_context["provider_config"] = {}
            output_context["provider_config"]["system_message"] = error_message
            return (output_context, error_message)

        # Build the system prompt from the style data
        system_prompt = build_system_prompt(selected_style_data)

        # Update the context with the new system prompt
        # This will be picked up by downstream nodes like the TextGenerator
        provider_config = output_context.get("provider_config", {})
        if not isinstance(provider_config, dict):
            print(f"Warning: Overwriting non-dict 'provider_config' in context.")
            provider_config = {}

        provider_config["system_message"] = system_prompt
        output_context["provider_config"] = provider_config

        print(f"StylePromptGenerator: Generated prompt for style '{style}' and updated context.")

        return (output_context, system_prompt)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "StylePromptGenerator": StylePromptGenerator
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StylePromptGenerator": "Style Prompt Generator (LLMToolkit)"
} 