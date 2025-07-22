import sys
import json
import logging
from typing import Optional, Union, List, Any

# Ensure repository root is on sys.path for context_payload import
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Initialize logger
logger = logging.getLogger(__name__)

# Helper to extract context from payload objects
from context_payload import extract_context

def _remove_thinking_tags(text: str) -> str:
    """Remove <think>...</think> or ◁think▷...◁/think▷ blocks from text."""
    import re
    # Pattern to match <think>...</think> or ◁think▷...◁/think▷ blocks
    pattern = r'<think>.*?</think>|◁think▷.*?◁/think▷'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    # Clean up any extra whitespace/newlines left behind
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Replace multiple newlines with double
    return cleaned.strip()

class Display_Text:
    """
    Displays text extracted from a wildcard input type, typically containing LLM responses.
    Allows selecting specific lines for output while passing through the original data structure.
    """
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("*", {}), # Accept wildcard input
                "select": ("STRING", {
                    "default": "0",
                    "tooltip": "Select which line to output (cycles through available lines)"
                }),
                "hide_thinking": ("BOOLEAN", {"default": True, "tooltip": "Hide model thinking process (content between <think> tags)"})
            },
            "optional": {},
            "hidden": {},
        }

    # Output the original 'context' data first, then the processed text outputs
    RETURN_TYPES = ("*", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("context", "text_list", "count", "selected", "text_full")
    OUTPUT_IS_LIST = (False, True, False, False, False) # text_list is the only list output
    FUNCTION = "display_llm_text"
    OUTPUT_NODE = True
    CATEGORY = "llm_toolkit/utils/text" # Changed category to llm_toolkit

    def display_llm_text(self, context: Any, select: str, hide_thinking: bool):
        # --- Safe conversion for 'select' input string ---
        select_int = 0 # Default value
        try:
            # Attempt conversion only if string is not empty after stripping whitespace
            stripped_select = select.strip()
            if stripped_select:
                select_int = int(stripped_select)
            else:
                logger.debug("Received empty string for 'select' input. Using default value 0.")
                # select_int remains 0
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid numeric value received for 'select': '{select}'. Using default 0. Error: {e}")
            select_int = 0
        # --- End safe conversion ---

        text_to_display = "" # Default to empty string
        
        # --- Extract dict from ContextPayload if necessary ---
        if not isinstance(context, dict):
            # Try to unwrap using helper (returns {} if nothing found)
            unwrapped = extract_context(context)
            if unwrapped:
                context_dict = unwrapped
            else:
                context_dict = None
        else:
            context_dict = context

        # --- Text Extraction Logic ---
        if isinstance(context_dict, dict):
            if "llm_response" in context_dict and isinstance(context_dict["llm_response"], str):
                text_to_display = context_dict["llm_response"]
                logger.info("Extracted text from 'llm_response' key.")
            # Add fallbacks for other common keys if needed
            elif "response" in context_dict and isinstance(context_dict["response"], str):
                text_to_display = context_dict["response"]
                logger.info("Extracted text from 'response' key.")
            elif "text" in context_dict and isinstance(context_dict["text"], str):
                text_to_display = context_dict["text"]
                logger.info("Extracted text from 'text' key.")
            elif "content" in context_dict and isinstance(context_dict["content"], str):
                text_to_display = context_dict["content"]
                logger.info("Extracted text from 'content' key.")
            else:
                logger.warning(f"Could not find a standard text key ('llm_response', 'response', 'text', 'content') in input dict. Stringifying the dict for display.")
                try:
                    # Pretty print the dict if possible
                    text_to_display = json.dumps(context_dict, indent=2)
                except TypeError:
                    text_to_display = str(context_dict) # Fallback stringification
        elif isinstance(context, str):
            text_to_display = context
            logger.info("Input is a string.")
        elif isinstance(context, list):
            # Try to join if list of strings, otherwise stringify
            if all(isinstance(item, str) for item in context):
                text_to_display = "\n".join(context)
                logger.info("Input is a list of strings, joined with newline.")
            else:
                logger.warning("Input is a list with non-string elements. Stringifying.")
                text_to_display = str(context)
        elif context is not None:
            logger.warning(f"Input is of unexpected type {type(context)}. Stringifying.")
            text_to_display = str(context)
        else: # context is None
             logger.warning("Input 'context' is None. Displaying empty string.")
             text_to_display = ""
        # --- End Text Extraction ---

        # Apply thinking tag removal if requested
        if hide_thinking and text_to_display:
            text_to_display = _remove_thinking_tags(text_to_display)

        # Use the extracted text for display logic
        if text_to_display is None: # Should not happen with default ""
             logger.error("text_to_display is None unexpectedly.")
             text_to_display = ""

        print("======================")
        print("Display_Text Output:")
        print("----------------------")
        print(text_to_display)
        print("======================")

        # Initialize variables for line processing
        text_list = []

        # Split the extracted text into lines
        if isinstance(text_to_display, str):
            text_list = [line.strip() for line in text_to_display.split('\n') if line.strip()]
        else:
            # This case should ideally not be reached if extraction works correctly
            logger.error(f"text_to_display is not a string after extraction: {type(text_to_display)}")
            text_list = []

        count = len(text_list)

        # Select line using modulo to handle cycling
        if count == 0:
            selected = text_to_display # If no lines, selected is the whole (potentially empty) text
        else:
            # Ensure select is non-negative before modulo
            select_index = max(0, select_int)
            selected = text_list[select_index % count]

        # Prepare UI update - always use a list of strings for the UI text widget
        ui_text = [text_to_display]

        # Return UI update and the multiple outputs, including the original 'context'
        return {
            "ui": {"string": ui_text},
            "result": (
                context,         # Pass through the original input data (payload or dict)
                text_list,   # List of individual lines as separate string outputs
                count,       # Number of lines
                selected,    # Selected line based on select input
                text_to_display # Full extracted text
            )
        }

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "Display_Text": Display_Text # Renamed class
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Display_Text": "Display Text (LLMToolkit)" # Renamed display name
}
# --- End Node Mappings --- 