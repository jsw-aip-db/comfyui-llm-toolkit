import sys
import json
import logging
from typing import Optional, Union, List, Any

# Initialize logger
logger = logging.getLogger(__name__)

class IF_LLM_DisplayText:
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
    CATEGORY = "llm_toolkit" # Changed category to llm_toolkit

    def display_llm_text(self, context: Any, select: str):
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
        
        # --- Text Extraction Logic ---
        if isinstance(context, dict):
            if "llm_response" in context and isinstance(context["llm_response"], str):
                text_to_display = context["llm_response"]
                logger.info("Extracted text from 'llm_response' key.")
            # Add fallbacks for other common keys if needed
            elif "response" in context and isinstance(context["response"], str):
                text_to_display = context["response"]
                logger.info("Extracted text from 'response' key.")
            elif "text" in context and isinstance(context["text"], str):
                text_to_display = context["text"]
                logger.info("Extracted text from 'text' key.")
            elif "content" in context and isinstance(context["content"], str):
                text_to_display = context["content"]
                logger.info("Extracted text from 'content' key.")
            else:
                logger.warning(f"Could not find a standard text key ('llm_response', 'response', 'text', 'content') in input dict. Stringifying the dict for display.")
                try:
                    # Pretty print the dict if possible
                    text_to_display = json.dumps(context, indent=2)
                except TypeError:
                    text_to_display = str(context) # Fallback stringification
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

        # Use the extracted text for display logic
        if text_to_display is None: # Should not happen with default ""
             logger.error("text_to_display is None unexpectedly.")
             text_to_display = ""

        print("======================")
        print("IF_LLM_DisplayText Output:")
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
                context,         # Pass through the original input data
                text_list,   # List of individual lines as separate string outputs
                count,       # Number of lines
                selected,    # Selected line based on select input
                text_to_display # Full extracted text
            )
        }

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "IF_LLM_DisplayText": IF_LLM_DisplayText # Renamed class
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_LLM_DisplayText": "LLM Display Text (LLMToolkit)" # Renamed display name
}
# --- End Node Mappings --- 