import sys
import logging
from typing import Any, Tuple

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

WILDCARD = AnyType("*")

# Initialize logger
logger = logging.getLogger(__name__)

class SwitchAny:
    """
    A switch node that takes two inputs of any type and a boolean selector.
    Outputs the first input if the boolean is True, otherwise outputs the second input.
    Useful for conditional logic in ComfyUI workflows.
    """
    def __init__(self):
        self.type = "utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selector": ("BOOLEAN", {"default": True, "tooltip": "True = output input_a, False = output input_b"}),
            },
            "optional": {
                "input_a": ("*", {"tooltip": "First input (output when selector is True)"}),
                "input_b": ("*", {"tooltip": "Second input (output when selector is False)"}),
            },
            "hidden": {},
        }

    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    OUTPUT_NODE = False  # This is a utility node, not an output node
    CATEGORY = "llm_toolkit"

    def switch(self, selector: bool, input_a: Any = None, input_b: Any = None) -> Tuple[Any]:
        """
        Switches between two inputs based on the boolean selector.
        
        Args:
            selector: Boolean that determines which input to output
            input_a: First input (returned when selector is True)
            input_b: Second input (returned when selector is False)
            
        Returns:
            Tuple containing the selected input
        """
        logger.info(f"SwitchAny node executing with selector={selector}")
        
        try:
            if selector:
                selected_output = input_a
                logger.info("SwitchAny: Selected input_a (selector=True)")
            else:
                selected_output = input_b
                logger.info("SwitchAny: Selected input_b (selector=False)")
            
            # Log the type of the selected output for debugging
            output_type = type(selected_output).__name__
            logger.info(f"SwitchAny: Output type is {output_type}")
            
            return (selected_output,)
            
        except Exception as e:
            logger.error(f"Error in SwitchAny: {e}", exc_info=True)
            # In case of error, return None
            return (None,)

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "SwitchAny": SwitchAny
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwitchAny": "Switch Any (LLMToolkit)"
} 