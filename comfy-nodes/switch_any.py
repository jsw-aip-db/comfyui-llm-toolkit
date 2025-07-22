import sys
import logging
from typing import Any, Tuple

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

WILDCARD = AnyType("*")
# Alias used for readability
any = WILDCARD

# Initialize logger
logger = logging.getLogger(__name__)

class SwitchAny:
    """
    A switch node that takes two inputs of any type and a boolean selector.
    Outputs the first input if the boolean is True, otherwise outputs the second input.
    Useful for conditional logic in ComfyUI workflows.
    """
    def __init__(self):
        self.type = "llm_toolkit/utils/logic"

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
    CATEGORY = "llm_toolkit/utils/logic"

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


# ---------------------------------------------------------------------------
#  Original SwitchAnyRoute (two outputs, maintains exact type)
# ---------------------------------------------------------------------------


class SwitchAnyRoute:
    """
    Reverse switch node that takes a single input (any type) and a boolean selector.
    Routes the input to either `output_true` or `output_false` so that downstream
    nodes can branch based on the boolean flag.
    """

    def __init__(self):
        self.type = "llm_toolkit/utils/logic"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selector": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "True = route to output_true, False = output_false",
                    },
                ),
                "input": ("*", {"tooltip": "Input to be routed"}),
            },
            "hidden": {},
        }

    RETURN_TYPES = (WILDCARD, WILDCARD)
    RETURN_NAMES = ("output_true", "output_false")
    FUNCTION = "route"
    OUTPUT_NODE = False
    CATEGORY = "llm_toolkit/utils/logic"

    def route(self, selector: bool, input: Any):  # noqa: D401
        logger.info("SwitchAnyRoute executing with selector=%s", selector)
        try:
            if selector:
                return (input, None)
            return (None, input)
        except Exception as e:
            logger.error("Error in SwitchAnyRoute: %s", e, exc_info=True)
            return (None, None)


# ===============================================================
#  _wANY variants – preserve input/output dynamic typing
# ===============================================================


class SwitchAny_wANY:
    """
    Identical to SwitchAny but declares inputs/outputs using the `any` wildcard
    instance so ComfyUI treats the data type dynamically (avoids validation
    errors when routing IMAGE/LATENT/etc.).
    """

    def __init__(self):
        self.type = "llm_toolkit/utils/logic"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selector": ("BOOLEAN", {"default": True, "tooltip": "True = output input_a, False = output input_b"}),
            },
            "optional": {
                "input_a": (any, {"tooltip": "First input to output when selector is True"}),
                "input_b": (any, {"tooltip": "Second input to output when selector is False"}),
            },
            "hidden": {},
        }

    RETURN_TYPES = (WILDCARD,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    OUTPUT_NODE = False
    CATEGORY = "llm_toolkit/utils/logic"

    def switch(self, selector: bool, input_a: Any = None, input_b: Any = None):  # type: ignore[override]
        logger.info(f"SwitchAny_wANY executing with selector={selector}")
        try:
            selected_output = input_a if selector else input_b
            logger.info("SwitchAny_wANY selected %s", "input_a" if selector else "input_b")
            return (selected_output,)
        except Exception as e:
            logger.error("Error in SwitchAny_wANY: %s", e, exc_info=True)
            return (None,)


class SwitchAnyRoute_wANY:
    """
    Reverse switch that routes a single input to two outputs using wildcard type.
    """

    def __init__(self):
        self.type = "llm_toolkit/utils/logic"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selector": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "True = route to output_true, False = output_false",
                    },
                ),
                "input": (any, {"tooltip": "Input to be routed"}),
            },
            "hidden": {},
        }

    RETURN_TYPES = (WILDCARD, WILDCARD)
    RETURN_NAMES = ("output_true", "output_false")
    FUNCTION = "route"
    OUTPUT_NODE = False
    CATEGORY = "llm_toolkit/utils/logic"

    def route(self, selector: bool, input: Any):  # type: ignore[override]
        logger.info("SwitchAnyRoute_wANY executing with selector=%s", selector)
        try:
            if selector:
                return (input, None)
            return (None, input)
        except Exception as e:
            logger.error("Error in SwitchAnyRoute_wANY: %s", e, exc_info=True)
            return (None, None)

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "SwitchAny": SwitchAny,
    "SwitchAnyRoute": SwitchAnyRoute,
    "SwitchAny_wANY": SwitchAny_wANY,
    "SwitchAnyRoute_wANY": SwitchAnyRoute_wANY,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SwitchAny": "Switch Any (LLMToolkit)",
    "SwitchAnyRoute": "Switch Any Route (LLMToolkit)",
    "SwitchAny_wANY": "Switch Any _wANY (LLMToolkit)",
    "SwitchAnyRoute_wANY": "Switch Any Route _wANY (LLMToolkit)",
} 