# comfy-nodes/string_utils.py
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class JoinStringsMulti:
    """
    A node to join multiple strings with a specified delimiter.
    Can return either a single concatenated string or a list of strings.
    The number of string inputs is dynamically adjustable in the UI.
    """

    _last_input_count: int = 2  # Tracks the last known input count for dynamic UI generation

    # ------------------------------------------------------------------
    # ComfyUI metadata
    # ------------------------------------------------------------------
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "join_strings"
    CATEGORY = "llm_toolkit/utils/text"
    OUTPUT_NODE = False  # Utility node, not an output node

    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Defines the input types for the node (dynamic based on `input_count`)."""
        # Base fields that are always present
        inputs: Dict[str, Any] = {
            "required": {
                "input_count": ("INT", {"default": 2, "min": 2, "max": 100, "step": 1}),
                "delimiter": ("STRING", {"default": " ", "multiline": False}),
                "return_list": ("BOOLEAN", {"default": False}),
            }
        }

        # Add string_i fields (at least two by default, or `_last_input_count` if bigger)
        max_fields = max(2, getattr(cls, "_last_input_count", 2))
        for i in range(1, max_fields + 1):
            # Force first two inputs; others optional
            inputs["required"][f"string_{i}"] = (
                "STRING",
                {
                    "default": "",
                    "forceInput": i <= 2,  # First two must be wired, rest optional
                    "multiline": False,
                },
            )

        return inputs

    # ------------------------------------------------------------------
    # Dynamic update hook – triggers the "Update" button in ComfyUI
    # ------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, input_count: int = 2, **_ignored):
        """Return a value that changes when `input_count` changes to trigger UI update."""
        try:
            input_count = int(input_count)
        except Exception:
            input_count = 2

        input_count = max(2, min(100, input_count))
        cls._last_input_count = input_count
        return input_count

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def join_strings(self, **kwargs) -> Tuple[List[str] | str]:
        # Pull settings
        input_count = int(kwargs.get("input_count", 2))
        delimiter = kwargs.get("delimiter", " ")
        return_list = bool(kwargs.get("return_list", False))

        logger.debug(
            "JoinStringsMulti: Joining %s strings with delimiter '%s'.", input_count, delimiter
        )

        # Collect strings based on the current input_count
        collected: List[str] = []
        for i in range(1, input_count + 1):
            key = f"string_{i}"
            if key not in kwargs:
                continue  # Skip if not provided (shouldn't happen after update)
            val = kwargs[key]
            if isinstance(val, list):
                collected.extend([str(item) for item in val])
            else:
                collected.append(str(val))

        if return_list:
            return (collected,)
        else:
            return (delimiter.join(collected),)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "JoinStringsMulti": JoinStringsMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoinStringsMulti": "Join Strings Multi (LLMToolkit)",
} 