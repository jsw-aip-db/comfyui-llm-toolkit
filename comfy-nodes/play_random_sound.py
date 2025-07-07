# Random sound player node for ComfyUI – chooses or forces a .mp3 from ../sounds
# Part of LLM-Toolkit utils.

import os
import random
from typing import List, Dict, Any

# --- Wildcard AnyType matching behaviour (mirrors SwitchAny implementation) ---
class AnyType(str):
    def __ne__(self, __value: object) -> bool:  # noqa: D401,E501
        return False

# Wildcard token used by ComfyUI for "any" typing
any = AnyType("*")


class PlayRandomSound:
    """Play an audio notification from *sounds/*

    • If *sound* == "(random)" → picks a random .mp3 from the folder.
    • Otherwise the selected filename is played (if present).

    The sounds folder lives at:
        custom_nodes/llm-toolkit/sounds/
    """

    # Resolve sounds directory relative to llm-toolkit root
    _SOUNDS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sounds"))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @classmethod
    def _list_mp3_files(cls) -> List[str]:
        """Return list of .mp3 filenames (no paths) inside _SOUNDS_DIR."""
        if not os.path.isdir(cls._SOUNDS_DIR):
            return []
        return sorted([f for f in os.listdir(cls._SOUNDS_DIR) if f.lower().endswith(".mp3")])

    @classmethod
    def _get_random_mp3(cls) -> str | None:
        files = cls._list_mp3_files()
        if not files:
            return None
        return os.path.join(cls._SOUNDS_DIR, random.choice(files))

    # ------------------------------------------------------------------
    # ComfyUI node spec
    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        mp3_choices = cls._list_mp3_files()
        dropdown = ["(random)"] + mp3_choices if mp3_choices else ["(none)"]

        return {
            "required": {
                "any": (any, {}),
                "mode": (["always", "on empty queue"], {}),
                "volume": ("FLOAT", {"min": 0, "max": 1, "step": 0.1, "default": 0.5}),
                "sound": (dropdown, {"default": "(random)", "tooltip": "Select a sound or (random)"}),
            }
        }

    FUNCTION = "nop"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    RETURN_TYPES = (any,)
    CATEGORY = "llm_toolkit/utils"

    # ------------------------------------------------------------------
    def IS_CHANGED(self, **_kwargs):  # noqa: D401
        """Always considered changed so re-evaluates sound selection each run."""
        return float("NaN")

    # ------------------------------------------------------------------
    def nop(self, any, mode, volume, sound):  # noqa: D401
        """Pass through *any* and send UI message to play chosen sound."""
        if sound == "(random)":
            chosen = self._get_random_mp3()
        elif sound == "(none)":
            chosen = None
        else:
            candidate = os.path.join(self._SOUNDS_DIR, sound)
            chosen = candidate if os.path.isfile(candidate) else None

        ui_payload: Dict[str, List[str]] = {"a": []}
        if chosen:
            ui_payload["a"] = [chosen]
        # else keep empty list – frontend will treat as no-op

        return {"ui": ui_payload, "result": (any,)}


# ---------------------------------------------------------------------------
# Node registration – required for ComfyUI to discover the node
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "PlayRandomSound": PlayRandomSound,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlayRandomSound": "Play Random Sound (LLMToolkit)",
} 