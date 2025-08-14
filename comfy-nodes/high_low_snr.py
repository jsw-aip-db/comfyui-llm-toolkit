# high_low_snr.py
from __future__ import annotations

from typing import Tuple


class HighLowSNR:
    """Map diffusion total steps to the corresponding High-SNR steps.

    This is a lightweight utility node intended for quick schedule mapping.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TotalSteps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 4000,
                        "step": 1,
                        "tooltip": "Total sampling steps for your scheduler/pipeline.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("HighSteps",)
    FUNCTION = "map"
    CATEGORY = "llm_toolkit/utils"

    def map(self, TotalSteps: int) -> Tuple[int]:
        mapping = {
            4: 2,
            6: 2,
            8: 3,
            10: 4,
            12: 4,
            14: 5,
            16: 6,
            18: 7,
            20: 8,
        }
        return (mapping.get(int(TotalSteps), 3),)


# --- Node registration ---
NODE_CLASS_MAPPINGS = {
    "HighLowSNR": HighLowSNR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HighLowSNR": "High/Low SNR Mapper (LLMToolkit)",
}

