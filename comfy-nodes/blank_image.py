import logging
from typing import Tuple
import torch
import numpy as np

logger = logging.getLogger(__name__)

class BlankImage:
    """ComfyUI node that creates a solid-color image tensor of the desired size.

    The color is chosen via a color-picker UI (hex string) and size via width/height ints.
    Output format: torch.Tensor shaped (1, 3, H, W) with values in [0,1].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "tooltip": "Image width in pixels"}),
                "height": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1, "tooltip": "Image height in pixels"}),
                "color": ("COLOR", {"default": "#000000", "tooltip": "Solid fill color"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create"
    CATEGORY = "llm_toolkit/utils"

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex string (6- or 8-digit) to 0-1 float RGB tuple."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) not in {6, 8}:
            raise ValueError("Invalid hex color string")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return r / 255.0, g / 255.0, b / 255.0

    def create(self, width: int, height: int, color: str):
        try:
            r, g, b = self._hex_to_rgb(color)
            arr = np.stack([
                np.full((height, width), r, dtype=np.float32),
                np.full((height, width), g, dtype=np.float32),
                np.full((height, width), b, dtype=np.float32),
            ], axis=0)  # shape (3, H, W)
            img_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,3,H,W)
            return (img_tensor,)
        except Exception as e:
            logger.error("BlankImage: Failed to create image – %s", e, exc_info=True)
            return (torch.zeros((1, 3, height, width), dtype=torch.float32),)

# --------------------------------------------------------------------
# Mappings for ComfyUI
# --------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "BlankImage": BlankImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlankImage": "Blank Image (LLMToolkit)",
} 