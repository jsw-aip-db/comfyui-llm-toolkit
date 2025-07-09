import os
import logging
from typing import Tuple, Optional, List

import torch
import numpy as np

# ComfyUI utility for temp/output directories (if available)
try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None  # Fallback: will use working dir

logger = logging.getLogger(__name__)


def _hex_to_rgb_floats(hex_color: str) -> Tuple[float, float, float]:
    """Convert #RRGGBB hex to normalized floats (0-1)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Color must be #RRGGBB")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


class PreviewImageLogic:
    """Preview/save an image but fall back to a generated blank if no image provided.

    Useful in logic branches where an IMAGE might be None (e.g., from a switch).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_image": ("BOOLEAN", {"default": False, "tooltip": "Save the resulting image to disk"}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
                "color": ("COLOR", {"default": "#FFFFFF", "tooltip": "Fill color when generating blank image"}),
            },
            "optional": {
                "file_name": ("STRING", {"default": "logic_preview", "multiline": False, "tooltip": "Base filename when saving (no extension)"}),
                "image": ("IMAGE", {"tooltip": "Optional image tensor; if None blank is generated"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "preview"
    OUTPUT_NODE = False
    CATEGORY = "llm_toolkit/utils/logic"

    # ------------------------------------------------------------------
    def _make_blank(self, width: int, height: int, color: str) -> torch.Tensor:
        r, g, b = _hex_to_rgb_floats(color)
        arr = np.stack([
            np.full((height, width), r, dtype=np.float32),
            np.full((height, width), g, dtype=np.float32),
            np.full((height, width), b, dtype=np.float32),
        ], axis=-1)  # (H,W,3)
        arr = arr[np.newaxis, ...]  # (1,H,W,3)
        return torch.from_numpy(arr)

    def _save(self, img_tensor: torch.Tensor, name_prefix: str) -> str:
        # Lazy import PIL only when saving
        from PIL import Image  # type: ignore

        # Accept (B,H,W,3) or (B,3,H,W). Convert to (H,W,3) for PIL.
        if img_tensor.ndim != 4:
            raise ValueError("Expected image tensor (B,3,H,W) or (B,H,W,3)")
        t = img_tensor[0].clamp(0, 1).cpu()
        if t.shape[0] == 3:  # (3,H,W)
            img_np = (t.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        else:  # (H,W,3)
            img_np = (t.numpy() * 255).astype(np.uint8)
        h, w = img_np.shape[:2]
        img = Image.fromarray(img_np)

        output_dir = folder_paths.get_output_directory() if folder_paths else os.getcwd()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{name_prefix}_{os.getpid()}_{np.random.randint(0,1e6):06d}.png"
        path = os.path.join(output_dir, filename)
        img.save(path)
        return path

    # ------------------------------------------------------------------
    def preview(
        self,
        save_image: bool,
        width: int,
        height: int,
        color: str,
        file_name: str = "logic_preview",
        image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        try:
            # Handle ComfyUI IMAGE formats: could be torch.Tensor, list[torch.Tensor], or None
            if image is None:
                out_img = None
            elif isinstance(image, list):
                out_img = image[0] if image else None
            else:
                out_img = image  # assume torch.Tensor

            if out_img is None:
                logger.debug("PreviewImageLogic: Creating blank image %dx%d color %s", width, height, color)
                out_img = self._make_blank(width, height, color)

            if save_image:
                try:
                    path = self._save(out_img, file_name or "logic_preview")
                    logger.info("PreviewImageLogic: Saved image to %s", path)
                except Exception as save_err:
                    logger.error("PreviewImageLogic: Failed to save image – %s", save_err, exc_info=True)
            return (out_img,)
        except Exception as e:
            logger.error("PreviewImageLogic: %s", e, exc_info=True)
            # On catastrophic error, return blank 1x1 black to keep graph alive
            fallback = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
            return (fallback,)


# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "PreviewImageLogic": PreviewImageLogic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewImageLogic": "Preview Image Logic (LLMToolkit)",
} 