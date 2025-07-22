import sys
import logging
from typing import Optional, Any, Tuple
import torch

# Initialize logger
logger = logging.getLogger(__name__)

class CheckImageEmpty:
    """
    Checks if an incoming image is null or empty and returns a boolean.
    Returns True if the image is null, empty, or invalid, False if it contains valid image data.
    """
    def __init__(self):
        self.type = "utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image to check for null/empty state"}),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("is_empty", "status_message")
    FUNCTION = "check_image_empty"
    OUTPUT_NODE = False  # This is a utility node, not an output node
    CATEGORY = "llm_toolkit/utils/logic"

    def check_image_empty(self, image: Optional[torch.Tensor] = None) -> Tuple[bool, str]:
        """
        Checks if the provided image is null or empty.
        
        Args:
            image: Optional image tensor to check
            
        Returns:
            Tuple containing:
            - is_empty: Boolean indicating if image is null/empty
            - status_message: Human-readable status description
        """
        logger.info("CheckImageEmpty node executing...")
        
        # Initialize return values
        is_empty = True
        status_message = "Image is empty or null"
        
        try:
            # Check if image is None
            if image is None:
                is_empty = True
                status_message = "Image is None (null)"
                logger.info("Image check result: None (null)")
                
            # Check if image is a valid tensor
            elif not isinstance(image, torch.Tensor):
                is_empty = True
                status_message = f"Image is not a valid tensor (type: {type(image).__name__})"
                logger.info(f"Image check result: Invalid type {type(image).__name__}")
                
            # Check if tensor is empty or has no data
            elif image.numel() == 0:
                is_empty = True
                status_message = "Image tensor is empty (no elements)"
                logger.info("Image check result: Empty tensor")
                
            # Check if tensor has invalid dimensions for an image
            elif len(image.shape) < 3:
                is_empty = True
                status_message = f"Image tensor has insufficient dimensions: {image.shape}"
                logger.info(f"Image check result: Invalid dimensions {image.shape}")
                
            # Check if any dimension is zero
            elif any(dim == 0 for dim in image.shape):
                is_empty = True
                status_message = f"Image tensor has zero-sized dimension: {image.shape}"
                logger.info(f"Image check result: Zero dimension in {image.shape}")
                
            # Image appears to be valid
            else:
                is_empty = False
                status_message = f"Image is valid with shape: {image.shape}"
                logger.info(f"Image check result: Valid image with shape {image.shape}")
                
        except Exception as e:
            # If any error occurs during checking, consider image as empty/invalid
            is_empty = True
            status_message = f"Error checking image: {str(e)}"
            logger.error(f"Error in CheckImageEmpty: {e}", exc_info=True)
        
        # Log the final result
        logger.info(f"CheckImageEmpty completed: is_empty={is_empty}, message='{status_message}'")
        
        return (is_empty, status_message)

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "CheckImageEmpty": CheckImageEmpty
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckImageEmpty": "Check Image Empty (LLMToolkit)"
} 