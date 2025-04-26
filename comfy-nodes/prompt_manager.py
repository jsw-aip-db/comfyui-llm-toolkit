# prompt_manager.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple
import torch
from llmtoolkit_utils import tensor_to_base64, TENSOR_SUPPORT, ensure_rgba_mask, resize_mask_to_match_image

# Ensure parent directory is in path if running standalone for testing
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Relative import of utilities
try:
    from llmtoolkit_utils import tensor_to_base64, TENSOR_SUPPORT
except ImportError:
    # Fallback for standalone execution or environment issues
    try:
        from ..llmtoolkit_utils import tensor_to_base64, TENSOR_SUPPORT
    except ImportError:
        # Final fallback
        from llmtoolkit_utils import tensor_to_base64, TENSOR_SUPPORT

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages and structures prompt components (text, image, mask, paths)
    into a prompt_config dictionary within the main context object.
    Accepts various optional inputs and adds them to the context if provided.
    """
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "optional": {
                "context": ("*", {}),
                "text_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }
        # Conditionally add image/mask inputs if torch is available
        if TENSOR_SUPPORT:
            inputs["optional"]["image"] = ("IMAGE",)
            inputs["optional"]["mask"] = ("MASK",)
        else:
             logger.warning("PromptManager: Torch/Numpy/PIL not found. IMAGE and MASK inputs disabled.")

        # Add path inputs (always available)
        inputs["optional"]["audio_path"] = ("STRING", {"multiline": False, "default": ""})
        inputs["optional"]["file_path"] = ("STRING", {"multiline": False, "default": ""})

        return inputs

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "manage_prompt"
    CATEGORY = "prompt_manager"

    def manage_prompt(self, context: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        Assembles prompt components into a dictionary within the context.
        """
        logger.info("PromptManager executing...")

        # Initialize or copy the context
        if context is None:
            output_context = {}
            logger.debug("PromptManager: Initializing new context.")
        elif isinstance(context, dict):
            output_context = context.copy()
            logger.debug("PromptManager: Copied input context.")
        else:
            # Handle non-dict context input gracefully
            output_context = {"passthrough_data": context}
            logger.warning("PromptManager: Received non-dict context input. Wrapping it.")

        # Initialize prompt_config dictionary (get existing if present, else create)
        prompt_config = output_context.get("prompt_config", {})
        if not isinstance(prompt_config, dict):
            logger.warning("PromptManager: Existing 'prompt_config' in context is not a dict. Overwriting.")
            prompt_config = {}

        # Process optional inputs using kwargs
        text_prompt = kwargs.get("text_prompt", "").strip()
        image_tensor = kwargs.get("image", None)
        mask_tensor = kwargs.get("mask", None)
        audio_path = kwargs.get("audio_path", "").strip()
        file_path = kwargs.get("file_path", "").strip()

        if text_prompt:
            prompt_config["text"] = text_prompt
            logger.debug(f"PromptManager: Added text_prompt (length: {len(text_prompt)}).")

        # --- Handle IMAGE tensor or batch ---
        if image_tensor is not None and TENSOR_SUPPORT:
            logger.debug(f"PromptManager: Processing image tensor with shape: {image_tensor.shape}")
            # If tensor is batch (B, H, W, C), iterate; else single image
            if image_tensor.dim() == 4 and image_tensor.shape[0] > 1:
                img_list = []
                for idx in range(image_tensor.shape[0]):
                    single_img = image_tensor[idx:idx+1]  # Keep batch dim for util
                    b64 = tensor_to_base64(single_img, image_format="PNG")
                    if b64:
                        img_list.append(b64)
                if img_list:
                    prompt_config["image_base64"] = img_list
                    logger.debug(f"PromptManager: Added list of {len(img_list)} base64 images.")
                else:
                    logger.warning("PromptManager: Failed to convert batch images to base64.")
            else:
                img_base64 = tensor_to_base64(image_tensor, image_format="PNG")
                if img_base64:
                    prompt_config["image_base64"] = img_base64
                    logger.debug("PromptManager: Added image_base64.")
                else:
                    logger.warning("PromptManager: Failed to convert image tensor to base64.")

        # --- Handle MASK tensor or batch ---
        if mask_tensor is not None and TENSOR_SUPPORT:
            logger.debug(f"PromptManager: Processing mask tensor with shape: {mask_tensor.shape}")

            def _prep_mask(t):
                # Ensure grayscale channel dim ==1
                if t.dim() == 4 and t.shape[-1] in [3, 4]:
                    t = t.mean(dim=-1, keepdim=True)
                elif t.dim() == 3:  # B H W
                    t = t.unsqueeze(-1)
                return t

            if mask_tensor.dim() == 4 and mask_tensor.shape[0] > 1:
                mask_list = []
                for idx in range(mask_tensor.shape[0]):
                    single_mask = _prep_mask(mask_tensor[idx:idx+1])
                    b64 = tensor_to_base64(single_mask, image_format="PNG")
                    if b64:
                        mask_list.append(b64)
                if mask_list:
                    prompt_config["mask_base64"] = mask_list
                    logger.debug(f"PromptManager: Added list of {len(mask_list)} base64 masks.")
                else:
                    logger.warning("PromptManager: Failed to convert batch masks to base64.")
            else:
                # Resize mask if needed to match image dimensions
                if image_tensor is not None and isinstance(image_tensor, torch.Tensor):
                    mask_tensor = resize_mask_to_match_image(mask_tensor, image_tensor)

                try:
                    # For OpenAI edits transparent areas (alpha=0) are replaced. In many
                    # ComfyUI masks the area to *edit* is white (1).  Therefore invert
                    # the mask so edited pixels get alpha=0.
                    inv_mask_tensor = 1.0 - mask_tensor.clamp(0, 1)
                    mask_tensor_rgba = ensure_rgba_mask(inv_mask_tensor)
                except Exception:
                    mask_tensor_rgba = mask_tensor  # Fallback

                mask_base64 = tensor_to_base64(mask_tensor_rgba, image_format="PNG")
                if mask_base64:
                    prompt_config["mask_base64"] = mask_base64
                    logger.debug("PromptManager: Added mask_base64.")
                else:
                    logger.warning("PromptManager: Failed to convert mask tensor to base64.")

        if audio_path:
            prompt_config["audio_path"] = audio_path
            logger.debug(f"PromptManager: Added audio_path: {audio_path}")

        if file_path:
            prompt_config["file_path"] = file_path
            logger.debug(f"PromptManager: Added file_path: {file_path}")

        # Update the main context with the (potentially updated) prompt_config
        if prompt_config: # Only add if not empty
             output_context["prompt_config"] = prompt_config
             logger.info("PromptManager: Updated context with prompt_config.")
        else:
             logger.info("PromptManager: No prompt components provided.")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "PromptManager": PromptManager
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptManager": "Prompt Manager (LLMToolkit)"
} 