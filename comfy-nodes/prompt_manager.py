# prompt_manager.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple
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

# Lazy loading for optional dependencies
cv2 = None
def get_cv2():
    global cv2
    if cv2 is None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            logger.warning("cv2 not available. Video frame extraction disabled.")
    return cv2

# Lazy loading for torch
_torch = None
def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch

class LLMPromptManager:
    """
    Universal prompt manager with dynamic inputs.
    Automatically adds new input ports as you connect nodes.
    Auto-detects input types and merges them into a unified context.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Start with a single context input - JavaScript will add more dynamically
        inputs = {
            "optional": {
                "context": ("*", {
                    "tooltip": "Universal input - connect any type. New inputs appear automatically as you connect nodes."
                }),
            }
        }
        return inputs
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always reprocess to handle accumulation
        return float("nan")

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "manage_prompt"
    CATEGORY = "🔗llm_toolkit/prompt"

    def _detect_tensor_type(self, tensor) -> str:
        """Detect if a tensor is an image, video, or mask based on its shape and values."""
        if not get_torch().is_tensor(tensor):
            return "unknown"
        
        # Check dimensions
        if tensor.dim() == 2:
            # 2D tensor is likely a mask (H, W)
            return "mask"
        elif tensor.dim() == 3:
            # Could be single image (H, W, C) or mask (B, H, W)
            if tensor.shape[-1] in [1, 3, 4]:
                # Last dimension is channels - it's an image
                return "image"
            else:
                # Likely batch of masks
                return "mask"
        elif tensor.dim() == 4:
            # Check if it's video frames or batch of images
            if tensor.shape[0] > 1:
                # Multiple frames - could be video or image batch
                # Check if values suggest it's a mask (typically 0-1 range, single channel)
                if tensor.shape[-1] == 1 or (tensor.min() >= 0 and tensor.max() <= 1 and tensor.shape[-1] not in [3, 4]):
                    return "mask"
                # Assume video if more than 8 frames, otherwise image batch
                return "video" if tensor.shape[0] > 8 else "image"
            else:
                return "image"
        return "unknown"

    def _process_string_input(self, text: str, context: Dict[str, Any]) -> None:
        """Process string input - detect if it's text, file path, or URL."""
        text = text.strip()
        if not text:
            return
        
        prompt_config = context.setdefault("prompt_config", {})
        
        # Check if it's a URL
        if text.startswith(("http://", "https://", "ftp://", "file://")):
            urls = [u.strip() for u in text.split(",") if u.strip()]
            # Append to existing URLs if present
            if "urls" in prompt_config:
                existing = prompt_config["urls"]
                if isinstance(existing, list):
                    existing.extend(urls)
                else:
                    prompt_config["urls"] = [existing] + urls
            else:
                prompt_config["urls"] = urls if len(urls) > 1 else urls[0]
            logger.debug(f"LLMPromptManager: Detected URL input: {text[:50]}...")
        # Check if it's a file path
        elif any(text.startswith(p) for p in ["/", "./", "../", "~/"]) or \
             any(ext in text.lower() for ext in [".mp4", ".avi", ".mov", ".pdf", ".txt", ".mp3", ".wav", ".png", ".jpg"]):
            paths = [p.strip() for p in text.split(",") if p.strip()]
            # Check if any are video files
            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
            audio_exts = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
            
            if any(any(ext in p.lower() for ext in video_exts) for p in paths):
                # Append to existing file paths
                if "file_paths" in prompt_config:
                    existing = prompt_config["file_paths"]
                    if isinstance(existing, list):
                        existing.extend(paths)
                    else:
                        prompt_config["file_paths"] = [existing] + paths
                else:
                    prompt_config["file_paths"] = paths if len(paths) > 1 else paths[0]
                logger.debug(f"LLMPromptManager: Detected video file path: {text[:50]}...")
            elif any(any(ext in p.lower() for ext in audio_exts) for p in paths):
                prompt_config["audio_path"] = paths[0] if paths else ""
                logger.debug(f"LLMPromptManager: Detected audio file path: {text[:50]}...")
            else:
                # Append to existing file paths
                if "file_paths" in prompt_config:
                    existing = prompt_config["file_paths"]
                    if isinstance(existing, list):
                        existing.extend(paths)
                    else:
                        prompt_config["file_paths"] = [existing] + paths
                else:
                    prompt_config["file_paths"] = paths if len(paths) > 1 else paths[0]
                logger.debug(f"LLMPromptManager: Detected file path: {text[:50]}...")
        else:
            # It's regular text/prompt - combine multiple text inputs
            if "text" in prompt_config:
                # Append to existing text with a newline
                prompt_config["text"] = prompt_config["text"] + "\n" + text
            else:
                prompt_config["text"] = text
            logger.debug(f"LLMPromptManager: Detected text prompt (length: {len(text)})")

    def _process_tensor_input(self, tensor, context: Dict[str, Any]) -> None:
        """Process tensor input - detect if it's image, video, or mask."""
        if not TENSOR_SUPPORT or tensor is None:
            return
        
        tensor_type = self._detect_tensor_type(tensor)
        logger.debug(f"LLMPromptManager: Detected tensor type: {tensor_type}, shape: {tensor.shape}")
        
        if tensor_type == "image":
            # Handle multiple images - combine into batch if needed
            if "IMAGE" in context:
                existing = context["IMAGE"]
                if get_torch().is_tensor(existing):
                    # Concatenate along batch dimension
                    if existing.dim() == tensor.dim():
                        context["IMAGE"] = get_torch().cat([existing, tensor], dim=0)
                    else:
                        context["IMAGE"] = tensor
                else:
                    context["IMAGE"] = tensor
            else:
                context["IMAGE"] = tensor
        elif tensor_type == "video":
            # Handle multiple videos - combine frames if needed
            if "VIDEO" in context:
                existing = context["VIDEO"]
                if get_torch().is_tensor(existing) and get_torch().is_tensor(tensor):
                    # Concatenate video frames
                    if existing.dim() == 4 and tensor.dim() == 4:
                        context["VIDEO"] = get_torch().cat([existing, tensor], dim=0)
                    else:
                        context["VIDEO"] = tensor
                elif isinstance(existing, list):
                    existing.append(tensor)
                else:
                    context["VIDEO"] = [existing, tensor]
            else:
                context["VIDEO"] = tensor
        elif tensor_type == "mask":
            # Handle multiple masks
            if "MASK" in context:
                existing = context["MASK"]
                if get_torch().is_tensor(existing) and get_torch().is_tensor(tensor):
                    if existing.dim() == tensor.dim():
                        context["MASK"] = get_torch().cat([existing, tensor], dim=0)
                    else:
                        context["MASK"] = tensor
                else:
                    context["MASK"] = tensor
            else:
                context["MASK"] = tensor
        else:
            logger.warning(f"LLMPromptManager: Unknown tensor type with shape {tensor.shape}")

    def _process_list_input(self, input_list, context: Dict[str, Any]) -> None:
        """Process list/tuple input - could be multiple frames, multiple contexts, etc."""
        if not input_list:
            return
        
        # Check if all items are tensors (likely video frames)
        if all(get_torch().is_tensor(item) for item in input_list) and TENSOR_SUPPORT:
            # Assume it's video frames
            logger.debug(f"PromptManager: Detected list of {len(input_list)} tensors, treating as video frames")
            context["VIDEO"] = input_list
        # Check if all items are strings
        elif all(isinstance(item, str) for item in input_list):
            # Multiple strings - could be multiple prompts or paths
            combined = ", ".join(input_list)
            self._process_string_input(combined, context)
        # Check if all items are dicts (multiple contexts to merge)
        elif all(isinstance(item, dict) for item in input_list):
            # Merge all contexts
            for item_dict in input_list:
                self._merge_contexts(context, item_dict)
        else:
            # Mixed types - store as is
            context["mixed_input"] = input_list
            logger.debug(f"PromptManager: Received mixed-type list with {len(input_list)} items")

    def _merge_contexts(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source context into target context."""
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dicts
                    self._merge_contexts(target[key], value)
                elif key == "prompt_config" and isinstance(target[key], dict) and isinstance(value, dict):
                    # Special handling for prompt_config - merge instead of replace
                    target[key].update(value)
                else:
                    # For other types, source overwrites target
                    target[key] = value
            else:
                target[key] = value
        
        # Special handling: If provider_config has a system_message but prompt_config doesn't have text,
        # copy the system_message to prompt_config.text so downstream nodes can use it as the prompt
        if "provider_config" in target and "system_message" in target["provider_config"]:
            if "prompt_config" not in target:
                target["prompt_config"] = {}
            if "text" not in target.get("prompt_config", {}):
                target["prompt_config"]["text"] = target["provider_config"]["system_message"]
                logger.debug("Copied system_message to prompt_config.text for downstream compatibility")

    def manage_prompt(self, context=None, **kwargs) -> Tuple[Dict[str, Any]]:
        """
        Assembles prompt components from a single universal input into a unified context.
        The input can be a single item OR a list of items of ANY type.
        Automatically detects and processes each item type.
        """
        logger.info("LLMPromptManager executing with dynamic inputs")

        # Initialize output context
        output_context = {}
        
        # Collect all inputs - JavaScript sends them as a list in context
        # or they come through kwargs for additional dynamic inputs
        items = []
        
        # Handle the main context input
        if context is not None:
            if isinstance(context, (list, tuple)):
                # JavaScript collected multiple inputs into a list
                items.extend(context)
                logger.debug(f"Processing {len(context)} items from dynamic inputs")
            else:
                # Single input
                items.append(context)
                logger.debug(f"Processing single {type(context).__name__} input")
        
        # Also check kwargs for any additional context_N inputs
        # (in case JavaScript passes them separately)
        for key, value in kwargs.items():
            if key.startswith('context_') and value is not None:
                if isinstance(value, (list, tuple)):
                    items.extend(value)
                else:
                    items.append(value)
                logger.debug(f"Added {key}: {type(value).__name__}")
        
        if not items:
            logger.debug("No inputs provided")
        
        # Process each item
        for idx, item in enumerate(items, 1):
            if item is None:
                continue
                
            logger.debug(f"Processing item {idx}/{len(items)}: type={type(item).__name__}")
            
            # Auto-detect and handle each item type
            if isinstance(item, dict):
                # Context dictionary - merge it
                self._merge_contexts(output_context, item)
                logger.debug(f"Item {idx}: Merged dictionary context")
            elif isinstance(item, str):
                # String - could be text, file path, or URL
                self._process_string_input(item, output_context)
                logger.debug(f"Item {idx}: Processed string")
            elif TENSOR_SUPPORT and get_torch().is_tensor(item):
                # Tensor - detect if image, video, or mask
                self._process_tensor_input(item, output_context)
                logger.debug(f"Item {idx}: Processed tensor")
            elif isinstance(item, (list, tuple)):
                # Nested list - recursively process
                for nested_item in item:
                    if isinstance(nested_item, dict):
                        self._merge_contexts(output_context, nested_item)
                    elif isinstance(nested_item, str):
                        self._process_string_input(nested_item, output_context)
                    elif TENSOR_SUPPORT and get_torch().is_tensor(nested_item):
                        self._process_tensor_input(nested_item, output_context)
                    else:
                        output_context[f"nested_data_{idx}"] = nested_item
            else:
                # Unknown type - store it
                output_context[f"passthrough_data_{idx}"] = item
                logger.warning(f"Item {idx}: Unknown type {type(item)}, storing as passthrough")

        # Initialize prompt_config dictionary (preserve existing if present)
        prompt_config = output_context.get("prompt_config", {})
        if not isinstance(prompt_config, dict):
            prompt_config = {}

        # Extract any data that was already detected and stored
        text_prompt = prompt_config.get("text", "")
        image_tensor = output_context.get("IMAGE", None)
        mask_tensor = output_context.get("MASK", None)
        video_tensor = output_context.get("VIDEO", None)
        audio_path = prompt_config.get("audio_path", "")
        file_path_str = prompt_config.get("file_paths", "")
        if isinstance(file_path_str, list):
            file_path_str = ", ".join(file_path_str)
        url_str = prompt_config.get("urls", "")
        if isinstance(url_str, list):
            url_str = ", ".join(url_str)

        if text_prompt:
            prompt_config["text"] = text_prompt
            logger.debug(f"PromptManager: Added text_prompt (length: {len(text_prompt)}).")

        # Mark content style hint for GPT-5 Responses API if used downstream
        # Downstream code will detect gpt-5* models and consume images accordingly.
        prompt_config["content_style"] = "responses_api"

        # --- Helper to process tensor or list of tensors into b64 list ---
        def _tensor_or_list_to_b64(tensor_or_list, max_items: int = 16):
            """Return list of base64 strings given tensor or list of tensors."""
            b64_list = []
            if get_torch().is_tensor(tensor_or_list):
                if tensor_or_list.dim() == 4 and tensor_or_list.shape[0] > 1:
                    sample_count = min(tensor_or_list.shape[0], max_items)
                    for idx in range(sample_count):
                        b64 = tensor_to_base64(tensor_or_list[idx:idx+1], image_format="PNG")
                        if b64:
                            b64_list.append(b64)
                else:
                    b64 = tensor_to_base64(tensor_or_list, image_format="PNG")
                    if b64:
                        b64_list.append(b64)
            elif isinstance(tensor_or_list, list):
                sample_items = tensor_or_list[:max_items]
                for t in sample_items:
                    if get_torch().is_tensor(t):
                        # Ensure has batch dim
                        if t.dim() == 3:
                            t = t.unsqueeze(0)
                        b64 = tensor_to_base64(t, image_format="PNG")
                        if b64:
                            b64_list.append(b64)
            return b64_list

        # --- Handle IMAGE input (single, list, or batch tensor) ---
        if image_tensor is not None and TENSOR_SUPPORT:
            imgs_b64 = _tensor_or_list_to_b64(image_tensor)
            if imgs_b64:
                prompt_config["image_base64"] = imgs_b64 if len(imgs_b64) > 1 else imgs_b64[0]
                logger.debug(f"PromptManager: Added {len(imgs_b64)} image(s) to prompt_config.")
            else:
                logger.warning("PromptManager: Failed to convert provided image(s) to base64.")

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
                torch = get_torch()
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

        # --- Handle VIDEO tensor/batch/list ---
        # Video tensors from nodes (e.g., LoadVideo) are extracted to frames
        # for APIs like OpenAI that only support images.
        # This is different from video file paths which are kept intact.
        if video_tensor is not None and TENSOR_SUPPORT:
            logger.debug(f"PromptManager: Processing video tensor with shape: {video_tensor.shape if get_torch().is_tensor(video_tensor) else 'non-tensor'}")
            # Extract frames at intervals (every 16th frame, max 5 frames)
            extracted_frames = []
            
            if get_torch().is_tensor(video_tensor):
                # Video tensor shape is typically [frames, H, W, C] from LoadVideo
                # or [B, H, W, C] for batched images that could represent video frames
                if video_tensor.dim() == 4:
                    frame_count = video_tensor.shape[0]
                    # Adaptive interval based on frame count
                    max_frames = 10  # Increased for better coverage
                    if frame_count <= max_frames:
                        # If we have fewer frames than max, take all
                        interval = 1
                        frames_to_extract = frame_count
                    else:
                        # Calculate interval to get evenly spaced frames
                        interval = max(1, frame_count // max_frames)
                        frames_to_extract = min(frame_count, max_frames)
                    
                    for i in range(0, min(frame_count, interval * frames_to_extract), interval):
                        frame = video_tensor[i:i+1]  # Keep batch dim
                        b64 = tensor_to_base64(frame, image_format="JPEG")
                        if b64:
                            extracted_frames.append(b64)
                    
                    logger.debug(f"PromptManager: Extracted {len(extracted_frames)} frames from video tensor (every {interval}th frame from {frame_count} total).")
                elif video_tensor.dim() == 3:
                    # Single frame without batch dimension - add batch dim and treat as single frame
                    frame = video_tensor.unsqueeze(0)
                    b64 = tensor_to_base64(frame, image_format="JPEG")
                    if b64:
                        extracted_frames.append(b64)
                    logger.debug("PromptManager: Processed single video frame (3D tensor).")
                else:
                    # Unexpected shape - try to handle as image
                    b64 = tensor_to_base64(video_tensor, image_format="JPEG")
                    if b64:
                        extracted_frames.append(b64)
                    logger.debug(f"PromptManager: Processed video tensor with unexpected dims: {video_tensor.dim()}")
            elif isinstance(video_tensor, (list, tuple)):
                # Handle list of frames
                for idx, frame in enumerate(video_tensor[:10]):  # Limit to 10 frames
                    if get_torch().is_tensor(frame):
                        if frame.dim() == 3:
                            frame = frame.unsqueeze(0)
                        b64 = tensor_to_base64(frame, image_format="JPEG")
                        if b64:
                            extracted_frames.append(b64)
                logger.debug(f"PromptManager: Extracted {len(extracted_frames)} frames from video list/tuple.")
            
            if extracted_frames:
                prompt_config["video_frames_base64"] = extracted_frames
                # Also store metadata about the video
                if get_torch().is_tensor(video_tensor) and video_tensor.dim() == 4:
                    prompt_config["video_metadata"] = {
                        "frame_count": video_tensor.shape[0],
                        "height": video_tensor.shape[1],
                        "width": video_tensor.shape[2],
                        "extracted_frames": len(extracted_frames)
                    }
                logger.debug(f"PromptManager: Added {len(extracted_frames)} extracted video frame(s).")
            else:
                logger.warning("PromptManager: Failed to convert video input to base64.")

        if audio_path:
            prompt_config["audio_path"] = audio_path
            logger.debug(f"PromptManager: Added audio_path: {audio_path}")

        # --- Handle FILE PATHS (comma-separated) ---
        # Note: Video files (.mp4) in file_paths are NOT extracted to frames
        # because some APIs (Gemini, etc.) can process video files directly.
        # Only video tensors from nodes get frame extraction.
        if file_path_str:
            paths = [p.strip() for p in file_path_str.split(",") if p.strip()]
            if paths:
                prompt_config["file_paths"] = paths if len(paths) > 1 else paths[0]
                logger.debug(f"PromptManager: Added {len(paths)} file_path(s).")

        # --- Handle URLS (comma-separated) ---
        if url_str:
            urls = [u.strip() for u in url_str.split(",") if u.strip()]
            if urls:
                prompt_config["urls"] = urls if len(urls) > 1 else urls[0]
                logger.debug(f"PromptManager: Added {len(urls)} url(s).")

        # Update the main context with the (potentially updated) prompt_config
        if prompt_config: # Only add if not empty
             output_context["prompt_config"] = prompt_config
             logger.info("PromptManager: Updated context with prompt_config.")
        else:
             logger.info("PromptManager: No prompt components provided.")
        
        # Final check: If provider_config has a system_message but prompt_config doesn't have text,
        # copy the system_message to prompt_config.text so downstream nodes can use it as the prompt
        if "provider_config" in output_context and "system_message" in output_context["provider_config"]:
            if "prompt_config" not in output_context:
                output_context["prompt_config"] = {}
            if "text" not in output_context.get("prompt_config", {}) or not output_context["prompt_config"]["text"]:
                output_context["prompt_config"]["text"] = output_context["provider_config"]["system_message"]
                logger.info("Copied system_message to prompt_config.text for downstream compatibility")

        # Add metadata about processed items
        if items:
            output_context["_processed_info"] = {
                "count": len(items),
                "types": [type(item).__name__ for item in items]
            }
            logger.info(f"Processed {len(items)} inputs")

        return (output_context,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "LLMPromptManager": LLMPromptManager
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptManager": "LLM Prompt Manager"
} 