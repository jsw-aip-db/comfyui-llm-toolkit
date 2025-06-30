# generate_image.py
import os
import sys
import logging
from typing import Any, Dict, Optional, Tuple

# Ensure parent directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Relative imports
try:
    from llmtoolkit_utils import tensor_to_base64, process_images_for_comfy, TENSOR_SUPPORT, get_api_key
    from send_request import run_async # Assuming send_request.py exists for run_async
    # Import the specific API call function from root directory
    from openai_api import send_openai_image_generation_request
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Failed relative imports in generate_image.py. Check file structure and __init__.py.")
    # Fallback to absolute imports if run standalone or structure differs
    try:
        from llmtoolkit_utils import tensor_to_base64, process_images_for_comfy, TENSOR_SUPPORT, get_api_key
        from send_request import run_async
        from openai_api import send_openai_image_generation_request
    except ImportError:
        logging.error("Failed to import required modules for generate_image.py")
        TENSOR_SUPPORT = False

logger = logging.getLogger(__name__)

# Attempt to import ComfyUI's PreviewImage helper for temporary previews. Fallback gracefully if not available.
try:
    from nodes import PreviewImage  # ComfyUI builtin located at ComfyUI/nodes.py
except Exception:
    PreviewImage = None

# Helper to extract dict from payload
from context_payload import extract_context

class GenerateImage:
    """
    Generates an image using configuration and prompt details from the context.
    Currently supports OpenAI DALL-E/GPT-Image providers.
    Outputs the generated image(s) as a ComfyUI IMAGE tensor and updates the context.
    """

    DEFAULT_PROVIDER = "openai"
    DEFAULT_MODEL = "gpt-image-1" # Favors the newer model
    # Default prompt shown in the node UI when no context is provided
    DEFAULT_PROMPT = (
        """
A children's book drawing of a veterinarian using a stethoscope to 
listen to the heartbeat of a baby otter.
"""
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Direct prompt input (overridden if context provides one)
                "prompt": ("STRING", {"multiline": True, "default": cls.DEFAULT_PROMPT}),
                "mode": (["generate", "edit", "variation"], {"default": "generate", "tooltip": "Choose the type of image request."}),
            },
            "optional": {
                "context": ("*", {}),  # Combined config and prompt info (optional)
            }
        }

    RETURN_TYPES = ("*", "IMAGE",) # Keep IMAGE for direct use/preview
    RETURN_NAMES = ("context", "image",)
    FUNCTION = "generate"
    CATEGORY = "llm_toolkit"
    OUTPUT_NODE = True

    def generate(self, prompt: str, mode: str = "generate", context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Any]:
        """
        Extracts config, calls the appropriate API, processes the response.
        """
        logger.info("GenerateImage node executing...")

        # Allow context to be optional; unwrap from ContextPayload if provided
        if context is None:
            context = {}
        elif not isinstance(context, dict):
            context = extract_context(context)
            if not isinstance(context, dict):
                context = {}

        # --- Extract Configurations ---
        provider_config = context.get("provider_config", {})
        prompt_config = context.get("prompt_config", {})
        generation_config = context.get("generation_config", {})

        # --- Determine Provider and Model ---
        llm_provider = provider_config.get("provider_name", self.DEFAULT_PROVIDER).lower()
        # Default to gpt-image-1 if provider is openai and model is missing/empty
        default_model_for_provider = self.DEFAULT_MODEL if llm_provider == "openai" else ""
        llm_model = provider_config.get("llm_model") or generation_config.get("model") or default_model_for_provider

        # --- Get API Key ---
        api_key = provider_config.get("api_key", "")

        # If API key is missing or placeholder, attempt automatic resolution via utils.get_api_key
        if (not api_key or api_key in {"1234", "", None}) and llm_provider == "openai":
            try:
                api_key = get_api_key("OPENAI_API_KEY", llm_provider)
                logger.info("GenerateImage: Retrieved API key via get_api_key helper.")
            except ValueError as _e:
                logger.warning("GenerateImage: get_api_key failed – %s", _e)

        # After retries, ensure we have a usable key for providers that need one
        if llm_provider == "openai" and not api_key:
            logger.error(f"GenerateImage: Missing API key for provider '{llm_provider}'.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = f"Missing API key for {llm_provider}"

            ui_dict = {}
            if PreviewImage is not None:
                try:
                    preview_node = PreviewImage()
                    preview_res = preview_node.save_images(placeholder_img, filename_prefix="GenerateImageError")
                    ui_dict = preview_res.get("ui", {})
                except Exception:
                    pass

            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        # --- Get Prompt Details ---
        # Choose the prompt text: prefer context's prompt_config if present, otherwise use the direct node input
        prompt_text = prompt_config.get("text") or prompt
        image_b64 = prompt_config.get("image_base64", None) # From PromptManager
        mask_b64 = prompt_config.get("mask_base64", None)   # From PromptManager

        if not prompt_text and not image_b64:
            logger.error("GenerateImage: Requires at least a text prompt or an input image.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = "Missing text prompt or input image"

            ui_dict = {}
            if PreviewImage is not None:
                try:
                    preview_node = PreviewImage()
                    preview_res = preview_node.save_images(placeholder_img, filename_prefix="GenerateImageError")
                    ui_dict = preview_res.get("ui", {})
                except Exception:
                    pass

            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        # --- Get Generation Settings (with defaults) ---
        n = generation_config.get("n", 1)
        size = generation_config.get("size")  # could be None
        response_format = generation_config.get("response_format", "b64_json")
        user = generation_config.get("user", None)

        # Model-specific settings
        quality = None
        style = None
        background = None
        output_format_gpt = None
        output_compression_gpt = None
        moderation_gpt = None

        if llm_model == "dall-e-3":
            quality = generation_config.get("quality_dalle3", "standard")
            style = generation_config.get("style_dalle3", "vivid")
        elif llm_model == "gpt-image-1":
            quality = generation_config.get("quality_gpt", "auto")
            background = generation_config.get("background_gpt", "auto")
            output_format_gpt = generation_config.get("output_format_gpt", "png")
            output_compression_gpt = generation_config.get("output_compression_gpt", None) # Default handled by API
            moderation_gpt = generation_config.get("moderation_gpt", "auto")
            response_format = "b64_json" # Override for gpt-image-1
        elif llm_model == "dall-e-2":
            quality = "standard" # Only option

        # --- Determine Mode (Generate, Edit, Variation) from dropdown ---
        mode = str(mode).lower().strip()
        if mode not in {"generate", "edit", "variation"}:
            logger.warning("GenerateImage: Unknown mode '%s', defaulting to 'generate'.", mode)
            mode = "generate"

        edit_mode = mode == "edit"
        variation_mode = mode == "variation"

        # If in edit mode with image input and no explicit size passed, choose size based on image dims
        if edit_mode and image_b64 and not size:
            from llmtoolkit_utils import get_dims_from_base64, choose_openai_size
            dims = get_dims_from_base64(image_b64)
            if dims:
                w, h = dims
                size = choose_openai_size(w, h, llm_model)
                logger.info(f"GenerateImage: Auto-selected size '{size}' for edit request based on input image {w}x{h}.")

        # Fallback default if still None
        if not size:
            size = "1024x1024"

        # Validate requirements for chosen mode
        if edit_mode and not image_b64:
            logger.error("GenerateImage: 'edit' mode requires at least one reference image.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = "Edit mode requires both image and mask"
            ui_dict = {}
            if PreviewImage is not None:
                try:
                    ui_dict = PreviewImage().save_images(placeholder_img, filename_prefix="GenerateImageError").get("ui", {})
                except Exception:
                    pass
            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        # Mask requirement specific for DALL·E 2 edits
        if edit_mode and llm_model == "dall-e-2" and not mask_b64:
            logger.error("GenerateImage: 'edit' mode with dall-e-2 requires a mask.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = "DALL·E 2 edit requires a mask"
            ui_dict = {}
            if PreviewImage is not None:
                try:
                    ui_dict = PreviewImage().save_images(placeholder_img, filename_prefix="GenerateImageError").get("ui", {})
                except Exception:
                    pass
            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        if variation_mode and not image_b64:
            logger.error("GenerateImage: 'variation' mode selected but input image missing.")
            placeholder_img, _ = process_images_for_comfy(None)
            context["error"] = "Variation mode requires an input image"
            ui_dict = {}
            if PreviewImage is not None:
                try:
                    ui_dict = PreviewImage().save_images(placeholder_img, filename_prefix="GenerateImageError").get("ui", {})
                except Exception:
                    pass
            return {"ui": ui_dict, "result": (context, placeholder_img,)}

        # --- Call API ---
        output_image_tensor = None
        raw_api_response = None
        error_message = None

        try:
            if llm_provider == "openai":
                logger.info(f"Calling OpenAI Image API (model: {llm_model}, mode: {'edit' if edit_mode else 'variation' if variation_mode else 'generate'})")
                # Use run_async to call the async API function from sync context
                raw_api_response = run_async(
                    send_openai_image_generation_request(
                        api_key=api_key,
                        model=llm_model,
                        prompt=prompt_text,
                        n=n,
                        size=size,
                        quality=quality,
                        style=style,
                        response_format=response_format,
                        user=user,
                        background=background,
                        output_format_gpt=output_format_gpt,
                        output_compression_gpt=output_compression_gpt,
                        moderation_gpt=moderation_gpt,
                        image_base64=image_b64,
                        mask_base64=mask_b64,
                        edit_mode=edit_mode,
                        variation_mode=variation_mode
                    )
                )

                # --- Process Response ---
                if raw_api_response and raw_api_response.get("data"):
                    logger.info(f"Received {len(raw_api_response['data'])} image(s) from API.")
                    # Use the utility to convert API response (list of {'b64_json':...} or {'url':...}) to ComfyUI tensor
                    output_image_tensor, _ = process_images_for_comfy(raw_api_response) # Discard mask from util
                else:
                    error_message = "API response did not contain expected image data."
                    logger.error(f"{error_message} Response: {raw_api_response}")

            else:
                error_message = f"Provider '{llm_provider}' not currently supported by GenerateImage node."
                logger.error(error_message)

        except Exception as e:
            error_message = f"Error during image generation API call: {str(e)}"
            logger.error(error_message, exc_info=True)

        # --- Update Context and Return ---
        output_context = context.copy() # Work on a copy

        if error_message:
            output_context["error"] = error_message
        if raw_api_response:
            output_context["image_generation_response"] = raw_api_response # Store raw response

        if output_image_tensor is not None:
            output_context["generated_image_tensor"] = output_image_tensor # Store tensor in context
            logger.info("GenerateImage node finished successfully.")

            # ---------------------------------------------
            # Prepare preview image for ComfyUI UI panel
            # ---------------------------------------------
            ui_dict = {}
            if PreviewImage is not None:
                try:
                    preview_node = PreviewImage()
                    preview_res = preview_node.save_images(output_image_tensor, filename_prefix="GenerateImage")
                    ui_dict = preview_res.get("ui", {})
                except Exception as e:
                    logger.warning("GenerateImage: Failed to create preview image – %s", e, exc_info=True)

            return {"ui": ui_dict, "result": (output_context, output_image_tensor,)}
        else:
            # Return placeholder image if generation failed
            logger.warning("GenerateImage node failed, returning placeholder image.")
            placeholder_img, _ = process_images_for_comfy(None)

            ui_dict = {}
            if PreviewImage is not None:
                try:
                    preview_node = PreviewImage()
                    preview_res = preview_node.save_images(placeholder_img, filename_prefix="GenerateImageError")
                    ui_dict = preview_res.get("ui", {})
                except Exception:
                    pass

            return {"ui": ui_dict, "result": (output_context, placeholder_img,)}

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "GenerateImage": GenerateImage
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateImage": "Generate Image (LLMToolkit)"
} 