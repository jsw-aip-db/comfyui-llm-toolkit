# transformers_provider.py
"""Provider wrapper that connects *send_request.py* to the local HuggingFace
Transformers backend.

This module lives inside the *comfy-nodes* package so it gets picked up by the
ComfyUI auto-import logic just like the other provider modules (e.g.
`openai_provider.py`).  The public coroutine `send_transformers_request` mirrors
other provider-specific functions such as `openai_api.send_openai_request` so
that *send_request.py* can treat it transparently.
"""

from __future__ import annotations

from typing import Any, Dict, List
import logging
import os
from pathlib import Path

import folder_paths  # Provided by ComfyUI runtime

# The heavy lifting (tokenisation, model loading, etc.) is delegated to
# *transformers_api.py* located at the repository root.  We simply re-export its
# main coroutine so callers can `from transformers_provider import
# send_transformers_request`.
from api.transformers_api import send_transformers_request  # type: ignore
from api.transformers_api import send_transformers_request as _hf_send_transformers_request  # type: ignore
from api.transformers_api import send_transformers_request_stream as _hf_send_transformers_request_stream  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Model discovery helpers – surf *models/transformers* and *models/LLM*
# -----------------------------------------------------------------------------

def _discover_models() -> List[str]:
    """Return a list of local transformer model directories and recommended models."""

    # Start with a curated list of recommended models that users can download.
    # These are identifiers, not local paths.
    recommended_models = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "Qwen/Qwen3-4B-AWQ",
        "SoybeanMilk/Kimi-VL-A3B-Thinking-2506-BNB-4bit",
    ]

    model_parent_dirs: List[Path] = []

    # Explicitly look inside the common folders used for HF style checkpoints.
    for fld in ("transformers", "LLM"):
        try:
            for p in folder_paths.get_folder_paths(fld):
                model_parent_dirs.append(Path(p))
        except Exception:
            continue

    # 1. Scan local directories and map model names to full paths.
    local_models_map: Dict[str, str] = {}
    for parent in model_parent_dirs:
        if not parent.is_dir():
            continue
        logger.debug("[Transformers-Provider] Recursively scanning for models in: %s", parent)
        for root, dirs, files in os.walk(parent, topdown=True):
            if "config.json" in files:
                model_path = Path(root)
                model_name = model_path.name
                
                # Prioritize existing entries, maybe from a more specific path
                if model_name not in local_models_map:
                    local_models_map[model_name] = model_path.as_posix()

                # Prune search: don't descend into model subdirectories.
                dirs[:] = []

    # 2. Build the final list, prioritizing local paths for recommended models.
    out: List[str] = []
    # Use a set on the final path/ID to prevent duplicates
    seen_entries: set[str] = set()

    for rec_model_id in recommended_models:
        rec_model_name = Path(rec_model_id).name
        
        # If a recommended model is found locally, use its local path.
        if rec_model_name in local_models_map:
            local_path = local_models_map[rec_model_name]
            if local_path not in seen_entries:
                 out.append(local_path)
                 seen_entries.add(local_path)
        else:
            # Otherwise, add the HuggingFace identifier so the user can download it.
            if rec_model_id not in seen_entries:
                out.append(rec_model_id)
                seen_entries.add(rec_model_id)

    # 3. Add any other local models that weren't in the recommended list.
    for model_name, model_path in local_models_map.items():
        if model_path not in seen_entries:
            out.append(model_path)
            seen_entries.add(model_path)
    
    return sorted(out)


# -----------------------------------------------------------------------------
# PUBLIC: async send_transformers_request – thin wrapper with logging
# -----------------------------------------------------------------------------

async def send_transformers_request_provider(
    *,
    base64_images: List[str],
    base64_audio: List[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]] | None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    precision: str = "fp16",
    **kwargs,
):
    """Thin wrapper that forwards to *transformers_api* but adds helpful logging."""
    logger.info("[Transformers-Provider] model=%s | precision=%s", model, precision)
    return await _hf_send_transformers_request(
        base64_images=base64_images,
        base64_audio=base64_audio,
        model=model,
        system_message=system_message,
        user_message=user_message,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        precision=precision,
    )


# Expose the coroutine under the canonical name so callers can simply do:
# `from transformers_provider import send_transformers_request`
send_transformers_request = send_transformers_request_provider 


# -----------------------------------------------------------------------------
# Streaming wrapper – exposes *send_transformers_request_stream* to callers
# -----------------------------------------------------------------------------

async def send_transformers_request_stream_provider(
    *,
    base64_images: List[str],
    base64_audio: List[str],
    model: str,
    system_message: str,
    user_message: str,
    messages: List[Dict[str, Any]] | None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    precision: str = "fp16",
    **kwargs,
):
    """Streaming variant that yields text chunks."""
    logger.info("[Transformers-Provider-Stream] model=%s | precision=%s", model, precision)
    async for chunk in _hf_send_transformers_request_stream(
        base64_images=base64_images,
        base64_audio=base64_audio,
        model=model,
        system_message=system_message,
        user_message=user_message,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        precision=precision,
    ):
        yield chunk


# Convenience alias
send_transformers_request_stream = send_transformers_request_stream_provider


# -----------------------------------------------------------------------------
# ComfyUI Node: Local Transformers Provider
# -----------------------------------------------------------------------------

try:
    from server import PromptServer  # Optional – only available inside ComfyUI runtime
    from aiohttp import web

    @PromptServer.instance.routes.get("/ComfyLLMToolkit/get_transformer_models")
    async def get_transformer_models_endpoint(request):
        """Frontend helper endpoint that returns the list of discovered local HF models."""
        try:
            models = _discover_models()
            if not models:
                models = ["No local models found"]
            return web.json_response(models)
        except Exception as e:
            logger.error("Error in get_transformer_models_endpoint: %s", e, exc_info=True)
            return web.json_response(["Error fetching models"], status=500)

    logger.info("TransformersProviderNode: /ComfyLLMToolkit/get_transformer_models endpoint registered")
except (ImportError, AttributeError):
    # Running outside the ComfyUI webserver; skip endpoint registration.
    pass


class LocalTransformersProviderNode:
    """Node that lets the user pick a local HF model and outputs provider_config."""

    PROVIDER_NAME = "transformers"

    @classmethod
    def INPUT_TYPES(cls):
        models = _discover_models()
        if not models:
            models = ["No local models found"]
        default_model = models[0] if models else ""
        return {
            "required": {
                "llm_model": (
                    models,
                    {
                        "default": default_model,
                        "tooltip": "Select a local HuggingFace model directory.",
                    },
                ),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure_transformers"
    CATEGORY = "llm_toolkit/providers"

    @classmethod
    def IS_CHANGED(cls, llm_model: str, context=None):
        # Recompute if model path changes
        return llm_model

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def configure_transformers(self, llm_model: str, context=None):
        logger.info("LocalTransformersProviderNode: configuring model '%s'", llm_model)

        # Build provider configuration dictionary
        provider_config = {
            "provider_name": self.PROVIDER_NAME,
            "llm_model": llm_model.strip(),
            "api_key": "1234",  # placeholder so downstream nodes treat as local provider (no key).
            "base_ip": None,
            "port": None,
        }

        # Merge with incoming context to retain pipeline data
        if context is not None and isinstance(context, dict):
            context["provider_config"] = provider_config
            output = context
        else:
            output = {"provider_config": provider_config}

        return (output,)


# Explicit node registration (optional, auto-discovery already handled by __init__)
NODE_CLASS_MAPPINGS = {
    "LocalTransformersProviderNode": LocalTransformersProviderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalTransformersProviderNode": "Local Transformers Provider (LLMToolkit)",
} 