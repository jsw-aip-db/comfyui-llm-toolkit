from __future__ import annotations

"""vLLM provider for LLM-Toolkit (ComfyUI).

This module exposes two public coroutines – `send_vllm_request` and
`send_vllm_request_stream` – whose signatures match the other provider_* 
helpers (openai_api.send_openai_request, transformers_provider.send_transformers_request, …).

It also registers a *Local vLLM Provider* node so that ComfyUI users can pick
local checkpoints from the usual `models/transformers` or `models/LLM` 
folders and run them via vLLM.

Currently only **text-only** generation is implemented.  The stub accepts
`base64_images` / `base64_audio` parameters but ignores them.  Multi-modal
support can be added later once vLLM’s Python API exposes a stable
interface for `multi_modal_data`.
"""

import asyncio
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, AsyncGenerator

# --- Optional dependency guard ------------------------------------------------
try:
    from vllm import LLM, SamplingParams  # type: ignore
    _VLLM_AVAILABLE = True
except Exception as _e:  # pragma: no cover – vllm missing
    _VLLM_AVAILABLE = False
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

# Re-use the local model discovery helper from transformers_provider
try:
    from transformers_provider import _discover_models as _hf_discover_models  # type: ignore
except Exception:
    def _hf_discover_models() -> List[str]:
        return []

# ComfyUI specific (optional – only when running inside the server)
try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None  # type: ignore

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
#  Engine cache – keep one LLM instance per model path to avoid reloads
# -----------------------------------------------------------------------------
_ENGINE_CACHE: Dict[str, LLM] = {}
_CACHE_LOCK = asyncio.Lock()

async def _ensure_engine(model: str) -> LLM:  # type: ignore[valid-type]
    """Load *model* once and return the cached vLLM engine."""
    if not _VLLM_AVAILABLE:
        raise RuntimeError("vllm_provider – vLLM not installed.  `pip install vllm`.")

    async with _CACHE_LOCK:
        if model in _ENGINE_CACHE:
            return _ENGINE_CACHE[model]

        logger.info("[vLLM] Loading model %s …", model)
        # NOTE: LLM() immediately loads the weights synchronously.  We therefore
        # off-load it to the default executor so the asyncio loop remains
        # responsive.
        loop = asyncio.get_running_loop()
        engine = await loop.run_in_executor(None, lambda: LLM(model=model))
        _ENGINE_CACHE[model] = engine
        return engine

# -----------------------------------------------------------------------------
#  Public non-streaming helper (mirrors other providers)
# -----------------------------------------------------------------------------
async def send_vllm_request(
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
    **kwargs,
) -> Dict[str, Any]:
    """Run inference with a local vLLM engine and return an OpenAI-style dict."""

    try:
        engine = await _ensure_engine(model)
    except Exception as exc:
        logger.error("vllm_provider – could not load engine: %s", exc, exc_info=True)
        return {"choices": [{"message": {"content": f"Error: {exc}"}}]}

    # Compose chat history in OpenAI style
    if messages is None:
        messages = []
    if system_message:
        messages = [{"role": "system", "content": system_message}] + messages
    if user_message:
        messages = messages + [{"role": "user", "content": user_message}]

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repeat_penalty,
    ) if _VLLM_AVAILABLE else None

    loop = asyncio.get_running_loop()
    try:
        # Prefer the newer chat interface (vLLM ≥0.9) – falls back to template.
        if hasattr(engine, "chat"):
            content: str = await loop.run_in_executor(
                None,
                lambda: engine.chat([messages], sampling)[0].outputs[0].text,
            )
        else:
            # Manual template for older versions
            try:
                from transformers import AutoTokenizer  # local import, optional
            except Exception as exc:
                err = f"transformers not available – required for vLLM <0.9 templates ({exc})"
                logger.error(err)
                return {"choices": [{"message": {"content": err}}]}

            tok = await loop.run_in_executor(None, lambda: AutoTokenizer.from_pretrained(model, trust_remote_code=True))
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            content: str = await loop.run_in_executor(
                None,
                lambda: engine.generate([prompt], sampling)[0].outputs[0].text,
            )
    except Exception as exc:
        logger.error("vLLM inference error: %s", exc, exc_info=True)
        return {"choices": [{"message": {"content": f"Error: {exc}"}}]}

    return {"choices": [{"message": {"content": content}}]}

# -----------------------------------------------------------------------------
#  Streaming helper – crude chunking until vLLM exposes token streaming
# -----------------------------------------------------------------------------
async def send_vllm_request_stream(
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
    chunk_size: int = 120,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """Yield text chunks.  Currently performs post-hoc chunking."""
    full = await send_vllm_request(
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
    )
    content = full.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        yield "[vLLM: empty response]"
        return
    for chunk in textwrap.wrap(content, chunk_size):
        yield chunk

# -----------------------------------------------------------------------------
#  ComfyUI Node – expose provider in the graph editor
# -----------------------------------------------------------------------------
class LocalVLLMProviderNode:
    """Node that outputs a provider_config for running *vLLM* locally."""

    PROVIDER_NAME = "vllm"

    @classmethod
    def INPUT_TYPES(cls):
        models = _hf_discover_models() or ["No local models found"]
        default = models[0] if models else ""
        return {
            "required": {
                "llm_model": (
                    models,
                    {"default": default, "tooltip": "Select a local HF model directory."},
                ),
            },
            "optional": {"context": ("*", {})},
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("context",)
    FUNCTION = "configure"
    CATEGORY = "llm_toolkit/providers"

    @classmethod
    def IS_CHANGED(cls, llm_model: str, context=None):  # noqa: N802 (ComfyUI convention)
        return llm_model  # recompute when model path changes

    def configure(self, llm_model: str, context=None):
        logger.info("LocalVLLMProviderNode: configuring model '%s'", llm_model)
        cfg = {
            "provider_name": self.PROVIDER_NAME,
            "llm_model": llm_model.strip(),
            # keep placeholders for consistency with other providers
            "api_key": "1234",
            "base_ip": None,
            "port": None,
        }
        if isinstance(context, dict):
            context["provider_config"] = cfg
            return (context,)
        return ({"provider_config": cfg},)

# Node registration dictionaries so ComfyUI auto-discovers the node
NODE_CLASS_MAPPINGS = {
    "LocalVLLMProviderNode": LocalVLLMProviderNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalVLLMProviderNode": "Local vLLM Provider (LLMToolkit)",
} 