import os
import sys
import logging
import json
import glob
import requests
import urllib.parse
import folder_paths
from typing import Any, Dict, Optional, Tuple

from io import BytesIO

# Ensure parent directory (project root) is on sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llmtoolkit_utils import TENSOR_SUPPORT
from context_payload import extract_context

logger = logging.getLogger(__name__)


class PreviewOutputs:  # noqa: N801 – stay consistent with other nodes
    """Read previously saved Suno outputs and expose tensors + lyrics for ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("*", {}),
            }
        }

    RETURN_TYPES = ("*", "AUDIO", "IMAGE", "STRING")
    RETURN_NAMES = ("context", "audio", "image", "lyrics")
    FUNCTION = "preview"
    CATEGORY = "🔗llm_toolkit/utils/audio"
    OUTPUT_NODE = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_audio_tensor(self, filepath: str) -> Dict[str, Any]:
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(filepath)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            logger.error("PreviewOutputs: Failed to load audio %s – %s", filepath, e)
            import torch
            return {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}

    def _load_image_tensor(self, filepath: str):
        try:
            if not TENSOR_SUPPORT:
                raise RuntimeError("tensor support disabled")
            from PIL import Image
            import numpy as np
            import torch
            
            img = Image.open(filepath).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            return img_tensor
        except Exception as e:
            logger.error("PreviewOutputs: Failed to load image %s – %s", filepath, e)
            import torch
            return torch.zeros((1, 64, 64, 3))

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def preview(self, context: Any) -> Tuple[Dict[str, Any], Any, Any, str]:
        # Unwrap context if needed
        if not isinstance(context, dict):
            context = extract_context(context) or {}

        output_dir = context.get("output_dir") or folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)

        def _download_file(url: str, dest_folder: str, filename: Optional[str] = None) -> str:
            try:
                if not filename:
                    filename = os.path.basename(urllib.parse.urlparse(url).path)
                dest_path = os.path.join(dest_folder, filename)
                if not os.path.isfile(dest_path):
                    with requests.get(url, stream=True, timeout=60) as resp:
                        resp.raise_for_status()
                        with open(dest_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                return dest_path
            except Exception as e:
                logger.warning("PreviewOutputs: failed to download %s – %s", url, e)
                return ""

        # -----------------------------
        # Load metadata
        # -----------------------------
        meta_path = context.get("metadata_path") or os.path.join(output_dir, "metadata.json")
        title = ""
        lyrics = ""
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                title = meta.get("title", "")
                lyrics = meta.get("lyrics", "")
            except Exception as e:
                logger.warning("PreviewOutputs: Failed to read metadata – %s", e)

        # Attempt to locate or download audio/image
        audio_files = []
        image_files = []

        # Always attempt to determine URLs from context
        if context:
            audio_url = context.get("suno_audio_url")
            image_url = context.get("suno_image_url")

            # Parse raw response to find urls if not present
            if not (audio_url and image_url):
                raw = context.get("suno_raw_response", {})
                if isinstance(raw, dict):
                    tracks = raw.get("data", {}).get("data", [])
                    for tr in tracks:
                        if not audio_url:
                            audio_url = tr.get("audioUrl") or tr.get("audio_url") or tr.get("sourceAudioUrl") or tr.get("source_audio_url")
                        if not image_url:
                            image_url = tr.get("imageUrl") or tr.get("image_url") or tr.get("sourceImageUrl") or tr.get("source_image_url")
                        if audio_url and image_url:
                            break

            if audio_url and not audio_files:
                downloaded = _download_file(audio_url, output_dir)
                if downloaded:
                    audio_files = [downloaded]

            if image_url and not image_files:
                downloaded = _download_file(image_url, output_dir)
                if downloaded:
                    image_files = [downloaded]

        # Load tensors (or placeholders)
        if audio_files:
            audio_tensor = self._load_audio_tensor(audio_files[0])
        else:
            import torch
            audio_tensor = {"waveform": torch.zeros((1, 1, 44100)), "sample_rate": 44100}

        if image_files:
            image_tensor = self._load_image_tensor(image_files[0])
        else:
            import torch
            image_tensor = torch.zeros((1, 64, 64, 3))

        combined_lyrics = f"{title}\n\n{lyrics}" if title or lyrics else ""
        return (context, audio_tensor, image_tensor, combined_lyrics)


# ---------------------------------------------
# Register node with ComfyUI
# ---------------------------------------------
NODE_CLASS_MAPPINGS = {
    "PreviewOutputs": PreviewOutputs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewOutputs": "Preview Audio/Image Outputs (🔗LLMToolkit)",
} 