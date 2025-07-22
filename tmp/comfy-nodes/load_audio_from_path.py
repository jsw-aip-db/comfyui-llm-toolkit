# comfy-nodes/load_audio_from_path.py
"""Load Audio From Path (LLMToolkit)

Utility node that takes a string path to an audio file, loads it,
previews it in the UI, and returns an `AUDIO` object for use with other
audio nodes. Unlike the built-in LoadAudio node, this one accepts an
arbitrary path from an input.
"""

from __future__ import annotations

import os
import shutil
import logging
import random

try:
    import torchaudio
    import torch
except ImportError:
    # This should not happen in a ComfyUI environment with audio nodes.
    torchaudio = None
    torch = None

try:
    import folder_paths
except ImportError:
    # Mock for local development
    class MockFolderPaths:
        def get_output_directory(self): return "output"
        def get_input_directory(self): return "input"
        def get_temp_directory(self): return "temp"
    folder_paths = MockFolderPaths()

# Import save_audio function from core nodes
try:
    from comfy_extras.nodes_audio import save_audio, SaveAudio
except ImportError:
    save_audio = None
    SaveAudio = object

logger = logging.getLogger(__name__)

class LoadAudioFromPath(SaveAudio if SaveAudio is not object else object):
    def __init__(self):
        if SaveAudio is not object:
            self.output_dir = folder_paths.get_temp_directory()
            self.type = "temp"
            self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "placeholder": "/absolute/or/relative/path.wav",
                        "tooltip": "Full path to the audio file.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "llm_toolkit/utils/audio"

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "load_and_preview"
    OUTPUT_NODE = True

    def load_and_preview(self, audio_path: str, prompt=None, extra_pnginfo=None):
        if torchaudio is None or torch is None:
            logger.error("torchaudio or torch is not installed. Cannot load or preview audio.")
            return (None, audio_path, {"ui": {"audio": []}})

        audio_path = audio_path.strip()
        if not audio_path or not os.path.exists(audio_path):
            logger.warning("LoadAudioFromPath: Audio path is empty or file does not exist: %s", audio_path)
            return (None, audio_path, {"ui": {"audio": []}})

        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_out = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except Exception as e:
            logger.error("Failed to load audio file %s: %s", audio_path, e, exc_info=True)
            return {"ui": {"audio": []}, "result": (None, audio_path)}

        # --- Preview generation --------------------------------------------------
        preview_dict = {"ui": {"audio": []}}
        if save_audio and SaveAudio is not object:
            try:
                preview_dict = save_audio(
                    self,
                    audio_out,
                    filename_prefix="preview",
                    format="flac",
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
            except Exception as e:
                logger.warning("save_audio preview failed: %s", e, exc_info=True)

        # Ensure we return the correct structure
        if not isinstance(preview_dict, dict):
            preview_dict = {"ui": {"audio": []}}

        preview_dict["result"] = (audio_out, audio_path)
        return preview_dict


NODE_CLASS_MAPPINGS = {"LoadAudioFromPath": LoadAudioFromPath}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadAudioFromPath": "Load Audio From Path (LLMToolkit)"} 