# comfy-nodes/load_audio_from_path.py
"""Load Audio From Path - Bridge node to load audio files into ComfyUI format."""

import os
import torch
import logging
from typing import Tuple, Dict, Any

# Try to import torchaudio for audio loading
try:
    import torchaudio
except ImportError:
    print("Warning: torchaudio is not installed. The LoadAudioFromPath node will not work.")
    print("Please run: pip install torchaudio")
    torchaudio = None

logger = logging.getLogger(__name__)


class LoadAudioFromPath:
    """Load an audio file from a file path string and convert to ComfyUI audio format."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # This input will connect to the 'audio_path' output of GenerateSpeech
                "audio_path": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    # We output the special AUDIO type that other audio nodes understand
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"
    CATEGORY = "llm_toolkit/utils/audio"  # Consistent with LoadVideoFromPath

    def load_audio(self, audio_path: str) -> Tuple[Dict[str, Any]]:
        """Load audio file and return as ComfyUI audio format."""
        
        if not torchaudio:
            raise ImportError("torchaudio is not installed. Cannot run LoadAudioFromPath.")
        
        # Validate the input path
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"LoadAudioFromPath: Audio path is empty or file does not exist: {audio_path}")
            # Return a silent, empty audio object to prevent crashing the workflow
            return ({"waveform": torch.zeros((1, 1, 1)), "sample_rate": 44100},)

        try:
            # Load the audio file using torchaudio
            # This is the most reliable way to get the waveform and sample rate
            waveform, sample_rate = torchaudio.load(audio_path)

            # Ensure the waveform tensor has a batch dimension for ComfyUI compatibility
            # Standard audio nodes expect [batch_size, num_channels, num_samples]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # Adds the batch dimension

            # Create the AUDIO dictionary object
            audio_output = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
            logger.info(f"Loaded audio from {audio_path} with shape {waveform.shape} and sample rate {sample_rate}")
            
            return (audio_output,)

        except Exception as e:
            logger.error(f"Error loading audio from path '{audio_path}': {e}", exc_info=True)
            # Return a silent, empty audio object on failure
            return ({"waveform": torch.zeros((1, 1, 1)), "sample_rate": 44100},)


# Add the new node to ComfyUI's list of recognized nodes
NODE_CLASS_MAPPINGS = {
    "LoadAudioFromPath": LoadAudioFromPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioFromPath": "Load Audio From Path (LLMToolkit)"
} 