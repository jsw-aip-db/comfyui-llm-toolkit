import logging
import math
from typing import Tuple, List, Union

# Initialize logger
logger = logging.getLogger(__name__)

class AudioDurationFrames:
    """ComfyUI node that reads an AUDIO dict (from LoadAudio or other audio-producing node),
    calculates its duration in milliseconds (named *a*), then applies the equation
        frames = a / 1000 * b
    where *b* is the chosen frame-rate (fps).

    The node outputs two integers:
        1. duration_ms – the audio length in milliseconds
        2. frames      – the computed frame count according to the selected fps
    """

    # ---------------------------------------------------------------------
    # ComfyUI required class attributes
    # ---------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs.
        `fps` is an integer slider ranging from 8 to 30 frames-per-second.
        """
        return {
            "required": {
                "audio": ("AUDIO", {}),
                # fps slider: integer between 8 and 30
                "fps": ("INT", {"default": 25, "min": 8, "max": 30, "step": 1, "tooltip": "Frames-per-second to use in the a/1000*b equation."}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 10000, "tooltip": "Value used to divide total frames (context = frames / num_frames)."}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "duration_ms",
        "frames",
        "context_times",
        "maximum_length_allowed",
        "end_time",
    )
    FUNCTION = "calculate"
    CATEGORY = "llm_toolkit/utils/audio"

    # ---------------------------------------------------------------------
    # Core logic
    # ---------------------------------------------------------------------
    def calculate(self, audio: dict, fps: Union[str, int], num_frames: int) -> Tuple[int, int, int, int, str]:
        """Compute duration in milliseconds and resulting frame count.

        Args:
            audio: Dict produced by LoadAudio (expects keys `waveform` and `sample_rate`).
            fps:   Selected frame-rate – integer between 8 and 30 (frames-per-second).
            num_frames: Value used to divide total frames (context = frames / num_frames).

        Returns:
            Tuple(duration_ms, frames, context_times, maximum_length_allowed, end_time)  all as ints and str.
        """
        try:
            waveform = audio.get("waveform")
            sample_rate = audio.get("sample_rate")
            if waveform is None or sample_rate is None:
                raise ValueError("Audio dictionary must contain 'waveform' and 'sample_rate'.")

            # Assume shape: (batch, channels, samples) or (channels, samples)
            # We compute duration based on the *first* sample in the batch.
            if waveform.ndim == 3:
                num_samples = waveform.shape[-1]
            elif waveform.ndim == 2:
                num_samples = waveform.shape[-1]
            else:
                raise ValueError(f"Unexpected waveform dimensions: {waveform.shape}")

            duration_ms = int(round(num_samples / sample_rate * 1000))

            fps_int = int(fps) if isinstance(fps, str) else fps
            frames = int(round(duration_ms / 1000 * fps_int))

            # Compute how many times num_frames fits into frames (floor division)
            if num_frames <= 0:
                raise ValueError("num_frames must be a positive integer")
            context_times = frames // num_frames
            maximum_length_allowed = context_times * num_frames

            # Compute end_time string in M:SS format based on maximum_length_allowed frames
            seconds_total = maximum_length_allowed / fps_int if fps_int else 0
            seconds_int = math.ceil(seconds_total)
            minutes = seconds_int // 60
            seconds_rem = seconds_int % 60
            end_time_str = f"{minutes}:{seconds_rem:02d}"

            logger.debug(
                "AudioDurationFrames: num_samples=%s, sample_rate=%s, duration_ms=%s, fps=%s, frames=%s", 
                num_samples, sample_rate, duration_ms, fps_int, frames,
            )

            return (
                duration_ms,
                frames,
                context_times,
                maximum_length_allowed,
                end_time_str,
            )
        except Exception as e:
            logger.error("AudioDurationFrames: error during calculation: %s", e, exc_info=True)
            # Return zeros on failure to avoid breaking the workflow
            return (0, 0, 0, 0, "0:00")


# -------------------------------------------------------------------------
# ComfyUI mappings so the node is discoverable
# -------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "AudioDurationFrames": AudioDurationFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioDurationFrames": "Audio Duration → Frames (LLMToolkit)",
} 