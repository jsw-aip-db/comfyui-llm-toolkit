import logging
from typing import Tuple

# Initialize logger
logger = logging.getLogger(__name__)

class FramesToSeconds:
    """ComfyUI node that converts a frame count into seconds based on the
    frames-per-second (fps) value.

    The node outputs both a floating-point and an integer representation of the
    duration in seconds, plus it echoes the fps and frames values.
    """

    # ------------------------------------------------------------------
    # ComfyUI required class attributes
    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs.
        `frames` is the number of frames to convert.
        `fps` is an integer slider ranging from 8 to 30 frames-per-second.
        """
        return {
            "required": {
                "frames": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 1_000_000,
                        "tooltip": "Frame count representing the duration to convert.",
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 25,
                        "min": 8,
                        "max": 30,
                        "step": 1,
                        "tooltip": "Frames-per-second for the conversion (8–30).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "INT")
    RETURN_NAMES = (
        "duration_seconds",
        "duration_seconds_int",
        "fps",
        "frames",
    )
    FUNCTION = "convert"
    CATEGORY = "llm_toolkit/utils/audio"

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def convert(self, frames: int, fps: int) -> Tuple[float, int, float, int]:
        """Convert a frame count into seconds.

        Args:
            frames: Total number of frames.
            fps:    Frames-per-second value used for conversion.

        Returns:
            Tuple(duration_seconds_float, duration_seconds_int, fps_float, frames_int)
        """
        try:
            if fps <= 0:
                raise ValueError("fps must be a positive integer")

            duration_seconds = frames / fps
            duration_seconds_int = int(round(duration_seconds))

            logger.debug(
                "FramesToSeconds: frames=%s, fps=%s, duration_seconds=%s",
                frames,
                fps,
                duration_seconds,
            )

            return duration_seconds, duration_seconds_int, float(fps), frames
        except Exception as e:
            logger.error("FramesToSeconds: error during conversion: %s", e, exc_info=True)
            # Return zeros on failure
            return 0.0, 0, 0.0, 0


# -------------------------------------------------------------------------
# ComfyUI mappings so the node is discoverable
# -------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "FramesToSeconds": FramesToSeconds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FramesToSeconds": "Frames → Seconds (LLMToolkit)",
} 