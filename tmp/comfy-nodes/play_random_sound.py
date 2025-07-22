# Random sound player node for ComfyUI – chooses or forces a .mp3 from ../sounds
# Part of LLM-Toolkit utils.

import os
import random
import logging
from typing import List, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# --- Wildcard AnyType matching behaviour (mirrors SwitchAny implementation) ---
class AnyType(str):
    def __ne__(self, __value: object) -> bool:  # noqa: D401,E501
        return False

# Wildcard token used by ComfyUI for "any" typing
any = AnyType("*")


class PlayRandomSound:
    """Play an audio notification from *sounds/*

    • If *sound* == "(random)" → picks a random .mp3 from the folder.
    • Otherwise the selected filename is played (if present).

    The sounds folder lives at:
        custom_nodes/llm-toolkit/sounds/
    """

    # Resolve sounds directory relative to llm-toolkit root
    _SOUNDS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "sounds"))

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @classmethod
    def _list_mp3_files(cls) -> List[str]:
        """Return list of .mp3 filenames (no paths) inside _SOUNDS_DIR."""
        if not os.path.isdir(cls._SOUNDS_DIR):
            return []
        return sorted([f for f in os.listdir(cls._SOUNDS_DIR) if f.lower().endswith(".mp3")])

    @classmethod
    def _get_random_mp3(cls) -> str | None:
        files = cls._list_mp3_files()
        if not files:
            return None
        return os.path.join(cls._SOUNDS_DIR, random.choice(files))

    # ------------------------------------------------------------------
    # ComfyUI node spec
    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        mp3_choices = cls._list_mp3_files()
        dropdown = ["(random)"] + mp3_choices if mp3_choices else ["(none)"]

        return {
            "required": {
                "any": (any, {}),
                "mode": (["always", "on empty queue"], {}),
                "volume": ("FLOAT", {"min": 0, "max": 1, "step": 0.1, "default": 0.5}),
                "sound": (dropdown, {"default": "(random)", "tooltip": "Select a sound or (random)"}),
            }
        }

    FUNCTION = "nop"
    INPUT_IS_LIST = True  # still accept list wrapping for convenience
    OUTPUT_IS_LIST = ()   # no data outputs
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    CATEGORY = "llm_toolkit/utils/audio"

    # ------------------------------------------------------------------
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):  # noqa: D401
        """Always considered changed so re-evaluates sound selection each run."""
        return float("NaN")

    # ------------------------------------------------------------------
    def nop(self, any, mode, volume, sound):  # noqa: D401
        """Pass through *any* and send UI message to play chosen sound."""
        # Normalise *sound* selection param (may arrive wrapped in list)
        if isinstance(sound, list):
            sound = sound[0] if sound else "(none)"
        
        # Normalise *volume* param (may arrive wrapped in list)
        if isinstance(volume, list):
            volume = volume[0] if volume else 0.5

        if sound == "(random)":
            chosen = self._get_random_mp3()
        elif sound == "(none)":
            chosen = None
        else:
            candidate = os.path.join(self._SOUNDS_DIR, str(sound))
            chosen = candidate if os.path.isfile(candidate) else None

        # Prepare UI payload expected by ComfyUI frontend.
        # Only the "audio" key should be present and mapped to a list of file paths.
        ui_payload: Dict[str, Any] = {"audio": []}
        if chosen:
            ui_payload["audio"] = [chosen]

            # Attempt local playback (server-side) for immediate feedback.
            # Run in a daemon thread so we don't block the graph execution.
            def _play_local(path: str):
                """Play audio with comprehensive error handling to prevent crashes."""
                try:
                    logger.info(f"PlayRandomSound: Attempting to play {path}")
                    
                    # Preferred: playsound (cross-platform, MP3 support via OS codecs)
                    try:
                        from playsound import playsound  # type: ignore
                        try:
                            playsound(path, block=False)  # non-blocking when supported
                            logger.info("PlayRandomSound: Successfully played with playsound (non-blocking)")
                            return
                        except TypeError:
                            # Older playsound versions don't support block kwarg
                            playsound(path)
                            logger.info("PlayRandomSound: Successfully played with playsound (blocking)")
                            return
                    except ImportError:
                        logger.debug("PlayRandomSound: playsound not available")
                    except Exception as e:
                        logger.warning(f"PlayRandomSound: playsound failed - {type(e).__name__}: {str(e)}")
                    
                    # Fallback: simpleaudio (requires PCM WAV) – attempt conversion on the fly
                    try:
                        import simpleaudio  # type: ignore
                        from pydub import AudioSegment  # type: ignore
                        
                        seg = AudioSegment.from_file(path)
                        # Set volume based on the node's volume parameter
                        seg = seg + (20 * (volume - 0.5))  # Adjust dB based on volume slider
                        play_obj = simpleaudio.play_buffer(
                            seg.raw_data, 
                            num_channels=seg.channels, 
                            bytes_per_sample=seg.sample_width, 
                            sample_rate=seg.frame_rate
                        )
                        play_obj.wait_done()
                        logger.info("PlayRandomSound: Successfully played with simpleaudio")
                        return
                    except ImportError:
                        logger.debug("PlayRandomSound: simpleaudio/pydub not available")
                    except Exception as e:
                        logger.warning(f"PlayRandomSound: simpleaudio failed - {type(e).__name__}: {str(e)}")
                    
                    # Windows-only final fallback using winsound for .wav files
                    try:
                        import sys
                        if sys.platform.startswith("win") and path.lower().endswith(".wav"):
                            import winsound  # type: ignore
                            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                            logger.info("PlayRandomSound: Successfully played with winsound")
                            return
                    except Exception as e:
                        logger.warning(f"PlayRandomSound: winsound failed - {type(e).__name__}: {str(e)}")
                    
                    logger.warning(f"PlayRandomSound: All playback methods failed for {path}")
                    
                except Exception as e:
                    # Catch-all to prevent any crash
                    logger.error(f"PlayRandomSound: Unexpected error in audio thread - {type(e).__name__}: {str(e)}", exc_info=True)

            import threading
            
            # Create thread with explicit exception handling
            thread = threading.Thread(target=_play_local, args=(chosen,), daemon=True)
            thread.start()

        # No data outputs – only side-effect via UI payload
        return {"ui": ui_payload}


# ---------------------------------------------------------------------------
# Node registration – required for ComfyUI to discover the node
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "PlayRandomSound": PlayRandomSound,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlayRandomSound": "Play Random Sound (LLMToolkit)",
} 