# comfy-nodes/generate_video.py
"""Generate Video using Veo 2 via Gemini API.

The node supports text-to-video for now. It polls the long-running
operation until completion and downloads the resulting MP4(s).
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from context_payload import extract_context

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from llmtoolkit_utils import get_api_key
except ImportError:
    get_api_key = None  # type: ignore

try:
    import folder_paths
except ImportError:
    folder_paths = None  # type: ignore

logger = logging.getLogger(__name__)


class GenerateVideo:
    DEFAULT_MODEL = "veo-2.0-generate-001"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic shot of sunrise over mountains"}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "video_paths")
    FUNCTION = "generate"
    CATEGORY = "llm_toolkit/generators"

    # ------------------------------------------------------------------
    def _save_videos(self, client, videos: List[Any], out_dir: Path, base_name: str) -> List[str]:
        paths: List[str] = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, vid in enumerate(videos):
            fname = f"{base_name}_{idx}.mp4"
            tgt = out_dir / fname
            try:
                client.files.download(file=vid.video)  # ensures local cache
                vid.video.save(str(tgt))
                paths.append(str(tgt))
            except Exception as dl_exc:
                logger.error("Failed saving video %s – %s", fname, dl_exc, exc_info=True)
        return paths

    # ------------------------------------------------------------------
    def generate(self, prompt: str, context: Optional[Any] = None):
        logger.info("GenerateVideo node executing…")

        # ------------ Context handling -------------
        if context is None:
            ctx: Dict[str, Any] = {}
        elif isinstance(context, dict):
            ctx = context.copy()
        else:
            ctx = extract_context(context)
            if not isinstance(ctx, dict):
                ctx = {"passthrough_data": context}

        provider_cfg = ctx.get("provider_config", {})
        gen_cfg = ctx.get("generation_config", {})

        # Default to gemini provider if none
        llm_provider = provider_cfg.get("provider_name", "gemini").lower()
        if llm_provider not in {"gemini", "google"}:
            err = f"GenerateVideo supports Gemini/Veo only, provider '{llm_provider}' unsupported."
            logger.error(err)
            ctx["error"] = err
            return (ctx, "")

        llm_model = provider_cfg.get("llm_model", self.DEFAULT_MODEL) or self.DEFAULT_MODEL

        # API key
        api_key = provider_cfg.get("api_key", "").strip()
        if (not api_key or api_key == "1234") and get_api_key is not None:
            try:
                api_key = get_api_key("GEMINI_API_KEY", "gemini")
            except ValueError:
                pass
        if not api_key:
            err = "GenerateVideo: missing Gemini API key"
            logger.error(err)
            ctx["error"] = err
            return (ctx, "")

        # Import SDK
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            err = "google-generativeai not installed – run 'pip install google-generativeai'"
            logger.error(err)
            ctx["error"] = err
            return (ctx, "")

        aspect_ratio = gen_cfg.get("aspect_ratio", "16:9")
        person_generation = gen_cfg.get("person_generation", "dont_allow")
        
        # Map unsupported "allow_all" to "allow_adult" for backward compatibility
        if person_generation == "allow_all":
            logger.warning("person_generation 'allow_all' is not supported by Veo API, using 'allow_adult' instead")
            person_generation = "allow_adult"
            
        number_of_videos = int(gen_cfg.get("number_of_videos", 1))
        duration_seconds = int(gen_cfg.get("duration_seconds", 6))
        negative_prompt = gen_cfg.get("negative_prompt", "")
        enhance_prompt = bool(gen_cfg.get("enhance_prompt", True))

        video_cfg = types.GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            person_generation=person_generation,
            number_of_videos=number_of_videos,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt if negative_prompt else None,
            enhance_prompt=enhance_prompt,
        )

        try:
            client = genai.Client(api_key=api_key)
            operation = client.models.generate_videos(
                model=llm_model,
                prompt=prompt,
                config=video_cfg,
            )

            # Poll
            poll_sec = 20
            while not operation.done:
                logger.info("GenerateVideo: operation not done yet – sleeping %s sec", poll_sec)
                time.sleep(poll_sec)
                operation = client.operations.get(operation)

            videos = operation.response.generated_videos  # type: ignore
            if not videos:
                raise RuntimeError("Veo generation returned no videos")

            # Use ComfyUI's standard output directory
            if folder_paths and hasattr(folder_paths, 'get_output_directory'):
                base_output_dir = folder_paths.get_output_directory()
                out_dir = Path(base_output_dir) / "veo_videos"
            else:
                # Fallback to current directory if folder_paths not available
                out_dir = Path(os.getcwd()) / "output" / "veo_videos"
            
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving videos to: {out_dir}")
            
            base_name = f"veo_{os.getpid()}"
            paths = self._save_videos(client, videos, out_dir, base_name)
            if not paths:
                raise RuntimeError("Failed saving any generated videos")

            ctx["generated_video_paths"] = paths
            logger.info("GenerateVideo: saved %d video(s)", len(paths))
            # Return first path for convenience
            return (ctx, paths[0] if len(paths) == 1 else "|".join(paths))

        except Exception as exc:
            err = f"Veo API error: {exc}"
            logger.error(err, exc_info=True)
            ctx["error"] = err
            return (ctx, "")


NODE_CLASS_MAPPINGS = {"GenerateVideo": GenerateVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"GenerateVideo": "Generate Video (LLMToolkit)"} 
