# comfy-nodes/load_video_from_path.py
"""Load Video From Path (🔗LLMToolkit)

Utility node that takes a string path to a video file (e.g. the output of
GenerateVideo) and returns an `IO.VIDEO` object so it can be previewed or saved
with standard ComfyUI video nodes.  Unlike the built-in LoadVideo node, this
one accepts an arbitrary path — you don’t have to place the file in the
`input/` folder.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

from comfy.comfy_types import IO, ComfyNodeABC
from comfy_api.input_impl import VideoFromFile
from comfy_api.util import VideoContainer

class LoadVideoFromPath(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "placeholder": "/absolute/or/relative/path.mp4",
                        "tooltip": "Full path to the video file produced by GenerateVideo",
                    },
                ),
            }
        }

    CATEGORY = "🔗llm_toolkit/utils/video"

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "load"

    def load(self, video_path: str) -> Tuple[VideoFromFile, str]:
        video_path = video_path.strip().replace("\\", "/")
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"LoadVideoFromPath: file does not exist – {video_path}")

        video = VideoFromFile(video_path)
        return (video, video_path)


NODE_CLASS_MAPPINGS = {"LoadVideoFromPath": LoadVideoFromPath}
NODE_DISPLAY_NAME_MAPPINGS = {"LoadVideoFromPath": "Load Video From Path (🔗LLMToolkit)"} 