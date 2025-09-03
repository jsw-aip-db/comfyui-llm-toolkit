# comfy-nodes/preview_video.py
"""Preview Video (🔗LLMToolkit)

A node that displays a video from a given file path without re-encoding or
saving a new version. It's designed to preview videos that already exist on
the hard drive, such as those produced by the GenerateVideo or
LoadVideoFromPath nodes.
"""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path

try:
    from comfy.comfy_types import IO
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    # Mock for local development if comfy isn't available
    IO = type("IO", (), {"VIDEO": "VIDEO"})
    VideoFromFile = None

try:
    import folder_paths
except ImportError:
    # Mock for local development
    class MockFolderPaths:
        def get_output_directory(self): return "output"
        def get_input_directory(self): return "input"
        def get_temp_directory(self): return "temp"
    folder_paths = MockFolderPaths()

logger = logging.getLogger(__name__)

class PreviewVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "placeholder": "/path/to/video.mp4",
                        "tooltip": "Absolute or relative path to the video file to preview.",
                    },
                ),
            }
        }

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "🔗llm_toolkit/utils/video"

    def preview(self, video_path: str):
        if not video_path or not os.path.exists(video_path):
            logger.warning("PreviewVideo: Video path is empty or file does not exist: %s", video_path)
            return {"ui": {"images": []}}

        video_path = os.path.abspath(video_path)
        
        video = None
        if VideoFromFile:
            try:
                video = VideoFromFile(video_path)
            except Exception as e:
                logger.error("Failed to create VideoFromFile object: %s", e)

        # Check if the video is in a web-accessible directory
        for dir_type, dir_path in [
            ("output", folder_paths.get_output_directory()),
            ("input", folder_paths.get_input_directory()),
            ("temp", folder_paths.get_temp_directory()),
        ]:
            try:
                abs_dir_path = os.path.abspath(dir_path)
                if os.path.commonpath([video_path, abs_dir_path]) == abs_dir_path:
                    relative_path = os.path.relpath(video_path, abs_dir_path)
                    subfolder, filename = os.path.split(relative_path)
                    return {
                        "ui": {
                            "images": [
                                {
                                    "filename": filename,
                                    "subfolder": subfolder,
                                    "type": dir_type,
                                }
                            ],
                            "animated": (True,),
                        },
                        "result": (video, video_path),
                    }
            except Exception as e:
                logger.error("Error checking path %s against %s: %s", video_path, dir_path, e)


        # If not, copy to temp directory to make it accessible
        try:
            temp_dir = folder_paths.get_temp_directory()
            filename = os.path.basename(video_path)
            dest_path = os.path.join(temp_dir, filename)
            
            # To avoid re-copying, check if it already exists
            if not os.path.exists(dest_path) or os.path.getmtime(video_path) != os.path.getmtime(dest_path):
                shutil.copy2(video_path, dest_path)
                logger.info("Copied video to temp for preview: %s", dest_path)

            return {
                "ui": {
                    "images": [
                        {"filename": filename, "subfolder": "", "type": "temp"}
                    ],
                    "animated": (True,),
                },
                "result": (video, video_path),
            }
        except Exception as e:
            logger.error("Failed to copy video to temp directory for preview: %s", e, exc_info=True)
            return {"ui": {"images": []}}


NODE_CLASS_MAPPINGS = {"PreviewVideo": PreviewVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"PreviewVideo": "Preview Video Form Path (🔗LLMToolkit)"} 