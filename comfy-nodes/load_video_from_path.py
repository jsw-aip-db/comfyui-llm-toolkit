# comfy-nodes/load_video_from_path.py
"""Load Video From Path - Bridge node to load video files into ComfyUI format."""

import os
import torch
import numpy as np
import logging
from typing import Tuple

# A popular library for video processing
try:
    import cv2
except ImportError:
    print("Warning: opencv-python is not installed. The LoadVideoFromPath node will not work.")
    print("Please run: pip install opencv-python")
    cv2 = None

logger = logging.getLogger(__name__)


class LoadVideoFromPath:
    """Load a video file from a file path string and convert to ComfyUI image format."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # The video_paths string from your GenerateVideo node
                "video_path": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000, 
                                      "tooltip": "Maximum frames to load (0 = all frames)"}),
                "skip_frames": ("INT", {"default": 0, "min": 0, "max": 100,
                                       "tooltip": "Load every Nth frame (1 = all, 2 = every other, etc.)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("images", "frame_count", "fps", "width", "height")
    FUNCTION = "load_video"
    CATEGORY = "llm_toolkit/utils/video"

    def load_video(self, video_path: str, max_frames: int = 0, skip_frames: int = 0) -> Tuple:
        """Load video and return frames as ComfyUI image tensor."""
        
        if not cv2:
            raise ImportError("opencv-python is not installed. Cannot run LoadVideoFromPath.")

        # Handle multiple paths separated by |
        if "|" in video_path:
            video_path = video_path.split("|")[0].strip()
            logger.info(f"Multiple paths provided, using first: {video_path}")

        if not video_path or not os.path.exists(video_path):
            logger.error(f"LoadVideoFromPath: Video path is empty or file does not exist: {video_path}")
            # Return empty tensors to prevent crashing the workflow
            return (torch.zeros((1, 64, 64, 3)), 0, 0.0, 64, 64)

        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video info: {width}x{height} @ {fps}fps, {total_frames} total frames")

            frames = []
            frame_idx = 0
            frames_loaded = 0
            skip_interval = max(1, skip_frames)
            
            while True:
                # Read one frame
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Apply frame skipping
                if frame_idx % skip_interval == 0:
                    # OpenCV loads images in BGR format, so we convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert the numpy array frame to a torch tensor
                    # and normalize from [0, 255] to [0.0, 1.0]
                    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                    frames.append(frame_tensor)
                    frames_loaded += 1
                    
                    # Check if we've loaded enough frames
                    if max_frames > 0 and frames_loaded >= max_frames:
                        break
                
                frame_idx += 1
            
            cap.release()

            if not frames:
                raise ValueError("No frames could be read from the video.")

            # Stack all frame tensors into a single tensor (batch of images)
            video_tensor = torch.stack(frames)
            frame_count = len(frames)
            
            logger.info(f"Loaded {frame_count} frames from {video_path}")
            
            return (video_tensor, frame_count, fps, width, height)

        except Exception as e:
            logger.error(f"Error loading video from path '{video_path}': {e}", exc_info=True)
            return (torch.zeros((1, 64, 64, 3)), 0, 0.0, 64, 64)


# Add the new node to ComfyUI's list of recognized nodes
NODE_CLASS_MAPPINGS = {
    "LoadVideoFromPath": LoadVideoFromPath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoFromPath": "Load Video From Path (LLMToolkit)"
} 