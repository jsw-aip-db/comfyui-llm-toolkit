class ResolutionSelector:
    """Select width & height for video/image generation.

    Modes supported:
    - I2V480p, I2V720p
    - T2V1.3B, T2V14B
    - IMG (general image) with Cinematic AR option
    """

    RESOLUTIONS = {
        "I2V720p": {
            "Horizontal": {"HQ": (1280, 720), "MQ": (832, 480), "LQ": (704, 544)},
            "Vertical":   {"HQ": (720, 1280), "MQ": (480, 832), "LQ": (544, 704)},
            "Squarish":   {"HQ": (624, 624),  "MQ": (624, 624),  "LQ": (624, 624)},
        },
        "I2V480p": {
            "Horizontal": {"HQ": (832, 480), "MQ": (704, 544), "LQ": (704, 544)},
            "Vertical":   {"HQ": (480, 832), "MQ": (544, 704), "LQ": (544, 704)},
            "Squarish":   {"HQ": (624, 624),  "MQ": (624, 624),  "LQ": (624, 624)},
        },
        "T2V14B": {
            "Horizontal": {"HQ": (1280, 720), "MQ": (1088, 832), "LQ": (832, 480)},
            "Vertical":   {"HQ": (720, 1280), "MQ": (832, 1088), "LQ": (480, 832)},
            "Squarish":   {"HQ": (960, 960),  "MQ": (624, 624),  "LQ": (544, 704)},
        },
        "T2V1.3B": {
            "Horizontal": {"HQ": (832, 480), "MQ": (704, 544), "LQ": (704, 544)},
            "Vertical":   {"HQ": (480, 832), "MQ": (544, 704), "LQ": (544, 704)},
            "Squarish":   {"HQ": (624, 624),  "MQ": (624, 624),  "LQ": (624, 624)},
        },
        "IMG": {
            "Horizontal": {"HQ": (1600, 900), "MQ": (1280, 720), "LQ": (1024, 576)},
            "Vertical":   {"HQ": (900, 1600), "MQ": (720, 1280), "LQ": (576, 1024)},
            "Squarish":   {"HQ": (1600, 1600), "MQ": (1024, 1024), "LQ": (512, 512)},
            "Cinematic":  {"HQ": (1600, 688), "MQ": (1280, 550), "LQ": (1024, 440)},  # ≈2.35:1
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        modes = list(cls.RESOLUTIONS.keys())
        return {
            "required": {
                "mode": (modes, {"default": modes[0], "tooltip": "Generation mode"}),
                "aspect_ratio": (["Horizontal", "Vertical", "Squarish", "Cinematic"], {"default": "Horizontal"}),
                "quality": (["HQ", "MQ", "LQ"], {"default": "HQ"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "llm_toolkit/utils"

    def get_resolution(self, mode, aspect_ratio, quality):
        try:
            w, h = self.RESOLUTIONS[mode][aspect_ratio][quality]
            return (w, h)
        except KeyError:
            # fallback default
            return (832, 480)


NODE_CLASS_MAPPINGS = {
    "ResolutionSelector": ResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionSelector": "Resolution Selector (LLMToolkit)",
} 