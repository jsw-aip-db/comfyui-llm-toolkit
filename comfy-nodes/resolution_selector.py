import math

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
RADIAL_ALIGNMENT = VAE_STRIDE[1] * PATCH_SIZE[1]  # 16 for height/width

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def calculate_radial_compatible_resolution(width, height, mode="closest", block_size=128, length=None, patch_divisor=16):
    """
    Calculate radial attention compatible resolution for images or videos.
    The user-provided 'length' is a fixed constraint and will not be modified.
    
    The function will search for a (width, height) pair that satisfies the divisibility constraints
    for the given length.

    Args:
        width (int): Original width
        height (int): Original height  
        mode (str): "upscale", "downscale", or "closest"
        block_size (int): Radial attention block size (64 or 128)
        length (int, optional): Length of video in frames. This value is NOT changed.
        patch_divisor (int): The spatial patch size divisor (e.g., 16 or 32).
    
    Returns:
        tuple: (compatible_width, compatible_height)
    """

    if length is not None:
        if (length + 3) % 4 != 0:
            print(f"Warning: Radial attention may not work with length {length}. It should be 4k-3 (e.g., 1, 5, ..., 81).")
            length_factor = 1
        else:
            length_factor = (length + 3) // 4
        
        common_divisor = math.gcd(length_factor, block_size)
        target_divisor = block_size // common_divisor
    else:
        target_divisor = block_size

    best_res = (width, height)
    min_dist = float('inf')

    # Heuristic search area around the original resolution
    # Search in a 256-pixel radius, with patch_divisor steps
    for w_candidate in range(width - 256, width + 256 + patch_divisor, patch_divisor):
        if w_candidate <= 0: continue
        
        for h_candidate in range(height - 256, height + 256 + patch_divisor, patch_divisor):
            if h_candidate <= 0: continue
            
            w_p = w_candidate // patch_divisor
            h_p = h_candidate // patch_divisor
            
            if (w_p * h_p) % target_divisor == 0:
                dist = math.sqrt((w_candidate - width)**2 + (h_candidate - height)**2)

                is_candidate = False
                if mode == 'closest':
                    is_candidate = True
                elif mode == 'upscale':
                    if w_candidate >= width and h_candidate >= height:
                        is_candidate = True
                elif mode == 'downscale':
                    if w_candidate <= width and h_candidate <= height:
                        is_candidate = True

                if is_candidate and dist < min_dist:
                    min_dist = dist
                    best_res = (w_candidate, h_candidate)

    if min_dist == float('inf'):
        print(f"Warning: Could not find a compatible resolution for {width}x{height} with length {length}. Returning original values.")
        return width, height

    return best_res

class ResolutionSelector:
    """Select width & height for video/image generation.

    Modes supported:
    - I2V480p, I2V720p
    - T2V1.3B, T2V14B
    - IMG (general image) with Cinematic AR option
    - KONTEXT (optimized resolutions)
    - QWEN (image edit compatible resolutions)
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
        "KONTEXT": {
            "Vertical":   {"HQ": (672, 1568), "MQ": (720, 1456), "LQ": (832, 1248)},
            "Horizontal": {"HQ": (1568, 672), "MQ": (1456, 720), "LQ": (1248, 832)},
            "Squarish":   {"HQ": (1024, 1024), "MQ": (944, 1104), "LQ": (880, 1184)},
        },
        "QWEN": {
            "Square":     {"HQ": (1024, 1024), "MQ": (768, 768), "LQ": (512, 512)},
            "Landscape":  {"HQ": (1280, 720), "MQ": (1024, 768), "LQ": (832, 624)},
            "Portrait":   {"HQ": (720, 1280), "MQ": (768, 1024), "LQ": (624, 832)},
            "Wide":       {"HQ": (1536, 768), "MQ": (1280, 640), "LQ": (1024, 512)},
            "Tall":       {"HQ": (768, 1536), "MQ": (640, 1280), "LQ": (512, 1024)},
            "UltraWide":  {"HQ": (1792, 768), "MQ": (1536, 640), "LQ": (1280, 544)},
            "UltraTall":  {"HQ": (768, 1792), "MQ": (640, 1536), "LQ": (544, 1280)},
        },
        "GPT_IMAGE_1": {
            "Square":   {"HQ": (1024, 1024), "MQ": (1024, 1024), "LQ": (1024, 1024)},
            "Portrait": {"HQ": (1024, 1536), "MQ": (1024, 1536), "LQ": (1024, 1536)},
            "Landscape":{"HQ": (1536, 1024), "MQ": (1536, 1024), "LQ": (1536, 1024)},
        },
        "GEMINI_IMAGEN": {
            "Square (1:1)":      {"HQ": (1024, 1024), "MQ": (1024, 1024), "LQ": (1024, 1024)},
            "Portrait (3:4)":    {"HQ": (896, 1200),  "MQ": (768, 1024),  "LQ": (672, 896)},
            "Landscape (4:3)":   {"HQ": (1200, 896),  "MQ": (1024, 768),  "LQ": (896, 672)},
            "Portrait (9:16)":   {"HQ": (864, 1536),  "MQ": (720, 1280), "LQ": (576, 1024)},
            "Landscape (16:9)":  {"HQ": (1536, 864),  "MQ": (1280, 720), "LQ": (1024, 576)},
        },
        "BFL": {
            "1:1":   {"HQ": (1024, 1024), "MQ": (768, 768), "LQ": (512, 512)},
            "3:4":   {"HQ": (768, 1024), "MQ": (512, 682), "LQ": (384, 512)},
            "4:3":   {"HQ": (1024, 768), "MQ": (682, 512), "LQ": (512, 384)},
            "9:16":  {"HQ": (720, 1280), "MQ": (576, 1024), "LQ": (405, 720)},
            "16:9":  {"HQ": (1280, 720), "MQ": (1024, 576), "LQ": (720, 405)},
            "21:9":  {"HQ": (1536, 658), "MQ": (1280, 548), "LQ": (1024, 438)},
            "9:21":  {"HQ": (658, 1536), "MQ": (548, 1280), "LQ": (438, 1024)},
        },
        "SEEDREAM_V4": {
            "Square (1:1)":      {"HQ": (2048, 2048), "MQ": (1536, 1536), "LQ": (1024, 1024)},
            "Landscape (16:9)":  {"HQ": (2048, 1152), "MQ": (1536, 864), "LQ": (1024, 576)},
            "Portrait (9:16)":   {"HQ": (1152, 2048), "MQ": (864, 1536), "LQ": (576, 1024)},
            "Landscape (4:3)":   {"HQ": (2048, 1536), "MQ": (1536, 1152), "LQ": (1024, 768)},
            "Portrait (3:4)":    {"HQ": (1536, 2048), "MQ": (1152, 1536), "LQ": (768, 1024)},
        },
        "HUNYUAN": {
            "Landscape (16:9)": {"HQ": (2560, 1536), "MQ": (2560, 1536), "LQ": (2560, 1536)},
            "Landscape (4:3)":  {"HQ": (2304, 1792), "MQ": (2304, 1792), "LQ": (2304, 1792)},
            "Square (1:1)":     {"HQ": (2048, 2048), "MQ": (2048, 2048), "LQ": (2048, 2048)},
            "Portrait (3:4)":   {"HQ": (1792, 2304), "MQ": (1792, 2304), "LQ": (1792, 2304)},
            "Portrait (9:16)":  {"HQ": (1536, 2560), "MQ": (1536, 2560), "LQ": (1536, 2560)},
        },
    }
    # Create NANO_BANANA as an alias for GEMINI_IMAGEN
    RESOLUTIONS["NANO_BANANA"] = RESOLUTIONS["GEMINI_IMAGEN"]
    # Add FLUX_DEV as an alias for IMG
    RESOLUTIONS["FLUX_DEV"] = RESOLUTIONS["IMG"]

    ASPECT_RATIO_STRING_MAP = {
        "Square (1:1)": "1:1",
        "Portrait (3:4)": "3:4",
        "Landscape (4:3)": "4:3",
        "Portrait (9:16)": "9:16",
        "Landscape (16:9)": "16:9",
    }

    @classmethod
    def get_valid_aspect_ratios_for_mode(cls, mode):
        """Get valid aspect ratios for a specific mode."""
        if mode in cls.RESOLUTIONS:
            return list(cls.RESOLUTIONS[mode].keys())
        return []

    @classmethod
    def INPUT_TYPES(cls):
        modes = list(cls.RESOLUTIONS.keys())
        # Get all possible aspect ratios across all modes
        all_aspect_ratios = set()
        for mode_data in cls.RESOLUTIONS.values():
            all_aspect_ratios.update(mode_data.keys())
        
        return {
            "required": {
                "mode": (modes, {"default": modes[0], "tooltip": "Generation mode"}),
                "aspect_ratio": (sorted(list(all_aspect_ratios)), {"default": "Horizontal"}),
                "quality": (["HQ", "MQ", "LQ"], {"default": "HQ"}),
                "radial_attention_mode": (["disabled", "image", "video"], {"default": "disabled", "tooltip": "Enable and configure radial attention compatibility"}),
            },
            "optional": {
                "context": ("*", {}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4, "tooltip": "Video length in frames (for video radial attention)"}),
                "patch_divisor": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8, "tooltip": "Spatial patch divisor for radial attention (e.g., 16 or 32)"}),
                "block_size": ([128, 64], {"default": 128, "tooltip": "Radial attention block size"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING", "*")
    RETURN_NAMES = ("width", "height", "length", "size_string", "context")
    FUNCTION = "get_resolution"
    CATEGORY = "🔗llm_toolkit/utils"

    def get_resolution(self, mode, aspect_ratio, quality, radial_attention_mode, context=None, length=81, patch_divisor=16, block_size=128):
        # Initialize or copy the context
        if context is None:
            output_context = {}
        else:
            # Simple shallow copy is fine here
            output_context = context.copy()
            
        try:
            # Check if the aspect ratio is valid for this mode
            if aspect_ratio not in self.RESOLUTIONS[mode]:
                # Fall back to first available aspect ratio for this mode
                valid_aspect_ratios = list(self.RESOLUTIONS[mode].keys())
                if valid_aspect_ratios:
                    aspect_ratio = valid_aspect_ratios[0]
                    print(f"Warning: Invalid aspect ratio for mode {mode}, falling back to {aspect_ratio}")
            
            w, h = self.RESOLUTIONS[mode][aspect_ratio][quality]
            
            # Apply radial attention compatibility if enabled
            if radial_attention_mode in ["video", "image"]:
                video_length = length if radial_attention_mode == "video" else None
                w, h = calculate_radial_compatible_resolution(w, h, "upscale", block_size, video_length, patch_divisor)

            # Determine the string output based on the mode
            if mode in ["GEMINI_IMAGEN", "NANO_BANANA"]:
                size_string = self.ASPECT_RATIO_STRING_MAP.get(aspect_ratio, f"{w}x{h}")
            elif mode == "BFL":
                size_string = aspect_ratio
            else:
                size_string = f"{w}x{h}"
            
            # --- Update context ---
            generation_config = output_context.get("generation_config", {})
            if not isinstance(generation_config, dict):
                generation_config = {}

            generation_config["size"] = f"{w}x{h}"
            
            if mode in ["GEMINI_IMAGEN", "NANO_BANANA"]:
                 generation_config["aspect_ratio"] = self.ASPECT_RATIO_STRING_MAP.get(aspect_ratio)
            elif mode == "BFL":
                 generation_config["aspect_ratio"] = aspect_ratio

            output_context["generation_config"] = generation_config
            # --- End Update context ---

            return (w, h, length, size_string, output_context)
        except KeyError:
            # fallback default
            return (832, 480, length, "832x480", output_context)


NODE_CLASS_MAPPINGS = {
    "ResolutionSelector": ResolutionSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResolutionSelector": "Resolution Selector (🔗LLMToolkit)",
}

# Export for potential JavaScript access
WEB_DIRECTORY = "./web" 