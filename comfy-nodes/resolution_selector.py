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
    
    For images (length=None):
    (width/patch_divisor * height/patch_divisor) must be divisible by block_size.
    
    For videos:
    (width/patch_divisor * height/patch_divisor * (length+3)/4) must be divisible by block_size
    AND the total number of tokens must be a power of two.
    
    Args:
        width (int): Original width
        height (int): Original height  
        mode (str): "upscale", "downscale", or "closest"
        block_size (int): Radial attention block size (64 or 128)
        length (int, optional): Length of video in frames. If None, calculates for an image.
        patch_divisor (int): The spatial patch size divisor (e.g., 16 or 32).
    
    Returns:
        tuple: (compatible_width, compatible_height)
    """
    
    def is_power_of_two(n):
        return (n & (n - 1) == 0) and n != 0

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

    def find_compatible_dimension(target_size, other_dim_patched, mode):
        base_size = (target_size // patch_divisor) * patch_divisor
        search_range = range(max(patch_divisor, base_size - 128), base_size + 128 + patch_divisor, patch_divisor)
        
        candidates = []
        for test_size in search_range:
            patched_dim = test_size // patch_divisor
            total_tokens = patched_dim * other_dim_patched * (length_factor if length is not None else 1)
            
            if (patched_dim * other_dim_patched) % target_divisor == 0 and \
               (length is None or is_power_of_two(total_tokens)):
                distance = abs(test_size - target_size)
                candidates.append((distance, test_size))
        
        if not candidates:
            return base_size if base_size >= patch_divisor else patch_divisor
        
        candidates.sort()
        
        if mode == "upscale":
            valid_candidates = [size for dist, size in candidates if size >= target_size]
            return valid_candidates[0] if valid_candidates else candidates[-1][1]
        elif mode == "downscale":
            valid_candidates = [size for dist, size in candidates if size <= target_size]
            return valid_candidates[0] if valid_candidates else candidates[0][1]
        else:  # closest
            return candidates[0][1]

    if width == height:
        target_patched = width // patch_divisor
        candidates = []
        
        for offset in range(0, 32):
            test_patched = target_patched + offset
            if test_patched > 0:
                total_tokens = (test_patched * test_patched) * (length_factor if length is not None else 1)
                if (test_patched * test_patched) % target_divisor == 0 and \
                   (length is None or is_power_of_two(total_tokens)):
                    test_size = test_patched * patch_divisor
                    distance = abs(test_size - width)
                    candidates.append((distance, test_size, test_size >= width))
        
        if not candidates:
            for offset in range(-1, -32, -1):
                test_patched = target_patched + offset
                if test_patched > 0:
                    total_tokens = (test_patched * test_patched) * (length_factor if length is not None else 1)
                    if (test_patched * test_patched) % target_divisor == 0 and \
                       (length is None or is_power_of_two(total_tokens)):
                        test_size = test_patched * patch_divisor
                        distance = abs(test_size - width)
                        candidates.append((distance, test_size, test_size >= width))
        
        if candidates:
            higher_candidates = [c for c in candidates if c[2]]
            if higher_candidates:
                higher_candidates.sort(key=lambda x: x[0])
                res = higher_candidates[0][1]
                return res, res
            else:
                candidates.sort(key=lambda x: x[0])
                res = candidates[0][1]
                return res, res
    
    # For non-square, we can't easily solve for both. Let's adjust one dimension first.
    # We'll adjust width first, then find a compatible height.
    base_width = (width // patch_divisor) * patch_divisor
    base_height = (height // patch_divisor) * patch_divisor
    
    w_p = base_width // patch_divisor
    h_p = base_height // patch_divisor
    
    total_tokens = w_p * h_p * (length_factor if length is not None else 1)
    
    if (w_p * h_p) % target_divisor == 0 and \
       (length is None or is_power_of_two(total_tokens)):
        return base_width, base_height

    # If not compatible, find a new height for the current width
    compatible_height = find_compatible_dimension(height, w_p, mode)
    
    return base_width, compatible_height

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
            if radial_attention_mode == "video":
                w, h = calculate_radial_compatible_resolution(w, h, "upscale", block_size, length, patch_divisor)
            elif radial_attention_mode == "image":
                w, h = calculate_radial_compatible_resolution(w, h, "upscale", block_size, None, patch_divisor)

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