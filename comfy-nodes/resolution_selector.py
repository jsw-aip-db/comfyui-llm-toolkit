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

def calculate_radial_compatible_resolution(width, height, mode="closest", block_size=64):
    """
    Calculate radial attention compatible resolution.
    
    For radial attention to work, patches_per_frame must be divisible by block_size:
    patches_per_frame = (height//8) * (width//8) // 4
    
    Args:
        width (int): Original width
        height (int): Original height  
        mode (str): "upscale", "downscale", or "closest"
        block_size (int): Radial attention block size (64 or 128)
    
    Returns:
        tuple: (compatible_width, compatible_height)
    """
    
    def find_compatible_dimension(target_size, mode, block_size):
        # Start with VAE-aligned size
        base_size = (target_size // VAE_STRIDE[1]) * VAE_STRIDE[1]
        
        # Search for a size where patches_per_frame % block_size == 0
        search_range = range(max(VAE_STRIDE[1], base_size - 64), base_size + 80, VAE_STRIDE[1])
        
        candidates = []
        for test_size in search_range:
            lat_dim = test_size // VAE_STRIDE[1]
            patches_per_frame = (lat_dim * lat_dim) // (PATCH_SIZE[1] * PATCH_SIZE[2])
            
            if patches_per_frame % block_size == 0:
                distance = abs(test_size - target_size)
                candidates.append((distance, test_size))
        
        if not candidates:
            # Fallback: just ensure VAE alignment
            return base_size if base_size >= VAE_STRIDE[1] else VAE_STRIDE[1]
        
        candidates.sort()  # Sort by distance
        
        if mode == "upscale":
            valid_candidates = [size for dist, size in candidates if size >= target_size]
            return valid_candidates[0] if valid_candidates else candidates[-1][1]
        elif mode == "downscale":
            valid_candidates = [size for dist, size in candidates if size <= target_size]
            return valid_candidates[0] if valid_candidates else candidates[0][1]
        else:  # closest
            return candidates[0][1]
    
    # Handle square resolutions specially (both dimensions must work together)
    if width == height:
        # For square resolutions, find size where lat_dim^2 // 4 % block_size == 0
        target_lat = width // VAE_STRIDE[1]
        
        # Always prefer higher values - search upward first
        candidates = []
        
        # Search upward for compatible sizes
        for offset in range(0, 16):  # Search upward from target
            test_lat = target_lat + offset
            if test_lat <= 0:
                continue
                
            patches_per_frame = (test_lat * test_lat) // (PATCH_SIZE[1] * PATCH_SIZE[2])
            if patches_per_frame % block_size == 0:
                test_size = test_lat * VAE_STRIDE[1]
                distance = abs(test_size - width)
                candidates.append((distance, test_size, test_size >= width))
        
        # If no upward candidates, search downward
        if not candidates:
            for offset in range(-1, -16, -1):  # Search downward from target
                test_lat = target_lat + offset
                if test_lat <= 0:
                    continue
                    
                patches_per_frame = (test_lat * test_lat) // (PATCH_SIZE[1] * PATCH_SIZE[2])
                if patches_per_frame % block_size == 0:
                    test_size = test_lat * VAE_STRIDE[1]
                    distance = abs(test_size - width)
                    candidates.append((distance, test_size, test_size >= width))
        
        if candidates:
            # Always prefer higher values (>= original size)
            higher_candidates = [c for c in candidates if c[2]]  # c[2] is "is_higher_or_equal"
            if higher_candidates:
                # Sort by distance and return the closest higher value
                higher_candidates.sort(key=lambda x: x[0])
                return higher_candidates[0][1], higher_candidates[0][1]
            else:
                # Fallback to closest if no higher candidates
                candidates.sort(key=lambda x: x[0])
                return candidates[0][1], candidates[0][1]
        
        # If no perfect match found, use the original if it's already compatible
        orig_lat = width // VAE_STRIDE[1]
        orig_patches = (orig_lat * orig_lat) // 4
        if orig_patches % block_size == 0:
            return width, height
    
    # For non-square or fallback, handle dimensions independently
    compatible_width = find_compatible_dimension(width, mode, block_size)
    compatible_height = find_compatible_dimension(height, mode, block_size)
    
    return compatible_width, compatible_height

def update_resolutions_for_radial_attention(resolutions_dict, mode="closest", block_size=64):
    """
    Update resolution dictionary to make all resolutions radial attention compatible.
    
    Args:
        resolutions_dict (dict): Original resolutions dictionary
        mode (str): "upscale", "downscale", or "closest"
        block_size (int): Radial attention block size (64 or 128)
    
    Returns:
        dict: Updated resolutions dictionary
    """
    updated_resolutions = {}
    
    for model_type, orientations in resolutions_dict.items():
        updated_resolutions[model_type] = {}
        
        for orientation, qualities in orientations.items():
            updated_resolutions[model_type][orientation] = {}
            
            for quality, (width, height) in qualities.items():
                new_width, new_height = calculate_radial_compatible_resolution(width, height, mode, block_size)
                updated_resolutions[model_type][orientation][quality] = (new_width, new_height)
                
                # Log changes if resolution was modified
                if new_width != width or new_height != height:
                    print(f"Radial Attention (block_size={block_size}): {model_type}-{orientation}-{quality}: {width}x{height} -> {new_width}x{new_height}")
    
    return updated_resolutions

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
            },
            "optional": {
                "enable_radial_attention": ("BOOLEAN", {"default": False, "tooltip": "Enable radial attention compatibility"}),
                "block_size": ([128, 64], {"default": 128, "tooltip": "Radial attention block size"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_resolution"
    CATEGORY = "llm_toolkit/utils"

    def get_resolution(self, mode, aspect_ratio, quality, enable_radial_attention=False, block_size=128):
        try:
            # Check if the aspect ratio is valid for this mode
            if aspect_ratio not in self.RESOLUTIONS[mode]:
                # Fall back to first available aspect ratio for this mode
                valid_aspect_ratios = list(self.RESOLUTIONS[mode].keys())
                if valid_aspect_ratios:
                    aspect_ratio = valid_aspect_ratios[0]
                    print(f"Warning: Invalid aspect ratio for mode {mode}, falling back to {aspect_ratio}")
            
            w, h = self.RESOLUTIONS[mode][aspect_ratio][quality]
            
            # Apply radial attention compatibility if enabled (always prefer higher values)
            if enable_radial_attention:
                w, h = calculate_radial_compatible_resolution(w, h, "upscale", block_size)
            
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

# Export for potential JavaScript access
WEB_DIRECTORY = "./web"

def get_radial_resolutions(mode="closest", block_size=64):
    """Get radial attention compatible resolutions."""
    return update_resolutions_for_radial_attention(ResolutionSelector.RESOLUTIONS, mode, block_size)

if __name__ == "__main__":
    # Test the calculator
    print("Testing radial attention compatible resolutions:")
    
    for block_size in [64, 128]:
        print(f"\n=== BLOCK SIZE {block_size} ===")
        print(f"Original problematic size: 624x624")
        print("Upscale mode:", calculate_radial_compatible_resolution(624, 624, "upscale", block_size))
        print("Downscale mode:", calculate_radial_compatible_resolution(624, 624, "downscale", block_size))
        print("Closest mode:", calculate_radial_compatible_resolution(624, 624, "closest", block_size))
        
        print(f"\n--- Resolution Sets (block_size={block_size}) ---")
        
        for mode in ["upscale", "downscale", "closest"]:
            print(f"\n{mode.upper()} MODE:")
            radial_resolutions = get_radial_resolutions(mode, block_size)
            
            # Show squarish resolutions (most affected)
            print("Squarish resolutions:")
            for model_type in radial_resolutions:
                if "Squarish" in radial_resolutions[model_type]:
                    for quality, (w, h) in radial_resolutions[model_type]["Squarish"].items():
                        original = ResolutionSelector.RESOLUTIONS[model_type]["Squarish"][quality]
                        changed = "✓" if (w, h) != original else " "
                        print(f"  {model_type}-{quality}: {w}x{h} {changed}")
                        
            # Test a few other problematic ones
            print("Other potentially problematic:")
            test_cases = [
                ("T2V14B", "Horizontal", "MQ"),  # 1088x832
                ("IMG", "Cinematic", "LQ"),      # 1024x440
            ]
            
            for model, orient, qual in test_cases:
                if orient in radial_resolutions[model] and qual in radial_resolutions[model][orient]:
                    w, h = radial_resolutions[model][orient][qual]
                    original = ResolutionSelector.RESOLUTIONS[model][orient][qual]
                    changed = "✓" if (w, h) != original else " "
                    print(f"  {model}-{orient}-{qual}: {w}x{h} {changed}") 