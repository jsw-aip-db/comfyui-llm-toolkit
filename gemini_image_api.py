"""Gemini Image Generation API
Supports both native Gemini multimodal image generation (Gemini 2.0 Flash) 
and dedicated Imagen models (Imagen 3, Imagen 4, Imagen 4 Ultra).
"""

import asyncio
import base64
import logging
from typing import Optional, Dict, Any, List, Union
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Aspect ratio to dimensions mapping
ASPECT_RATIO_DIMENSIONS = {
    "1:1": (1024, 1024),
    "16:9": (1408, 768),
    "9:16": (768, 1408), 
    "4:3": (1280, 896),
    "3:4": (896, 1280),
}

def _convert_size_to_aspect_ratio(size: str) -> str:
    """Convert OpenAI-style size to aspect ratio."""
    size_to_aspect = {
        "1024x1024": "1:1",
        "1792x1024": "16:9",
        "1024x1792": "9:16",
    }
    return size_to_aspect.get(size, "1:1")

async def send_gemini_native_image_request(
    api_key: str,
    prompt: str,
    model: str = "gemini-2.0-flash-preview-image-generation",
    n: int = 1,
    size: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    seed: Optional[int] = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    **kwargs
) -> Dict[str, Any]:
    """Generate images using Gemini native multimodal capabilities"""
    
    if not api_key:
        raise ValueError("Gemini image generation requires a valid API key")
    
    # Create client with API key
    client = genai.Client(api_key=api_key)
    
    # Determine aspect ratio and dimensions
    if not aspect_ratio and size:
        aspect_ratio = _convert_size_to_aspect_ratio(size)
    elif not aspect_ratio:
        aspect_ratio = "1:1"
    
    # Get target dimensions
    target_width, target_height = ASPECT_RATIO_DIMENSIONS.get(aspect_ratio, (1024, 1024))
    logger.info(f"Using resolution {target_width}x{target_height} for aspect ratio {aspect_ratio}")
    
    # Include dimensions in the prompt
    prompt_with_dimensions = f"Generate a detailed, high-quality image with dimensions {target_width}x{target_height} of: {prompt}"
    
    all_images = []
    
    for i in range(n):
        try:
            # Set up generation config
            config_args = {
                "temperature": temperature,
                "response_modalities": ["TEXT", "IMAGE"],  # Critical for image generation!
            }
            
            # Generate a unique seed for each image if seed is provided
            if seed is not None:
                config_args["seed"] = (seed + i) % (2**31 - 1)
                logger.info(f"Generating image {i+1}/{n} with seed {config_args['seed']}")
            
            generation_config = types.GenerateContentConfig(**config_args)
            
            # Generate content
            response = client.models.generate_content(
                model=model,
                contents=prompt_with_dimensions,
                config=generation_config
            )
            
            # Extract images from response
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            # Extract image data
                            if hasattr(part, 'inline_data') and part.inline_data:
                                try:
                                    # The image data is already in bytes
                                    image_bytes = part.inline_data.data
                                    
                                    # Convert to base64
                                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                    all_images.append({"b64_json": image_base64})
                                    
                                except Exception as e:
                                    logger.error(f"Error processing image data: {e}")
            
            # Add small delay between requests if generating multiple
            if i < n - 1:
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error generating image {i+1}: {e}")
            continue
    
    if not all_images:
        logger.error("No images were generated")
        return {"error": "No images generated", "data": []}
    
    return {"data": all_images}


async def send_imagen_image_request(
    api_key: str,
    prompt: str,
    model: str = "imagen-3.0-generate-002",
    n: int = 1,
    aspect_ratio: str = "1:1",
    person_generation: str = "allow_adult",
    safety_filter_level: str = "block_some", 
    language: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate images using Google's Imagen models"""
    
    if not api_key:
        raise ValueError("Imagen generation requires a valid API key")
    
    # Create client with API key
    client = genai.Client(api_key=api_key)
    
    # Note: For Imagen 4 Ultra, only 1 image can be generated at a time
    if "ultra" in model.lower() and n > 1:
        logger.warning("Imagen 4 Ultra only supports generating 1 image at a time. Setting n=1.")
        n = 1
    
    try:
        # Set up generation config for Imagen
        config_args = {
            # Required / supported fields
            "number_of_images": n,
            "aspect_ratio": aspect_ratio,
        }

        # Optional fields with mapping / validation -------------------------
        # person_generation is allowed unless it's the default (allow_adult)
        if person_generation and person_generation != "allow_adult":
            config_args["person_generation"] = person_generation

        # Map legacy safety level names (block_few / some / most) ➜ new names
        legacy_map = {
            "block_few": "block_low_and_above",
            "block_some": "block_medium_and_above",
            "block_most": "block_high_and_above",
        }
        if safety_filter_level:
            mapped_level = legacy_map.get(safety_filter_level, safety_filter_level)
            # Imagen preview models currently only support block_low_and_above
            if mapped_level not in {
                "block_low_and_above",
                "block_medium_and_above",
                "block_high_and_above",
            }:
                mapped_level = "block_low_and_above"

            # Preview Imagen endpoints currently support only block_low_and_above
            if "preview" in model.lower():
                mapped_level = "block_low_and_above"
            config_args["safety_filter_level"] = mapped_level

        # Language (omit if 'auto')
        if language and language != "auto":
            config_args["language"] = language

        generation_config = types.GenerateImagesConfig(**config_args)
        
        # Generate images using the Imagen-specific method
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=generation_config
        )
        
        # Process generated images
        all_images: List[Dict[str, str]] = []

        if getattr(response, "generated_images", None):
            gen_images = response.generated_images or []
            for generated_image in gen_images:
                try:
                    img_obj = getattr(generated_image, "image", None)
                    if img_obj is None:
                        continue
                    
                    image_bytes = None
                    # The SDK may return either raw bytes or a PIL Image
                    if isinstance(img_obj, bytes):
                        image_bytes = img_obj
                    else:
                        # It's an Image object. Its `save` method expects a file path.
                        import tempfile
                        import os
                        
                        fp, temp_path = tempfile.mkstemp(suffix=".png")
                        os.close(fp)

                        try:
                            # Save to the temporary path
                            img_obj.save(temp_path)
                            # Read the bytes back
                            with open(temp_path, 'rb') as f:
                                image_bytes = f.read()
                        except Exception as e:
                            logger.error(f"Failed to process image via temporary file: {e}")
                        finally:
                            # Clean up the temp file
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)

                    if image_bytes:
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        all_images.append({"b64_json": image_base64})
                    else:
                        logger.warning("Could not get image bytes for one returned image.")

                except Exception as e:
                    logger.error(f"Error processing Imagen response item: {e}")
        else:
            # Fallback: some Imagen responses embed images in candidates/parts (rare)
            parts = getattr(response, "candidates", None)
            if parts:
                for cand in parts:
                    if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                        for part in cand.content.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                try:
                                    image_bytes = part.inline_data.data
                                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                                    all_images.append({"b64_json": image_base64})
                                except Exception:
                                    pass
        
        if not all_images:
            logger.error("No images were generated by Imagen")
            return {"error": "No images generated", "data": []}
            
        return {"data": all_images}
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Imagen generation error: {error_msg}")
        
        # Check for specific error types
        if "404" in error_msg or "not found" in error_msg.lower():
            logger.info(f"Model {model} not found, trying alternative names...")
            # Try alternative model names
            alt_models = {
                "imagen-3.0-generate-001": "imagen-3.0-generate-002",
                "imagen-4.0-generate-001": "imagen-4.0-generate-preview-06-06",
                "imagen-4.0-ultra-generate-001": "imagen-4.0-ultra-generate-preview-06-06",
            }
            if model in alt_models:
                return await send_imagen_image_request(
                    api_key=api_key,
                    prompt=prompt,
                    model=alt_models[model],
                    n=n,
                    aspect_ratio=aspect_ratio,
                    person_generation=person_generation,
                    safety_filter_level=safety_filter_level,
                    language=language,
                    seed=seed,
                    **kwargs
                )
        
        raise


async def send_gemini_image_generation_unified(
    api_key: str,
    model: str,
    prompt: str,
    n: int = 1,
    size: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    seed: Optional[int] = None,
    edit_mode: bool = False,
    variation_mode: bool = False,
    input_image_base64: Optional[Union[str, List[str]]] = None,
    mask_base64: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Unified entry point for Gemini/Imagen image generation"""
    
    # Normalize model name (remove preview suffixes for routing)
    model_lower = model.lower()
    
    # Map preview models to their stable versions if needed
    model_mapping = {
        "imagen-3.0-generate-preview-06-06": "imagen-3.0-generate-002",
        "imagen-3-light-alpha": "imagen-3.0-generate-002",  # Legacy naming
    }
    
    if model in model_mapping:
        logger.info(f"Mapping model {model} to {model_mapping[model]}")
        model = model_mapping[model]
    
    # Route to appropriate function
    if "imagen" in model_lower:
        # Imagen models
        return await send_imagen_image_request(
            api_key=api_key,
            prompt=prompt,
            model=model,
            n=n,
            aspect_ratio=aspect_ratio or _convert_size_to_aspect_ratio(size or "1024x1024"),
            person_generation=kwargs.get("person_generation", "allow_adult"),
            safety_filter_level=kwargs.get("safety_filter_level", "block_low_and_above"),
            language=kwargs.get("language"),
            seed=seed,
        )
    else:
        # Gemini native models (including the preview image generation model)
        return await send_gemini_native_image_request(
            api_key=api_key,
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            aspect_ratio=aspect_ratio,
            seed=seed,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 8192),
        ) 