from nodes import PreviewImage


class ImageComparer(PreviewImage):
    """A node that compares two images in the UI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_images"
    CATEGORY = "llm_toolkit/utils"
    OUTPUT_NODE = True

    def compare_images(self,
                       image_a=None,
                       image_b=None,
                       filename_prefix="llmtoolkit.compare.",
                       prompt=None,
                       extra_pnginfo=None):

        result = {"ui": {"a_images": [], "b_images": []}}
        if image_a is not None and len(image_a) > 0:
            result['ui']['a_images'] = self.save_images(image_a, filename_prefix, prompt, extra_pnginfo)['ui']['images']

        if image_b is not None and len(image_b) > 0:
            result['ui']['b_images'] = self.save_images(image_b, filename_prefix, prompt, extra_pnginfo)['ui']['images']

        return result


NODE_CLASS_MAPPINGS = {
    "ImageComparer": ImageComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageComparer": "Image Comparer (LLMToolkit)",
}