from typing import Optional, List

class ContextPayload(str):
    """A string that carries a full context dictionary and optional images.

    Because it subclasses ``str`` it can be connected anywhere a plain STRING is
    expected in ComfyUI.  The extra attributes remain accessible to nodes that
    know about them (``.context`` and ``.images``).
    """

    def __new__(cls, text: str, context: Optional[dict] = None, images: Optional[List] = None):
        # Create the actual string instance
        obj = str.__new__(cls, text if text is not None else "")
        # Attach additional payload attributes
        obj.context = context or {}
        obj.images = images or []
        return obj


def extract_context(value):
    """Return the context dictionary embedded in *value* if possible.

    If *value* is already a ``dict`` it is returned unchanged.  If it is a
    :class:`ContextPayload` the embedded ``.context`` dict is returned.  In all
    other cases an empty dict is returned.
    """
    if isinstance(value, dict):
        return value
    return getattr(value, "context", {})


def extract_images(value):
    """Return list of images embedded in *value* (or empty list)."""
    return getattr(value, "images", []) 