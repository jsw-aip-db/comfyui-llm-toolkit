"""
@author: ComfyDeploy
@title: llm toolkit
@nickname: llm_toolkit
@description: llm toolkit
"""
import os
import sys
import inspect
import importlib
import re
import importlib.util

sys.path.append(os.path.join(os.path.dirname(__file__)))

ag_path = os.path.join(os.path.dirname(__file__))

def get_python_files(path):
    return [f[:-3] for f in os.listdir(path) if f.endswith(".py")]

def append_to_sys_path(path):
    if path not in sys.path:
        sys.path.append(path)

paths = ["comfy-nodes"]
files = []

for path in paths:
    full_path = os.path.join(ag_path, path)
    append_to_sys_path(full_path)
    files.extend(get_python_files(full_path))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def split_camel_case(name):
    # Split on underscores first, then split each part on camelCase
    parts = []
    for part in name.split('_'):
        # Find all camelCase boundaries
        words = re.findall('[A-Z][^A-Z]*', part)
        if not words:  # If no camelCase found, use the whole part
            words = [part]
        parts.extend(words)
    return parts

# Import all the modules and append their mappings
for file in files:
    module = importlib.import_module(file)
    
    # Check if the module has explicit mappings
    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    
    # Auto-discover classes with ComfyUI node attributes
    for name, obj in inspect.getmembers(module):
        # Check if it's a class and has the required ComfyUI node attributes
        if inspect.isclass(obj) and hasattr(obj, "INPUT_TYPES") and hasattr(obj, "RETURN_TYPES"):
            # Set or override class attributes here
            if not hasattr(obj, "CATEGORY"):
                setattr(obj, "CATEGORY", "llm_toolkit")
            if not hasattr(obj, "FUNCTION"):
                setattr(obj, "FUNCTION", "run")
            # Use the class name as the key if not already in mappings
            if name not in NODE_CLASS_MAPPINGS:
                NODE_CLASS_MAPPINGS[name] = obj
                # Create a display name by converting camelCase to Title Case with spaces
                words = split_camel_case(name)
                display_name = " ".join(word.capitalize() for word in words)
                # print(display_name, name)
                NODE_DISPLAY_NAME_MAPPINGS[name] = display_name

# WEB_DIRECTORY points to the directory where your frontend files should be served from
# WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "js") # Old path
WEB_DIRECTORY = "web/js" # Correct path

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
