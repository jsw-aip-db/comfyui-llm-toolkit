#!/usr/bin/env python3
import os
import sys
import importlib

# Current directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# Add current dir to path
sys.path.append(current_dir)
print(f"sys.path: {sys.path}")

# Try importing the modules
try:
    import send_request
    print("Successfully imported send_request")
except ImportError as e:
    print(f"Failed to import send_request: {e}")

try:
    import utils
    print("Successfully imported utils")
except ImportError as e:
    print(f"Failed to import utils: {e}")

# Try to import from comfy-nodes
try:
    # Use importlib to handle the hyphen in the directory name
    spec = importlib.util.spec_from_file_location(
        "generate_text", 
        os.path.join(current_dir, "comfy-nodes", "generate_text.py")
    )
    generate_text = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_text)
    print("Successfully imported generate_text")
except Exception as e:
    print(f"Failed to import generate_text: {e}")

print("Import test completed") 