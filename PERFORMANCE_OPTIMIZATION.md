# LLM Toolkit Performance Optimization Guide

## Current Load Time Issue
The LLM Toolkit takes ~1.0 seconds to load compared to 0.1-0.2 seconds for other nodes.

## Root Causes
1. **Heavy library imports at module level** (torch, numpy, torchaudio)
2. **Loading all 50+ node files during initialization**
3. **Cascading imports between modules**
4. **Unintended TensorFlow loading**

## Recommended Optimizations

### 1. Lazy Import Pattern
Convert heavy imports to lazy loading:

```python
# Instead of:
import torch
import numpy as np

# Use:
def _get_torch():
    import torch
    return torch

# Or use lazy imports inside functions:
def process_image(self, image):
    import torch  # Import only when needed
    return torch.tensor(image)
```

### 2. Selective Node Loading
Modify `__init__.py` to load nodes on-demand:

```python
# Load only essential nodes at startup
ESSENTIAL_NODES = [
    'api_key_input',
    'generate_text',
    'prompt_manager',
    # ... core nodes only
]

# Load others when first accessed
def lazy_load_node(node_name):
    if node_name not in NODE_CLASS_MAPPINGS:
        module = importlib.import_module(node_name)
        # ... register the node
```

### 3. Import Optimization in transformers_api.py
The transformers library is already using lazy loading pattern - this is good!
Ensure other heavy modules follow the same pattern.

### 4. Defer torch/numpy imports
Move imports inside class methods:

```python
class GenerateMusic:
    def run(self, ...):
        import torch  # Import when actually used
        import torchaudio
        # ... rest of the code
```

### 5. Profile-guided optimization
Use Python's `-X importtime` flag to identify slow imports:
```bash
python -X importtime main.py 2>&1 | grep "llm-toolkit"
```

## Quick Wins (Immediate improvements)

1. **Move torch imports to function level** in these files:
   - `blank_image.py`
   - `check_image_empty.py`
   - `logic_preview_image.py`
   - `prompt_manager.py`
   - `generate_music.py`
   - `preview_outputs.py`

2. **Use conditional imports** for optional dependencies:
   ```python
   try:
       import torchaudio
       AUDIO_SUPPORT = True
   except ImportError:
       AUDIO_SUPPORT = False
   ```

3. **Cache module imports** in `__init__.py`:
   ```python
   _module_cache = {}
   
   def get_module(name):
       if name not in _module_cache:
           _module_cache[name] = importlib.import_module(name)
       return _module_cache[name]
   ```

## Expected Impact
These optimizations should reduce load time from 1.0s to approximately 0.2-0.3s, bringing it in line with other ComfyUI nodes.

## Implementation Priority
1. High: Move torch/numpy imports to function level
2. High: Implement lazy loading for non-essential nodes
3. Medium: Add import caching
4. Low: Profile and optimize remaining bottlenecks