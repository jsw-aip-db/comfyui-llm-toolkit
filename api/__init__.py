"""
API sub-package providing namespacing for all *_api helper modules.

This file dynamically re-exports the original top-level modules under the
`api` namespace so that both of the following patterns are valid:

    from openai_api import send_openai_request  # legacy
    from api.openai_api import send_openai_request  # new preferred form

The implementation is **lazy** and incurs virtually zero overhead.
"""

import importlib
import sys
from typing import List

# ---------------------------------------------------------------------------
#  List every *_api module shipped with the toolkit.
#  Add new entries here whenever a new provider helper is introduced.
# ---------------------------------------------------------------------------
_api_modules: List[str] = [
    "bfl_api",
    "gemini_api",
    "gemini_image_api",
    "ollama_api",
    "openai_api",
    "suno_api",
    "transformers_api",
    "wavespeed_image_api",
]

_pkg_name = __name__
for _mod_name in _api_modules:
    # Prefer relative import inside this package (i.e. api.<name>)
    try:
        _module = importlib.import_module(f".{_mod_name}", package=_pkg_name)
    except ModuleNotFoundError:
        # Fall back to legacy top-level module if it still exists
        _module = importlib.import_module(_mod_name)

    # Expose under both the new and old namespaces
    sys.modules[f"{_pkg_name}.{_mod_name}"] = _module  # api.openai_api, etc.
    sys.modules[_mod_name] = _module  # openai_api, etc. (back-compat)
    setattr(sys.modules[_pkg_name], _mod_name, _module)

# Optional: make the modules discoverable via `api.__all__`
__all__ = _api_modules.copy()

# Cleanup internals to avoid leaking symbols
for _tmp in ("importlib", "sys", "List", "_pkg_name", "_api_modules", "_mod_name", "_module"):
    globals().pop(_tmp, None) 