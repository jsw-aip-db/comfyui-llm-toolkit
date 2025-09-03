import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import logging

# Ensure parent directory in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

# ------------------------- Task Loading -------------------------
_tasks_cache: Optional[Dict[str, Dict]] = None
_task_names_cache: Optional[List[str]] = None
_tasks_path: Optional[Path] = None
_load_error: Optional[str] = None


def get_tasks_path() -> Path:
    """Get the path to banana-tasks.json."""
    global _tasks_path
    if _tasks_path is None:
        project_root = Path(current_dir).parent
        _tasks_path = project_root / "presets" / "banana-tasks.json"
    return _tasks_path


def get_task_names() -> List[str]:
    """Get just the task names without loading full task data - faster for UI."""
    global _task_names_cache, _load_error
    
    if _task_names_cache is not None:
        return _task_names_cache
    
    if _load_error is not None:
        return ["Error: " + _load_error]
    
    try:
        tasks_path = get_tasks_path()
        if not tasks_path.exists():
            _load_error = "banana-tasks.json not found"
            return ["Error: " + _load_error]
        
        # Quick parse just for task names
        with open(tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _task_names_cache = [entry["task"] for entry in data if isinstance(entry, dict) and entry.get("task")]
            logger.info(f"Loaded {len(_task_names_cache)} banana task names")
            return _task_names_cache
    except Exception as e:
        _load_error = str(e)
        logger.error(f"Error loading banana task names: {e}")
        return ["Error: " + _load_error]


def load_tasks() -> Dict[str, Dict]:
    """Lazily load full banana task data only when needed."""
    global _tasks_cache, _load_error
    
    if _tasks_cache is not None:
        return _tasks_cache
    
    try:
        tasks_path = get_tasks_path()
        if not tasks_path.exists():
            raise FileNotFoundError(f"banana-tasks.json not found at {tasks_path}")
        
        with open(tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # map by task name
            _tasks_cache = {entry["task"]: entry for entry in data if isinstance(entry, dict) and entry.get("task")}
            logger.debug(f"Loaded {len(_tasks_cache)} banana tasks (full data)")
            return _tasks_cache
    except Exception as e:
        logger.error(f"Error loading banana tasks: {e}")
        _load_error = str(e)
        return {}

# ------------------------- Node Definition -------------------------


class BananaTaskGenerator:
    """ComfyUI node to generate a system prompt based on predefined banana tasks."""

    @classmethod
    def INPUT_TYPES(cls):
        # Use the optimized task name loading - doesn't load full task data
        task_names = get_task_names()
        
        return {
            "required": {
                "task": (task_names, {"default": task_names[0] if task_names else ""}),
                "output_as_text": ("BOOLEAN", {"default": False, "tooltip": "If enabled, outputs prompt as text only without adding to context"}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "system_prompt")
    FUNCTION = "generate_prompt"
    CATEGORY = "🔗llm_toolkit/prompt"

    def generate_prompt(self, task: str, output_as_text: bool = False, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        # Prepare context copy / init
        if context is None:
            output_context: Dict[str, Any] = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            output_context = {"passthrough_data": context}

        # Only load full task data when actually executing
        tasks = load_tasks()
        if not tasks:
            err = f"Error: Failed to load banana tasks."
            logger.error(err)
            if not output_as_text:
                if "provider_config" not in output_context:
                    output_context["provider_config"] = {}
                output_context["provider_config"]["system_message"] = err
            return (output_context, err)
        
        task_data = tasks.get(task)
        if not task_data:
            err = f"Error: Banana task '{task}' not found."
            logger.error(err)
            if not output_as_text:
                # ensure provider_config exists
                if "provider_config" not in output_context:
                    output_context["provider_config"] = {}
                output_context["provider_config"]["system_message"] = err
            return (output_context, err)

        system_prompt: str = task_data.get("prompt", "")

        if output_as_text:
            # When switch is ON: output prompt as text directly in prompt_config
            prompt_config = output_context.get("prompt_config", {})
            if not isinstance(prompt_config, dict):
                prompt_config = {}
            prompt_config["text"] = system_prompt
            output_context["prompt_config"] = prompt_config
            logger.info(f"BananaTaskGenerator: Generated prompt for task '{task}' as text output.")
        else:
            # When switch is OFF (default): use a dedicated banana_config to avoid conflicts
            banana_config = output_context.get("banana_config", {})
            if not isinstance(banana_config, dict):
                banana_config = {}
            
            banana_config["system_message"] = system_prompt
            
            # Optionally add meta like num_instructions
            if task_data.get("num_instructions"):
                banana_config["num_instructions"] = task_data["num_instructions"]
            
            output_context["banana_config"] = banana_config
            logger.info(f"BananaTaskGenerator: Generated prompt for task '{task}' into banana_config.")

        return (output_context, system_prompt)


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "BananaTaskGenerator": BananaTaskGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BananaTaskGenerator": "Banana System Prompt text (🔗LLMToolkit)",
}


